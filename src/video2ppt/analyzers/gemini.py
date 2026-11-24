"""Pluggable Vision LLM base helper + Gemini Vision API analyzer backend."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import cv2
import httpx

from video2ppt.analyzers.base import ProgressCallback, SlideAnalyzer
from video2ppt.config import PipelineConfig
from video2ppt.inpainter import inpaint_slide_background
from video2ppt.models import (
    BBox,
    DetectedTable,
    RGBColor,
    SlideLayout,
    SlideSegment,
    TableCell,
    TextBlock,
    TextLine,
)
from video2ppt.modular import extract_modular_elements
from video2ppt.styling import analyze_line_typography, estimate_slide_background_color

VLM_EXTRACTION_PROMPT = """Analyze this presentation slide image and extract ALL readable text blocks and any structured tables so it can be rebuilt as an editable PowerPoint slide.

Return ONLY valid JSON matching this exact schema:
{
  "text_blocks": [
    {
      "text": "Exact text (use \\n for explicit multi-line paragraphs belonging to one visual block)",
      "bbox_1000": [ymin, xmin, ymax, xmax],
      "alignment": "left" | "center" | "right",
      "role": "title" | "subtitle" | "heading" | "body" | "bullet" | "caption" | "number",
      "color_hex": "#1A1A1A",
      "bold": false,
      "italic": false
    }
  ],
  "tables": [
    {
      "bbox_1000": [ymin, xmin, ymax, xmax],
      "rows": 3,
      "cols": 4,
      "cells": [
        {"row": 0, "col": 0, "text": "Header", "bold": true, "bbox_1000": [ymin, xmin, ymax, xmax]}
      ]
    }
  ],
  "watermark_bbox_1000": [ymin, xmin, ymax, xmax] | null
}

Rules:
- Coordinates in bbox_1000 are integers normalized to 0..1000 ([ymin, xmin, ymax, xmax]).
- Exclude any tiny corner platform watermark (like "Gemini Notebook") from text_blocks; put its box in watermark_bbox_1000 instead.
- Group lines that clearly form a single paragraph or multi-line heading into one text_block; keep isolated diagram labels as separate text_blocks.
- Do not include text inside tables in text_blocks.
- Preserve exact capitalization, punctuation, and bullet symbols.
"""


def _norm_box_to_px(box_1000: list[float], w: int, h: int) -> BBox:
    ymin, xmin, ymax, xmax = [float(v) for v in box_1000]
    x = max(0.0, min(float(w - 1), xmin / 1000.0 * w))
    y = max(0.0, min(float(h - 1), ymin / 1000.0 * h))
    x2 = max(x + 2.0, min(float(w), xmax / 1000.0 * w))
    y2 = max(y + 2.0, min(float(h), ymax / 1000.0 * h))
    return BBox(x=x, y=y, w=x2 - x, h=y2 - y)


def _strip_json_fences(raw: str) -> str:
    txt = raw.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    return txt


def build_slide_layout_from_vlm_json(
    payload: dict[str, Any],
    seg: SlideSegment,
    img_bgr: Any,
    work_dir: Path,
    config: PipelineConfig,
) -> SlideLayout:
    """Fuse semantic VLM JSON extraction with pixel-exact OpenCV styling, inpainting, and modular crops."""
    h, w = img_bgr.shape[:2]
    clean_bg_dir = work_dir / "clean_bg"
    assets_dir = work_dir / "modular_assets"
    clean_bg_dir.mkdir(parents=True, exist_ok=True)

    bg_color = estimate_slide_background_color(img_bgr)
    watermark_bboxes: list[BBox] = []
    wm_raw = payload.get("watermark_bbox_1000")
    if isinstance(wm_raw, list) and len(wm_raw) == 4:
        watermark_bboxes.append(_norm_box_to_px(wm_raw, w, h))

    text_blocks: list[TextBlock] = []
    all_lines: list[TextLine] = []
    for idx, item in enumerate(payload.get("text_blocks", [])):
        txt = str(item.get("text", "")).strip()
        raw_box = item.get("bbox_1000")
        if not txt or not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        bbox = _norm_box_to_px(raw_box, w, h)
        # Split multi-line block into even vertical slices for pixel-accurate typography & glyph inpainting
        sub_texts = [t.strip() for t in txt.splitlines() if t.strip()]
        n_lines = max(1, len(sub_texts))
        sub_h = bbox.h / n_lines
        block_lines: list[TextLine] = []
        for l_i, sub_txt in enumerate(sub_texts):
            lb = BBox(x=bbox.x, y=bbox.y + l_i * sub_h, w=bbox.w, h=sub_h)
            poly = [[lb.x, lb.y], [lb.x2, lb.y], [lb.x2, lb.y2], [lb.x, lb.y2]]
            style = analyze_line_typography(img_bgr, lb, poly, sub_txt, h, config)
            color = (
                RGBColor.from_hex(item["color_hex"])
                if item.get("color_hex")
                else style["color"]
            )
            ln = TextLine(
                text=sub_txt,
                bbox=lb,
                polygon=poly,
                confidence=1.0,
                font_size_pt=style["font_size_pt"],
                cap_height_px=style["cap_height_px"],
                color=color,
                bg_color=style["bg_color"],
                bold=bool(item.get("bold", style["bold"])),
                italic=bool(item.get("italic", False)),
                font_family=style["font_family"],
                highlights=style["highlights"],
            )
            block_lines.append(ln)
            all_lines.append(ln)

        font_size_pt = min(l.font_size_pt for l in block_lines)
        align = item.get("alignment", "left")
        if align not in ("left", "center", "right"):
            align = "left"
        role = item.get("role", "body")
        if role not in ("title", "subtitle", "heading", "body", "bullet", "caption", "number", "table_cell"):
            role = "body"

        text_blocks.append(
            TextBlock(
                block_id=f"tb_{idx + 1:02d}",
                text="\n".join(sub_texts),
                lines=block_lines,
                bbox=bbox,
                alignment=align,
                role=role,
                font_size_pt=font_size_pt,
                color=block_lines[0].color,
                bold=bool(item.get("bold", block_lines[0].bold)),
                italic=bool(item.get("italic", False)),
                font_family=block_lines[0].font_family,
                highlights=[hl for l in block_lines for hl in l.highlights],
            )
        )

    tables: list[DetectedTable] = []
    if config.detect_tables:
        for t_item in payload.get("tables", []):
            raw_tb = t_item.get("bbox_1000")
            if not isinstance(raw_tb, list) or len(raw_tb) != 4:
                continue
            tb_box = _norm_box_to_px(raw_tb, w, h)
            rows = max(1, int(t_item.get("rows", 1)))
            cols = max(1, int(t_item.get("cols", 1)))
            cells: list[TableCell] = []
            for c_item in t_item.get("cells", []):
                c_box_raw = c_item.get("bbox_1000")
                c_box = (
                    _norm_box_to_px(c_box_raw, w, h)
                    if isinstance(c_box_raw, list) and len(c_box_raw) == 4
                    else BBox(
                        x=tb_box.x + int(c_item.get("col", 0)) * (tb_box.w / cols),
                        y=tb_box.y + int(c_item.get("row", 0)) * (tb_box.h / rows),
                        w=tb_box.w / cols,
                        h=tb_box.h / rows,
                    )
                )
                cells.append(
                    TableCell(
                        row=int(c_item.get("row", 0)),
                        col=int(c_item.get("col", 0)),
                        text=str(c_item.get("text", "")),
                        bbox=c_box,
                        font_size_pt=13.0,
                        bold=bool(c_item.get("bold", int(c_item.get("row", 0)) == 0)),
                    )
                )
            tables.append(
                DetectedTable(
                    bbox=tb_box,
                    rows=rows,
                    cols=cols,
                    cells=cells,
                    col_widths_px=[tb_box.w / cols] * cols,
                    row_heights_px=[tb_box.h / rows] * rows,
                )
            )

    clean_bg_path = clean_bg_dir / f"slide_{seg.slide_index:03d}_clean.png"
    clean_bg_bgr = inpaint_slide_background(
        img_bgr=img_bgr,
        text_blocks=text_blocks,
        tables=tables,
        watermark_bboxes=watermark_bboxes,
        remove_watermark=config.remove_watermark,
        out_path=clean_bg_path,
    )

    shapes = []
    graphics = []
    if config.mode == "modular":
        shapes, graphics = extract_modular_elements(
            clean_bg_bgr=clean_bg_bgr,
            canvas_bg=bg_color,
            text_blocks=text_blocks,
            assets_dir=assets_dir,
            slide_index=seg.slide_index,
            watermark_bboxes=watermark_bboxes,
        )

    return SlideLayout(
        slide_index=seg.slide_index,
        start_sec=seg.start_sec,
        end_sec=seg.end_sec,
        width=w,
        height=h,
        raw_frame_path=seg.frame_path,
        clean_bg_path=str(clean_bg_path),
        background_color=bg_color,
        text_blocks=text_blocks,
        raw_lines=all_lines,
        shapes=shapes,
        graphics=graphics,
        tables=tables,
        watermark_bboxes=watermark_bboxes,
        is_outro=seg.is_outro,
    )


class GeminiVisionAnalyzer(SlideAnalyzer):
    """Optional Gemini Vision API backend (requires GEMINI_API_KEY or GOOGLE_API_KEY)."""

    name = "gemini"

    def __init__(self, config: PipelineConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Gemini backend requires GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable."
            )

    def _call_gemini(self, frame_path: str) -> dict[str, Any]:
        img_b64 = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent"
        )
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": VLM_EXTRACTION_PROMPT},
                        {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
            },
        }
        with httpx.Client(timeout=self.config.api_timeout_sec) as client:
            resp = client.post(url, params={"key": self.api_key}, json=body)
            resp.raise_for_status()
            data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(_strip_json_fences(raw_text))

    def analyze_deck(
        self,
        segments: list[SlideSegment],
        work_dir: Path,
        progress_cb: ProgressCallback | None = None,
    ) -> list[SlideLayout]:
        layouts: list[SlideLayout] = []
        total = len(segments)
        for idx, seg in enumerate(segments):
            if progress_cb:
                progress_cb(idx + 1, total, f"Gemini Vision analyzing slide {seg.slide_index}/{total}")
            img_bgr = cv2.imread(seg.frame_path)
            payload = self._call_gemini(seg.frame_path)
            layouts.append(
                build_slide_layout_from_vlm_json(payload, seg, img_bgr, work_dir, self.config)
            )
        return layouts
