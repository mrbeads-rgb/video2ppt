"""100% local, zero-API-key slide analyzer powered by RapidOCR + OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from video2ppt.analyzers.base import ProgressCallback, SlideAnalyzer
from video2ppt.config import PipelineConfig
from video2ppt.grouper import detect_tables_and_recover_cells, group_lines_into_blocks
from video2ppt.inpainter import inpaint_slide_background
from video2ppt.models import BBox, SlideLayout, SlideSegment, TextLine
from video2ppt.modular import extract_modular_elements
from video2ppt.ocr_post import (
    detect_deck_is_latin,
    is_doodle_false_positive,
    mark_recurring_watermarks,
    merge_same_line_fragments,
    normalize_ocr_text,
    recover_leading_bullet,
    recover_trailing_punctuation,
    restore_missing_word_spaces,
    upscale_re_recognize_small_line,
)
from video2ppt.styling import analyze_line_typography, estimate_slide_background_color


class LocalSlideAnalyzer(SlideAnalyzer):
    """Local extraction backend using RapidOCR (ONNX) + OpenCV styling & surgical inpainting."""

    name = "local"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._ocr = RapidOCR(
            det_limit_side_len=1280,
            det_box_thresh=0.35,
            det_unclip_ratio=1.8,
        )

    def _recognize_crop(self, crop_bgr: np.ndarray) -> tuple[str, float] | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        res, _ = self._ocr(crop_bgr, use_det=False, use_cls=False, use_rec=True)
        if not res:
            return None
        first = res[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return str(first[0]), float(first[1])
        return None

    def analyze_deck(
        self,
        segments: list[SlideSegment],
        work_dir: Path,
        progress_cb: ProgressCallback | None = None,
    ) -> list[SlideLayout]:
        clean_bg_dir = work_dir / "clean_bg"
        assets_dir = work_dir / "modular_assets"
        clean_bg_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: Run OCR across all slides so we have deck-wide statistics (language, AI mentions, watermarks)
        raw_per_slide: list[tuple[SlideSegment, np.ndarray, list[tuple[BBox, list[list[float]], str, float]]]] = []
        raw_text_pairs: list[list[tuple[str, float]]] = []

        total = len(segments)
        for idx, seg in enumerate(segments):
            if progress_cb:
                progress_cb(idx + 1, total, f"OCR scanning slide {seg.slide_index}/{total}")
            img_bgr = cv2.imread(seg.frame_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read extracted slide frame: {seg.frame_path}")

            ocr_out, _ = self._ocr(img_bgr)
            items: list[tuple[BBox, list[list[float]], str, float]] = []
            text_pairs: list[tuple[str, float]] = []
            for box, txt, conf in ocr_out or []:
                c = float(conf)
                t = str(txt).strip()
                poly = [[float(pt[0]), float(pt[1])] for pt in box]
                bbox = BBox.from_polygon(poly)
                items.append((bbox, poly, t, c))
                text_pairs.append((t, c))
            raw_per_slide.append((seg, img_bgr, items))
            raw_text_pairs.append(text_pairs)

        deck_is_latin = detect_deck_is_latin(raw_text_pairs)
        deck_mentions_ai = any(
            "AI" in txt or "ai-" in txt.lower()
            for slide_pairs in raw_text_pairs
            for txt, conf in slide_pairs
            if conf >= 0.85
        )

        # Pass 2: Filter false positives, merge same-line overlaps, recover punctuation & bullets, measure typography
        lines_per_slide: list[list[TextLine]] = []
        slide_dims = (1280, 720)

        for idx, (seg, img_bgr, raw_items) in enumerate(raw_per_slide):
            h, w = img_bgr.shape[:2]
            slide_dims = (w, h)
            if progress_cb:
                progress_cb(idx + 1, total, f"Analyzing typography on slide {seg.slide_index}/{total}")

            # Filter confidence & doodle hallucinations
            filtered = [
                it
                for it in raw_items
                if it[3] >= self.config.ocr_min_confidence
                and not is_doodle_false_positive(
                    it[2], it[3], it[0], deck_is_latin=deck_is_latin, lang=self.config.lang
                )
            ]

            # Merge overlapping / fragmented same-line detections
            if self.config.ocr_postprocess:
                filtered = merge_same_line_fragments(img_bgr, filtered, self._recognize_crop)

            # Build rich styled TextLines
            styled_lines: list[TextLine] = []
            for bbox, poly, txt, conf in filtered:
                if self.config.ocr_postprocess:
                    txt = upscale_re_recognize_small_line(img_bgr, bbox, txt, self._recognize_crop)
                    bbox, poly, txt = recover_leading_bullet(img_bgr, bbox, poly, txt)
                    bbox, poly, txt = recover_trailing_punctuation(
                        img_bgr, bbox, poly, txt, self._recognize_crop
                    )
                    txt = restore_missing_word_spaces(img_bgr, bbox, txt, self._recognize_crop)
                    txt = normalize_ocr_text(txt, deck_mentions_ai=deck_mentions_ai)

                if not txt.strip():
                    continue

                style = analyze_line_typography(
                    img_bgr=img_bgr,
                    bbox=bbox,
                    polygon=poly,
                    text=txt,
                    slide_height_px=h,
                    config=self.config,
                )
                styled_lines.append(
                    TextLine(
                        text=txt,
                        bbox=bbox,
                        polygon=poly,
                        confidence=round(conf, 4),
                        font_size_pt=style["font_size_pt"],
                        cap_height_px=style["cap_height_px"],
                        color=style["color"],
                        bg_color=style["bg_color"],
                        bold=style["bold"],
                        italic=style["italic"],
                        font_family=style["font_family"],
                        highlights=style["highlights"],
                    )
                )

            lines_per_slide.append(styled_lines)

        # Deck-wide recurring corner watermark detection
        mark_recurring_watermarks(lines_per_slide, width=slide_dims[0], height=slide_dims[1])

        # Pass 3: Table detection, paragraph grouping, surgical inpainting, and modular extraction
        layouts: list[SlideLayout] = []
        for idx, ((seg, img_bgr, _), styled_lines) in enumerate(zip(raw_per_slide, lines_per_slide)):
            h, w = img_bgr.shape[:2]
            if progress_cb:
                progress_cb(idx + 1, total, f"Reconstructing slide {seg.slide_index}/{total}")

            bg_color = estimate_slide_background_color(img_bgr)
            watermark_bboxes = [ln.bbox for ln in styled_lines if ln.is_watermark]

            # Detect structured tables (with missing-cell recovery!)
            tables = []
            consumed: set[int] = set()
            if self.config.detect_tables:
                tables, consumed = detect_tables_and_recover_cells(
                    img_bgr, styled_lines, recognize_fn=self._recognize_crop
                )

            # Group remaining lines into editable multi-line TextBlocks
            text_blocks = group_lines_into_blocks(
                img_bgr=img_bgr,
                lines=styled_lines,
                slide_width=w,
                slide_height=h,
                consumed_indices=consumed,
            )

            # Surgically inpaint text (and watermark if requested)
            clean_bg_path = clean_bg_dir / f"slide_{seg.slide_index:03d}_clean.png"
            clean_bg_bgr = inpaint_slide_background(
                img_bgr=img_bgr,
                text_blocks=text_blocks,
                tables=tables,
                watermark_bboxes=watermark_bboxes,
                remove_watermark=self.config.remove_watermark,
                out_path=clean_bg_path,
            )

            # In modular mode, decompose clean canvas into native cards, highlights, and transparent RGBA graphics
            shapes = []
            graphics = []
            if self.config.mode == "modular":
                shapes, graphics = extract_modular_elements(
                    clean_bg_bgr=clean_bg_bgr,
                    canvas_bg=bg_color,
                    text_blocks=text_blocks,
                    assets_dir=assets_dir,
                    slide_index=seg.slide_index,
                    watermark_bboxes=watermark_bboxes,
                )

            layouts.append(
                SlideLayout(
                    slide_index=seg.slide_index,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    width=w,
                    height=h,
                    raw_frame_path=seg.frame_path,
                    clean_bg_path=str(clean_bg_path),
                    background_color=bg_color,
                    text_blocks=text_blocks,
                    raw_lines=styled_lines,
                    shapes=shapes,
                    graphics=graphics,
                    tables=tables,
                    watermark_bboxes=watermark_bboxes,
                    is_outro=seg.is_outro,
                )
            )

        return layouts
