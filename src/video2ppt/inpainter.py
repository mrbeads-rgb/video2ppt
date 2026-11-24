"""Surgical glyph-aware text and optional watermark inpainting."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video2ppt.models import BBox, DetectedTable, TextBlock, TextLine
from video2ppt.styling import extract_line_ink_mask


def _build_text_inpaint_mask(
    img_bgr: np.ndarray,
    text_lines: list[TextLine],
    tables: list[DetectedTable] | None = None,
    watermark_bboxes: list[BBox] | None = None,
    remove_watermark: bool = False,
) -> np.ndarray:
    """Build a tight glyph-level binary mask covering only text strokes (+ small halo)."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Collect every line & table cell that will be rendered natively
    targets: list[tuple[BBox, list[list[float]] | None]] = []
    for ln in text_lines:
        if ln.is_watermark:
            continue
        targets.append((ln.bbox, ln.polygon if ln.polygon else None))

    for tbl in tables or []:
        for cell in tbl.cells:
            targets.append((cell.bbox, None))

    for bbox, poly in targets:
        ink_mask, _, (x0, y0, x1, y1) = extract_line_ink_mask(img_bgr, bbox, poly)
        if ink_mask.size == 0 or int((ink_mask > 0).sum()) == 0:
            continue

        # Protect large adjacent graphic strokes: check if any component touches box edge
        # AND extends far beyond normal glyph aspect
        # Dilate tight glyph strokes by 2-3px to cleanly cover H.264 chroma & AA halos
        k_size = 7 if bbox.h >= 34 else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(ink_mask, kernel, iterations=1)

        # Clip dilated mask tightly to padded polygon/bbox so neighboring doodles stay crisp
        clip_mask = np.zeros(ink_mask.shape, dtype=np.uint8)
        if poly and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32) - np.array([x0, y0])
            cv2.fillPoly(clip_mask, [pts], 255)
            clip_mask = cv2.dilate(
                clip_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1
            )
        else:
            clip_mask[:, :] = 255

        final_roi = cv2.bitwise_and(dilated, clip_mask)
        mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], final_roi)

    if remove_watermark and watermark_bboxes:
        for wb in watermark_bboxes:
            # Expand slightly left to also cover tiny logo icon next to watermark text,
            # while staying inside outer 4px video frame border to avoid bleeding dark edges
            pad_left = max(24.0, wb.h * 1.6)
            x0 = max(0, int(np.floor(wb.x - pad_left)))
            y0 = max(0, int(np.floor(wb.y - 4)))
            x1 = min(max(x0 + 1, w - 4), int(np.ceil(wb.x2 + 6)))
            y1 = min(max(y0 + 1, h - 4), int(np.ceil(wb.y2 + 4)))
            mask[y0:y1, x0:x1] = 255

    return mask


def inpaint_slide_background(
    img_bgr: np.ndarray,
    text_blocks: list[TextBlock],
    tables: list[DetectedTable] | None = None,
    watermark_bboxes: list[BBox] | None = None,
    remove_watermark: bool = False,
    out_path: str | Path | None = None,
) -> np.ndarray:
    """Erase editable text (and optionally corner watermark) while preserving artwork & highlights."""
    # Flatten all lines across blocks
    all_lines: list[TextLine] = []
    for blk in text_blocks:
        if blk.lines:
            all_lines.extend(blk.lines)
        else:
            # Fallback for VLM backends that emit TextBlocks without per-line polygons
            all_lines.append(
                TextLine(
                    text=blk.text,
                    bbox=blk.bbox,
                    font_size_pt=blk.font_size_pt,
                    color=blk.color,
                    bold=blk.bold,
                    font_family=blk.font_family,
                )
            )

    mask = _build_text_inpaint_mask(
        img_bgr,
        text_lines=all_lines,
        tables=tables,
        watermark_bboxes=watermark_bboxes,
        remove_watermark=remove_watermark,
    )

    if int((mask > 0).sum()) == 0:
        result = img_bgr.copy()
    else:
        # Telea inpainting bridges thin glyph strokes cleanly without smearing grids or highlights
        result = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), result)

    return result
