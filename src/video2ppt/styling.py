"""Pixel-level typography, color, highlight, and font metric analysis."""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from video2ppt.config import PipelineConfig
from video2ppt.models import BBox, HighlightSpan, RGBColor

# Known Linux / macOS / Windows TrueType metric-equivalent font paths for width calibration
_FONT_PATH_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "Arial": [
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
    ],
    "Calibri": [
        (
            "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
            "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
        ),
    ],
    "Times New Roman": [
        (
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        ),
    ],
}


@functools.lru_cache(maxsize=64)
def _load_metric_font(font_family: str, bold: bool, size_px: int) -> ImageFont.FreeTypeFont | None:
    size = max(8, int(round(size_px)))
    candidates = _FONT_PATH_CANDIDATES.get(font_family, _FONT_PATH_CANDIDATES["Arial"])
    for reg_path, bold_path in candidates:
        p = Path(bold_path if bold else reg_path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return None


def measure_rendered_text_width_px(
    text: str, font_family: str, bold: bool, font_size_pt: float, px_per_pt: float
) -> float:
    """Estimate rendered text width in video pixels using metric-compatible TTF fonts."""
    size_px = max(8, int(round(font_size_pt * px_per_pt)))
    font = _load_metric_font(font_family, bold, size_px)
    if font is None:
        return len(text) * font_size_pt * px_per_pt * 0.56
    bbox = font.getbbox(text)
    return float(max(1, bbox[2] - bbox[0]))


def estimate_slide_background_color(img_bgr: np.ndarray) -> RGBColor:
    """Estimate dominant canvas background color from image borders and low-gradient regions."""
    h, w = img_bgr.shape[:2]
    # Sample border strips + uniform interior grid
    samples = np.concatenate(
        [
            img_bgr[0 : max(2, h // 40), :].reshape(-1, 3),
            img_bgr[h - max(2, h // 40) : h, :].reshape(-1, 3),
            img_bgr[:, 0 : max(2, w // 40)].reshape(-1, 3),
            img_bgr[:, w - max(2, w // 40) : w].reshape(-1, 3),
        ],
        axis=0,
    )
    # Quantize to 8-level bins to find true mode (resistant to doodles on edges)
    q = (samples // 8) * 8
    vals, counts = np.unique(q, axis=0, return_counts=True)
    mode_q = vals[int(np.argmax(counts))]
    inliers = samples[np.all(np.abs(samples.astype(np.int16) - mode_q.astype(np.int16)) <= 12, axis=1)]
    med = np.median(inliers, axis=0) if len(inliers) > 0 else np.median(samples, axis=0)
    return RGBColor(r=int(round(med[2])), g=int(round(med[1])), b=int(round(med[0])))


def extract_line_ink_mask(
    img_bgr: np.ndarray,
    bbox: BBox,
    polygon: list[list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Segment exact glyph ink pixels inside a text line region.

    Uses per-column local background estimation so text crossing highlight pills
    or pastel backgrounds is segmented cleanly without erasing the highlight itself.

    Returns:
        (ink_mask_uint8, col_bg_bgr, (x0, y0, x1, y1))
    """
    h, w = img_bgr.shape[:2]
    x0 = max(0, int(np.floor(bbox.x)) - 2)
    y0 = max(0, int(np.floor(bbox.y)) - 2)
    x1 = min(w, int(np.ceil(bbox.x2)) + 2)
    y1 = min(h, int(np.ceil(bbox.y2)) + 2)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 3), dtype=np.uint8), (x0, y0, x1, y1)

    roi_bgr = img_bgr[y0:y1, x0:x1]
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Per-column local background from top & bottom 2 rows of ROI (plus 2px outside if available)
    top_y0 = max(0, y0 - 2)
    bot_y1 = min(h, y1 + 2)
    top_strip = cv2.cvtColor(img_bgr[top_y0 : y0 + 2, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    bot_strip = cv2.cvtColor(img_bgr[max(0, y1 - 2) : bot_y1, x0:x1], cv2.COLOR_BGR2LAB).astype(
        np.float32
    )
    tb_lab = np.vstack([top_strip, bot_strip])
    col_bg_lab = np.median(tb_lab, axis=0)  # (rw, 3)

    top_bgr = img_bgr[top_y0 : y0 + 2, x0:x1]
    bot_bgr = img_bgr[max(0, y1 - 2) : bot_y1, x0:x1]
    col_bg_bgr = np.median(np.vstack([top_bgr, bot_bgr]), axis=0).astype(np.uint8)

    # Also compute overall box border median as fallback if a doodle touches one column's top/bottom
    box_border = np.vstack([roi_lab[0:2, :, :].reshape(-1, 3), roi_lab[-2:, :, :].reshape(-1, 3)])
    med_bg_lab = np.median(box_border, axis=0)

    # Distance from per-column bg
    dist = np.linalg.norm(roi_lab - col_bg_lab[None, :, :], axis=2)

    # Adaptive threshold: detect high-contrast ink even for pastel text (like pink numbers '1'..'5')
    p95 = float(np.percentile(dist, 95))
    thresh = max(11.0, min(24.0, p95 * 0.38))
    ink = (dist > thresh).astype(np.uint8) * 255

    # Clip to polygon (slightly padded) if provided
    if polygon and len(polygon) >= 3:
        poly_mask = np.zeros(ink.shape, dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32) - np.array([x0, y0])
        cv2.fillPoly(poly_mask, [pts], 255)
        poly_mask = cv2.dilate(
            poly_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
        )
        ink = cv2.bitwise_and(ink, poly_mask)

    # Protect adjacent colored doodles that happen to graze the edge of the box:
    # Find dominant ink color from interior (middle 60% vertically) ink pixels
    rh, rw = ink.shape
    core_ink = ink[int(rh * 0.18) : max(int(rh * 0.82), int(rh * 0.18) + 1), :] > 0
    if int(core_ink.sum()) >= 12:
        core_pixels = roi_lab[int(rh * 0.18) : max(int(rh * 0.82), int(rh * 0.18) + 1), :][core_ink]
        # Take the strongest-contrast half of core ink pixels to avoid antialiasing blur
        core_dists = np.linalg.norm(core_pixels - med_bg_lab[None, :], axis=1)
        strong_thresh = float(np.percentile(core_dists, 45))
        strong_px = core_pixels[core_dists >= strong_thresh]
        ink_lab = np.median(strong_px if len(strong_px) > 5 else core_pixels, axis=0)

        # Keep pixels that are reasonably close to the line's ink color OR intermediate antialiasing
        # between ink_lab and col_bg_lab (rejects bright orange gears grazing black text!)
        to_ink = np.linalg.norm(roi_lab - ink_lab[None, None, :], axis=2)
        bg_to_ink = np.linalg.norm(col_bg_lab - ink_lab[None, :], axis=1)  # (rw,)
        # Project pixel onto segment [col_bg_lab, ink_lab]
        vec_bi = ink_lab[None, None, :] - col_bg_lab[None, :, :]
        vec_bp = roi_lab - col_bg_lab[None, :, :]
        norm_sq = np.maximum(1.0, np.sum(vec_bi * vec_bi, axis=2))
        proj_t = np.sum(vec_bp * vec_bi, axis=2) / norm_sq
        perp = np.linalg.norm(vec_bp - np.clip(proj_t, 0.0, 1.2)[:, :, None] * vec_bi, axis=2)
        color_consistent = (to_ink < np.maximum(38.0, bg_to_ink[None, :] * 0.65)) | (
            (proj_t > 0.28) & (perp < 28.0)
        )
        ink = np.where(color_consistent, ink, 0).astype(np.uint8)

    return ink, col_bg_bgr, (x0, y0, x1, y1)


def detect_highlights_for_line(
    img_bgr: np.ndarray,
    bbox: BBox,
    ink_mask: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[HighlightSpan]:
    """Detect colored highlighter rectangles (e.g. yellow markers) directly behind text."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = roi_coords
    rw = x1 - x0
    rh = y1 - y0
    if rw < 10 or rh < 10:
        return []

    # Sample outer ring well outside the text box to know the surrounding card/canvas color
    pad_y = max(8, int(rh * 0.45))
    pad_x = max(10, int(rh * 0.5))
    oy0, oy1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    ox0, ox1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    outer_patch = img_bgr[oy0:oy1, ox0:ox1]
    outer_mask = np.ones((oy1 - oy0, ox1 - ox0), dtype=bool)
    outer_mask[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    outer_px = outer_patch[outer_mask]
    if len(outer_px) < 20:
        return []

    outer_hsv = cv2.cvtColor(outer_px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    outer_lab = cv2.cvtColor(outer_px.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(
        np.float32
    )
    outer_bg_lab = np.median(outer_lab, axis=0)
    outer_sat = float(np.median(outer_hsv[:, 1]))

    # Examine non-ink pixels inside the text box
    roi_bgr = img_bgr[y0:y1, x0:x1]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    non_ink = cv2.dilate(ink_mask, np.ones((3, 3), np.uint8), iterations=1) == 0
    dist_from_outer = np.linalg.norm(roi_lab - outer_bg_lab[None, None, :], axis=2)
    # A highlight pill is saturated, bright-ish, and significantly distinct from surrounding canvas
    hl_px = (
        non_ink
        & (dist_from_outer > 32.0)
        & (roi_hsv[:, :, 1] > max(70.0, outer_sat + 35.0))
        & (roi_hsv[:, :, 2] > 140)
    )

    # Check column coverage of highlight pixels
    col_ratio = hl_px.mean(axis=0)
    hl_cols = col_ratio > 0.32
    if not np.any(hl_cols):
        return []

    # Find contiguous horizontal spans (allow tiny 4px gaps for dense letters)
    hl_cols_u8 = hl_cols.astype(np.uint8).reshape(1, -1)
    hl_cols_closed = cv2.morphologyEx(
        hl_cols_u8, cv2.MORPH_CLOSE, np.ones((1, max(7, int(rh * 0.25))), np.uint8)
    ).ravel() > 0

    spans: list[HighlightSpan] = []
    in_span = False
    s_col = 0
    for c in range(rw + 1):
        val = bool(hl_cols_closed[c]) if c < rw else False
        if val and not in_span:
            s_col = c
            in_span = True
        elif not val and in_span:
            in_span = False
            if (c - s_col) >= max(14, int(rh * 0.45)):
                # Measure exact vertical extent of this highlight pill (may extend slightly outside box)
                ey0, ey1 = max(0, y0 - 12), min(h, y1 + 12)
                ex0, ex1 = max(0, x0 + s_col - 6), min(w, x0 + c + 6)
                sub_bgr = img_bgr[ey0:ey1, ex0:ex1]
                sub_hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
                sub_lab = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                # Representative color of highlight inside the span
                span_hl_pixels = roi_bgr[:, s_col:c][hl_px[:, s_col:c]]
                if len(span_hl_pixels) < 10:
                    continue
                hl_med_bgr = np.median(span_hl_pixels, axis=0)
                hl_med_lab = cv2.cvtColor(
                    hl_med_bgr.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2LAB
                ).reshape(3).astype(np.float32)
                sub_match = (
                    (np.linalg.norm(sub_lab - hl_med_lab[None, None, :], axis=2) < 28.0)
                    & (sub_hsv[:, :, 1] > 60)
                )
                ys, xs = np.where(sub_match)
                if len(xs) >= 30:
                    hb = BBox(
                        x=float(ex0 + int(xs.min())),
                        y=float(ey0 + int(ys.min())),
                        w=float(int(xs.max()) - int(xs.min()) + 1),
                        h=float(int(ys.max()) - int(ys.min()) + 1),
                    )
                    spans.append(
                        HighlightSpan(
                            bbox=hb,
                            color=RGBColor(
                                r=int(round(hl_med_bgr[2])),
                                g=int(round(hl_med_bgr[1])),
                                b=int(round(hl_med_bgr[0])),
                            ),
                        )
                    )
    return spans


def analyze_line_typography(
    img_bgr: np.ndarray,
    bbox: BBox,
    polygon: list[list[float]],
    text: str,
    slide_height_px: int,
    config: PipelineConfig,
    slide_height_pt: float = 540.0,
) -> dict:
    """Measure font size (pt), cap height, color, bg color, bold, font family, and highlights."""
    ink_mask, col_bg_bgr, roi_coords = extract_line_ink_mask(img_bgr, bbox, polygon)
    x0, y0, x1, y1 = roi_coords
    roi_bgr = img_bgr[y0:y1, x0:x1]
    pt_per_px = slide_height_pt / max(1.0, float(slide_height_px))
    px_per_pt = 1.0 / pt_per_px

    # Background color
    med_bg_bgr = np.median(col_bg_bgr, axis=0) if len(col_bg_bgr) > 0 else np.array([255, 255, 255])
    bg_color = RGBColor(
        r=int(round(med_bg_bgr[2])),
        g=int(round(med_bg_bgr[1])),
        b=int(round(med_bg_bgr[0])),
    )

    # Ink color (median of top 50% strongest contrast ink pixels)
    ink_bool = ink_mask > 0
    if int(ink_bool.sum()) >= 8:
        ink_pixels = roi_bgr[ink_bool]
        dists = np.linalg.norm(
            ink_pixels.astype(np.float32) - med_bg_bgr.astype(np.float32)[None, :], axis=1
        )
        top_half = ink_pixels[dists >= np.percentile(dists, 50)]
        med_ink_bgr = np.median(top_half if len(top_half) > 3 else ink_pixels, axis=0)
        color = RGBColor(
            r=int(round(med_ink_bgr[2])),
            g=int(round(med_ink_bgr[1])),
            b=int(round(med_ink_bgr[0])),
        )
    else:
        color = RGBColor(r=26, g=26, b=26)

    # Measure true glyph vertical span (cap height) from connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    comp_heights: list[int] = []
    for lbl in range(1, num_labels):
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        # Exclude tiny dots (i-dots, periods) and wide non-letter borders
        if area >= 8 and ch >= max(5, int(bbox.h * 0.28)) and cw < max(12, int(bbox.h * 1.8)):
            comp_heights.append(ch)

    has_caps_or_asc = any(ch.isupper() or ch.isdigit() or ch in "bdfhklt?!" for ch in text)
    has_descenders = any(ch in "gjpqy" for ch in text)

    if comp_heights:
        # 78th percentile height of letter components approximates cap/ascender height
        cap_h_px = float(np.percentile(comp_heights, 78))
        if not has_caps_or_asc:
            # Pure lowercase x-height (~0.72 of cap height)
            cap_h_px = cap_h_px / 0.72
        # Never let noisy tall blobs exceed realistic cap height inside the OCR box
        cap_h_px = min(cap_h_px, bbox.h * (0.68 if has_descenders else 0.76))
    else:
        factor = 0.60 if has_descenders else 0.70
        cap_h_px = bbox.h * factor

    # Convert cap height in pixels to typographic points (cap height ~ 0.72 * em in Arial/Calibri)
    raw_font_pt = (cap_h_px / 0.72) * pt_per_px
    # Hard safety cap from OCR box height so headings never overshoot vertically
    raw_font_pt = min(raw_font_pt, bbox.h * 0.86 * pt_per_px)

    # Boldness estimation via stroke width / reference cap-height ratio (distance transform)
    bold = False
    if int(ink_bool.sum()) >= 20 and cap_h_px >= 8.0:
        dist_tf = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 3)
        fg_dist = dist_tf[ink_bool]
        if len(fg_dist) > 10:
            ref_h = max(10.0, cap_h_px, bbox.h * 0.62)
            stroke_w = float(np.percentile(fg_dist, 85)) * 2.0
            stroke_ratio = stroke_w / ref_h
            # Regular sans-serif is ~0.09-0.145; genuinely bold headings are >= 0.168
            bold = stroke_ratio >= 0.168

    # Font family hint (display serif vs clean sans-serif)
    font_family = config.default_sans_font
    if cap_h_px >= 38.0 and len(text) >= 8:
        sobel_x = np.abs(cv2.Sobel(ink_mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)).sum()
        sobel_y = np.abs(cv2.Sobel(ink_mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)).sum()
        if sobel_y > 0 and (sobel_x / sobel_y) > 1.52:
            font_family = config.default_serif_font

    # Calibrate font_size_pt strictly against metric-compatible TTF rendered width so text
    # never overflows into nearby graphics or wraps mid-word on narrow labels
    est_w = measure_rendered_text_width_px(text, font_family, bold, raw_font_pt, px_per_pt)
    if est_w > bbox.w * 0.97 and len(text.strip()) >= 3:
        scale = (bbox.w * 0.95) / est_w
        raw_font_pt *= max(0.58, scale)
    elif est_w < bbox.w * 0.72 and len(text.strip()) >= 5:
        scale = min(1.08, (bbox.w * 0.84) / max(1.0, est_w))
        raw_font_pt *= scale

    font_size_pt = round(max(7.5, min(96.0, raw_font_pt)) * 2.0) / 2.0  # nearest 0.5pt

    # Detect yellow / colored highlight spans behind text
    highlights = detect_highlights_for_line(img_bgr, bbox, ink_mask, roi_coords)

    return {
        "font_size_pt": font_size_pt,
        "cap_height_px": round(cap_h_px, 2),
        "color": color,
        "bg_color": bg_color,
        "bold": bold,
        "italic": False,
        "font_family": font_family,
        "highlights": highlights,
    }
