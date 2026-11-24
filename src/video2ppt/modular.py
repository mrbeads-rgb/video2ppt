"""Modular slide element extraction: native vector cards, highlights, and movable transparent graphics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video2ppt.models import (
    BBox,
    DetectedShape,
    GraphicRegion,
    HighlightSpan,
    RGBColor,
    TextBlock,
)


def _bgr_to_rgb(bgr: np.ndarray) -> RGBColor:
    return RGBColor(
        r=int(np.clip(round(float(bgr[2])), 0, 255)),
        g=int(np.clip(round(float(bgr[1])), 0, 255)),
        b=int(np.clip(round(float(bgr[0])), 0, 255)),
    )


def detect_card_shapes(
    clean_bg_bgr: np.ndarray,
    canvas_bg: RGBColor,
    text_blocks: list[TextBlock],
) -> list[DetectedShape]:
    """Detect clean rectangular / rounded-rectangular container cards on the text-erased canvas."""
    h, w = clean_bg_bgr.shape[:2]
    slide_area = float(w * h)
    gray = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    canvas_bgr = np.array(canvas_bg.to_bgr(), dtype=np.uint8)
    canvas_lab = (
        cv2.cvtColor(canvas_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    )

    candidates: list[tuple[BBox, float, bool]] = []  # (bbox, rectangularity, has_border)

    # 1. Bordered cards (dark outlines like Slide 4, Slide 9, Slide 18)
    edges = cv2.Canny(gray, 50, 150)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < slide_area * 0.035 or area > slide_area * 0.82:
            continue
        if cw < 120 or ch < 70:
            continue
        cnt_area = cv2.contourArea(cnt)
        rect_ratio = cnt_area / max(1.0, float(area))
        if rect_ratio >= 0.80:
            candidates.append((BBox(x=float(x), y=float(y), w=float(cw), h=float(ch)), rect_ratio, True))

    # 2. Borderless tinted panels (like Slide 3 pastel pink cards or Slide 14 big pink card or Slide 16 cards)
    dist_canvas = np.linalg.norm(lab - canvas_lab[None, None, :], axis=2)
    tinted = (dist_canvas > 16.0).astype(np.uint8) * 255
    tinted = cv2.morphologyEx(tinted, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    tinted = cv2.morphologyEx(tinted, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    t_contours, _ = cv2.findContours(tinted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in t_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < slide_area * 0.04 or area > slide_area * 0.82:
            continue
        if cw < 130 or ch < 75:
            continue
        cnt_area = cv2.contourArea(cnt)
        rect_ratio = cnt_area / max(1.0, float(area))
        if rect_ratio >= 0.84:
            candidates.append((BBox(x=float(x), y=float(y), w=float(cw), h=float(ch)), rect_ratio, False))

    # Deduplicate overlapping candidates (prefer larger / higher rectangularity)
    candidates.sort(key=lambda item: item[0].area, reverse=True)
    kept: list[DetectedShape] = []
    for box, rect_ratio, has_border_hint in candidates:
        if any(box.iou(s.bbox) > 0.55 for s in kept):
            continue

        # Verify card interior has a reasonably uniform fill color (not a noisy photo/doodle cluster)
        ix0 = int(box.x + box.w * 0.12)
        iy0 = int(box.y + box.h * 0.12)
        ix1 = int(box.x2 - box.w * 0.12)
        iy1 = int(box.y2 - box.h * 0.12)
        if ix1 <= ix0 + 10 or iy1 <= iy0 + 10:
            continue

        interior = clean_bg_bgr[iy0:iy1, ix0:ix1]
        # Quantize to find dominant fill color (mode) so small doodle icons inside the card don't break detection
        flat = interior.reshape(-1, 3)
        q = (flat // 12) * 12
        vals, counts = np.unique(q, axis=0, return_counts=True)
        top_idx = int(np.argmax(counts))
        dom_ratio = float(counts[top_idx]) / max(1, len(flat))
        if dom_ratio < 0.45:
            continue

        mode_q = vals[top_idx]
        inliers = flat[np.all(np.abs(flat.astype(np.int16) - mode_q.astype(np.int16)) <= 18, axis=1)]
        fill_bgr = np.median(inliers, axis=0) if len(inliers) > 0 else np.median(flat, axis=0)
        fill_color = _bgr_to_rgb(fill_bgr)

        # Check if card actually has a visible dark border along its top/left edges
        bx0, by0 = max(0, int(box.x)), max(0, int(box.y))
        bx1, by1 = min(w, int(box.x2)), min(h, int(box.y2))
        top_edge = clean_bg_bgr[by0 : min(h, by0 + 5), bx0 + int(box.w * 0.2) : bx1 - int(box.w * 0.2)]
        border_color: RGBColor | None = None
        border_width_pt = 0.0
        if top_edge.size > 0:
            top_gray = cv2.cvtColor(top_edge, cv2.COLOR_BGR2GRAY)
            dark_mask = top_gray < 95
            if float(dark_mask.mean()) >= 0.15:
                dark_px = top_edge[dark_mask]
                border_color = _bgr_to_rgb(np.median(dark_px, axis=0))
                border_width_pt = 1.5

        # Require card to either have a border, differ visibly from canvas bg, or contain text
        fill_dist = float(
            np.linalg.norm(
                np.array(fill_color.to_tuple(), dtype=np.float32)
                - np.array(canvas_bg.to_tuple(), dtype=np.float32)
            )
        )
        contains_text = any(
            tb.bbox.cx >= box.x and tb.bbox.cx <= box.x2 and tb.bbox.cy >= box.y and tb.bbox.cy <= box.y2
            for tb in text_blocks
        )
        if border_color is None and fill_dist < 18.0:
            continue
        if not contains_text and border_color is None and fill_dist < 30.0:
            continue

        # Estimate rounded vs sharp corners
        is_rounded = rect_ratio < 0.965
        kept.append(
            DetectedShape(
                shape_type="rounded_rect" if is_rounded else "rect",
                bbox=box,
                fill_color=fill_color,
                border_color=border_color,
                border_width_pt=border_width_pt,
                corner_radius_ratio=0.08 if is_rounded else 0.0,
                z_order=len(kept),
            )
        )

    return kept


def extract_modular_elements(
    clean_bg_bgr: np.ndarray,
    canvas_bg: RGBColor,
    text_blocks: list[TextBlock],
    assets_dir: Path,
    slide_index: int,
    watermark_bboxes: list[BBox] | None = None,
) -> tuple[list[DetectedShape], list[GraphicRegion]]:
    """Decompose a text-erased slide into native vector cards, highlights, and movable RGBA graphics."""
    h, w = clean_bg_bgr.shape[:2]
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect native highlight pills from text blocks (for z-order just behind text!)
    highlights: list[HighlightSpan] = []
    for tb in text_blocks:
        highlights.extend(tb.highlights)

    # 2. Detect native card / panel shapes, skipping any duplicate box matching a highlight pill
    raw_cards = detect_card_shapes(clean_bg_bgr, canvas_bg, text_blocks)
    cards = [
        c
        for c in raw_cards
        if not any(c.bbox.iou(hl.bbox) > 0.45 for hl in highlights)
    ]

    shapes: list[DetectedShape] = list(cards)
    for hl in highlights:
        shapes.append(
            DetectedShape(
                shape_type="highlight",
                bbox=hl.bbox,
                fill_color=hl.color,
                border_color=None,
                border_width_pt=0.0,
                corner_radius_ratio=0.18,
                z_order=50,
            )
        )

    # 3. Build foreground mask of illustrations / doodles (excluding uniform canvas, grid, highlights, and card fills)
    lab = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    canvas_bgr = np.array(canvas_bg.to_bgr(), dtype=np.uint8)
    canvas_lab = (
        cv2.cvtColor(canvas_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    )

    bg_lab_map = np.broadcast_to(canvas_lab[None, None, :], (h, w, 3)).copy()
    card_border_mask = np.zeros((h, w), dtype=np.uint8)

    # Inside detected cards, replace local expected background with the card's fill color so we still
    # extract doodle icons that sit INSIDE cards (like Slide 9, Slide 14, Slide 16)!
    for card in cards:
        x0, y0 = max(0, int(card.bbox.x)), max(0, int(card.bbox.y))
        x1, y1 = min(w, int(card.bbox.x2)), min(h, int(card.bbox.y2))
        c_bgr = np.array(card.fill_color.to_bgr(), dtype=np.uint8)
        c_lab = cv2.cvtColor(c_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        bg_lab_map[y0:y1, x0:x1] = c_lab
        # Mask out only the card's own outline ring so we don't re-extract the border as a bitmap graphic
        cv2.rectangle(card_border_mask, (x0, y0), (x1 - 1, y1 - 1), 255, thickness=10)

    # Mask out highlight pills (they are already native shapes)
    hl_mask = np.zeros((h, w), dtype=np.uint8)
    for hl in highlights:
        hx0, hy0 = max(0, int(hl.bbox.x) - 4), max(0, int(hl.bbox.y) - 4)
        hx1, hy1 = min(w, int(hl.bbox.x2) + 4), min(h, int(hl.bbox.y2) + 4)
        hl_mask[hy0:hy1, hx0:hx1] = 255

    # Mask out watermarks from graphic crops
    wm_mask = np.zeros((h, w), dtype=np.uint8)
    for wb in watermark_bboxes or []:
        wx0 = max(0, int(wb.x - max(24.0, wb.h * 1.6)))
        wy0 = max(0, int(wb.y - 4))
        wx1 = min(w, int(wb.x2 + 6))
        wy1 = min(h, int(wb.y2 + 6))
        wm_mask[wy0:wy1, wx0:wx1] = 255

    dist = np.linalg.norm(lab - bg_lab_map, axis=2)
    # Slightly higher threshold (26) cleanly ignores subtle pink/purple sketch watermarks & graph-paper grids
    fg = (dist > 26.0).astype(np.uint8) * 255
    fg[card_border_mask > 0] = 0
    fg[hl_mask > 0] = 0
    fg[wm_mask > 0] = 0
    # Ignore outermost 6px video framing border bars so they never fuse disparate graphics across the slide
    fg[:6, :] = 0
    fg[h - 6 :, :] = 0
    fg[:, :6] = 0
    fg[:, w - 6 :] = 0

    # Clean tiny dust specs, then cluster only tightly-proximate icon strokes (9x9) so nearby icons
    # (e.g. 3 traffic lights on Slide 6 or top row doodles on Slide 2) stay independently movable/animatable!
    fg_clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    clustered = cv2.dilate(
        fg_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(clustered, connectivity=8)
    graphics: list[GraphicRegion] = []
    g_idx = 0

    for lbl in range(1, num_labels):
        gx = int(stats[lbl, cv2.CC_STAT_LEFT])
        gy = int(stats[lbl, cv2.CC_STAT_TOP])
        gw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        gh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        # Skip tiny fragments
        if gw < 24 or gh < 24 or area < 350:
            continue

        # Tighten bounding box back to actual non-dilated foreground pixels inside this cluster
        sub_fg = fg_clean[gy : gy + gh, gx : gx + gw]
        ys, xs = np.where(sub_fg > 0)
        if len(xs) < 100:
            continue
        pad = 4
        rx0 = max(0, gx + int(xs.min()) - pad)
        ry0 = max(0, gy + int(ys.min()) - pad)
        rx1 = min(w, gx + int(xs.max()) + pad + 1)
        ry1 = min(h, gy + int(ys.max()) + pad + 1)
        rw = rx1 - rx0
        rh = ry1 - ry0
        if rw < 22 or rh < 22:
            continue

        # Build crisp RGBA cutout: make local background transparent, preserve filled doodle interiors
        crop_bgr = clean_bg_bgr[ry0:ry1, rx0:rx1]
        crop_dist = dist[ry0:ry1, rx0:rx1]

        # Fill closed interiors of doodles (e.g. white eyes / white shirt / white gear centers!)
        # by flood-filling background from the crop's outer border!
        seed_bg = (crop_dist <= 20.0).astype(np.uint8)
        padded_bg = cv2.copyMakeBorder(seed_bg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        flood_mask = np.zeros((padded_bg.shape[0] + 2, padded_bg.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(padded_bg, flood_mask, (0, 0), 2)
        exterior_bg = padded_bg[1:-1, 1:-1] == 2

        # Soft feathered alpha edge around foreground
        alpha = np.where(exterior_bg, 0, 255).astype(np.uint8)
        soft_edge = exterior_bg & (crop_dist > 11.0)
        alpha[soft_edge] = np.clip((crop_dist[soft_edge] - 11.0) / 11.0 * 255.0, 0, 255).astype(
            np.uint8
        )

        # Ensure highlight/watermark regions remain transparent
        alpha[hl_mask[ry0:ry1, rx0:rx1] > 0] = 0
        alpha[wm_mask[ry0:ry1, rx0:rx1] > 0] = 0

        if int((alpha > 128).sum()) < 120:
            continue

        rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        g_idx += 1
        img_path = assets_dir / f"slide_{slide_index:03d}_graphic_{g_idx:02d}.png"
        cv2.imwrite(str(img_path), rgba)

        graphics.append(
            GraphicRegion(
                bbox=BBox(x=float(rx0), y=float(ry0), w=float(rw), h=float(rh)),
                image_path=str(img_path),
                z_order=20 + g_idx,
                label=f"graphic_{g_idx:02d}",
            )
        )

    return shapes, graphics
