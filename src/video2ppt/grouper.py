"""Layout grouping: table grid detection, missing cell recovery, and paragraph TextBlock clustering."""

from __future__ import annotations

import cv2
import numpy as np

from video2ppt.models import (
    BBox,
    DetectedTable,
    HighlightSpan,
    RGBColor,
    TableCell,
    TextAlignment,
    TextBlock,
    TextLine,
    TextRole,
)
from video2ppt.ocr_post import RecognizeFn


def _cluster_1d(values: list[float], tol: float) -> list[float]:
    """Simple 1D single-linkage clustering returning cluster centers."""
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if abs(v - np.mean(clusters[-1])) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def detect_tables_and_recover_cells(
    img_bgr: np.ndarray,
    lines: list[TextLine],
    recognize_fn: RecognizeFn | None = None,
) -> tuple[list[DetectedTable], set[int]]:
    """Detect tabular text grids (e.g. headers + aligned numeric/text rows) and recover missed cells.

    Returns:
        (detected_tables, indices_of_lines_consumed_by_tables)
    """
    if len(lines) < 6:
        return [], set()

    # Candidate table cells are compact single-line entries (short text, small/medium height)
    cands = [
        (idx, ln)
        for idx, ln in enumerate(lines)
        if not ln.is_watermark and len(ln.text.strip()) <= 26 and ln.bbox.h <= 42
    ]
    if len(cands) < 6:
        return [], set()

    # Group candidates into horizontal rows by y-center
    row_centers = _cluster_1d([ln.bbox.cy for _, ln in cands], tol=18.0)
    rows_map: dict[int, list[tuple[int, TextLine]]] = {r: [] for r in range(len(row_centers))}
    for idx, ln in cands:
        r_idx = int(np.argmin([abs(ln.bbox.cy - rc) for rc in row_centers]))
        if abs(ln.bbox.cy - row_centers[r_idx]) <= 18.0:
            rows_map[r_idx].append((idx, ln))

    # Keep rows having >= 3 cells spaced horizontally
    multi_rows = [
        (r_idx, sorted(items, key=lambda it: it[1].bbox.x))
        for r_idx, items in rows_map.items()
        if len(items) >= 3
    ]
    if len(multi_rows) < 2:
        return [], set()

    # Check if at least 2 multi-cell rows share >= 2 aligned column x-starts
    # Group compatible rows into table candidates
    all_table_items: list[tuple[int, int, TextLine]] = []  # (row_rank, line_idx, line)
    multi_rows.sort(key=lambda r: row_centers[r[0]])

    # Build column clusters from x-starts across all multi-rows
    all_xs = [ln.bbox.x for _, items in multi_rows for _, ln in items]
    col_centers = _cluster_1d(all_xs, tol=28.0)

    # Keep columns that appear in >= 2 distinct multi-rows
    col_row_counts: dict[int, set[int]] = {c: set() for c in range(len(col_centers))}
    for r_pos, (_, items) in enumerate(multi_rows):
        for _, ln in items:
            c_idx = int(np.argmin([abs(ln.bbox.x - cc) for cc in col_centers]))
            if abs(ln.bbox.x - col_centers[c_idx]) <= 28.0:
                col_row_counts[c_idx].add(r_pos)

    valid_cols = sorted([c for c, rset in col_row_counts.items() if len(rset) >= 2])
    if len(valid_cols) < 3:
        return [], set()

    # Also keep rows that align with at least 2 valid_cols and have consistent vertical spacing
    valid_col_xs = [col_centers[c] for c in valid_cols]
    selected_rows: list[tuple[float, list[tuple[int, TextLine]]]] = []
    for r_idx, items in multi_rows:
        aligned = []
        for l_idx, ln in items:
            c_dists = [abs(ln.bbox.x - cx) for cx in valid_col_xs]
            if min(c_dists) <= 32.0:
                aligned.append((l_idx, ln))
        if len(aligned) >= 2:
            selected_rows.append((row_centers[r_idx], aligned))

    if len(selected_rows) < 2:
        return [], set()

    # Verify rows are spatially cohesive (not unrelated labels at top and bottom of slide)
    y_gaps = [
        selected_rows[k + 1][0] - selected_rows[k][0] for k in range(len(selected_rows) - 1)
    ]
    if max(y_gaps) > 220.0:
        return [], set()

    # Reject multi-card layouts (e.g. separate bordered cards side-by-side) from table detection:
    # 1) Require at least 2 numeric/percentage data cells in non-header rows (row >= 1)
    body_lines = [ln for _, aligned in selected_rows[1:] for _, ln in aligned]
    body_numeric = sum(1 for ln in body_lines if any(ch.isdigit() for ch in ln.text))
    if body_numeric < 2:
        return [], set()

    # 2) Reject if heavy dark vertical card outlines separate adjacent columns
    h_img, w_img = img_bgr.shape[:2]
    top_y = max(0, int(min(ln.bbox.y for _, aligned in selected_rows for _, ln in aligned)))
    bot_y = min(h_img, int(max(ln.bbox.y2 for _, aligned in selected_rows for _, ln in aligned)))
    if bot_y > top_y + 20:
        dark_separators = 0
        for c_i in range(len(valid_col_xs) - 1):
            sx0 = max(0, int(valid_col_xs[c_i] + 35))
            sx1 = min(w_img, int(valid_col_xs[c_i + 1] - 5))
            if sx1 > sx0 + 4:
                strip = cv2.cvtColor(img_bgr[top_y:bot_y, sx0:sx1], cv2.COLOR_BGR2GRAY)
                # Separate cards have 2 parallel dark border columns spanning most of the card height
                col_dark_ratio = (strip < 90).mean(axis=0)
                if int((col_dark_ratio > 0.65).sum()) >= 2:
                    dark_separators += 1
        if dark_separators >= len(valid_col_xs) - 1:
            return [], set()

    table_lines = [ln for _, aligned in selected_rows for _, ln in aligned]

    # Build grid matrix and recover any missing cells (like single digit '9' in Slide 11!)
    n_rows = len(selected_rows)
    n_cols = len(valid_col_xs)
    grid: dict[tuple[int, int], TextLine] = {}
    consumed: set[int] = set()
    for r_i, (_, aligned) in enumerate(selected_rows):
        for l_idx, ln in aligned:
            c_i = int(np.argmin([abs(ln.bbox.x - cx) for cx in valid_col_xs]))
            if (r_i, c_i) not in grid:
                grid[(r_i, c_i)] = ln
                consumed.add(l_idx)

    # Compute average row heights & column extents
    row_ys = [ry for ry, _ in selected_rows]
    avg_line_h = float(np.median([ln.bbox.h for ln in table_lines]))
    avg_font_pt = float(np.median([ln.font_size_pt for ln in table_lines]))
    h_img, w_img = img_bgr.shape[:2]

    # Missing cell recovery
    if recognize_fn is not None:
        for r_i in range(n_rows):
            for c_i in range(n_cols):
                if (r_i, c_i) in grid:
                    continue
                # Don't probe empty top-left corner cell if row 0 is a header row starting at col 1
                cx = valid_col_xs[c_i]
                cy = row_ys[r_i]
                probe_w = (
                    (valid_col_xs[c_i + 1] - cx) * 0.75
                    if c_i + 1 < n_cols
                    else max(65.0, avg_line_h * 3.0)
                )
                px0 = max(0, int(cx - 8))
                py0 = max(0, int(cy - avg_line_h * 0.75))
                px1 = min(w_img, int(cx + probe_w))
                py1 = min(h_img, int(cy + avg_line_h * 0.75))
                if px1 <= px0 + 8 or py1 <= py0 + 8:
                    continue
                crop = img_bgr[py0:py1, px0:px1]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                bg = float(np.median(gray))
                ink_pixels = int((np.abs(gray.astype(np.float32) - bg) > 38.0).sum())
                if ink_pixels < 12:
                    continue
                # Tighten crop to ink bounding box
                ys, xs = np.where(np.abs(gray.astype(np.float32) - bg) > 38.0)
                ix0 = max(0, int(xs.min()) - 5)
                iy0 = max(0, int(ys.min()) - 4)
                ix1 = min(crop.shape[1], int(xs.max()) + 6)
                iy1 = min(crop.shape[0], int(ys.max()) + 5)
                tight = crop[iy0:iy1, ix0:ix1]
                rec = recognize_fn(tight)
                if rec is not None and rec[0].strip() and rec[1] >= 0.85:
                    rec_txt = rec[0].strip()
                    if len(rec_txt) <= 18 and any(ch.isalnum() for ch in rec_txt):
                        recovered_box = BBox(
                            x=float(px0 + ix0),
                            y=float(py0 + iy0),
                            w=float(max(8, ix1 - ix0)),
                            h=float(max(10, iy1 - iy0)),
                        )
                        grid[(r_i, c_i)] = TextLine(
                            text=rec_txt,
                            bbox=recovered_box,
                            polygon=[
                                [recovered_box.x, recovered_box.y],
                                [recovered_box.x2, recovered_box.y],
                                [recovered_box.x2, recovered_box.y2],
                                [recovered_box.x, recovered_box.y2],
                            ],
                            confidence=rec[1],
                            font_size_pt=avg_font_pt,
                            color=table_lines[0].color,
                            font_family=table_lines[0].font_family,
                        )

    # Assemble DetectedTable
    all_cells_boxes = [ln.bbox for ln in grid.values()]
    tbl_bbox = BBox.union(all_cells_boxes).expanded(10, 8, w_img, h_img)

    col_widths: list[float] = []
    for c_i in range(n_cols):
        if c_i + 1 < n_cols:
            col_widths.append(max(40.0, valid_col_xs[c_i + 1] - valid_col_xs[c_i]))
        else:
            max_x2 = max((ln.bbox.x2 for (r, c), ln in grid.items() if c == c_i), default=tbl_bbox.x2)
            col_widths.append(max(50.0, max_x2 - valid_col_xs[c_i] + 16.0))

    row_heights: list[float] = []
    for r_i in range(n_rows):
        if r_i + 1 < n_rows:
            row_heights.append(max(24.0, row_ys[r_i + 1] - row_ys[r_i]))
        else:
            row_heights.append(max(28.0, avg_line_h * 1.8))

    cells: list[TableCell] = []
    for (r_i, c_i), ln in sorted(grid.items()):
        cells.append(
            TableCell(
                row=r_i,
                col=c_i,
                text=ln.text,
                bbox=ln.bbox,
                font_size_pt=ln.font_size_pt,
                color=ln.color,
                bold=ln.bold or (r_i == 0),
                alignment="left",
            )
        )

    table = DetectedTable(
        bbox=tbl_bbox,
        rows=n_rows,
        cols=n_cols,
        cells=cells,
        col_widths_px=col_widths,
        row_heights_px=row_heights,
    )
    return [table], consumed


def _has_barrier_between(img_bgr: np.ndarray, upper: BBox, lower: BBox) -> bool:
    """Return True if a dark horizontal border line or sharp background shift separates two lines."""
    h_img, w_img = img_bgr.shape[:2]
    y0 = int(min(upper.y2, lower.y))
    y1 = int(max(upper.y2, lower.y))
    x0 = max(0, int(max(upper.x, lower.x)))
    x1 = min(w_img, int(min(upper.x2, lower.x2)))
    if y1 <= y0 + 2 or x1 <= x0 + 8:
        return False

    strip = img_bgr[max(0, y0) : min(h_img, y1), x0:x1]
    if strip.size == 0:
        return False
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    # A solid card outline spans >= 70% of the overlapping width with dark pixels
    row_dark_ratio = (gray < 85).mean(axis=1)
    if float(row_dark_ratio.max()) >= 0.70:
        return True
    return False


def _color_distance(a: RGBColor, b: RGBColor) -> float:
    return float(np.sqrt((a.r - b.r) ** 2 + (a.g - b.g) ** 2 + (a.b - b.b) ** 2))


def _can_merge_vertical(
    img_bgr: np.ndarray, prev: TextLine, curr: TextLine
) -> tuple[bool, TextAlignment | None]:
    """Check whether curr belongs to the same multi-line text block as prev."""
    # Never merge standalone card numbers ('1', '2', '3') with card titles if font/color differs
    if _color_distance(prev.color, curr.color) > 55.0:
        return False, None

    # Check font size compatibility
    size_ratio = min(prev.font_size_pt, curr.font_size_pt) / max(
        1.0, max(prev.font_size_pt, curr.font_size_pt)
    )
    if size_ratio < 0.78:
        return False, None

    # Check bold mismatch on title+subtitle pairs (keep bold heading separate from regular body!)
    if prev.bold != curr.bold and (
        abs(prev.font_size_pt - curr.font_size_pt) > 2.5 or len(prev.text) < 28
    ):
        return False, None

    # Vertical gap check
    avg_h = 0.5 * (prev.bbox.h + curr.bbox.h)
    v_gap = curr.bbox.y - prev.bbox.y2
    if v_gap < -0.20 * avg_h or v_gap > 0.78 * avg_h:
        return False, None

    # Must overlap horizontally
    if prev.bbox.horizontal_overlap_ratio(curr.bbox) <= 0.12:
        return False, None

    # Don't cross solid card/box borders
    if _has_barrier_between(img_bgr, prev.bbox, curr.bbox):
        return False, None

    # Check alignment consistency (left, center, or right)
    left_diff = abs(prev.bbox.x - curr.bbox.x)
    center_diff = abs(prev.bbox.cx - curr.bbox.cx)
    right_diff = abs(prev.bbox.x2 - curr.bbox.x2)

    left_tol = max(14.0, avg_h * 0.48)
    center_tol = max(18.0, avg_h * 0.62)
    right_tol = max(14.0, avg_h * 0.48)

    if left_diff <= left_tol:
        return True, "left"
    if center_diff <= center_tol:
        return True, "center"
    if right_diff <= right_tol:
        return True, "right"

    return False, None


def group_lines_into_blocks(
    img_bgr: np.ndarray,
    lines: list[TextLine],
    slide_width: int,
    slide_height: int,
    consumed_indices: set[int] | None = None,
) -> list[TextBlock]:
    """Group non-watermark, non-table TextLines into clean editable TextBlocks."""
    skip = consumed_indices or set()
    active = [
        ln for idx, ln in enumerate(lines) if idx not in skip and not ln.is_watermark
    ]
    if not active:
        return []

    # Sort top-to-bottom, left-to-right
    active.sort(key=lambda ln: (ln.bbox.y, ln.bbox.x))

    groups: list[tuple[list[TextLine], TextAlignment | None]] = []
    for ln in active:
        best_g_idx = -1
        best_gap = float("inf")
        best_align: TextAlignment | None = None

        for g_idx, (g_lines, g_align) in enumerate(groups):
            last = g_lines[-1]
            ok, align = _can_merge_vertical(img_bgr, last, ln)
            if not ok:
                continue
            # Keep alignment consistent across 3+ lines
            if g_align is not None and align is not None and g_align != align:
                # Allow center/left ambiguity when lines have similar width
                if not (
                    abs(last.bbox.x - ln.bbox.x) <= max(14.0, last.bbox.h * 0.45)
                    and g_align == "left"
                ):
                    continue
            gap = ln.bbox.y - last.bbox.y2
            if gap < best_gap:
                best_gap = gap
                best_g_idx = g_idx
                best_align = g_align or align

        if best_g_idx >= 0:
            g_lines, _ = groups[best_g_idx]
            g_lines.append(ln)
            groups[best_g_idx] = (g_lines, best_align)
        else:
            groups.append(([ln], None))

    # Precompute all group boxes so single-line blocks can check sibling left-edge alignment
    group_boxes = [BBox.union([l.bbox for l in g_lines]) for g_lines, _ in groups]
    pt_per_px = 540.0 / max(1.0, float(slide_height))

    # Convert each group into a typed TextBlock
    blocks: list[TextBlock] = []
    for b_idx, (g_lines, inferred_align) in enumerate(groups):
        g_lines.sort(key=lambda l: l.bbox.y)
        box = group_boxes[b_idx]

        # Determine alignment cleanly
        if len(g_lines) == 1:
            shares_left_sibling = any(
                k != b_idx
                and abs(group_boxes[k].x - box.x) <= 24.0
                and abs(group_boxes[k].cy - box.cy) <= slide_height * 0.35
                for k in range(len(group_boxes))
            )
            if (
                not shares_left_sibling
                and abs(box.cx - slide_width * 0.5) <= slide_width * 0.045
                and box.w >= slide_width * 0.22
            ):
                alignment: TextAlignment = "center"
            else:
                alignment = "left"
        else:
            xs = [l.bbox.x for l in g_lines]
            cxs = [l.bbox.cx for l in g_lines]
            x2s = [l.bbox.x2 for l in g_lines]
            spread_left = max(xs) - min(xs)
            spread_center = max(cxs) - min(cxs)
            spread_right = max(x2s) - min(x2s)
            if spread_left <= min(spread_center, spread_right) + 4.0:
                alignment = "left"
            elif spread_center <= spread_right:
                alignment = "center"
            else:
                alignment = "right"

        # Compute exact point line step (center-to-center distance in pt) + proportional ratio
        if len(g_lines) >= 2:
            c_steps = [
                g_lines[k + 1].bbox.cy - g_lines[k].bbox.cy for k in range(len(g_lines) - 1)
            ]
            med_step_px = float(np.median(c_steps))
            avg_h = float(np.mean([l.bbox.h for l in g_lines]))
            line_spacing = float(np.clip(med_step_px / max(1.0, avg_h), 1.0, 1.5))
            line_step_pt: float | None = round(med_step_px * pt_per_px, 2)
        else:
            line_spacing = 1.15
            line_step_pt = None

        # Dominant font size, color, bold, font family
        font_size_pt = round(float(np.median([l.font_size_pt for l in g_lines])) * 2.0) / 2.0
        bold = sum(1 for l in g_lines if l.bold) > (len(g_lines) / 2.0)
        color = g_lines[0].color
        font_family = g_lines[0].font_family

        # Semantic role heuristic
        joined_text = "\n".join(l.text for l in g_lines)
        if any(l.text.lstrip().startswith(("•", "-", "·", "* ")) for l in g_lines):
            role: TextRole = "bullet"
        elif font_size_pt >= 34.0:
            role = "title"
        elif font_size_pt >= 22.0:
            role = "heading"
        elif len(joined_text.strip()) <= 2 and joined_text.strip().isdigit():
            role = "number"
        elif font_size_pt <= 15.0:
            role = "caption"
        else:
            role = "body"

        all_highlights: list[HighlightSpan] = []
        for l in g_lines:
            all_highlights.extend(l.highlights)

        blocks.append(
            TextBlock(
                block_id=f"tb_{b_idx + 1:02d}",
                text=joined_text,
                lines=g_lines,
                bbox=box,
                alignment=alignment,
                role=role,
                font_size_pt=font_size_pt,
                color=color,
                bold=bold,
                italic=any(l.italic for l in g_lines),
                font_family=font_family,
                line_spacing=round(line_spacing, 2),
                line_step_pt=line_step_pt,
                highlights=all_highlights,
            )
        )

    return blocks
