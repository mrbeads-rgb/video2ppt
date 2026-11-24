"""OCR post-processing: overlap merging, re-recognition, script & watermark filtering."""

from __future__ import annotations

import re
from typing import Callable

import cv2
import numpy as np

from video2ppt.config import PipelineConfig
from video2ppt.models import BBox, TextLine

RecognizeFn = Callable[[np.ndarray], tuple[str, float] | None]

_CJK_RE = re.compile(r"[\u3000-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF]")
_LATIN_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_PURE_SYMBOL_RE = re.compile(r"^[\W_→←↑↓↔•·\-=+*/\\|?؟!.,:;()\[\]{}'\"`~<>]+$")
_WATERMARK_RE = re.compile(r"(gemini\s*notebook|notebooklm|made\s+with|canva|gamma\.app)", re.IGNORECASE)


def _is_cjk_dominant(text: str) -> bool:
    cjk_count = len(_CJK_RE.findall(text))
    lat_count = len(_LATIN_ALNUM_RE.findall(text))
    return cjk_count > 0 and cjk_count >= lat_count


def detect_deck_is_latin(all_slide_raw_texts: list[list[tuple[str, float]]]) -> bool:
    """Return True if the deck's high-confidence text is overwhelmingly Latin script."""
    lat = 0
    cjk = 0
    for slide_items in all_slide_raw_texts:
        for txt, conf in slide_items:
            if conf >= 0.88:
                lat += len(_LATIN_ALNUM_RE.findall(txt))
                cjk += len(_CJK_RE.findall(txt))
    return lat >= 20 and (cjk / max(1, lat + cjk)) < 0.08


def is_doodle_false_positive(
    text: str, conf: float, bbox: BBox, deck_is_latin: bool, lang: str
) -> bool:
    """Filter out hand-drawn icon / arrow / flowchart OCR hallucinations."""
    clean = text.strip()
    if not clean:
        return True

    # Reject pure punctuation / arrows / symbols (e.g. '?', '→', '↑', '[A')
    if _PURE_SYMBOL_RE.match(clean):
        return True

    # On Latin-script decks, filter CJK doodle hallucinations (e.g. '口双', '自', '具', '一#一具')
    if (lang == "en" or (lang == "auto" and deck_is_latin)) and _is_cjk_dominant(clean):
        return True

    # Reject stray 1-2 character low/medium confidence icon misreads (keep strong single digits like '1'..'5', 'C', '9')
    if len(clean) == 1:
        if clean.isalnum():
            if conf < 0.88:
                return True
        else:
            return True
    elif len(clean) == 2:
        # e.g., '[A' on a chip icon
        if not any(ch.isalnum() for ch in clean):
            return True
        if not clean.isalnum() and conf < 0.90:
            return True

    # Reject repeated '10101010' / 'OOOOO' chain-diagram misreads
    if len(clean) >= 6 and re.fullmatch(r"[01OoIl\-]+", clean) and conf < 0.92:
        return True

    # Reject suspiciously tall square-ish blob with very short low-conf text (doodle box)
    if bbox.h > 85 and len(clean) <= 5 and conf < 0.85:
        return True

    return False


def merge_same_line_fragments(
    img_bgr: np.ndarray,
    raw_items: list[tuple[BBox, list[list[float]], str, float]],
    recognize_fn: RecognizeFn | None = None,
) -> list[tuple[BBox, list[list[float]], str, float]]:
    """Merge horizontally overlapping or tightly adjacent fragments on the same baseline.

    Fixes cases like 'Chief' + 'f of' + 'Staff?' where highlight boundaries split a single line
    into overlapping boxes, by re-recognizing the union crop cleanly.
    """
    if len(raw_items) <= 1:
        return raw_items

    # Sort top-to-bottom, left-to-right
    items = sorted(raw_items, key=lambda it: (it[0].cy, it[0].x))
    used = [False] * len(items)
    merged: list[tuple[BBox, list[list[float]], str, float]] = []
    h_img, w_img = img_bgr.shape[:2]

    for i in range(len(items)):
        if used[i]:
            continue
        group = [items[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            g_box = BBox.union([g[0] for g in group])
            for j in range(len(items)):
                if used[j]:
                    continue
                cand_box = items[j][0]
                v_ov = g_box.vertical_overlap_ratio(cand_box)
                h_ratio = min(g_box.h, cand_box.h) / max(1.0, max(g_box.h, cand_box.h))
                if v_ov < 0.62 or h_ratio < 0.68:
                    continue

                # Horizontal gap (negative means overlap!)
                if cand_box.x >= g_box.x:
                    gap = cand_box.x - g_box.x2
                else:
                    gap = g_box.x - cand_box.x2

                # Merge if overlapping OR very tight word gap (< 0.42 * line height)
                # Avoid merging separate table columns or distinct cards (which have wider gaps)
                if gap <= g_box.h * 0.40:
                    # Also check there isn't a hard dark card border line between them!
                    if gap > 2:
                        mid_x0 = int(min(g_box.x2, cand_box.x2))
                        mid_x1 = int(max(g_box.x, cand_box.x))
                        my0 = int(max(g_box.y, cand_box.y))
                        my1 = int(min(g_box.y2, cand_box.y2))
                        if mid_x1 > mid_x0 and my1 > my0:
                            strip = cv2.cvtColor(
                                img_bgr[my0:my1, mid_x0:mid_x1], cv2.COLOR_BGR2GRAY
                            )
                            # A vertical card border has a dark column spanning >= 85% of height
                            col_dark = (strip < 80).mean(axis=0)
                            if len(col_dark) > 0 and float(col_dark.max()) > 0.85:
                                continue

                    group.append(items[j])
                    used[j] = True
                    changed = True

        if len(group) == 1:
            merged.append(group[0])
            continue

        group.sort(key=lambda g: g[0].x)
        union_box = BBox.union([g[0] for g in group])
        union_poly = [
            [union_box.x, union_box.y],
            [union_box.x2, union_box.y],
            [union_box.x2, union_box.y2],
            [union_box.x, union_box.y2],
        ]
        # Check if any fragments overlapped horizontally
        has_overlap = any(
            group[k + 1][0].x < group[k][0].x2 - 3.0 for k in range(len(group) - 1)
        )

        # Re-recognize merged crop when overlapping or multi-fragment large heading
        new_text = " ".join(g[2].strip() for g in group if g[2].strip())
        new_conf = float(np.mean([g[3] for g in group]))
        if recognize_fn is not None and (has_overlap or len(group) >= 2):
            pad_x = max(2, int(union_box.h * 0.06))
            pad_y = max(2, int(union_box.h * 0.06))
            cx0 = max(0, int(np.floor(union_box.x)) - pad_x)
            cy0 = max(0, int(np.floor(union_box.y)) - pad_y)
            cx1 = min(w_img, int(np.ceil(union_box.x2)) + pad_x)
            cy1 = min(h_img, int(np.ceil(union_box.y2)) + pad_y)
            crop = img_bgr[cy0:cy1, cx0:cx1]
            rec = recognize_fn(crop)
            if rec is not None and rec[0].strip() and rec[1] >= 0.80:
                new_text, new_conf = rec[0].strip(), rec[1]

        merged.append((union_box, union_poly, new_text, new_conf))

    return merged


def recover_leading_bullet(
    img_bgr: np.ndarray,
    bbox: BBox,
    polygon: list[list[float]],
    text: str,
) -> tuple[BBox, list[list[float]], str]:
    """Detect an orphaned bullet dot just left of a text line so it is erased and editable."""
    stripped = text.lstrip()
    if not stripped or stripped[0] in "•·-*▪▫‣◦" or bbox.h > 46 or bbox.h < 12:
        return bbox, polygon, text

    h_img, w_img = img_bgr.shape[:2]
    scan_w = max(14, int(round(bbox.h * 0.95)))
    lx1 = max(0, int(np.floor(bbox.x)))
    lx0 = max(0, lx1 - scan_w)
    ly0 = max(0, int(np.floor(bbox.y)))
    ly1 = min(h_img, int(np.ceil(bbox.y2)))
    if lx1 - lx0 < 8 or ly1 - ly0 < 10:
        return bbox, polygon, text

    roi = img_bgr[ly0:ly1, lx0:lx1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bg_val = float(np.median(np.concatenate([gray[0, :], gray[-1, :]])))
    ink = (np.abs(gray.astype(np.float32) - bg_val) > 48.0).astype(np.uint8) * 255
    if int((ink > 0).sum()) < 4:
        return bbox, polygon, text

    num, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    rh = ly1 - ly0
    rw = lx1 - lx0
    for lbl in range(1, num):
        cx = int(stats[lbl, cv2.CC_STAT_LEFT])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        cent_y = float(centroids[lbl][1])
        aspect = cw / max(1.0, float(ch))
        if (
            cx > 1
            and (cx + cw) < rw - 1
            and 3 <= cw <= max(10, int(bbox.h * 0.38))
            and 3 <= ch <= max(10, int(bbox.h * 0.38))
            and 0.60 <= aspect <= 1.50
            and area >= 6
            and rh * 0.28 <= cent_y <= rh * 0.72
        ):
            new_x = max(0.0, float(lx0 + cx - 3))
            new_box = BBox(x=new_x, y=bbox.y, w=bbox.x2 - new_x, h=bbox.h)
            new_poly = [
                [new_box.x, new_box.y],
                [new_box.x2, new_box.y],
                [new_box.x2, new_box.y2],
                [new_box.x, new_box.y2],
            ]
            return new_box, new_poly, f"• {stripped}"

    return bbox, polygon, text


def upscale_re_recognize_small_line(
    img_bgr: np.ndarray,
    bbox: BBox,
    text: str,
    recognize_fn: RecognizeFn | None = None,
) -> str:
    """Upscale tiny text crops 2.5x before re-recognition to restore dropped inter-word spaces."""
    if recognize_fn is None or bbox.h >= 30.0 or len(text.strip()) < 7:
        return text

    clean = text.strip()
    needs_retry = (
        (len(clean) >= 10 and clean.count(" ") < max(1, len(clean) // 11))
        or bool(re.search(r",[A-Za-z]", clean))
        or bool(re.search(r"\b(asa|tothe|inthe|ofthe)\b", clean, re.IGNORECASE))
    )
    if not needs_retry:
        return text

    h_img, w_img = img_bgr.shape[:2]
    x0 = max(0, int(np.floor(bbox.x)) - 3)
    y0 = max(0, int(np.floor(bbox.y)) - 3)
    x1 = min(w_img, int(np.ceil(bbox.x2)) + 3)
    y1 = min(h_img, int(np.ceil(bbox.y2)) + 3)
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return text

    up = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    rec = recognize_fn(up)
    if rec is not None and rec[0].strip() and rec[1] >= 0.88:
        cand = rec[0].strip()
        if (
            cand.count(" ") >= clean.count(" ")
            and abs(len(cand.replace(" ", "")) - len(clean.replace(" ", ""))) <= 3
        ):
            return cand
    return text


def recover_trailing_punctuation(
    img_bgr: np.ndarray,
    bbox: BBox,
    polygon: list[list[float]],
    text: str,
    recognize_fn: RecognizeFn | None = None,
) -> tuple[BBox, list[list[float]], str]:
    """Check immediately to the right of a text line for clipped trailing punctuation (., ?, !)."""
    if not text or text[-1] in ".?!,:;\"')":
        return bbox, polygon, text

    h_img, w_img = img_bgr.shape[:2]
    scan_w = max(8, int(round(bbox.h * 0.42)))
    rx0 = int(np.ceil(bbox.x2))
    rx1 = min(w_img, rx0 + scan_w)
    ry0 = max(0, int(np.floor(bbox.y)))
    ry1 = min(h_img, int(np.ceil(bbox.y2)))
    if rx1 - rx0 < 4 or ry1 - ry0 < 8:
        return bbox, polygon, text

    # Compare against background around the right scan region
    roi = img_bgr[ry0:ry1, rx0:rx1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bg_val = float(np.median(np.concatenate([gray[0, :], gray[-1, :]])))
    ink = (np.abs(gray.astype(np.float32) - bg_val) > 45.0).astype(np.uint8) * 255
    if int((ink > 0).sum()) < 4:
        return bbox, polygon, text

    # Look for a small isolated punctuation mark close to the right edge (not a huge doodle!)
    num, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    best_x2 = None
    for lbl in range(1, num):
        cx = int(stats[lbl, cv2.CC_STAT_LEFT])
        cy = int(stats[lbl, cv2.CC_STAT_TOP])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        # Punctuation: near left of scan window, small width, does not touch right edge of scan
        if (
            cx <= bbox.h * 0.25
            and cw <= bbox.h * 0.35
            and ch <= bbox.h * 0.85
            and area >= 4
            and (cx + cw) < (rx1 - rx0 - 1)
            and cy >= bbox.h * 0.08
        ):
            best_x2 = max(best_x2 or 0, rx0 + cx + cw + 3)

    if best_x2 is None:
        return bbox, polygon, text

    # Verify with recognizer on expanded crop
    new_box = BBox(x=bbox.x, y=bbox.y, w=float(best_x2) - bbox.x, h=bbox.h)
    if recognize_fn is not None:
        cx0 = max(0, int(np.floor(new_box.x)) - 3)
        cy0 = max(0, int(np.floor(new_box.y)) - 3)
        cx1 = min(w_img, int(np.ceil(new_box.x2)) + 4)
        cy1 = min(h_img, int(np.ceil(new_box.y2)) + 3)
        rec = recognize_fn(img_bgr[cy0:cy1, cx0:cx1])
        if rec is not None and rec[0].strip() and rec[1] >= 0.82:
            cand = rec[0].strip()
            if cand[-1] in ".?!,:;" and len(cand) >= len(text):
                new_poly = [
                    [new_box.x, new_box.y],
                    [new_box.x2, new_box.y],
                    [new_box.x2, new_box.y2],
                    [new_box.x, new_box.y2],
                ]
                return new_box, new_poly, cand

    return bbox, polygon, text


def restore_missing_word_spaces(
    img_bgr: np.ndarray,
    bbox: BBox,
    text: str,
    recognize_fn: RecognizeFn | None = None,
) -> str:
    """Restore missing spaces in joined OCR tokens using camelCase/digit rules + ink-gap inspection."""
    # 1. Unambiguous camelCase / number-letter / comma joins: 'BudgetUtilised' -> 'Budget Utilised', '11.5DAYS' -> '11.5 DAYS'
    fixed = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    fixed = re.sub(r"(\d)([A-Z]{2,})", r"\1 \2", fixed)
    fixed = re.sub(r",([A-Za-z])", r", \1", fixed)
    # 2. If a single long lowercase token (>= 10 chars, no space) has a wide visual whitespace gap
    # in the actual glyph projection (e.g. 'careerpivots'), split and recognize each word!
    if (
        recognize_fn is not None
        and " " not in fixed
        and len(fixed) >= 10
        and fixed.isalpha()
        and fixed.islower()
    ):
        h_img, w_img = img_bgr.shape[:2]
        x0, y0 = max(0, int(bbox.x)), max(0, int(bbox.y))
        x1, y1 = min(w_img, int(bbox.x2)), min(h_img, int(bbox.y2))
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            bg = float(np.median(np.concatenate([gray[0, :], gray[-1, :]])))
            col_ink = (np.abs(gray.astype(np.float32) - bg) > 38.0).any(axis=0)
            # Find whitespace runs between ink columns
            gaps: list[tuple[int, int]] = []
            seen_ink = False
            g_start = 0
            in_gap = False
            for idx, has in enumerate(col_ink):
                if has:
                    if in_gap and seen_ink:
                        gaps.append((g_start, idx))
                    seen_ink = True
                    in_gap = False
                elif seen_ink and not in_gap:
                    g_start = idx
                    in_gap = True
            if len(gaps) >= 3:
                widths = [e - s for s, e in gaps]
                med_gap = float(np.median(widths))
                max_gap_idx = int(np.argmax(widths))
                max_gap_w = widths[max_gap_idx]
                if max_gap_w >= max(5.0, med_gap * 2.4):
                    split_x = (gaps[max_gap_idx][0] + gaps[max_gap_idx][1]) // 2
                    left_rec = recognize_fn(crop[:, : max(4, split_x + 2)])
                    right_rec = recognize_fn(crop[:, max(0, split_x - 2) :])
                    if (
                        left_rec
                        and right_rec
                        and left_rec[1] >= 0.85
                        and right_rec[1] >= 0.85
                        and len(left_rec[0].strip()) >= 2
                        and len(right_rec[0].strip()) >= 2
                    ):
                        combined = f"{left_rec[0].strip()} {right_rec[0].strip()}"
                        if combined.replace(" ", "").lower() == fixed.lower():
                            return combined
    return fixed


def normalize_ocr_text(text: str, deck_mentions_ai: bool = True) -> str:
    """Clean common sans-serif OCR confusions (e.g. 'Al' -> 'AI', 'S AM' -> '5 AM', stray quotes)."""
    clean = re.sub(r"\s+", " ", text).strip()
    if clean.startswith("· "):
        clean = "• " + clean[2:]
    elif clean.startswith("·") and len(clean) > 2 and clean[1].isupper():
        clean = "• " + clean[1:]
    if clean.startswith('"') and clean.count('"') == 1 and len(clean) <= 12:
        clean = clean[1:].strip()
    clean = re.sub(r"\bS\s+(AM|PM)\b", r"5 \1", clean)
    if deck_mentions_ai:
        # In sans-serif fonts, capital 'AI' is frequently misread as 'Al'
        # Replace standalone 'Al' when not followed by a capitalized surname (like 'Al Gore')
        clean = re.sub(r"\bAl\b(?!\s+[A-Z][a-z]{2,})", "AI", clean)
    return clean


def mark_recurring_watermarks(all_slide_lines: list[list[TextLine]], width: int, height: int) -> None:
    """Identify recurring corner platform watermarks across slides and flag them."""
    if not all_slide_lines:
        return

    # Count occurrences in the 4 small corner zones
    corner_hits: dict[str, list[TextLine]] = {"br": [], "bl": [], "tr": [], "tl": []}
    for lines in all_slide_lines:
        seen_in_slide: set[str] = set()
        for line in lines:
            b = line.bbox
            is_small = b.h <= height * 0.055 and b.w <= width * 0.25
            if not is_small:
                continue
            zone = None
            if b.x >= width * 0.78 and b.y >= height * 0.90:
                zone = "br"
            elif b.x2 <= width * 0.22 and b.y >= height * 0.90:
                zone = "bl"
            elif b.x >= width * 0.78 and b.y2 <= height * 0.10:
                zone = "tr"
            elif b.x2 <= width * 0.22 and b.y2 <= height * 0.10:
                zone = "tl"

            if zone:
                if _WATERMARK_RE.search(line.text):
                    line.is_watermark = True
                if zone not in seen_in_slide:
                    seen_in_slide.add(zone)
                corner_hits[zone].append(line)

    # If a corner zone repeats on >= 40% of slides (and at least 2 slides), mark all as watermarks
    min_slides = max(2, int(len(all_slide_lines) * 0.40))
    for zone, z_lines in corner_hits.items():
        if len(z_lines) >= min_slides:
            for ln in z_lines:
                ln.is_watermark = True
