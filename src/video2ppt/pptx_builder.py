"""High-precision PowerPoint (.pptx) generator supporting Hybrid and Modular modes."""

from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor as PptxRGB
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from video2ppt.models import (
    BBox,
    DeckPlan,
    DetectedShape,
    DetectedTable,
    RGBColor,
    SlideLayout,
    TextAlignment,
    TextBlock,
)
from video2ppt.styling import measure_rendered_text_width_px


def _to_pptx_rgb(c: RGBColor) -> PptxRGB:
    return PptxRGB(c.r, c.g, c.b)


def _to_pp_align(align: TextAlignment) -> PP_ALIGN:
    if align == "center":
        return PP_ALIGN.CENTER
    if align == "right":
        return PP_ALIGN.RIGHT
    return PP_ALIGN.LEFT


def _slide_dimensions_inches(width_px: int, height_px: int) -> tuple[float, float]:
    """Compute PowerPoint slide dimensions preserving the source video aspect ratio."""
    aspect = width_px / max(1.0, float(height_px))
    if aspect >= 1.0:
        w_in = 13.333333
        h_in = w_in / aspect
    else:
        h_in = 13.333333
        w_in = h_in * aspect
    return w_in, h_in


def _set_cell_no_fill_and_subtle_border(cell, border_hex: str | None = None) -> None:
    """Keep table cells transparent with clean light borders (or no borders on hybrid canvas)."""
    cell.fill.background()
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        existing = tcPr.find(qn(tag))
        if existing is not None:
            tcPr.remove(existing)
        ln = etree.SubElement(tcPr, qn(tag), w="6350")  # 0.5pt
        if border_hex:
            solid = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(solid, qn("a:srgbClr"), val=border_hex.lstrip("#"))
        else:
            etree.SubElement(ln, qn("a:noFill"))


def _add_native_shape(
    slide,
    shape_spec: DetectedShape,
    px_to_x,
    px_to_y,
) -> None:
    """Add an editable native PowerPoint rectangle / rounded rectangle / highlight pill."""
    mso_type = (
        MSO_SHAPE.ROUNDED_RECTANGLE
        if shape_spec.shape_type in ("rounded_rect", "highlight")
        else MSO_SHAPE.RECTANGLE
    )
    b = shape_spec.bbox
    shp = slide.shapes.add_shape(
        mso_type,
        Inches(px_to_x(b.x)),
        Inches(px_to_y(b.y)),
        Inches(px_to_x(b.w)),
        Inches(px_to_y(b.h)),
    )
    if mso_type == MSO_SHAPE.ROUNDED_RECTANGLE and len(shp.adjustments) > 0:
        shp.adjustments[0] = max(0.02, min(0.35, shape_spec.corner_radius_ratio))

    shp.fill.solid()
    shp.fill.fore_color.rgb = _to_pptx_rgb(shape_spec.fill_color)

    if shape_spec.border_color is not None and shape_spec.border_width_pt > 0:
        shp.line.color.rgb = _to_pptx_rgb(shape_spec.border_color)
        shp.line.width = Pt(shape_spec.border_width_pt)
    else:
        shp.line.fill.background()

    # Remove default PowerPoint drop shadow for clean flat match
    shp.shadow.inherit = False


def _add_native_table(
    slide,
    tbl_spec: DetectedTable,
    px_to_x,
    px_to_y,
    modular_mode: bool,
) -> None:
    """Insert an editable native PowerPoint table at the exact detected grid position."""
    b = tbl_spec.bbox
    shape = slide.shapes.add_table(
        tbl_spec.rows,
        tbl_spec.cols,
        Inches(px_to_x(b.x)),
        Inches(px_to_y(b.y)),
        Inches(px_to_x(b.w)),
        Inches(px_to_y(b.h)),
    )
    table = shape.table

    # Disable banded row / default blue PowerPoint table styling
    tblPr = table._tbl.tblPr
    if tblPr is not None:
        for attr in ("bandRow", "firstRow", "bandCol", "lastRow", "firstCol", "lastCol"):
            tblPr.set(attr, "0")

    # Apply exact column widths & row heights
    if len(tbl_spec.col_widths_px) == tbl_spec.cols:
        total_cw = sum(tbl_spec.col_widths_px)
        scale_x = b.w / max(1.0, total_cw)
        for c_i, cw in enumerate(tbl_spec.col_widths_px):
            table.columns[c_i].width = Inches(px_to_x(cw * scale_x))

    if len(tbl_spec.row_heights_px) == tbl_spec.rows:
        total_rh = sum(tbl_spec.row_heights_px)
        scale_y = b.h / max(1.0, total_rh)
        for r_i, rh in enumerate(tbl_spec.row_heights_px):
            table.rows[r_i].height = Inches(px_to_y(rh * scale_y))

    border_hex = "CCCCCC" if modular_mode else None
    for r_i in range(tbl_spec.rows):
        for c_i in range(tbl_spec.cols):
            cell = table.cell(r_i, c_i)
            _set_cell_no_fill_and_subtle_border(cell, border_hex=border_hex)
            cell.margin_left = Inches(0.03)
            cell.margin_right = Inches(0.03)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text = ""

    for c_spec in tbl_spec.cells:
        if c_spec.row >= tbl_spec.rows or c_spec.col >= tbl_spec.cols:
            continue
        cell = table.cell(c_spec.row, c_spec.col)
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = c_spec.text
        p.alignment = _to_pp_align(c_spec.alignment)
        run = p.runs[0] if p.runs else p.add_run()
        run.font.name = "Arial"
        run.font.size = Pt(max(8.5, c_spec.font_size_pt))
        run.font.bold = c_spec.bold
        run.font.color.rgb = _to_pptx_rgb(c_spec.color)


def _add_editable_textbox(
    slide,
    tb: TextBlock,
    slide_w_px: int,
    slide_h_px: int,
    px_to_x,
    px_to_y,
) -> None:
    """Create a pixel-aligned, zero-margin editable PowerPoint text box."""
    b = tb.bbox
    lines_list = tb.lines if tb.lines else []
    raw_text_lines = tb.text.splitlines() if tb.text else [""]
    is_single_line = len(raw_text_lines) <= 1

    # Add gentle horizontal breathing room so minor font-metric differences never trigger mid-word wrapping
    extra_w = max(14.0, b.w * 0.08)
    if tb.alignment == "center":
        x_px = max(0.0, b.x - extra_w * 0.5)
        w_px = min(float(slide_w_px) - x_px, b.w + extra_w)
    elif tb.alignment == "right":
        x_px = max(0.0, b.x - extra_w)
        w_px = min(float(slide_w_px) - x_px, b.w + extra_w)
    else:
        x_px = max(0.0, b.x - 2.0)
        w_px = min(float(slide_w_px) - x_px, b.w + extra_w)

    # TTF metric width guard: ensure no line exceeds w_px and causes unwanted extra line wraps
    px_per_pt = float(slide_h_px) / 540.0
    block_font_pt = max(7.5, tb.font_size_pt)
    block_font_name = tb.font_family or "Arial"
    block_bold = bool(lines_list[0].bold if is_single_line and lines_list else tb.bold)
    max_rendered_w = max(
        (
            measure_rendered_text_width_px(
                ln_txt, block_font_name, block_bold, block_font_pt, px_per_pt
            )
            for ln_txt in raw_text_lines
            if ln_txt.strip()
        ),
        default=0.0,
    )
    if max_rendered_w > w_px * 0.94:
        fit_scale = (w_px * 0.92) / max_rendered_w
        block_font_pt = max(7.5, round(block_font_pt * fit_scale * 2.0) / 2.0)

    # Give a tiny vertical pad so ascenders/descenders are never clipped
    pad_y = max(2.0, b.h * 0.04)
    y_px = max(0.0, b.y - pad_y)
    h_px = min(float(slide_h_px) - y_px, b.h + pad_y * 2.0)

    box = slide.shapes.add_textbox(
        Inches(px_to_x(x_px)),
        Inches(px_to_y(y_px)),
        Inches(px_to_x(w_px)),
        Inches(px_to_y(h_px)),
    )
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0)
    tf.margin_right = Inches(0)
    tf.margin_top = Inches(0)
    tf.margin_bottom = Inches(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE if is_single_line else MSO_ANCHOR.TOP

    for l_idx, line_text in enumerate(raw_text_lines):
        p = tf.paragraphs[0] if l_idx == 0 else tf.add_paragraph()
        p.alignment = _to_pp_align(tb.alignment)
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        if not is_single_line:
            if tb.line_step_pt is not None and tb.line_step_pt > 0:
                # Exact point leading matches video baseline-to-baseline spacing without drifting into subtitles
                exact_pt = max(block_font_pt * 1.05, min(block_font_pt * 1.55, tb.line_step_pt))
                p.line_spacing = Pt(exact_pt)
            else:
                p.line_spacing = max(0.95, min(1.35, tb.line_spacing))

        run = p.add_run()
        run.text = line_text

        src_line = lines_list[l_idx] if l_idx < len(lines_list) else None
        run.font.name = (src_line.font_family if src_line else block_font_name) or "Arial"
        run.font.size = Pt(block_font_pt)
        run.font.bold = block_bold
        run.font.italic = bool(src_line.italic if src_line else tb.italic)
        run.font.color.rgb = _to_pptx_rgb(src_line.color if src_line else tb.color)


def build_pptx(
    plan: DeckPlan,
    output_pptx: str | Path,
    mode_override: str | None = None,
) -> Path:
    """Assemble an editable PowerPoint (.pptx) presentation from a DeckPlan."""
    out_path = Path(output_pptx).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = mode_override or plan.mode
    prs = Presentation()
    slide_w_in, slide_h_in = _slide_dimensions_inches(
        plan.video_meta.width, plan.video_meta.height
    )
    prs.slide_width = Inches(slide_w_in)
    prs.slide_height = Inches(slide_h_in)
    blank_layout = prs.slide_layouts[6]

    for s_layout in plan.slides:
        slide = prs.slides.add_slide(blank_layout)
        w_px = max(1, s_layout.width)
        h_px = max(1, s_layout.height)

        px_to_x = lambda px, _w=w_px: (float(px) / float(_w)) * slide_w_in
        px_to_y = lambda py, _h=h_px: (float(py) / float(_h)) * slide_h_in

        # Always set native solid background color matching the slide canvas
        bg = slide.background
        bg.fill.solid()
        bg.fill.fore_color.rgb = _to_pptx_rgb(s_layout.background_color)

        if mode == "hybrid":
            # Layer 1: Surgically inpainted background image (artwork + highlights intact, text erased)
            bg_img_path = s_layout.clean_bg_path or s_layout.raw_frame_path
            if bg_img_path and Path(bg_img_path).exists():
                slide.shapes.add_picture(
                    bg_img_path,
                    Inches(0),
                    Inches(0),
                    width=Inches(slide_w_in),
                    height=Inches(slide_h_in),
                )
        else:
            # Modular mode:
            # Layer 1: Native vector cards/panels (z_order < 50)
            card_shapes = sorted(
                [s for s in s_layout.shapes if s.shape_type != "highlight"],
                key=lambda s: s.z_order,
            )
            for shp in card_shapes:
                _add_native_shape(slide, shp, px_to_x, px_to_y)

            # Layer 2: Movable transparent RGBA extracted graphic assets
            for gr in sorted(s_layout.graphics, key=lambda g: g.z_order):
                if gr.image_path and Path(gr.image_path).exists():
                    slide.shapes.add_picture(
                        gr.image_path,
                        Inches(px_to_x(gr.bbox.x)),
                        Inches(px_to_y(gr.bbox.y)),
                        width=Inches(px_to_x(gr.bbox.w)),
                        height=Inches(px_to_y(gr.bbox.h)),
                    )

            # Layer 3: Native highlight pills directly behind editable text
            hl_shapes = [s for s in s_layout.shapes if s.shape_type == "highlight"]
            for hl in hl_shapes:
                _add_native_shape(slide, hl, px_to_x, px_to_y)

        # Layer 4: Native editable tables
        for tbl in s_layout.tables:
            _add_native_table(
                slide, tbl, px_to_x, px_to_y, modular_mode=(mode == "modular")
            )

        # Layer 5: Native editable text boxes
        for tb in s_layout.text_blocks:
            _add_editable_textbox(slide, tb, w_px, h_px, px_to_x, px_to_y)

        # Speaker notes (from audio transcription or custom plan)
        if s_layout.speaker_notes.strip():
            notes_slide = slide.notes_slide
            notes_tf = notes_slide.notes_text_frame
            notes_tf.text = s_layout.speaker_notes.strip()

    prs.save(str(out_path))
    return out_path
