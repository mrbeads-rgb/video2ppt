"""Unit and integration tests for the video2ppt pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from pptx import Presentation

from video2ppt.analyzers import create_analyzer
from video2ppt.analyzers.gemini import build_slide_layout_from_vlm_json
from video2ppt.config import PipelineConfig
from video2ppt.grouper import detect_tables_and_recover_cells, group_lines_into_blocks
from video2ppt.inpainter import inpaint_slide_background
from video2ppt.models import (
    BBox,
    DeckPlan,
    RGBColor,
    SlideLayout,
    SlideSegment,
    TextBlock,
    TextLine,
    VideoMeta,
)
from video2ppt.modular import extract_modular_elements
from video2ppt.ocr_post import (
    detect_deck_is_latin,
    is_doodle_false_positive,
    mark_recurring_watermarks,
    normalize_ocr_text,
    restore_missing_word_spaces,
)
from video2ppt.pptx_builder import build_pptx


def _make_synthetic_slide(tmp_path: Path) -> tuple[np.ndarray, Path]:
    """Create a synthetic 1280x720 slide image with a bordered card, yellow highlight, doodle, and text."""
    img = np.full((720, 1280, 3), (248, 248, 250), dtype=np.uint8)
    # Draw rounded-style card on left
    cv2.rectangle(img, (80, 120), (540, 420), (255, 255, 255), thickness=-1)
    cv2.rectangle(img, (80, 120), (540, 420), (20, 20, 20), thickness=3)
    # Draw yellow highlight strip inside card
    cv2.rectangle(img, (120, 220), (360, 275), (85, 225, 255), thickness=-1)
    # Draw text inside card and on highlight
    cv2.putText(img, "Card Title", (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 3)
    cv2.putText(img, "Highlighted", (125, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    # Draw a colorful doodle icon on right (blue circle + orange square)
    cv2.circle(img, (900, 300), 70, (210, 140, 60), thickness=-1)
    cv2.circle(img, (900, 300), 70, (20, 20, 20), thickness=3)
    cv2.rectangle(img, (860, 420), (960, 500), (60, 160, 240), thickness=-1)

    path = tmp_path / "synthetic_slide.png"
    cv2.imwrite(str(path), img)
    return img, path


def test_bbox_and_rgb_helpers() -> None:
    a = BBox(x=10, y=20, w=100, h=40)
    b = BBox(x=60, y=20, w=100, h=40)
    assert a.x2 == 110
    assert a.y2 == 60
    assert a.vertical_overlap_ratio(b) == pytest.approx(1.0)
    assert 0.3 < a.iou(b) < 0.4
    u = BBox.union([a, b])
    assert (u.x, u.y, u.w, u.h) == (10, 20, 150, 40)

    c = RGBColor.from_hex("#1E2761")
    assert c.to_tuple() == (30, 39, 97)
    assert c.to_hex() == "#1E2761"


def test_ocr_postprocessing_filters_and_watermarks() -> None:
    assert detect_deck_is_latin([[("Hello World AI Presentation", 0.99), ("Slide Two", 0.98)]])
    assert is_doodle_false_positive("口双", 0.74, BBox(x=0, y=0, w=20, h=40), True, "auto")
    assert is_doodle_false_positive("→", 0.95, BBox(x=0, y=0, w=40, h=20), True, "auto")
    assert is_doodle_false_positive("1010101010", 0.86, BBox(x=0, y=0, w=200, h=20), True, "auto")
    assert not is_doodle_false_positive("1", 0.94, BBox(x=0, y=0, w=20, h=30), True, "auto")

    assert normalize_ocr_text("What if Al was your orderly Al?", deck_mentions_ai=True) == (
        "What if AI was your orderly AI?"
    )
    # Preserve proper name "Al Gore"
    assert normalize_ocr_text("Al Gore spoke about Al systems", deck_mentions_ai=True) == (
        "Al Gore spoke about AI systems"
    )

    dummy_img = np.full((100, 300, 3), 255, dtype=np.uint8)
    assert restore_missing_word_spaces(dummy_img, BBox(x=0, y=0, w=100, h=20), "BudgetUtilised") == (
        "Budget Utilised"
    )
    assert restore_missing_word_spaces(dummy_img, BBox(x=0, y=0, w=100, h=20), "11.5DAYS") == (
        "11.5 DAYS"
    )

    # Recurring corner watermark across slides
    s1 = [TextLine(text="Gemini Notebook", bbox=BBox(x=1150, y=695, w=120, h=18))]
    s2 = [TextLine(text="Gemini Notebook", bbox=BBox(x=1152, y=696, w=118, h=17))]
    s3 = [TextLine(text="Real Heading", bbox=BBox(x=100, y=100, w=400, h=50))]
    mark_recurring_watermarks([s1, s2, s3], width=1280, height=720)
    assert s1[0].is_watermark is True
    assert s2[0].is_watermark is True
    assert s3[0].is_watermark is False


def test_table_detection_and_paragraph_grouping() -> None:
    img = np.full((720, 1280, 3), 255, dtype=np.uint8)
    lines = [
        # 3x3 table
        TextLine(text="Planned", bbox=BBox(x=440, y=200, w=80, h=24), font_size_pt=14),
        TextLine(text="Completed", bbox=BBox(x=585, y=200, w=100, h=24), font_size_pt=14),
        TextLine(text="Budget", bbox=BBox(x=760, y=200, w=90, h=24), font_size_pt=14),
        TextLine(text="Kano", bbox=BBox(x=320, y=300, w=50, h=24), font_size_pt=14),
        TextLine(text="18", bbox=BBox(x=442, y=300, w=24, h=24), font_size_pt=14),
        TextLine(text="9", bbox=BBox(x=586, y=300, w=18, h=24), font_size_pt=14),
        TextLine(text="50.8%", bbox=BBox(x=762, y=300, w=60, h=24), font_size_pt=14),
        TextLine(text="Lagos", bbox=BBox(x=320, y=430, w=60, h=24), font_size_pt=14),
        TextLine(text="30", bbox=BBox(x=441, y=430, w=28, h=24), font_size_pt=14),
        TextLine(text="22", bbox=BBox(x=585, y=430, w=26, h=24), font_size_pt=14),
        TextLine(text="99.2%", bbox=BBox(x=761, y=430, w=60, h=24), font_size_pt=14),
        # Two-line paragraph outside table
        TextLine(text="Summary line one", bbox=BBox(x=80, y=540, w=300, h=30), font_size_pt=20),
        TextLine(text="Summary line two", bbox=BBox(x=80, y=580, w=280, h=30), font_size_pt=20),
    ]
    tables, consumed = detect_tables_and_recover_cells(img, lines)
    assert len(tables) == 1
    assert tables[0].rows == 3
    assert tables[0].cols == 4
    assert len(consumed) == 11

    blocks = group_lines_into_blocks(img, lines, 1280, 720, consumed_indices=consumed)
    assert len(blocks) == 1
    assert blocks[0].text == "Summary line one\nSummary line two"
    assert blocks[0].alignment == "left"


def test_inpainting_preserves_yellow_highlight_and_modular_extracts_elements(tmp_path: Path) -> None:
    img, _ = _make_synthetic_slide(tmp_path)
    lines = [
        TextLine(text="Card Title", bbox=BBox(x=116, y=154, w=240, h=42), font_size_pt=26),
        TextLine(text="Highlighted", bbox=BBox(x=120, y=224, w=220, h=42), font_size_pt=24),
    ]
    blocks = [
        TextBlock(
            block_id="tb_01",
            text="Card Title",
            lines=[lines[0]],
            bbox=lines[0].bbox,
            font_size_pt=26,
        ),
        TextBlock(
            block_id="tb_02",
            text="Highlighted",
            lines=[lines[1]],
            bbox=lines[1].bbox,
            font_size_pt=24,
        ),
    ]
    clean_path = tmp_path / "clean.png"
    clean = inpaint_slide_background(img, text_blocks=blocks, out_path=clean_path)
    assert clean_path.exists()
    # Yellow highlight pixels around y=245, x=200 should still be bright yellow (high R & G, low B)
    patch = clean[230:270, 140:340]
    med_bgr = np.median(patch.reshape(-1, 3), axis=0)
    assert med_bgr[2] > 200 and med_bgr[1] > 180 and med_bgr[0] < 130

    # Modular extraction should find the white bordered card + right-hand graphic doodle(s)
    shapes, graphics = extract_modular_elements(
        clean_bg_bgr=clean,
        canvas_bg=RGBColor(r=250, g=248, b=248),
        text_blocks=blocks,
        assets_dir=tmp_path / "assets",
        slide_index=1,
    )
    assert len(shapes) >= 1
    assert len(graphics) >= 1
    assert Path(graphics[0].image_path).exists()


def test_pptx_builder_hybrid_and_modular_roundtrip(tmp_path: Path) -> None:
    img, raw_path = _make_synthetic_slide(tmp_path)
    clean_path = tmp_path / "clean.png"
    cv2.imwrite(str(clean_path), img)
    graphic_path = tmp_path / "g1.png"
    rgba = np.full((80, 80, 4), (60, 160, 240, 255), dtype=np.uint8)
    cv2.imwrite(str(graphic_path), rgba)

    from video2ppt.models import DetectedShape, GraphicRegion

    slide = SlideLayout(
        slide_index=1,
        start_sec=0.0,
        end_sec=15.0,
        width=1280,
        height=720,
        raw_frame_path=str(raw_path),
        clean_bg_path=str(clean_path),
        background_color=RGBColor(r=250, g=250, b=250),
        text_blocks=[
            TextBlock(
                block_id="tb_01",
                text="Editable Headline\nSecond Line",
                bbox=BBox(x=100, y=100, w=500, h=100),
                alignment="center",
                font_size_pt=32.0,
                bold=True,
                color=RGBColor(r=20, g=20, b=20),
            )
        ],
        shapes=[
            DetectedShape(
                shape_type="rounded_rect",
                bbox=BBox(x=80, y=80, w=540, h=200),
                fill_color=RGBColor(r=255, g=255, b=255),
                border_color=RGBColor(r=20, g=20, b=20),
            )
        ],
        graphics=[
            GraphicRegion(
                bbox=BBox(x=800, y=200, w=120, h=120),
                image_path=str(graphic_path),
            )
        ],
        speaker_notes="Welcome to the AI-Powered Productive Life overview.",
    )
    plan = DeckPlan(
        video_path=str(tmp_path / "demo.mp4"),
        video_meta=VideoMeta(path=str(tmp_path / "demo.mp4"), width=1280, height=720, fps=24.0, duration_sec=15.0),
        mode="hybrid",
        backend="local",
        slides=[slide],
    )

    hybrid_out = build_pptx(plan, tmp_path / "hybrid.pptx", mode_override="hybrid")
    prs_h = Presentation(str(hybrid_out))
    assert len(prs_h.slides) == 1
    texts_h = [s.text_frame.text for s in prs_h.slides[0].shapes if s.has_text_frame]
    assert "Editable Headline\nSecond Line" in texts_h
    assert "Welcome to the AI-Powered Productive Life" in prs_h.slides[0].notes_slide.notes_text_frame.text

    modular_out = build_pptx(plan, tmp_path / "modular.pptx", mode_override="modular")
    prs_m = Presentation(str(modular_out))
    assert len(prs_m.slides) == 1
    # Modular slide has card shape + graphic picture + editable textbox
    assert len(prs_m.slides[0].shapes) == 3


def test_analyzer_factory_fallback_and_vlm_fusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Graceful fallback to local when API key missing
    cfg_fb = PipelineConfig(backend="gemini", fallback_local=True)
    analyzer = create_analyzer(cfg_fb)
    assert analyzer.name == "local"

    # Strict failure when fallback disabled
    cfg_strict = PipelineConfig(backend="openai", fallback_local=False)
    with pytest.raises(EnvironmentError):
        create_analyzer(cfg_strict)

    # Verify VLM JSON -> SlideLayout fusion
    img, raw_path = _make_synthetic_slide(tmp_path)
    seg = SlideSegment(
        slide_index=1, start_sec=0.0, end_sec=10.0, representative_sec=5.0, frame_path=str(raw_path)
    )
    payload = {
        "text_blocks": [
            {
                "text": "Card Title",
                "bbox_1000": [210, 90, 275, 280],
                "alignment": "left",
                "role": "heading",
                "color_hex": "#141414",
                "bold": True,
            }
        ],
        "tables": [],
        "watermark_bbox_1000": None,
    }
    layout = build_slide_layout_from_vlm_json(payload, seg, img, tmp_path / "work", PipelineConfig())
    assert len(layout.text_blocks) == 1
    assert layout.text_blocks[0].text == "Card Title"
    assert Path(layout.clean_bg_path).exists()
