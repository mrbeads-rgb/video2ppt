"""Configuration schema for the video2ppt pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """All knobs governing video segmentation, extraction, inpainting, and PPTX export."""

    # Core modes
    mode: Literal["hybrid", "modular"] = Field(
        default="hybrid",
        description="Slide reconstruction mode: 'hybrid' (inpainted artwork + editable text) or 'modular' (native shapes + movable graphics + editable text).",
    )
    backend: Literal["local", "gemini", "openai"] = Field(
        default="local",
        description="Text & layout extraction backend.",
    )
    fallback_local: bool = Field(
        default=True,
        description="Fallback to local OCR if a Vision LLM backend fails or lacks an API key.",
    )

    # Video segmentation
    sample_fps: float = Field(default=2.0, gt=0.1, le=10.0)
    diff_threshold: float = Field(default=6.0, ge=1.0, le=60.0)
    min_slide_duration: float = Field(default=1.0, ge=0.2)
    dedupe_threshold: float = Field(default=2.5, ge=0.0)
    skip_outro: bool = Field(
        default=False,
        description="Automatically drop trailing platform logo/outro splash slides.",
    )

    # OCR & layout analysis
    lang: Literal["auto", "en", "multi"] = "auto"
    ocr_min_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    ocr_postprocess: bool = Field(
        default=True,
        description="Normalize common OCR artifacts (overlapping spans, spacing, AI/Al).",
    )
    detect_tables: bool = Field(
        default=True,
        description="Detect tabular text grids and reconstruct them as native PowerPoint tables.",
    )
    remove_watermark: bool = Field(
        default=False,
        description="Inpaint out recurring corner platform watermarks.",
    )

    # Typography
    default_sans_font: str = "Arial"
    default_serif_font: str = "Times New Roman"

    # Audio transcription -> Speaker notes
    speaker_notes: bool = Field(
        default=False,
        description="Transcribe narration audio with faster-whisper into per-slide speaker notes.",
    )
    whisper_model: str = "base.en"

    # Optional Vision LLM backend settings
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o"
    api_timeout_sec: float = 90.0

    # Output & QA
    work_dir: Path | None = None
    run_qa: bool = False
    save_plan_json: bool = True
