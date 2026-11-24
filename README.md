# `video2ppt` — Video to Editable PowerPoint Slide Pipeline

Convert narrated slide-deck videos (e.g., NotebookLM Video Overviews, recorded presentations, explainer videos) into genuinely **editable PowerPoint (`.pptx`)** decks.

## Key Features

- **Adaptive Slide Segmentation** (`FFmpeg` + structural diff): Detects static slide intervals, skips crossfade transitions, preserves short section dividers (down to 1s), and optionally strips trailing outro logo screens (`--skip-outro`).
- **Dual Reconstruction Modes**:
  - **`--mode hybrid` (default — Highest Visual Fidelity)**: Surgically inpaints text glyph strokes off the slide canvas (preserving illustrations, hand-drawn doodles, graph-paper grids, and colored highlight pills intact) and overlays pixel-aligned native editable PowerPoint text boxes and tables.
  - **`--mode modular` (Highest Layout Flexibility)**: Decomposes each slide into native vector PowerPoint shapes (cards, panels, highlight pills), independently movable transparent RGBA graphic cutouts, native tables, and editable text boxes.
- **Pluggable Extraction Backends**:
  - **`--backend local` (default — 100% local & free)**: RapidOCR (ONNX) + pixel-level typography measurement (cap-height font sizing, stroke-width boldness, ink color, highlight pill detection, metric-compatible font calibration, overlapping-span re-recognition, table grid detection with missing-cell recovery).
  - **`--backend gemini`**: Google Gemini Vision API (`GEMINI_API_KEY`) fused with local OpenCV surgical inpainting and modular cropping.
  - **`--backend openai`**: OpenAI Vision API (`OPENAI_API_KEY`) fused with local OpenCV surgical inpainting and modular cropping.
  - Automatic graceful fallback to local OCR if an API key is missing (disable with `--no-fallback`).
- **Optional Audio Speaker Notes (`--speaker-notes`)**: Transcribes the video narration locally via `faster-whisper` and maps spoken narration directly into each slide's native PowerPoint Speaker Notes.
- **Editable Intermediate Plan (`deck_plan.json`)**: Edit extracted text, styles, or notes in JSON and rebuild a `.pptx` instantaneously with `video2ppt build`.
- **Built-in Visual QA (`--qa`)**: Renders the generated `.pptx` via LibreOffice + Poppler and produces side-by-side comparison contact sheets (`Original | Clean Inpainted | Rendered PPTX`) with PSNR/MAE fidelity scores.

## Installation

```bash
cd /home/antigravity/video2ppt
uv venv .venv
uv pip install --python .venv/bin/python -e ".[whisper,dev]"
```

System prerequisites: `ffmpeg` / `ffprobe` (required); `libreoffice-impress` + `poppler-utils` (optional, for `--qa`).

## Quick Start (CLI)

### 1. Hybrid Mode (default, 100% local)
```bash
.venv/bin/video2ppt convert input/AI-Powered_Productive_Life.mp4 \
  -o output/AI-Powered_Productive_Life_hybrid.pptx \
  --mode hybrid \
  --skip-outro \
  --qa
```

### 2. Modular Mode (movable cards, highlights, and transparent graphics)
```bash
.venv/bin/video2ppt convert input/AI-Powered_Productive_Life.mp4 \
  -o output/AI-Powered_Productive_Life_modular.pptx \
  --mode modular \
  --skip-outro
```

### 3. With Local Whisper Speaker Notes & Watermark Removal
```bash
.venv/bin/video2ppt convert input/AI-Powered_Productive_Life.mp4 \
  -o output/deck_with_notes.pptx \
  --mode hybrid \
  --speaker-notes \
  --whisper-model base.en \
  --remove-watermark \
  --skip-outro
```

### 4. Using Gemini or OpenAI Vision Backend
```bash
export GEMINI_API_KEY="your-key"
.venv/bin/video2ppt convert input/video.mp4 -o output/deck.pptx --backend gemini

export OPENAI_API_KEY="your-key"
.venv/bin/video2ppt convert input/video.mp4 -o output/deck.pptx --backend openai
```

### 5. Instant Rebuild After Editing `deck_plan.json`
```bash
.venv/bin/video2ppt build output/work/deck_plan.json -o output/deck_edited.pptx
```

## Python API

```python
from video2ppt import PipelineConfig, VideoToPptPipeline

cfg = PipelineConfig(
    mode="hybrid",          # or "modular"
    backend="local",        # or "gemini" / "openai"
    skip_outro=True,
    speaker_notes=False,
    remove_watermark=False,
)
pipeline = VideoToPptPipeline(cfg)
plan, pptx_path, qa_summary = pipeline.run(
    "input/AI-Powered_Productive_Life.mp4",
    "output/deck.pptx",
)
```

## Running Tests

```bash
.venv/bin/pytest -v
```
