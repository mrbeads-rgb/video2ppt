"""Visual QA: render generated PPTX via LibreOffice + Poppler and produce side-by-side audits."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from video2ppt.models import DeckPlan


def render_pptx_to_images(pptx_path: str | Path, out_dir: Path, dpi: int = 150) -> list[Path]:
    """Render each slide of a .pptx file to PNG using LibreOffice Impress + pdftoppm."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    pdftoppm = shutil.which("pdftoppm")
    if not soffice or not pdftoppm:
        raise RuntimeError(
            "Visual QA requires LibreOffice (`soffice`) and Poppler (`pdftoppm`) on PATH."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pptx_p = Path(pptx_path).resolve()

    # Convert PPTX -> PDF
    subprocess.run(
        [
            soffice,
            "--headless",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(pptx_p),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pdf_path = out_dir / f"{pptx_p.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected rendered PDF not found: {pdf_path}")

    # Convert PDF -> PNGs
    prefix = out_dir / "rendered_slide"
    subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )

    rendered = sorted(out_dir.glob("rendered_slide-*.png"))
    return rendered


def run_visual_qa(plan: DeckPlan, pptx_path: str | Path, qa_dir: str | Path) -> dict:
    """Render generated PPTX slides, compute fidelity metrics, and build side-by-side comparison sheets."""
    qa_path = Path(qa_dir).resolve()
    qa_path.mkdir(parents=True, exist_ok=True)

    rendered_paths = render_pptx_to_images(pptx_path, qa_path / "rendered", dpi=135)
    metrics: list[dict] = []
    rows_images: list[Image.Image] = []

    tw, th = 426, 240  # Compact crisp 16:9 tile width/height
    for idx, slide in enumerate(plan.slides):
        orig_bgr = cv2.imread(slide.raw_frame_path)
        clean_bgr = (
            cv2.imread(slide.clean_bg_path)
            if slide.clean_bg_path and Path(slide.clean_bg_path).exists()
            else orig_bgr
        )
        rend_bgr = (
            cv2.imread(str(rendered_paths[idx])) if idx < len(rendered_paths) else orig_bgr
        )
        if orig_bgr is None or rend_bgr is None:
            continue

        oh, ow = orig_bgr.shape[:2]
        rend_resized = cv2.resize(rend_bgr, (ow, oh), interpolation=cv2.INTER_AREA)

        # Compute fidelity metrics (MAE and PSNR) between original video frame and rendered editable PPTX
        diff = np.abs(orig_bgr.astype(np.float32) - rend_resized.astype(np.float32))
        mae = float(diff.mean())
        mse = float(np.mean(diff ** 2))
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse))) if mse > 1e-6 else 99.0

        metrics.append(
            {
                "slide_index": slide.slide_index,
                "text_blocks": len(slide.text_blocks),
                "tables": len(slide.tables),
                "shapes": len(slide.shapes),
                "graphics": len(slide.graphics),
                "mae": round(mae, 2),
                "psnr_db": round(psnr, 2),
                "rendered_png": str(rendered_paths[idx]) if idx < len(rendered_paths) else "",
            }
        )

        # Build 3-column comparison row: Original | Clean Inpainted | Rendered Editable PPTX
        row = Image.new("RGB", (tw * 3 + 16, th + 24), (24, 24, 28))
        draw = ImageDraw.Draw(row)
        panels = [
            (f"S{slide.slide_index:02d} Original ({slide.start_sec:.0f}s-{slide.end_sec:.0f}s)", orig_bgr),
            (f"S{slide.slide_index:02d} Clean Inpainted", clean_bgr),
            (f"S{slide.slide_index:02d} Editable PPTX (PSNR {psnr:.1f}dB, {len(slide.text_blocks)} blocks)", rend_resized),
        ]
        for col, (label, bgr_im) in enumerate(panels):
            rgb_im = Image.fromarray(cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)).resize(
                (tw, th), Image.Resampling.LANCZOS
            )
            ox = col * (tw + 8)
            row.paste(rgb_im, (ox, 22))
            draw.text((ox + 6, 4), label, fill=(245, 245, 130))
        rows_images.append(row)

    # Save paged comparison sheets (7 slides per sheet so each image stays sharp and easy to view)
    sheet_paths: list[str] = []
    per_sheet = 7
    for page_idx in range(0, len(rows_images), per_sheet):
        chunk = rows_images[page_idx : page_idx + per_sheet]
        total_h = sum(im.height for im in chunk)
        sheet = Image.new("RGB", (chunk[0].width, total_h), (24, 24, 28))
        y_off = 0
        for im in chunk:
            sheet.paste(im, (0, y_off))
            y_off += im.height
        sp = qa_path / f"comparison_page_{(page_idx // per_sheet) + 1:02d}.jpg"
        sheet.save(sp, quality=92)
        sheet_paths.append(str(sp))

    avg_psnr = float(np.mean([m["psnr_db"] for m in metrics])) if metrics else 0.0
    avg_mae = float(np.mean([m["mae"] for m in metrics])) if metrics else 0.0
    summary = {
        "pptx_path": str(Path(pptx_path).resolve()),
        "total_slides": len(metrics),
        "avg_psnr_db": round(avg_psnr, 2),
        "avg_mae": round(avg_mae, 2),
        "comparison_sheets": sheet_paths,
        "slides": metrics,
    }
    (qa_path / "qa_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
