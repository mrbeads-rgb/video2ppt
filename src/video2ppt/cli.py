"""Command-line interface for video2ppt."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from video2ppt.config import PipelineConfig
from video2ppt.models import DeckPlan
from video2ppt.pipeline import VideoToPptPipeline
from video2ppt.qa import run_visual_qa

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="video2ppt")
def main() -> None:
    """Convert presentation/explainer videos into editable PowerPoint (.pptx) decks."""


@main.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output .pptx file path (defaults to <video_stem>.pptx).",
)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "modular"], case_sensitive=False),
    default="hybrid",
    show_default=True,
    help="Reconstruction mode: 'hybrid' (inpainted artwork + editable text) or 'modular' (native cards + movable graphics + text).",
)
@click.option(
    "--backend",
    type=click.Choice(["local", "gemini", "openai"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Text/layout extraction backend.",
)
@click.option(
    "--no-fallback",
    is_flag=True,
    default=False,
    help="Fail instead of falling back to local OCR when a Vision LLM backend is unavailable.",
)
@click.option(
    "--speaker-notes",
    is_flag=True,
    default=False,
    help="Transcribe video narration locally via faster-whisper into per-slide speaker notes.",
)
@click.option(
    "--whisper-model",
    default="base.en",
    show_default=True,
    help="faster-whisper model name for --speaker-notes.",
)
@click.option(
    "--remove-watermark",
    is_flag=True,
    default=False,
    help="Inpaint out recurring corner platform watermarks.",
)
@click.option(
    "--skip-outro",
    is_flag=True,
    default=False,
    help="Automatically drop trailing platform logo/splash outro slide.",
)
@click.option(
    "--sample-fps",
    type=float,
    default=2.0,
    show_default=True,
    help="Frame sampling rate (FPS) used for slide transition detection.",
)
@click.option(
    "--diff-threshold",
    type=float,
    default=6.0,
    show_default=True,
    help="Frame difference threshold for detecting a new slide.",
)
@click.option(
    "--min-duration",
    type=float,
    default=1.0,
    show_default=True,
    help="Minimum duration (seconds) of a valid slide.",
)
@click.option(
    "--font",
    default="Arial",
    show_default=True,
    help="Default sans-serif font family for editable text boxes.",
)
@click.option(
    "--no-tables",
    is_flag=True,
    default=False,
    help="Disable native PowerPoint table reconstruction.",
)
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to store extracted frames, inpainted backgrounds, assets, and deck_plan.json.",
)
@click.option(
    "--qa",
    is_flag=True,
    default=False,
    help="Render the generated PPTX with LibreOffice and produce visual QA comparison sheets.",
)
def convert(
    video: Path,
    output: Path | None,
    mode: str,
    backend: str,
    no_fallback: bool,
    speaker_notes: bool,
    whisper_model: str,
    remove_watermark: bool,
    skip_outro: bool,
    sample_fps: float,
    diff_threshold: float,
    min_duration: float,
    font: str,
    no_tables: bool,
    work_dir: Path | None,
    qa: bool,
) -> None:
    """Convert a video file into an editable PowerPoint (.pptx) presentation."""
    out_path = output or video.with_suffix(".pptx")
    cfg = PipelineConfig(
        mode=mode.lower(),  # type: ignore[arg-type]
        backend=backend.lower(),  # type: ignore[arg-type]
        fallback_local=not no_fallback,
        speaker_notes=speaker_notes,
        whisper_model=whisper_model,
        remove_watermark=remove_watermark,
        skip_outro=skip_outro,
        sample_fps=sample_fps,
        diff_threshold=diff_threshold,
        min_slide_duration=min_duration,
        default_sans_font=font,
        detect_tables=not no_tables,
        work_dir=work_dir,
        run_qa=qa,
    )

    pipeline = VideoToPptPipeline(cfg)
    plan, built_pptx, qa_summary = pipeline.run(
        video_path=video,
        output_pptx=out_path,
        stage_cb=lambda msg: console.print(f"[cyan]▸[/cyan] {msg}"),
    )

    tbl = Table(title="video2ppt Summary", show_header=True)
    tbl.add_column("Slide", justify="right")
    tbl.add_column("Time Range")
    tbl.add_column("Text Blocks", justify="right")
    tbl.add_column("Tables", justify="right")
    tbl.add_column("Shapes", justify="right")
    tbl.add_column("Graphics", justify="right")
    tbl.add_column("Notes")

    for s in plan.slides:
        tbl.add_row(
            str(s.slide_index),
            f"{s.start_sec:.1f}s – {s.end_sec:.1f}s",
            str(len(s.text_blocks)),
            str(len(s.tables)),
            str(len(s.shapes)),
            str(len(s.graphics)),
            f"{len(s.speaker_notes.split())} words" if s.speaker_notes else "-",
        )
    console.print(tbl)
    console.print(f"\n[bold green]✓ Generated editable PPTX:[/bold green] {built_pptx}")
    if qa_summary:
        console.print(
            f"[bold blue]✓ Visual QA:[/bold blue] avg PSNR = {qa_summary['avg_psnr_db']} dB, "
            f"avg MAE = {qa_summary['avg_mae']}, sheets = {len(qa_summary['comparison_sheets'])}"
        )


@main.command()
@click.argument("plan_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output .pptx file path.",
)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "modular"], case_sensitive=False),
    default=None,
    help="Optional mode override when rebuilding from deck_plan.json.",
)
def build(plan_json: Path, output: Path, mode: str | None) -> None:
    """Rebuild an editable .pptx directly from a saved/edited deck_plan.json."""
    _, built = VideoToPptPipeline.build_from_plan(
        plan_json, output, mode_override=mode.lower() if mode else None
    )
    console.print(f"[bold green]✓ Rebuilt PPTX from plan:[/bold green] {built}")


@main.command()
@click.argument("plan_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("pptx_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("qa_output"),
    show_default=True,
    help="Directory for rendered slides and side-by-side comparison sheets.",
)
def qa(plan_json: Path, pptx_file: Path, out_dir: Path) -> None:
    """Render an existing .pptx via LibreOffice and run side-by-side visual QA against the deck plan."""
    plan = DeckPlan.model_validate_json(plan_json.read_text(encoding="utf-8"))
    summary = run_visual_qa(plan, pptx_file, out_dir)
    console.print(
        f"[bold blue]✓ QA complete:[/bold blue] {summary['total_slides']} slides | "
        f"avg PSNR = {summary['avg_psnr_db']} dB | avg MAE = {summary['avg_mae']}"
    )
    for sheet in summary["comparison_sheets"]:
        console.print(f"  Sheet: {sheet}")


if __name__ == "__main__":
    main()
