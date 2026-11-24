"""End-to-end Video-to-Editable-PPTX pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from video2ppt.analyzers import create_analyzer
from video2ppt.config import PipelineConfig
from video2ppt.models import DeckPlan
from video2ppt.pptx_builder import build_pptx
from video2ppt.qa import run_visual_qa
from video2ppt.transcriber import transcribe_speaker_notes
from video2ppt.video import detect_slide_segments, probe_video

logger = logging.getLogger(__name__)
StageCallback = Callable[[str], None]


class VideoToPptPipeline:
    """Orchestrates video segmentation, layout analysis, inpainting, transcription, and PPTX export."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        video_path: str | Path,
        output_pptx: str | Path,
        stage_cb: StageCallback | None = None,
    ) -> tuple[DeckPlan, Path, dict | None]:
        """Run the full video-to-editable-PPTX pipeline.

        Returns:
            (deck_plan, output_pptx_path, qa_summary_or_none)
        """
        v_path = Path(video_path).resolve()
        out_pptx = Path(output_pptx).resolve()
        work_dir = (
            Path(self.config.work_dir).resolve()
            if self.config.work_dir is not None
            else out_pptx.parent / f".{out_pptx.stem}_work"
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        if stage_cb:
            stage_cb(f"Probing video metadata: {v_path.name}")
        video_meta = probe_video(v_path)

        if stage_cb:
            stage_cb(
                f"Detecting slide transitions ({video_meta.width}x{video_meta.height}, "
                f"{video_meta.duration_sec:.1f}s @ {self.config.sample_fps} fps scan)..."
            )
        segments = detect_slide_segments(
            video_meta=video_meta,
            config=self.config,
            frames_dir=work_dir / "frames",
        )
        if stage_cb:
            stage_cb(f"Detected {len(segments)} distinct slide(s).")

        analyzer = create_analyzer(self.config)
        if stage_cb:
            stage_cb(
                f"Running '{analyzer.name}' analyzer in '{self.config.mode}' reconstruction mode..."
            )

        def _prog(cur: int, tot: int, msg: str) -> None:
            if stage_cb and (cur == 1 or cur == tot or cur % 5 == 0):
                stage_cb(f"  [{cur}/{tot}] {msg}")

        try:
            slides = analyzer.analyze_deck(segments, work_dir=work_dir, progress_cb=_prog)
        except Exception as exc:
            if self.config.backend != "local" and self.config.fallback_local:
                logger.warning(
                    "Backend '%s' failed (%s); falling back to local OCR.",
                    self.config.backend,
                    exc,
                )
                if stage_cb:
                    stage_cb(f"Backend '{self.config.backend}' failed; falling back to local OCR...")
                fallback_cfg = self.config.model_copy(update={"backend": "local"})
                analyzer = create_analyzer(fallback_cfg)
                slides = analyzer.analyze_deck(segments, work_dir=work_dir, progress_cb=_prog)
            else:
                raise

        # Optional audio transcription -> per-slide speaker notes
        if self.config.speaker_notes and video_meta.has_audio:
            if stage_cb:
                stage_cb(
                    f"Transcribing audio into speaker notes (Whisper '{self.config.whisper_model}')..."
                )
            notes_map = transcribe_speaker_notes(
                video_path=v_path,
                segments=segments,
                work_dir=work_dir,
                model_size=self.config.whisper_model,
            )
            for s_layout in slides:
                s_layout.speaker_notes = notes_map.get(s_layout.slide_index, "")

        plan = DeckPlan(
            video_path=str(v_path),
            video_meta=video_meta,
            mode=self.config.mode,
            backend=analyzer.name,  # type: ignore[arg-type]
            slides=slides,
        )

        if self.config.save_plan_json:
            plan_path = work_dir / "deck_plan.json"
            plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            if stage_cb:
                stage_cb(f"Saved editable intermediate deck plan: {plan_path}")

        if stage_cb:
            stage_cb(f"Building editable PowerPoint deck ({self.config.mode} mode): {out_pptx}")
        built_path = build_pptx(plan, out_pptx)

        qa_summary = None
        if self.config.run_qa:
            if stage_cb:
                stage_cb("Rendering PPTX slides via LibreOffice for visual QA audit...")
            qa_summary = run_visual_qa(plan, built_path, work_dir / "qa")

        return plan, built_path, qa_summary

    @staticmethod
    def build_from_plan(
        plan_json_path: str | Path,
        output_pptx: str | Path,
        mode_override: str | None = None,
    ) -> tuple[DeckPlan, Path]:
        """Rebuild a .pptx directly from a saved/edited deck_plan.json."""
        plan = DeckPlan.model_validate_json(
            Path(plan_json_path).read_text(encoding="utf-8")
        )
        out = build_pptx(plan, output_pptx, mode_override=mode_override)
        return plan, out
