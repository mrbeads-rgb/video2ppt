"""Local audio transcription with faster-whisper aligned to slide timestamp ranges."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from video2ppt.models import SlideSegment


def extract_audio_wav(video_path: str | Path, out_wav: Path) -> Path:
    """Extract mono 16kHz WAV audio track from video for speech transcription."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def _load_wav_float32(wav_path: Path) -> np.ndarray:
    """Load mono 16-bit PCM WAV directly as float32 [-1, 1] (avoids PyAV>=19 metadata_errors bug)."""
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def transcribe_speaker_notes(
    video_path: str | Path,
    segments: list[SlideSegment],
    work_dir: Path,
    model_size: str = "base.en",
) -> dict[int, str]:
    """Transcribe video narration locally and map spoken sentences to each slide index."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for --speaker-notes. Install with: pip install faster-whisper"
        ) from exc

    wav_path = extract_audio_wav(video_path, work_dir / "audio_16k.wav")
    audio_f32 = _load_wav_float32(wav_path)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    whisper_segs, _ = model.transcribe(audio_f32, vad_filter=True, beam_size=3)

    notes_by_slide: dict[int, list[str]] = {seg.slide_index: [] for seg in segments}
    for w_seg in whisper_segs:
        text = w_seg.text.strip()
        if not text:
            continue
        mid_t = 0.5 * (float(w_seg.start) + float(w_seg.end))
        # Assign to slide containing midpoint (or closest slide boundary)
        matched_idx = None
        for slide in segments:
            if slide.start_sec <= mid_t <= slide.end_sec:
                matched_idx = slide.slide_index
                break
        if matched_idx is None and segments:
            closest = min(
                segments,
                key=lambda s: min(abs(mid_t - s.start_sec), abs(mid_t - s.end_sec)),
            )
            matched_idx = closest.slide_index
        if matched_idx is not None:
            notes_by_slide[matched_idx].append(text)

    return {idx: " ".join(parts).strip() for idx, parts in notes_by_slide.items()}
