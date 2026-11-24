"""Video probing, adaptive slide segmentation, and keyframe extraction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from video2ppt.config import PipelineConfig
from video2ppt.models import SlideSegment, VideoMeta


def probe_video(video_path: str | Path) -> VideoMeta:
    """Inspect video metadata using ffprobe."""
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))

    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "24/1"
    if "/" in fps_raw:
        num, den = fps_raw.split("/", 1)
        fps = float(num) / max(1.0, float(den))
    else:
        fps = float(fps_raw)

    duration = float(
        video_stream.get("duration") or data.get("format", {}).get("duration") or 0.0
    )

    return VideoMeta(
        path=str(path),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        duration_sec=duration,
        has_audio=has_audio,
    )


def _stream_downscaled_frames(
    video_path: str, sample_fps: float, thumb_w: int = 160, thumb_h: int = 90
) -> list[tuple[float, np.ndarray]]:
    """Decode small RGB thumbnails via FFmpeg pipe for fast transition scanning."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={sample_fps},scale={thumb_w}:{thumb_h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = thumb_w * thumb_h * 3
    frames: list[tuple[float, np.ndarray]] = []
    idx = 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((thumb_h, thumb_w, 3)).copy()
        ts = (idx + 0.5) / sample_fps
        frames.append((ts, arr))
        idx += 1
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg frame decode failed: {err}")
    return frames


def _frame_diff_score(a: np.ndarray, b: np.ndarray) -> float:
    """Combined mean + localized structural difference score between two RGB thumbnails."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    mean_diff = float(diff.mean())
    # Also catch smaller localized slide builds (e.g. a new bullet appearing)
    p97_diff = float(np.percentile(diff, 97))
    return max(mean_diff, p97_diff * 0.22)


def _is_likely_outro_frame(frame_rgb: np.ndarray) -> bool:
    """Heuristic check for a sparse centered logo/splash outro screen."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    # Check outer margins (top 30%, bottom 30%, left 20%, right 20%)
    bg_val = float(np.median(gray))
    diff = np.abs(gray.astype(np.float32) - bg_val)
    non_bg = diff > 15.0
    total_ink_ratio = float(non_bg.mean())

    # Outer border band should be virtually blank
    border_mask = np.ones_like(non_bg, dtype=bool)
    border_mask[int(h * 0.35) : int(h * 0.65), int(w * 0.15) : int(w * 0.85)] = False
    border_ink_ratio = float(non_bg[border_mask].mean())

    # Typical logo outro: < 4% total non-bg pixels, < 0.25% in outer band
    return total_ink_ratio < 0.055 and border_ink_ratio < 0.004


def extract_frame_at(video_path: str, timestamp_sec: float, out_path: Path) -> None:
    """Extract one lossless full-resolution PNG frame at the given timestamp."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def detect_slide_segments(
    video_meta: VideoMeta,
    config: PipelineConfig,
    frames_dir: Path,
) -> list[SlideSegment]:
    """Segment a video into distinct stable slides and extract full-res keyframes."""
    sampled = _stream_downscaled_frames(video_meta.path, sample_fps=config.sample_fps)
    if not sampled:
        raise RuntimeError("No frames could be decoded from video.")

    n = len(sampled)
    step_diffs = [0.0]
    for i in range(1, n):
        step_diffs.append(_frame_diff_score(sampled[i - 1][1], sampled[i][1]))

    # Mark transition boundaries whenever diff spikes above threshold
    # Also collapse consecutive transition frames (e.g. a 0.5-1.5s crossfade)
    is_transition = [d >= config.diff_threshold for d in step_diffs]

    # Collect contiguous stable runs of frames
    raw_runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if is_transition[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and not is_transition[i + 1]:
            i += 1
        end = i
        raw_runs.append((start, end))
        i += 1

    # Filter out ultra-short transition artifacts and pick representative frames
    min_frames = max(1, int(round(config.min_slide_duration * config.sample_fps)))
    kept_runs: list[tuple[int, int, int]] = []  # (start_idx, end_idx, rep_idx)

    for s_idx, e_idx in raw_runs:
        dur = (e_idx - s_idx + 1) / config.sample_fps
        if (e_idx - s_idx + 1) < min_frames and dur < config.min_slide_duration:
            continue
        # Pick representative frame strictly inside the stable plateau (skip edge frames)
        margin = max(0, min(2, (e_idx - s_idx) // 4))
        inner_s = s_idx + margin
        inner_e = e_idx - margin
        rep_idx = (inner_s + inner_e) // 2
        # Refine rep_idx to the frame with minimal local diff to neighbors
        best_idx = rep_idx
        best_motion = float("inf")
        for cand in range(inner_s, inner_e + 1):
            motion = step_diffs[cand] + (step_diffs[cand + 1] if cand + 1 < n else 0.0)
            if motion < best_motion:
                best_motion = motion
                best_idx = cand
        kept_runs.append((s_idx, e_idx, best_idx))

    if not kept_runs:
        # Fallback: single slide from midpoint of entire video
        mid = n // 2
        kept_runs = [(0, n - 1, mid)]

    # Deduplicate adjacent near-identical runs (e.g. minor encoding blip or cursor flicker)
    merged_runs: list[tuple[int, int, int]] = [kept_runs[0]]
    for s_idx, e_idx, rep_idx in kept_runs[1:]:
        prev_s, prev_e, prev_rep = merged_runs[-1]
        rep_diff = _frame_diff_score(sampled[prev_rep][1], sampled[rep_idx][1])
        if rep_diff <= config.dedupe_threshold:
            # Merge into previous slide, keep representative from longer plateau
            new_rep = prev_rep if (prev_e - prev_s) >= (e_idx - s_idx) else rep_idx
            merged_runs[-1] = (prev_s, e_idx, new_rep)
        else:
            merged_runs.append((s_idx, e_idx, rep_idx))

    # Build contiguous time boundaries covering the full video for accurate audio alignment
    segments: list[SlideSegment] = []
    frames_dir.mkdir(parents=True, exist_ok=True)
    for k, (s_idx, e_idx, rep_idx) in enumerate(merged_runs):
        start_sec = 0.0 if k == 0 else max(0.0, sampled[s_idx][0] - 0.5 / config.sample_fps)
        if k + 1 < len(merged_runs):
            next_start = max(
                0.0, sampled[merged_runs[k + 1][0]][0] - 0.5 / config.sample_fps
            )
            end_sec = next_start
        else:
            end_sec = video_meta.duration_sec

        rep_sec = sampled[rep_idx][0]
        is_last = k == len(merged_runs) - 1
        is_outro = bool(
            is_last
            and (end_sec - start_sec) <= 8.0
            and _is_likely_outro_frame(sampled[rep_idx][1])
        )
        if is_outro and config.skip_outro:
            continue

        slide_num = len(segments) + 1
        frame_path = frames_dir / f"slide_{slide_num:03d}.png"
        extract_frame_at(video_meta.path, rep_sec, frame_path)

        segments.append(
            SlideSegment(
                slide_index=slide_num,
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                representative_sec=round(rep_sec, 3),
                frame_path=str(frame_path),
                is_outro=is_outro,
            )
        )

    return segments
