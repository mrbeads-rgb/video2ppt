"""Pluggable OpenAI Vision API analyzer backend."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import cv2
import httpx

from video2ppt.analyzers.base import ProgressCallback, SlideAnalyzer
from video2ppt.analyzers.gemini import (
    VLM_EXTRACTION_PROMPT,
    _strip_json_fences,
    build_slide_layout_from_vlm_json,
)
from video2ppt.config import PipelineConfig
from video2ppt.models import SlideLayout, SlideSegment


class OpenAIVisionAnalyzer(SlideAnalyzer):
    """Optional OpenAI Vision API backend (requires OPENAI_API_KEY)."""

    name = "openai"

    def __init__(self, config: PipelineConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OpenAI backend requires OPENAI_API_KEY environment variable."
            )

    def _call_openai(self, frame_path: str) -> dict[str, Any]:
        img_b64 = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.config.openai_model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
        with httpx.Client(timeout=self.config.api_timeout_sec) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        return json.loads(_strip_json_fences(raw_text))

    def analyze_deck(
        self,
        segments: list[SlideSegment],
        work_dir: Path,
        progress_cb: ProgressCallback | None = None,
    ) -> list[SlideLayout]:
        layouts: list[SlideLayout] = []
        total = len(segments)
        for idx, seg in enumerate(segments):
            if progress_cb:
                progress_cb(idx + 1, total, f"OpenAI Vision analyzing slide {seg.slide_index}/{total}")
            img_bgr = cv2.imread(seg.frame_path)
            payload = self._call_openai(seg.frame_path)
            layouts.append(
                build_slide_layout_from_vlm_json(payload, seg, img_bgr, work_dir, self.config)
            )
        return layouts
