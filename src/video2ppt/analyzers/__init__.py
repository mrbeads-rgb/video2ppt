"""Pluggable slide analyzer factory."""

from __future__ import annotations

import logging

from video2ppt.analyzers.base import SlideAnalyzer
from video2ppt.analyzers.gemini import GeminiVisionAnalyzer
from video2ppt.analyzers.local import LocalSlideAnalyzer
from video2ppt.analyzers.openai import OpenAIVisionAnalyzer
from video2ppt.config import PipelineConfig

logger = logging.getLogger(__name__)


def create_analyzer(config: PipelineConfig) -> SlideAnalyzer:
    """Instantiate the requested slide analyzer backend, optionally falling back to local OCR."""
    if config.backend == "local":
        return LocalSlideAnalyzer(config)

    try:
        if config.backend == "gemini":
            return GeminiVisionAnalyzer(config)
        if config.backend == "openai":
            return OpenAIVisionAnalyzer(config)
    except EnvironmentError as exc:
        if config.fallback_local:
            logger.warning("%s Falling back to local OCR backend.", exc)
            return LocalSlideAnalyzer(config)
        raise

    raise ValueError(f"Unsupported backend: {config.backend}")


__all__ = [
    "GeminiVisionAnalyzer",
    "LocalSlideAnalyzer",
    "OpenAIVisionAnalyzer",
    "SlideAnalyzer",
    "create_analyzer",
]
