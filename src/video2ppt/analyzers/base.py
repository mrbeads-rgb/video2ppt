"""Abstract interface for pluggable slide extraction backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from video2ppt.config import PipelineConfig
from video2ppt.models import SlideLayout, SlideSegment

ProgressCallback = Callable[[int, int, str], None]


class SlideAnalyzer(ABC):
    """Base class for all slide text, typography, and visual layout extraction engines."""

    name: str = "base"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @abstractmethod
    def analyze_deck(
        self,
        segments: list[SlideSegment],
        work_dir: Path,
        progress_cb: ProgressCallback | None = None,
    ) -> list[SlideLayout]:
        """Extract structured SlideLayouts (including clean backgrounds / assets) for all slides."""
        raise NotImplementedError
