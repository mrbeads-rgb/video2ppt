"""video2ppt: Convert presentation videos into editable PowerPoint (.pptx) decks."""

from video2ppt.config import PipelineConfig
from video2ppt.models import DeckPlan, SlideLayout, SlideSegment, VideoMeta
from video2ppt.pipeline import VideoToPptPipeline

__all__ = [
    "DeckPlan",
    "PipelineConfig",
    "SlideLayout",
    "SlideSegment",
    "VideoMeta",
    "VideoToPptPipeline",
]
__version__ = "0.1.0"
