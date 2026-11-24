"""Typed data models for the video-to-editable-PPTX pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RGBColor(BaseModel):
    """24-bit RGB color."""

    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)

    def to_hex(self) -> str:
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_bgr(self) -> tuple[int, int, int]:
        return (self.b, self.g, self.r)

    @classmethod
    def from_hex(cls, value: str) -> RGBColor:
        clean = value.strip().lstrip("#")
        if len(clean) == 3:
            clean = "".join(ch * 2 for ch in clean)
        if len(clean) != 6:
            return cls(r=26, g=26, b=26)
        return cls(
            r=int(clean[0:2], 16),
            g=int(clean[2:4], 16),
            b=int(clean[4:6], 16),
        )


class BBox(BaseModel):
    """Axis-aligned pixel bounding box (top-left origin)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def expanded(self, pad_x: float, pad_y: float, max_w: int, max_h: int) -> BBox:
        x1 = max(0.0, self.x - pad_x)
        y1 = max(0.0, self.y - pad_y)
        x2 = min(float(max_w), self.x2 + pad_x)
        y2 = min(float(max_h), self.y2 + pad_y)
        return BBox(x=x1, y=y1, w=max(1.0, x2 - x1), h=max(1.0, y2 - y1))

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def vertical_overlap_ratio(self, other: BBox) -> float:
        oy = max(0.0, min(self.y2, other.y2) - max(self.y, other.y))
        denom = min(self.h, other.h)
        return oy / denom if denom > 0 else 0.0

    def horizontal_overlap_ratio(self, other: BBox) -> float:
        ox = max(0.0, min(self.x2, other.x2) - max(self.x, other.x))
        denom = min(self.w, other.w)
        return ox / denom if denom > 0 else 0.0

    @classmethod
    def union(cls, boxes: list[BBox]) -> BBox:
        x1 = min(b.x for b in boxes)
        y1 = min(b.y for b in boxes)
        x2 = max(b.x2 for b in boxes)
        y2 = max(b.y2 for b in boxes)
        return cls(x=x1, y=y1, w=max(1.0, x2 - x1), h=max(1.0, y2 - y1))

    @classmethod
    def from_polygon(cls, polygon: list[list[float]]) -> BBox:
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        return cls(
            x=min(xs),
            y=min(ys),
            w=max(1.0, max(xs) - min(xs)),
            h=max(1.0, max(ys) - min(ys)),
        )


class VideoMeta(BaseModel):
    """Metadata probed from an input video file."""

    path: str
    width: int
    height: int
    fps: float
    duration_sec: float
    has_audio: bool = True


class SlideSegment(BaseModel):
    """Detected static slide segment in the source video."""

    slide_index: int
    start_sec: float
    end_sec: float
    representative_sec: float
    frame_path: str
    is_outro: bool = False

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


class HighlightSpan(BaseModel):
    """Colored marker / highlight pill behind text."""

    bbox: BBox
    color: RGBColor


class TextLine(BaseModel):
    """Single detected text line with geometry and visual typography."""

    text: str
    bbox: BBox
    polygon: list[list[float]] = Field(default_factory=list)
    confidence: float = 1.0
    font_size_pt: float = 18.0
    cap_height_px: float = 20.0
    color: RGBColor = Field(default_factory=lambda: RGBColor(r=26, g=26, b=26))
    bg_color: RGBColor = Field(default_factory=lambda: RGBColor(r=255, g=255, b=255))
    bold: bool = False
    italic: bool = False
    font_family: str = "Arial"
    highlights: list[HighlightSpan] = Field(default_factory=list)
    is_watermark: bool = False


TextRole = Literal[
    "title",
    "subtitle",
    "heading",
    "body",
    "bullet",
    "caption",
    "number",
    "table_cell",
]
TextAlignment = Literal["left", "center", "right"]


class TextBlock(BaseModel):
    """Grouped editable text box containing one or more lines."""

    block_id: str = ""
    text: str
    lines: list[TextLine] = Field(default_factory=list)
    bbox: BBox
    alignment: TextAlignment = "left"
    role: TextRole = "body"
    font_size_pt: float = 18.0
    color: RGBColor = Field(default_factory=lambda: RGBColor(r=26, g=26, b=26))
    bold: bool = False
    italic: bool = False
    font_family: str = "Arial"
    line_spacing: float = 1.15
    line_step_pt: float | None = None
    highlights: list[HighlightSpan] = Field(default_factory=list)


class DetectedShape(BaseModel):
    """Native PowerPoint vector shape (card, panel, highlight pill)."""

    shape_type: Literal["rounded_rect", "rect", "highlight"] = "rounded_rect"
    bbox: BBox
    fill_color: RGBColor
    border_color: RGBColor | None = None
    border_width_pt: float = 1.5
    corner_radius_ratio: float = 0.08
    z_order: int = 0


class GraphicRegion(BaseModel):
    """Extracted movable illustration or icon asset for modular mode."""

    bbox: BBox
    image_path: str
    z_order: int = 10
    label: str = ""


class TableCell(BaseModel):
    """Single cell within a detected native table."""

    row: int
    col: int
    text: str
    bbox: BBox
    font_size_pt: float = 14.0
    color: RGBColor = Field(default_factory=lambda: RGBColor(r=26, g=26, b=26))
    bold: bool = False
    alignment: TextAlignment = "left"


class DetectedTable(BaseModel):
    """Grid-aligned tabular structure reconstructed as a native PPTX table."""

    bbox: BBox
    rows: int
    cols: int
    cells: list[TableCell] = Field(default_factory=list)
    col_widths_px: list[float] = Field(default_factory=list)
    row_heights_px: list[float] = Field(default_factory=list)


class SlideLayout(BaseModel):
    """Complete intermediate representation of one editable slide."""

    slide_index: int
    start_sec: float
    end_sec: float
    width: int
    height: int
    raw_frame_path: str
    clean_bg_path: str = ""
    background_color: RGBColor = Field(default_factory=lambda: RGBColor(r=250, g=250, b=250))
    text_blocks: list[TextBlock] = Field(default_factory=list)
    raw_lines: list[TextLine] = Field(default_factory=list)
    shapes: list[DetectedShape] = Field(default_factory=list)
    graphics: list[GraphicRegion] = Field(default_factory=list)
    tables: list[DetectedTable] = Field(default_factory=list)
    watermark_bboxes: list[BBox] = Field(default_factory=list)
    speaker_notes: str = ""
    is_outro: bool = False


class DeckPlan(BaseModel):
    """Serializable full deck representation emitted before PPTX assembly."""

    video_path: str
    video_meta: VideoMeta
    mode: Literal["hybrid", "modular"] = "hybrid"
    backend: Literal["local", "gemini", "openai"] = "local"
    slides: list[SlideLayout] = Field(default_factory=list)
