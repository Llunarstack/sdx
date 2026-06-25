"""Shared types for the retrieve → transform → compose video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class VideoMode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"


class TransitionType(str, Enum):
    CUT = "cut"
    DISSOLVE = "dissolve"
    MATCH_ACTION = "match_action"
    WHIP = "whip"
    FLASH = "flash"
    DIP = "dip_to_black"


class RetrievalSource(str, Enum):
    LOCAL = "local"
    USER_UPLOAD = "user_upload"
    PEXELS = "pexels"
    WEB_CATALOG = "web_catalog"
    SYNTHETIC = "synthetic"


@dataclass(slots=True)
class MasterTimeline:
    """Output timeline specification."""

    fps: float = 24.0
    width: int = 1280
    height: int = 720
    duration_sec: float = 4.0
    aspect_ratio: str = "16:9"


@dataclass(slots=True)
class ShotSpec:
    """One planned shot in the edit."""

    index: int
    prompt: str
    duration_sec: float
    shot_type: str = "medium"
    lens_hint: str = ""
    negative: str = ""
    motion_hint: str = ""
    must_preserve: List[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoPlan:
    """Full plan from user prompt → shot list + timeline."""

    mode: VideoMode
    user_prompt: str
    timeline: MasterTimeline
    shots: List[ShotSpec]
    global_negative: str = ""
    style_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClipCandidate:
    """A retrievable reference clip."""

    source: RetrievalSource
    path: str
    title: str = ""
    tags: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    fps: float = 24.0
    width: int = 0
    height: int = 0
    license: str = ""
    url: str = ""
    score: float = 0.0
    motion_score: float = 0.0


@dataclass(slots=True)
class SegmentAssignment:
    """Maps a shot to a retrieved clip (or synthetic)."""

    shot: ShotSpec
    clip: Optional[ClipCandidate]
    in_sec: float = 0.0
    out_sec: float = 0.0
    use_motion_only: bool = True
    keyframe_interval: int = 6
    edit_strength: float = 0.55
    transition: TransitionType = TransitionType.CUT
    start_image: str = ""
    end_image: str = ""
    flf2v: bool = False
    motion_brush: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProvenanceRecord:
    """Audit trail for one segment."""

    segment_index: int
    source: str
    source_path: str
    license: str
    url: str
    operations: List[str] = field(default_factory=list)
    retrieved_at: str = ""


@dataclass(slots=True)
class KeyframeEditJob:
    """One keyframe to edit via sample.py."""

    segment_index: int
    frame_index: int
    source_frame_path: str
    output_path: str
    prompt: str
    negative: str = ""
    init_strength: float = 0.65
    mask_path: str = ""
    sample_extra_args: List[str] = field(default_factory=list)
    init_image_override: str = ""


@dataclass(slots=True)
class SegmentQualityReport:
    segment_index: int
    temporal_score: float
    sharpness_score: float
    passed: bool
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoPipelineResult:
    output_path: str
    plan: VideoPlan
    provenance: List[ProvenanceRecord]
    quality: List[SegmentQualityReport]
    segment_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


EditStrategy = Literal["keyframes", "full_frame", "motion_transfer"]
