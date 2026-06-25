"""Pipeline processing options (from scene edit block)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["ProcessOptions", "parse_process_options"]


@dataclass(slots=True)
class ProcessOptions:
    motion_transfer: bool = True
    motion_transfer_retrieved: bool = True
    region_motion: bool = False
    identity_lock: bool = True
    identity_lock_strength: float = 0.82
    propagate_masks: bool = True
    depth_interpolate: bool = False
    camera_stabilize: bool = False
    camera_stabilize_strength: float = 0.65
    deflicker: bool = True
    deflicker_strength: float = 0.75
    motion_beat_keyframes: bool = False
    flow_consistency: bool = True
    frame_enhance: bool = False
    frame_enhance_amount: float = 0.32
    semantic_drift_repair: bool = True
    drift_threshold: float = 0.55
    drift_blend_strength: float = 0.45
    velocity_ease: bool = False
    velocity_ease_mode: str = "smooth"
    quality_retry: bool = True
    max_retries: int = 2
    temporal_alpha: float = 0.10
    temporal_smooth: int = 2
    post_grade: str = ""
    pose_control: bool = False
    keyframe_interval: int = 6
    edit_strength: float = 0.55
    audio_from_source: bool = False
    parallel_segments: bool = False
    max_segment_workers: int = 2
    thumbnail_pass: bool = False
    thumbnail_size: int = 128


def parse_process_options(raw: Mapping[str, Any] | None) -> ProcessOptions:
    r = dict(raw or {})
    return ProcessOptions(
        motion_transfer=bool(r.get("motion_transfer", True)),
        motion_transfer_retrieved=bool(r.get("motion_transfer_retrieved", True)),
        region_motion=bool(r.get("region_motion", False)),
        identity_lock=bool(r.get("identity_lock", True)),
        identity_lock_strength=float(r.get("identity_lock_strength", 0.82) or 0.82),
        propagate_masks=bool(r.get("propagate_masks", True)),
        depth_interpolate=bool(r.get("depth_interpolate", False)),
        camera_stabilize=bool(r.get("camera_stabilize", False)),
        camera_stabilize_strength=float(r.get("camera_stabilize_strength", 0.65) or 0.65),
        deflicker=bool(r.get("deflicker", True)),
        deflicker_strength=float(r.get("deflicker_strength", 0.75) or 0.75),
        motion_beat_keyframes=bool(r.get("motion_beat_keyframes", False)),
        flow_consistency=bool(r.get("flow_consistency", True)),
        frame_enhance=bool(r.get("frame_enhance", False)),
        frame_enhance_amount=float(r.get("frame_enhance_amount", 0.32) or 0.32),
        semantic_drift_repair=bool(r.get("semantic_drift_repair", True)),
        drift_threshold=float(r.get("drift_threshold", 0.55) or 0.55),
        drift_blend_strength=float(r.get("drift_blend_strength", 0.45) or 0.45),
        velocity_ease=bool(r.get("velocity_ease", False)),
        velocity_ease_mode=str(r.get("velocity_ease_mode") or "smooth"),
        quality_retry=bool(r.get("quality_retry", True)),
        max_retries=int(r.get("max_retries", 2) or 2),
        temporal_alpha=float(r.get("temporal_alpha", 0.10) or 0.10),
        temporal_smooth=int(r.get("temporal_smooth", 2) or 2),
        post_grade=str(r.get("post_grade") or ""),
        pose_control=bool(r.get("pose_control", False)),
        keyframe_interval=int(r.get("keyframe_interval", 6) or 6),
        edit_strength=float(r.get("edit_strength", 0.55) or 0.55),
        audio_from_source=bool(r.get("audio_from_source", False)),
        parallel_segments=bool(r.get("parallel_segments", False)),
        max_segment_workers=int(r.get("max_segment_workers", 2) or 2),
        thumbnail_pass=bool(r.get("thumbnail_pass", False)),
        thumbnail_size=int(r.get("thumbnail_size", 128) or 128),
    )
