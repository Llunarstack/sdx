"""Pipeline processing options (from scene edit block)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
    # quality / sample pack (video_helpers)
    sample: Any = None
    video_quality: str = "none"
    # VIDEOMAX / superiority
    videomax: bool = False
    permanence_repair: bool = True
    identity_bind: bool = True
    extremity_lock: bool = True
    physics_gate: bool = True
    physics_repair: bool = False
    contact_ground: bool = True
    secondary_track: bool = True
    occlusion_resolve: bool = False
    hf_deshimmer: bool = True
    count_bind: bool = True
    count_bind_repair: bool = False
    shot_chain: bool = False
    shot_chain_strength: float = 0.35
    glyph_lock: bool = False
    motion_shutter: bool = False
    motion_grammar: str = "auto"
    motion_grammar_positive: str = ""
    motion_grammar_negative: str = ""
    invention_stack: str = ""
    log_feedback: bool = False
    lip_sync: bool = True
    native_audio: bool = False
    leaders_stack: bool = False
    in_gen_cuts: bool = False
    camera_path: bool = True
    motion_intel: bool = True
    prompt_ground: bool = True
    use_permanent_dit: bool = False
    neural_engine: bool = False
    min_permanence: float = 0.45
    min_artifact: float = 0.40
    min_identity: float = 0.0
    min_extremity: float = 0.0
    min_glyph: float = 0.0
    min_physics: float = 0.0
    min_contact: float = 0.0
    min_count: float = 0.0
    min_shimmer: float = 0.0
    min_secondary: float = 0.0
    min_occlusion: float = 0.0
    min_adherence: float = 0.0
    min_lip_sync: float = 0.0
    identity_refs: tuple[str, ...] = ()
    style_route: dict[str, Any] = field(default_factory=dict)
    videomax_plan: dict[str, Any] = field(default_factory=dict)
    director_events: list[Any] = field(default_factory=list)
    director_timeline: dict[str, Any] = field(default_factory=dict)
    cross_keyframe_identity: bool = False
    cross_keyframe_identity_strength: float = 0.28


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
        sample=r.get("sample"),
        video_quality=str(r.get("video_quality") or "none"),
        videomax=bool(r.get("videomax", False) or str(r.get("video_quality", "")).lower() in ("max", "videomax")),
        permanence_repair=bool(r.get("permanence_repair", True)),
        identity_bind=bool(r.get("identity_bind", True)),
        extremity_lock=bool(r.get("extremity_lock", True)),
        physics_gate=bool(r.get("physics_gate", True)),
        physics_repair=bool(r.get("physics_repair", False)),
        contact_ground=bool(r.get("contact_ground", True)),
        secondary_track=bool(r.get("secondary_track", True)),
        occlusion_resolve=bool(r.get("occlusion_resolve", False)),
        hf_deshimmer=bool(r.get("hf_deshimmer", True)),
        count_bind=bool(r.get("count_bind", True)),
        count_bind_repair=bool(r.get("count_bind_repair", False)),
        shot_chain=bool(r.get("shot_chain", False)),
        shot_chain_strength=float(r.get("shot_chain_strength", 0.35) or 0.35),
        glyph_lock=bool(r.get("glyph_lock", False)),
        motion_shutter=bool(r.get("motion_shutter", False)),
        motion_grammar=str(r.get("motion_grammar", "auto") or "auto"),
        invention_stack=str(r.get("invention_stack") or ""),
        log_feedback=bool(r.get("log_feedback", False)),
        min_permanence=float(r.get("min_permanence", 0.45) or 0.45),
        min_artifact=float(r.get("min_artifact", 0.40) or 0.40),
        min_identity=float(r.get("min_identity", 0.0) or 0.0),
        min_occlusion=float(r.get("min_occlusion", 0.0) or 0.0),
        min_adherence=float(r.get("min_adherence", 0.0) or 0.0),
    )
