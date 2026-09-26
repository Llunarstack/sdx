"""Video quality helper stack — image-gen critic analogues for VIDEOMAX.

Maps Hugging Face helpers → SDX video failure axes (identity drift, flicker,
hand morph, prompt drift, …). Foundations (Wan/LTX) generate; these *score and
guide repairs*.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "VIDEO_QUALITY_MAP",
    "list_video_quality_helpers",
    "resolve_video_quality_bundle",
]


# Image-stack parallel → video helper → VIDEOMAX / superior_pass axis
VIDEO_QUALITY_MAP: dict[str, dict[str, str]] = {
    "DOVER": {
        "image_analogue": "Aesthetic-Predictor / technical sharpness critics",
        "axis": "flicker + artifact + aesthetic",
        "use": "Frame/clip aesthetic + technical score for quality_retry",
        "path_fn": "default_dover_path",
    },
    "VideoScore-v1.1": {
        "image_analogue": "HPSv3 / ImageReward / PickScore",
        "axis": "prompt_drift + overall preference",
        "use": "AI-generated video preference for pick-best / retry",
        "path_fn": "default_videoscore_path",
    },
    "VisionReward-Video": {
        "image_analogue": "HPSv3 multi-axis (heavy)",
        "axis": "physics + stability + preservation + fidelity",
        "use": "21-dim preference (scaffold / large weights on hub)",
        "path_fn": "default_visionreward_video_path",
    },
    "XCLIP-base-patch32": {
        "image_analogue": "CLIP / SigLIP adherence",
        "axis": "prompt_drift / temporal_adherence",
        "use": "Prompt↔video embedding similarity",
        "path_fn": "default_xclip_path",
    },
    "InternVideo2-1B": {
        "image_analogue": "Qwen3-VL / DINOv3 video twin",
        "axis": "prompt_drift + retrieval",
        "use": "Video-text features for adherence / clip retrieval",
        "path_fn": "default_internvideo_path",
    },
    "RAFT-OpticalFlow": {
        "image_analogue": "(no stills twin — temporal only)",
        "axis": "flicker / flow_consistency / motion_shutter",
        "use": "Optical flow for deflicker + physics_gate",
        "path_fn": "default_raft_path",
    },
    "CoTracker3": {
        "image_analogue": "GroundingDINO track over time",
        "axis": "object_vanish / permanence / count_bind",
        "use": "Point tracking for object permanence repair",
        "path_fn": "default_cotracker_path",
    },
    "DWPose": {
        "image_analogue": "OpenPose ControlNet / anatomy guidance",
        "axis": "hand_morph / extremity_lock",
        "use": "2D pose for hand/limb coherence",
        "path_fn": "default_dwpose_path",
    },
    "ArcFace": {
        "image_analogue": "IP-Adapter-FaceID / identity lock",
        "axis": "identity_drift",
        "use": "Face embedding lock across frames",
        "path_fn": "default_arcface_path",
    },
    "Buffalo-L": {
        "image_analogue": "InsightFace pack",
        "axis": "identity_drift",
        "use": "Detection + recognition pack for identity_bind",
        "path_fn": "default_arcface_path",
    },
    "Whisper-Large-V3-Turbo": {
        "image_analogue": "(audio — lip_sync)",
        "axis": "lip_desync",
        "use": "ASR timing for native_audio / lip_sync gate",
        "path_fn": "default_whisper_path",
    },
    "VideoMAE-base": {
        "image_analogue": "DINOv2 temporal backbone",
        "axis": "artifact / shimmer",
        "use": "Motion feature backbone for critics",
        "path_fn": "default_videomae_path",
    },
    # Shared image pretrained already wired into video
    "SAM2-Hiera-Large": {
        "image_analogue": "SAM2 (same)",
        "axis": "occlusion_pop / count",
        "use": "Temporal masks for occlusion_resolve",
        "path_fn": "default_sam2_hiera_large_path",
    },
    "Depth-Anything-V3-Large": {
        "image_analogue": "Depth Anything (same)",
        "axis": "floating_feet / contact_ground",
        "use": "Per-frame depth for contact + depth_interpolate",
        "path_fn": "default_depth_anything_path",
    },
    "Qwen3-VL-8B-Instruct": {
        "image_analogue": "VLM critic (same)",
        "axis": "prompt_drift",
        "use": "Frame/clip caption critique",
        "path_fn": "default_qwen_vl_path",
    },
    "GroundingDINO-Base": {
        "image_analogue": "GroundingDINO (same)",
        "axis": "count_bind / object_vanish",
        "use": "Open-vocab detect for count permanence",
        "path_fn": "default_grounding_dino_base_path",
    },
    "HPSv3": {
        "image_analogue": "HPSv3 (frame-wise)",
        "axis": "aesthetic preference on keyframes",
        "use": "Score keyframe stills inside VIDEOMAX pick-best",
        "path_fn": "default_hps_path",
    },
}


def list_video_quality_helpers() -> list[str]:
    return sorted(VIDEO_QUALITY_MAP.keys())


def resolve_video_quality_bundle() -> dict[str, Any]:
    """Resolve local/hub paths for the curated quality shortlist."""
    from utils.modeling import model_paths as mp

    out: dict[str, Any] = {}
    for name, row in VIDEO_QUALITY_MAP.items():
        fn = getattr(mp, row["path_fn"], None)
        path = fn() if callable(fn) else ""
        out[name] = {**row, "resolved": path}
    return out
