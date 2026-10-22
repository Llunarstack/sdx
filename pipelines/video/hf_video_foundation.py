"""Resolve open video foundation models (Wan / LTX / CogVideoX / …) for neural path.

Scaffolds live under ``pretrained/<Name>/`` (config-only). Full weights stay on
the hub until explicitly downloaded — see ``scripts/download/download_video_models.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "VideoFoundation",
    "resolve_video_foundation",
    "list_video_foundations",
]


@dataclass(frozen=True, slots=True)
class VideoFoundation:
    name: str
    path_or_id: str
    role: str  # t2v | i2v
    notes: str = ""


def list_video_foundations() -> list[VideoFoundation]:
    from utils.modeling.model_paths import (
        default_cogvideox_i2v_path,
        default_cogvideox_path,
        default_flf2v_foundation_path,
        default_hunyuan_i2v_path,
        default_hunyuan_video_path,
        default_live_portrait_path,
        default_ltx_video_path,
        default_mochi_path,
        default_open_sora_path,
        default_svd_path,
        default_v2v_path,
        default_vace_path,
        default_video_foundation_path,
        default_wan_animate_path,
        default_wan_i2v_path,
        default_wan_t2v_path,
    )

    return [
        VideoFoundation("default", default_video_foundation_path(), "t2v", "Wan TI2V prefer"),
        VideoFoundation("wan_t2v", default_wan_t2v_path(), "t2v"),
        VideoFoundation("wan_i2v", default_wan_i2v_path(), "i2v"),
        VideoFoundation("ltx", default_ltx_video_path(), "t2v", "LTX-2.3 prefer"),
        VideoFoundation("cogvideox", default_cogvideox_path(), "t2v", "STDiT twin to VideoDiT"),
        VideoFoundation("cogvideox_i2v", default_cogvideox_i2v_path(), "i2v"),
        VideoFoundation("hunyuan", default_hunyuan_video_path(), "t2v"),
        VideoFoundation("hunyuan_i2v", default_hunyuan_i2v_path(), "i2v"),
        VideoFoundation("mochi", default_mochi_path(), "t2v"),
        VideoFoundation("open_sora", default_open_sora_path(), "t2v"),
        VideoFoundation("svd", default_svd_path(), "i2v"),
        VideoFoundation("vace", default_vace_path(), "video_edit", "All-in-one edit / V2V control"),
        VideoFoundation("v2v", default_v2v_path(), "v2v", "SkyReels / VACE restyle"),
        VideoFoundation("flf2v", default_flf2v_foundation_path(), "flf2v", "First–last-frame Wan"),
        VideoFoundation("animate", default_wan_animate_path(), "video_animate"),
        VideoFoundation("portrait", default_live_portrait_path(), "video_portrait"),
    ]


def resolve_video_foundation(prefer: str = "default") -> VideoFoundation:
    """Pick a foundation by short name (``wan``, ``ltx``, ``vace``, ``v2v``, …)."""
    key = (prefer or "default").strip().lower().replace("-", "_")
    aliases = {
        "": "default",
        "auto": "default",
        "wan": "wan_t2v",
        "wan2": "wan_t2v",
        "wan2.2": "wan_t2v",
        "i2v": "wan_i2v",
        "cogvideo": "cogvideox",
        "cogvideox5b": "cogvideox",
        "opensora": "open_sora",
        "stable_video": "svd",
        "svd_xt": "svd",
        "edit": "vace",
        "video_edit": "vace",
        "vid2vid": "v2v",
        "video2video": "v2v",
        "first_last": "flf2v",
        "flf": "flf2v",
        "liveportrait": "portrait",
        "reenact": "portrait",
    }
    key = aliases.get(key, key)
    for f in list_video_foundations():
        if f.name == key:
            return f
    return list_video_foundations()[0]


def foundation_sample_hint(prefer: str = "default") -> dict[str, Any]:
    """Metadata for VIDEOMAX / neural_sample logs (no Diffusers load here)."""
    f = resolve_video_foundation(prefer)
    return {"foundation": f.name, "path_or_id": f.path_or_id, "role": f.role, "notes": f.notes}
