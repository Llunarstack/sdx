"""
Deprecated import path for advanced sampling helpers.

Use ``diffusion.sampling`` for new code. Names resolve lazily against that package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Kept in sync with ``diffusion.sampling.__all__`` (update both when extending API).
__all__: list[str] = [
    "HolyGrailRecipe",
    "HolyGrailStepPlan",
    "build_holy_grail_step_plan",
    "cads_noise_std",
    "apply_condition_noise",
    "attention_entropy",
    "adaptive_cfg_from_attention",
    "fuse_condition_scales",
    "unsharp_mask_latent",
    "dynamic_percentile_clamp",
    "consistency_blend_latent",
    "attention_token_coverage",
    "weighted_patch_alignment_score",
    "coverage_shortfall_loss",
    "style_detail_mix_for_progress",
    "bounded_scale",
    "HolyGrailPreset",
    "HOLY_GRAIL_PRESETS",
    "list_holy_grail_presets",
    "get_holy_grail_preset",
    "apply_holy_grail_preset_to_args",
    "sanitize_holy_grail_kwargs",
    "recommend_holy_grail_preset",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module("diffusion.sampling")
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
