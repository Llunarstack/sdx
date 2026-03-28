"""
Holy Grail diffusion experiments.

This package holds aggressive-but-safe runtime ideas:
- Adaptive CFG from attention entropy.
- Prompt coverage penalties/signals.
- Latent refinement helpers for detail retention.
- Unified per-step planning for adapter/control/guidance scales.
"""

from .blueprint import HolyGrailRecipe, HolyGrailStepPlan, build_holy_grail_step_plan
from .condition_annealing import apply_condition_noise, cads_noise_std
from .guidance_fusion import (
    adaptive_cfg_from_attention,
    attention_entropy,
    fuse_condition_scales,
)
from .latent_refiner import (
    consistency_blend_latent,
    dynamic_percentile_clamp,
    unsharp_mask_latent,
)
from .prompt_coverage import (
    attention_token_coverage,
    coverage_shortfall_loss,
    weighted_patch_alignment_score,
)
from .presets import (
    HOLY_GRAIL_PRESETS,
    HolyGrailPreset,
    apply_holy_grail_preset_to_args,
    get_holy_grail_preset,
    list_holy_grail_presets,
)
from .recommender import recommend_holy_grail_preset
from .runtime_guard import sanitize_holy_grail_kwargs
from .style_router import bounded_scale, style_detail_mix_for_progress

__all__ = [
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

