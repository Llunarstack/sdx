"""
Research prototypes (ViT / AR / DiT helpers) — not wired into train/sample by default.
"""

from __future__ import annotations

from . import (
    autoregressive_plans,
    creature_character_guidance,
    diffusion_noise_structures,
    hybrid_sampling_schedules,
    latent_agreement,
    physics_visual_guidance,
    quality_timestep_weights,
    visual_quality,
)

__all__ = [
    "autoregressive_plans",
    "creature_character_guidance",
    "diffusion_noise_structures",
    "hybrid_sampling_schedules",
    "latent_agreement",
    "physics_visual_guidance",
    "quality_timestep_weights",
    "visual_quality",
]
