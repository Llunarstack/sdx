"""
Ordered stage ids for the text-to-image inference path.

These labels align with the README **System diagram** (prompt → encoders → DiT →
diffusion → Holy Grail → VAE). Use for logging, UI steppers, or docs— they are
not wired into ``sample.py`` yet.
"""

from __future__ import annotations

# Stable public tuple: append-only preferred; rename only with a major release.
INFERENCE_PIPELINE_STAGES: tuple[str, ...] = (
    "prompt",
    "text_encoders",
    "fused_conditioning",
    "dit_backbone",
    "diffusion_engine",
    "holy_grail_and_extras",
    "vae_decode",
    "image_output",
)


def inference_stage_index(stage_id: str) -> int:
    """Return zero-based index of ``stage_id`` or raise ``ValueError``."""
    try:
        return INFERENCE_PIPELINE_STAGES.index(stage_id)
    except ValueError as e:
        raise ValueError(f"unknown inference stage: {stage_id!r}") from e
