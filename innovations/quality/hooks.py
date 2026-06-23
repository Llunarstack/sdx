"""
Bridge from ultra-quality research modules to the production SDX stack.

Production paths:
  - VAE decode + optional refine: ``sample.py --refine-t``, ``utils/quality/``
  - Face/region enhance: ``utils/quality/face_region_enhance.py``
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from .engine import UltraQualityEngine

MaterialType = Literal["metallic", "skin", "cloth", "liquid", "photorealistic"]


def material_hint_from_prompt(prompt: str) -> MaterialType:
    """Heuristic material route from prompt text (no model weights required)."""
    p = (prompt or "").lower()
    if any(w in p for w in ("metal", "chrome", "steel", "gold", "silver", "brass")):
        return "metallic"
    if any(w in p for w in ("skin", "portrait", "face", "freckle", "pore")):
        return "skin"
    if any(w in p for w in ("fabric", "silk", "cotton", "wool", "dress", "cloth")):
        return "cloth"
    if any(w in p for w in ("water", "liquid", "ocean", "wine", "glass", "caustic")):
        return "liquid"
    return "photorealistic"


def refine_latent_embedding(
    latent_or_embedding: torch.Tensor,
    *,
    prompt: str = "",
    material_type: Optional[MaterialType] = None,
    engine: Optional[UltraQualityEngine] = None,
) -> torch.Tensor:
    """
    Optional post-conditioning on a pooled latent vector before DiT cross-attn.

    For image tensors (B,C,H,W) with C=3, use ``SubpixelRefinement`` via the engine's
    subpixel module directly in training experiments.
    """
    eng = engine or UltraQualityEngine()
    mat = material_type or material_hint_from_prompt(prompt)
    return eng.render_photorealistic(latent_or_embedding, mat)


__all__ = ["MaterialType", "material_hint_from_prompt", "refine_latent_embedding"]
