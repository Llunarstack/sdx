"""
Ultra-photorealism facade — routes to per-material renderers (INNOVATION_GUIDE §1).
"""

import torch

from .cloth import ClothFabricSimulator
from .global_light import GlobalIlluminationApproximator
from .liquid import LiquidPhysicsRenderer
from .metallic import MetallicMaterialRenderer
from .skin import SkinTextureAuthenticator
from .subpixel import SubpixelRefinement

__all__ = [
    "ClothFabricSimulator",
    "GlobalIlluminationApproximator",
    "LiquidPhysicsRenderer",
    "MetallicMaterialRenderer",
    "SkinTextureAuthenticator",
    "SubpixelRefinement",
    "UltraQualityEngine",
]


class UltraQualityEngine:
    """Unified ultra-quality image generation pipeline."""

    def __init__(self):
        self.subpixel = SubpixelRefinement(3, 4)
        self.metallic = MetallicMaterialRenderer()
        self.skin = SkinTextureAuthenticator()
        self.cloth = ClothFabricSimulator()
        self.liquid = LiquidPhysicsRenderer()
        self.gi = GlobalIlluminationApproximator()

    def render_photorealistic(self, latent: torch.Tensor, material_type: str) -> torch.Tensor:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if material_type == "metallic":
            return self.metallic(latent, torch.randn(latent.shape[0], 3))
        if material_type == "skin":
            return self.skin(latent)
        if material_type == "cloth":
            return self.cloth(latent, torch.randint(0, 8, (latent.shape[0],)))
        if material_type == "liquid":
            return self.liquid(latent, torch.randn(latent.shape[0], 3, 64, 64))
        return self.gi(latent)
