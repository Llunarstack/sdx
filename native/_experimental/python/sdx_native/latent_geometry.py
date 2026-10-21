"""Pure-Python latent grid math matching the ``native/cpp`` C ABI (``sdx/latent.h``).

``sdx_native.native_tools.LatentLib`` uses the C++ build when present and these as the fallback.
"""

from __future__ import annotations

import re


def latent_spatial_size(image_hw: int, vae_scale: int) -> int:
    """Latent side length: ``image_hw // vae_scale`` (0 if not divisible or invalid)."""
    if image_hw <= 0 or vae_scale <= 0 or image_hw % vae_scale != 0:
        return 0
    return image_hw // vae_scale


def patch_grid_dim(latent_hw: int, patch_size: int) -> int:
    """Patches per side: ``latent_hw // patch_size``."""
    if latent_hw <= 0 or patch_size <= 0 or latent_hw % patch_size != 0:
        return 0
    return latent_hw // patch_size


def num_patch_tokens(image_hw: int, vae_scale: int, patch_size: int) -> int:
    """DiT patch token count (square grid, no register tokens): ``(latent/patch)^2``."""
    g = patch_grid_dim(latent_spatial_size(image_hw, vae_scale), patch_size)
    return g * g


def latent_numel(channels: int, latent_h: int, latent_w: int) -> int:
    """``C * H * W`` for a latent tensor (0 if any dimension invalid or product overflows int)."""
    if channels <= 0 or latent_h <= 0 or latent_w <= 0:
        return 0
    n = channels * latent_h * latent_w
    return n if n <= 2147483647 else 0


def dit_patch_size_from_variant_name(model_name: str) -> int:
    """Patch size from a registry name like ``DiT-XL/2-Text`` or ``DiT-B/4-Text``; 2 when there is no ``/N``."""
    m = re.search(r"/(\d+)(?:-|/|$)", model_name)
    return max(1, int(m.group(1))) if m else 2
