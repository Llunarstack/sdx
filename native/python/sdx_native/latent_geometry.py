"""
Pure-Python latent grid math (matches ``native/cpp`` C ABI in ``sdx/latent.h``).

Use this for CLI sanity checks and logging without building the shared library.
When ``libsdx_latent`` is loaded via :mod:`sdx_native.native_tools`, prefer that for
parity testing against the C++ build.
"""

from __future__ import annotations


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
    lh = latent_spatial_size(image_hw, vae_scale)
    g = patch_grid_dim(lh, patch_size)
    if g <= 0:
        return 0
    return g * g


def latent_numel(channels: int, latent_h: int, latent_w: int) -> int:
    """``C * H * W`` for a latent tensor (0 if any dimension invalid or product overflows int)."""
    if channels <= 0 or latent_h <= 0 or latent_w <= 0:
        return 0
    n = channels * latent_h * latent_w
    if n > 2147483647:
        return 0
    return int(n)


def dit_patch_size_from_variant_name(model_name: str) -> int:
    """
    Parse DiT registry names like ``DiT-XL/2-Text`` or ``DiT-B/4-Text`` → patch size (2 or 4).
    Defaults to **2** if no ``/N`` segment is found.
    """
    import re

    m = re.search(r"/(\d+)(?:-|/|$)", model_name)
    if m:
        return max(1, int(m.group(1)))
    return 2
