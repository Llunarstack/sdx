"""
DiT + block-AR + ViT scorer alignment for latent editing (img2img / inpaint).

- **DiT**: latent tensors ``(B, 4, H, W)`` must match ``PatchEmbed`` ``img_size`` on the loaded model.
- **AR**: block-causal masks depend on ``sqrt(num_patches)``; refresh after changing ``num_ar_blocks``
  at runtime (curriculum-style) so attention matches the layout.
- **ViT quality**: use the same ``num_ar_blocks`` tags / one-hot as training (``ar_block_conditioning``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from models.attention import create_block_causal_mask_2d

from utils.architecture.ar_block_conditioning import (
    ar_conditioning_vector,
    ar_regime_label,
    normalize_num_ar_blocks,
    tag_manifest_row_ar,
)


def dit_latent_hw_from_model(model: Any) -> Tuple[int, int]:
    """
    Spatial ``(H, W)`` of the **latent** grid expected by a DiT-like ``x_embedder`` (timm ``PatchEmbed``).
    """
    emb = getattr(model, "x_embedder", None)
    if emb is None:
        raise ValueError("expected a DiT-like module with attribute x_embedder (PatchEmbed)")
    img_size = getattr(emb, "img_size", None)
    if img_size is None:
        raise ValueError("x_embedder has no img_size; cannot infer latent spatial shape")
    if isinstance(img_size, int):
        s = int(img_size)
        return s, s
    return int(img_size[0]), int(img_size[1])


def dit_patch_grid_side_from_model(model: Any) -> int:
    """Patch grid side ``P`` where ``P * P == num_patches`` (integer ``sqrt``)."""
    np_ = int(getattr(model, "num_patches", 0) or 0)
    if np_ <= 0:
        raise ValueError("model.num_patches is missing or zero")
    p = int(round(math.sqrt(float(np_))))
    if p * p != np_:
        raise ValueError(f"num_patches={np_} is not a perfect square; cannot build 2D AR mask")
    return p


def validate_latent_edit_tensors(
    z0: torch.Tensor,
    mask_latent: Optional[torch.Tensor],
    model: Any,
) -> None:
    """
    Ensure ``z0`` (and optional inpaint mask) spatial dims match the DiT latent ``img_size``.
    """
    if z0.dim() != 4:
        raise ValueError(f"z0 must be (B, C, H, W), got dim={z0.dim()}")
    eh, ew = dit_latent_hw_from_model(model)
    _, _, h, w = z0.shape
    if int(h) != eh or int(w) != ew:
        raise ValueError(f"z0 spatial ({h},{w}) != DiT latent ({eh},{ew}) from model.x_embedder.img_size")
    if mask_latent is not None:
        if mask_latent.dim() != 4 or int(mask_latent.shape[0]) != int(z0.shape[0]):
            raise ValueError("mask_latent must be (B, 1, H, W) with same B as z0")
        _, _, hm, wm = mask_latent.shape
        if int(hm) != eh or int(wm) != ew:
            raise ValueError(f"mask_latent spatial ({hm},{wm}) != DiT latent ({eh},{ew})")


def refresh_block_ar_mask_on_model(
    model: Any,
    *,
    num_ar_blocks: Optional[int] = None,
    ar_block_order: Optional[str] = None,
) -> None:
    """
    Rebuild ``model._ar_mask`` from current ``num_ar_blocks`` / ``ar_block_order`` and ``num_patches``.

    Mirrors ``train._apply_runtime_ar`` so inference scripts can switch AR regime without reloading weights.
    """
    b = int(num_ar_blocks if num_ar_blocks is not None else getattr(model, "num_ar_blocks", 0) or 0)
    order = str(ar_block_order if ar_block_order is not None else getattr(model, "ar_block_order", "raster") or "raster")
    setattr(model, "num_ar_blocks", b)
    setattr(model, "ar_block_order", order)
    if b <= 0:
        setattr(model, "_ar_mask", None)
        return
    p = dit_patch_grid_side_from_model(model)
    mask = create_block_causal_mask_2d(p, p, b, block_order=order.strip().lower())
    ref = getattr(model, "pos_embed", None)
    if ref is not None and hasattr(ref, "device"):
        mask = mask.to(device=ref.device)
    setattr(model, "_ar_mask", mask)


def vit_scorer_ar_vector(num_ar_blocks: Union[int, Any], *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    ``(4,)`` one-hot for ViT quality / dataset QA (same as ``ar_conditioning_vector``).
    """
    v = normalize_num_ar_blocks(num_ar_blocks)
    return ar_conditioning_vector(v, device=device, dtype=dtype)


def tag_row_for_vit_scorer(
    row: Mapping[str, Any],
    num_ar_blocks: Union[int, Any],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Attach AR fields a ViT scorer / JSONL digest expects (wraps ``tag_manifest_row_ar``)."""
    v = normalize_num_ar_blocks(num_ar_blocks)
    return tag_manifest_row_ar(row, v, overwrite=overwrite)


def generation_edit_metadata(
    *,
    image_size_px: int,
    num_ar_blocks: Union[int, Any],
    strength: float,
    inpaint: bool,
    inpaint_mode: str = "",
) -> Dict[str, Any]:
    """Small dict for manifests / logs (book pipeline, tooling)."""
    v = normalize_num_ar_blocks(num_ar_blocks)
    return {
        "image_size_px": int(image_size_px),
        "latent_hw": int(image_size_px) // 8,
        "num_ar_blocks": int(v) if v in (0, 2, 4) else v,
        "ar_regime": ar_regime_label(v),
        "img2img_strength": float(strength),
        "inpaint": bool(inpaint),
        "inpaint_mode": str(inpaint_mode or ""),
    }


__all__ = [
    "dit_latent_hw_from_model",
    "dit_patch_grid_side_from_model",
    "generation_edit_metadata",
    "refresh_block_ar_mask_on_model",
    "tag_row_for_vit_scorer",
    "validate_latent_edit_tensors",
    "vit_scorer_ar_vector",
]
