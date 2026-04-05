"""
Latent tensor views aligned with block-wise AR macro-blocks (inference / research).

Does not change the trained mask; useful for partial noising, diagnostics, or future block-wise refine.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import torch
from models.ar_masks_extended import block_grid_dims


def iter_macro_block_views(
    x: torch.Tensor,
    *,
    num_ar_blocks: int,
    latent_h: int,
    latent_w: int,
) -> Iterator[Tuple[Tuple[int, int], torch.Tensor]]:
    """
    Yield ``((bi, bj), x_block)`` for each macro-block over spatial latent grid.

    x: (B, C, H, W) with H==latent_h, W==latent_w.
    """
    if x.dim() != 4:
        raise ValueError("x must be (B, C, H, W)")
    _b, _c, h, w = x.shape
    if h != latent_h or w != latent_w:
        raise ValueError(f"x spatial {h}x{w} != latent_h x latent_w {latent_h}x{latent_w}")
    bh, bw = block_grid_dims(h, w, num_ar_blocks)
    for bi in range(num_ar_blocks):
        for bj in range(num_ar_blocks):
            r0, r1 = bi * bh, min(h, (bi + 1) * bh)
            c0, c1 = bj * bw, min(w, (bj + 1) * bw)
            if r0 >= h or c0 >= w:
                continue
            yield (bi, bj), x[:, :, r0:r1, c0:c1].contiguous()


def stack_macro_blocks(x: torch.Tensor, *, num_ar_blocks: int, latent_h: int, latent_w: int) -> List[torch.Tensor]:
    """Return list of macro-block tensors in raster (bi,bj) order."""
    return [t for _, t in iter_macro_block_views(x, num_ar_blocks=num_ar_blocks, latent_h=latent_h, latent_w=latent_w)]


def paste_macro_blocks(
    blocks: List[torch.Tensor],
    *,
    num_ar_blocks: int,
    latent_h: int,
    latent_w: int,
    batch_channels: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Inverse of ``stack_macro_blocks``: paste same-ordered blocks into empty latent.
    batch_channels: (B, C)
    """
    b, c = batch_channels
    out = torch.zeros(b, c, latent_h, latent_w, device=device, dtype=dtype)
    bh, bw = block_grid_dims(latent_h, latent_w, num_ar_blocks)
    idx = 0
    for bi in range(num_ar_blocks):
        for bj in range(num_ar_blocks):
            r0, r1 = bi * bh, min(latent_h, (bi + 1) * bh)
            c0, c1 = bj * bw, min(latent_w, (bj + 1) * bw)
            if r0 >= latent_h or c0 >= latent_w:
                continue
            if idx >= len(blocks):
                raise ValueError("blocks list too short for grid")
            blk = blocks[idx]
            idx += 1
            out[:, :, r0:r1, c0:c1] = blk
    if idx != len(blocks):
        raise ValueError("blocks list longer than grid")
    return out
