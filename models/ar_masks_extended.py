"""
Extended block-causal (AR) self-attention masks for DiT.

- **raster**: blocks visited row-major (default; matches historical SDX behavior).
- **zorder**: Morton (Z-order) curve over the block grid — different spatial bias, same parameter count.
- **snake**: boustrophedon row scan (alternating left↔right each block row).
- **spiral**: outside-in spiral over macro-blocks.

See ``docs/AR.md`` and ``docs/AR_EXTENSIONS.md``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def _morton_encode_2d(bi: int, bj: int, bits: int) -> int:
    z = 0
    for bit in range(bits):
        z |= ((bi >> bit) & 1) << (2 * bit)
        z |= ((bj >> bit) & 1) << (2 * bit + 1)
    return z


def block_grid_dims(h: int, w: int, num_ar_blocks: int) -> Tuple[int, int]:
    """Patch counts per block along H and W."""
    bh = max(1, (h + num_ar_blocks - 1) // num_ar_blocks)
    bw = max(1, (w + num_ar_blocks - 1) // num_ar_blocks)
    return bh, bw


def patch_to_block_indices(
    i: int, j: int, h: int, w: int, num_ar_blocks: int
) -> Tuple[int, int]:
    """Map patch row,col to block row,col."""
    bh, bw = block_grid_dims(h, w, num_ar_blocks)
    return i // bh, j // bw


def block_raster_rank(bi: int, bj: int, num_ar_blocks: int) -> int:
    return bi * num_ar_blocks + bj


def block_zorder_rank(bi: int, bj: int, num_ar_blocks: int) -> int:
    bits = max(1, (num_ar_blocks - 1).bit_length())
    return _morton_encode_2d(bi, bj, bits)


def block_snake_rank(bi: int, bj: int, num_ar_blocks: int) -> int:
    if (bi % 2) == 0:
        col = bj
    else:
        col = (num_ar_blocks - 1) - bj
    return bi * num_ar_blocks + col


def _spiral_cells(num_ar_blocks: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = num_ar_blocks - 1, num_ar_blocks - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            out.append((top, c))
        top += 1
        for r in range(top, bottom + 1):
            out.append((r, right))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                out.append((bottom, c))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                out.append((r, left))
            left += 1
    return out


def block_spiral_rank(bi: int, bj: int, num_ar_blocks: int) -> int:
    cells = _spiral_cells(num_ar_blocks)
    lut = {(r, c): i for i, (r, c) in enumerate(cells)}
    return int(lut.get((bi, bj), 0))


def block_visit_order(num_ar_blocks: int, block_order: str = "raster") -> List[Tuple[int, int]]:
    order = str(block_order or "raster").strip().lower()
    cells = [(bi, bj) for bi in range(num_ar_blocks) for bj in range(num_ar_blocks)]
    if order in ("z", "z-order", "zorder", "morton"):
        cells.sort(key=lambda t: block_zorder_rank(t[0], t[1], num_ar_blocks))
        return cells
    if order in ("snake", "boustrophedon"):
        cells.sort(key=lambda t: block_snake_rank(t[0], t[1], num_ar_blocks))
        return cells
    if order in ("spiral", "snail"):
        return _spiral_cells(num_ar_blocks)
    if order in ("raster", "row", "row_major", "default"):
        cells.sort(key=lambda t: block_raster_rank(t[0], t[1], num_ar_blocks))
        return cells
    raise ValueError(f"Unknown block_order {block_order!r}; use raster | zorder | snake | spiral")


def create_block_causal_mask_2d(
    h: int,
    w: int,
    num_ar_blocks: int,
    *,
    block_order: str = "raster",
) -> torch.Tensor:
    """
    h, w = patches per side. num_ar_blocks = blocks per dimension (e.g. 2 -> 2×2 macro grid).

    block_order:
        - ``raster``: macro-blocks in row-major order; within-block causal raster by flat index.
        - ``zorder``: macro-blocks ordered by Morton code; within-block same causal rule.
        - ``snake``: alternating row direction for macro-block traversal.
        - ``spiral``: outside-in macro-block traversal.

    Returns (h*w, h*w) additive mask: 0 = attend, -inf = blocked.
    """
    n = h * w
    if num_ar_blocks <= 0:
        return torch.zeros(n, n, dtype=torch.float32)

    order_cells = block_visit_order(num_ar_blocks, block_order=block_order)
    rank_lut = {(bi, bj): rank for rank, (bi, bj) in enumerate(order_cells)}

    bh, bw = block_grid_dims(h, w, num_ar_blocks)
    mask = torch.zeros(n, n, dtype=torch.float32)

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            bi, bj = i // bh, j // bw
            br = int(rank_lut.get((bi, bj), 0))
            for i2 in range(h):
                for j2 in range(w):
                    idx2 = i2 * w + j2
                    bi2, bj2 = i2 // bh, j2 // bw
                    br2 = int(rank_lut.get((bi2, bj2), 0))
                    if br2 > br:
                        mask[idx, idx2] = float("-inf")
                    elif br2 == br and idx2 > idx:
                        mask[idx, idx2] = float("-inf")
    return mask


def ar_mask_sparsity_stats(mask: torch.Tensor) -> Tuple[float, int]:
    """Fraction of allowed pairs (finite mask), and N for an N×N mask."""
    if mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be square (N, N)")
    n = int(mask.shape[0])
    ok = torch.isfinite(mask) & (mask > -1e18)
    frac = float(ok.float().mean().item())
    return frac, n


def soften_ar_mask(mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Differentiable-ish relaxation: scale finite penalties toward 0 (does not un-block -inf rows).

    temperature > 1.0 weakens causal constraint; 1.0 = unchanged. Experimental / research only.
    """
    t = max(1.0, float(temperature))
    if t == 1.0:
        return mask
    out = mask.clone()
    finite = torch.isfinite(out)
    out[finite] = out[finite] / t
    return out
