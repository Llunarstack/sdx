"""
Block-causal (AR) self-attention masks for DiT.

The patch grid is split into ``num_ar_blocks x num_ar_blocks`` macro-blocks visited in one of these orders:

- **raster**: row-major (default).
- **zorder**: Morton (Z-order) curve — different spatial bias, same parameter count.
- **snake**: boustrophedon row scan (alternating left↔right each block row).
- **spiral**: outside-in spiral.

A token sees every token of earlier blocks and the raster-earlier tokens of its own block.
See ``docs/AR.md`` and ``docs/AR_EXTENSIONS.md``.
"""

from __future__ import annotations

import torch


def _morton_code(bi: int, bj: int, bits: int) -> int:
    z = 0
    for bit in range(bits):
        z |= ((bi >> bit) & 1) << (2 * bit)
        z |= ((bj >> bit) & 1) << (2 * bit + 1)
    return z


def _spiral_cells(num_ar_blocks: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
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


def block_visit_order(num_ar_blocks: int, block_order: str = "raster") -> list[tuple[int, int]]:
    """Macro-block ``(row, col)`` cells in visit order."""
    order = str(block_order or "raster").strip().lower()
    n = num_ar_blocks
    cells = [(bi, bj) for bi in range(n) for bj in range(n)]
    if order in ("raster", "row", "row_major", "default"):
        return cells
    if order in ("z", "z-order", "zorder", "morton"):
        bits = max(1, (n - 1).bit_length())
        return sorted(cells, key=lambda c: _morton_code(c[0], c[1], bits))
    if order in ("snake", "boustrophedon"):
        return sorted(cells, key=lambda c: c[0] * n + (c[1] if c[0] % 2 == 0 else n - 1 - c[1]))
    if order in ("spiral", "snail"):
        return _spiral_cells(n)
    raise ValueError(f"Unknown block_order {block_order!r}; use raster | zorder | snake | spiral")


def create_block_causal_mask_2d(
    h: int,
    w: int,
    num_ar_blocks: int,
    *,
    block_order: str = "raster",
) -> torch.Tensor:
    """Additive ``(h*w, h*w)`` mask for an ``h x w`` patch grid: 0 = attend, -inf = blocked.

    ``num_ar_blocks`` is macro-blocks per side (2 -> 2x2); ``<= 0`` returns an all-zero (full attention) mask.
    """
    n = h * w
    if num_ar_blocks <= 0:
        return torch.zeros(n, n, dtype=torch.float32)
    lut = torch.zeros(num_ar_blocks, num_ar_blocks, dtype=torch.long)
    for rank, (bi, bj) in enumerate(block_visit_order(num_ar_blocks, block_order)):
        lut[bi, bj] = rank
    bh = max(1, -(-h // num_ar_blocks))
    bw = max(1, -(-w // num_ar_blocks))
    rows = torch.arange(h).repeat_interleave(w) // bh
    cols = torch.arange(w).repeat(h) // bw
    rank = lut[rows, cols]
    idx = torch.arange(n)
    blocked = (rank[None, :] > rank[:, None]) | ((rank[None, :] == rank[:, None]) & (idx[None, :] > idx[:, None]))
    return torch.zeros(n, n, dtype=torch.float32).masked_fill(blocked, float("-inf"))


def ar_mask_sparsity_stats(mask: torch.Tensor) -> tuple[float, int]:
    """Fraction of allowed pairs (finite mask), and N for an N×N mask."""
    if mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be square (N, N)")
    ok = torch.isfinite(mask) & (mask > -1e18)
    return float(ok.float().mean().item()), int(mask.shape[0])
