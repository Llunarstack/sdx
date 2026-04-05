"""
Layout helpers for block-wise AR (patch ↔ macro-block mapping, diagnostics).

Model masks live in ``models/ar_masks_extended.py``; this module is for tooling and docs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from models.ar_masks_extended import (
    block_grid_dims,
    block_raster_rank,
    block_zorder_rank,
    patch_to_block_indices,
)


def macro_block_centers_patch_space(
    h: int, w: int, num_ar_blocks: int
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    For each macro-block (bi, bj), return approximate center (row, col) in patch coordinates.
    """
    bh, bw = block_grid_dims(h, w, num_ar_blocks)
    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for bi in range(num_ar_blocks):
        for bj in range(num_ar_blocks):
            r0, r1 = bi * bh, min(h, (bi + 1) * bh)
            c0, c1 = bj * bw, min(w, (bj + 1) * bw)
            if r0 >= h or c0 >= w:
                continue
            out[(bi, bj)] = ((r0 + r1 - 1) * 0.5, (c0 + c1 - 1) * 0.5)
    return out


def block_visit_order(num_ar_blocks: int, order: str = "raster") -> List[Tuple[int, int]]:
    """Ordered list of (bi, bj) macro-block coordinates."""
    o = str(order or "raster").strip().lower()
    cells = [(bi, bj) for bi in range(num_ar_blocks) for bj in range(num_ar_blocks)]
    if o in ("z", "z-order", "zorder", "morton"):
        cells.sort(key=lambda t: block_zorder_rank(t[0], t[1], num_ar_blocks))
    else:
        cells.sort(key=lambda t: block_raster_rank(t[0], t[1], num_ar_blocks))
    return cells


def patch_block_map(h: int, w: int, num_ar_blocks: int) -> List[Tuple[int, int, int, int]]:
    """
    For each patch index in raster order, return (patch_row, patch_col, block_row, block_col).
    """
    rows = []
    for i in range(h):
        for j in range(w):
            bi, bj = patch_to_block_indices(i, j, h, w, num_ar_blocks)
            rows.append((i, j, bi, bj))
    return rows
