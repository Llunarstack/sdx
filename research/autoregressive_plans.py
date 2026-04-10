"""
**AR block visit / revisit** sketches for hybrid autoregressive + diffusion decoders.

Complements ``utils/architecture/ar_block_layout.py`` (geometry) with *when* to touch blocks.
No model imports — safe for tooling and tests.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def raster_block_order(num_ar_blocks: int) -> List[Tuple[int, int]]:
    """(bi, bj) in row-major order for an ``num_ar_blocks x num_ar_blocks`` macro grid."""
    out: List[Tuple[int, int]] = []
    for bi in range(num_ar_blocks):
        for bj in range(num_ar_blocks):
            out.append((bi, bj))
    return out


def serpentine_block_order(num_ar_blocks: int) -> List[Tuple[int, int]]:
    """Alternating row direction — sometimes better for locality in wide layouts."""
    out: List[Tuple[int, int]] = []
    for bi in range(num_ar_blocks):
        cols = list(range(num_ar_blocks))
        if bi % 2 == 1:
            cols.reverse()
        for bj in cols:
            out.append((bi, bj))
    return out


def revisit_schedule(
    base_order: Sequence[Tuple[int, int]],
    *,
    cycles: int,
    stride: int = 1,
) -> List[Tuple[int, int]]:
    """
    Repeat ``base_order`` ``cycles`` times, skipping ahead by ``stride`` modulo length each cycle.

    Sketch for "coarse AR pass then polish earlier blocks" without a full model API.
    """
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    if not base_order:
        raise ValueError("base_order must be non-empty")
    n = len(base_order)
    out: List[Tuple[int, int]] = []
    offset = 0
    for _ in range(cycles):
        for k in range(n):
            out.append(base_order[(offset + k) % n])
        offset = (offset + stride) % n
    return out


def blocks_visible_fraction(step: int, total_steps: int, num_ar_blocks: int) -> float:
    """
    Linearly grow visible macro-blocks from center outward (conceptual fraction in ``[0,1]``).

    ``step`` in ``[0, total_steps-1]``; returns approximate fraction of blocks "unmasked"
    for curriculum-style schedules.
    """
    if total_steps < 1 or num_ar_blocks < 1:
        raise ValueError("total_steps and num_ar_blocks must be >= 1")
    t = min(max(step / float(total_steps - 1), 0.0), 1.0) if total_steps > 1 else 1.0
    total_b = num_ar_blocks * num_ar_blocks
    # grow count from 1 .. total_b
    count = 1 + int(t * (total_b - 1) + 1e-6)
    return count / float(total_b)
