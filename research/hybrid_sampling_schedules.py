"""
**Step budgets** that split capacity between AR-style passes and diffusion denoising.

Pure math — plug counts into your own loop (no sampler dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class HybridStepBudget:
    ar_refine_steps: int
    diffusion_steps: int

    @property
    def total(self) -> int:
        return int(self.ar_refine_steps + self.diffusion_steps)


def split_budget_geometric(
    total_steps: int,
    *,
    ar_fraction: float,
    min_ar: int = 0,
    min_diffusion: int = 1,
) -> HybridStepBudget:
    """
    Allocate ``total_steps`` between AR refine steps and diffusion steps.

    ``ar_fraction`` in ``[0,1]`` is applied after reserving ``min_*`` floors.
    """
    if total_steps < min_ar + min_diffusion:
        raise ValueError("total_steps too small for min_ar + min_diffusion")
    af = max(0.0, min(1.0, float(ar_fraction)))
    remaining = total_steps - min_ar - min_diffusion
    ar_extra = int(round(af * remaining))
    diff_extra = remaining - ar_extra
    return HybridStepBudget(ar_refine_steps=min_ar + ar_extra, diffusion_steps=min_diffusion + diff_extra)


def interleave_phases(
    budget: HybridStepBudget,
    *,
    chunk_ar: int = 1,
    chunk_diffusion: int = 1,
) -> List[str]:
    """
    Return a list of phase tags ``\"ar\"`` / ``\"diff\"`` interleaved in chunks.

    Drains AR steps first in chunks of ``chunk_ar``, then diffusion in ``chunk_diffusion``.
    """
    if chunk_ar < 1 or chunk_diffusion < 1:
        raise ValueError("chunk sizes must be >= 1")
    out: List[str] = []
    ar_left = budget.ar_refine_steps
    d_left = budget.diffusion_steps
    while ar_left > 0:
        take = min(chunk_ar, ar_left)
        out.extend(["ar"] * take)
        ar_left -= take
    while d_left > 0:
        take = min(chunk_diffusion, d_left)
        out.extend(["diff"] * take)
        d_left -= take
    return out


def cosine_ar_ramp(total_ar_steps: int) -> List[float]:
    """
    Cosine ramp weights for AR refine passes (early steps small, late steps large).

    Length ``total_ar_steps``; sums to ~1.0 (normalized).
    """
    if total_ar_steps < 1:
        raise ValueError("total_ar_steps must be >= 1")
    raw: List[float] = []
    for i in range(total_ar_steps):
        t = i / max(total_ar_steps - 1, 1)
        raw.append(0.5 * (1.0 - math.cos(math.pi * t)))
    s = sum(raw) or 1.0
    return [x / s for x in raw]
