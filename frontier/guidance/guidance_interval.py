"""
CFG guidance intervals — skip early/late CFG, ramp mid-schedule.

Research: high CFG early harms diversity; late CFG can oversharpen.
See arXiv:2404.13040 (CFG weight schedulers) and masked-diffusion analysis 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceInterval:
    """When to apply CFG and how strongly."""

    skip_early_frac: float = 0.08
    skip_late_frac: float = 0.0
    ramp: str = "cosine"  # linear | cosine
    min_multiplier: float = 0.65
    max_multiplier: float = 1.0


def _progress(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return max(0.0, min(1.0, float(step_index) / float(total_steps - 1)))


def cfg_multiplier_for_step(
    step_index: int,
    total_steps: int,
    interval: GuidanceInterval,
) -> float:
    """
    Return CFG multiplier in ``[0, 1]`` for this step.

    ``0`` means skip CFG (use conditional only).
    """
    p = _progress(step_index, total_steps)
    if p < interval.skip_early_frac:
        return 0.0
    if p > 1.0 - interval.skip_late_frac:
        return 0.0

    # Renormalize progress inside active window.
    lo = interval.skip_early_frac
    hi = 1.0 - interval.skip_late_frac
    span = max(1e-6, hi - lo)
    q = (p - lo) / span

    if interval.ramp == "linear":
        m = interval.min_multiplier + (interval.max_multiplier - interval.min_multiplier) * q
    else:
        e = 0.5 * (1.0 - math.cos(math.pi * q))
        m = interval.min_multiplier + (interval.max_multiplier - interval.min_multiplier) * e
    return float(max(0.0, min(1.0, m)))
