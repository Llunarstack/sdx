"""
LAMIC-inspired region fusion schedule (AAAI 2026).

Early denoise: isolate regions (less cross-region semantic leakage).
Late denoise: allow fusion for coherent lighting and contact shadows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RegionFusionSchedule:
    """Controls how much regions blend with each other over steps."""

    isolate_until_frac: float = 0.45
    curve: str = "cosine"  # linear | cosine
    min_isolation: float = 0.85  # 1 = full isolation early
    max_fusion: float = 1.0


def fusion_weight_at_step(
    step_index: int,
    total_steps: int,
    schedule: RegionFusionSchedule,
) -> float:
    """
    Return cross-region fusion factor in ``[0, 1]``.

    Low early (isolated), high late (fused). Multiply regional mask overlap
    blending by this factor when implementing LAMIC-style attention hooks.
    """
    if total_steps <= 1:
        return schedule.max_fusion
    p = float(step_index) / float(total_steps - 1)
    if p <= schedule.isolate_until_frac:
        if schedule.curve == "linear":
            t = p / max(1e-6, schedule.isolate_until_frac)
            iso = schedule.min_isolation + (1.0 - schedule.min_isolation) * t
        else:
            t = p / max(1e-6, schedule.isolate_until_frac)
            iso = schedule.min_isolation + (1.0 - schedule.min_isolation) * (0.5 * (1.0 - math.cos(math.pi * t)))
        return float(1.0 - iso)
    # Late phase: ramp to full fusion
    q = (p - schedule.isolate_until_frac) / max(1e-6, 1.0 - schedule.isolate_until_frac)
    if schedule.curve == "cosine":
        q = 0.5 * (1.0 - math.cos(math.pi * q))
    return float(schedule.max_fusion * q)
