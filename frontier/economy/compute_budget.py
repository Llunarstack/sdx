"""
Adaptive compute allocation across denoise steps.

Ahead-of-curve idea: most products run full CFG + regional + attention every step.
This planner marks steps as cheap / standard / expensive so hooks can skip work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class GuidanceTier(str, Enum):
    SKIP = "skip"  # no CFG, no regional — trust momentum
    LITE = "lite"  # single CFG pass
    FULL = "full"  # regional + attention + batched CFG
    HEAVY = "heavy"  # best-of-N micro-rerank at this step


@dataclass(frozen=True)
class ComputeBudget:
    num_steps: int
    tiers: Tuple[GuidanceTier, ...]
    estimated_cost: float  # relative 0..1 vs all-FULL baseline


class ComputeBudgetPlanner:
    """
    Default policy (literature + product intuition):
    - early: FULL (layout lock)
    - mid: LITE (structure committed)
    - late: SKIP or LITE (texture/detail)
    - spike HEAVY at ~65% progress for composition rescue
    """

    def __init__(self, num_steps: int = 28) -> None:
        self.num_steps = max(4, int(num_steps))

    def plan(
        self,
        *,
        risk_score: float = 0.0,
        layout_regions: int = 0,
        heavy_at_progress: float = 0.65,
    ) -> ComputeBudget:
        n = self.num_steps
        tiers: List[GuidanceTier] = []
        cost_map = {GuidanceTier.SKIP: 0.0, GuidanceTier.LITE: 0.35, GuidanceTier.FULL: 1.0, GuidanceTier.HEAVY: 1.8}
        heavy_idx = int(heavy_at_progress * (n - 1))

        for i in range(n):
            p = i / max(1, n - 1)
            if layout_regions > 0 and p < 0.35:
                tiers.append(GuidanceTier.FULL)
            elif i == heavy_idx and risk_score > 0.35:
                tiers.append(GuidanceTier.HEAVY)
            elif p < 0.2:
                tiers.append(GuidanceTier.FULL if layout_regions else GuidanceTier.LITE)
            elif p < 0.75:
                tiers.append(GuidanceTier.LITE)
            else:
                tiers.append(GuidanceTier.SKIP if risk_score < 0.4 else GuidanceTier.LITE)

        baseline = float(n) * cost_map[GuidanceTier.FULL]
        est = sum(cost_map[t] for t in tiers) / baseline if baseline else 0.0
        return ComputeBudget(num_steps=n, tiers=tuple(tiers), estimated_cost=est)

    def tier_at(self, budget: ComputeBudget, step_index: int) -> GuidanceTier:
        if 0 <= step_index < len(budget.tiers):
            return budget.tiers[step_index]
        return GuidanceTier.LITE

    def should_run_regional(self, tier: GuidanceTier) -> bool:
        return tier in (GuidanceTier.FULL, GuidanceTier.HEAVY)

    def should_run_attention_layout(self, tier: GuidanceTier) -> bool:
        return tier == GuidanceTier.FULL

    def cfg_passes(self, tier: GuidanceTier) -> int:
        return {GuidanceTier.SKIP: 0, GuidanceTier.LITE: 1, GuidanceTier.FULL: 1, GuidanceTier.HEAVY: 2}[tier]


__all__ = ["ComputeBudget", "ComputeBudgetPlanner", "GuidanceTier"]
