"""
Allocate a fixed randomness budget across steps and spatial regions.

Prevents "everything is equally noisy" — spend entropy where it helps composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class EntropyBudget:
    total: float
    per_step: Tuple[float, ...]
    per_region: Tuple[float, ...]  # e.g. foreground vs background weights


class EntropyBudgetAllocator:
    """
    Split ``total`` entropy across ``num_steps`` and ``num_regions``.

    Default policy: 40% early (layout), 30% mid (objects), 30% late (texture).
    """

    def __init__(self, num_steps: int = 28, num_regions: int = 2) -> None:
        self.num_steps = max(4, int(num_steps))
        self.num_regions = max(1, int(num_regions))

    def allocate(
        self,
        total: float = 1.0,
        *,
        early_weight: float = 0.4,
        mid_weight: float = 0.3,
        late_weight: float = 0.3,
        foreground_share: float = 0.35,
    ) -> EntropyBudget:
        total = max(0.0, float(total))
        n = self.num_steps
        third = max(1, n // 3)
        per_step: List[float] = []
        for i in range(n):
            if i < third:
                w = early_weight
            elif i < 2 * third:
                w = mid_weight
            else:
                w = late_weight
            per_step.append(w)
        s = sum(per_step) or 1.0
        per_step = [total * w / s for w in per_step]

        fg = float(max(0.0, min(1.0, foreground_share)))
        per_region = (fg, 1.0 - fg) if self.num_regions == 2 else tuple(1.0 / self.num_regions for _ in range(self.num_regions))

        return EntropyBudget(total=total, per_step=tuple(per_step), per_region=per_region)

    def scale_noise(
        self,
        noise: torch.Tensor,
        step_index: int,
        budget: EntropyBudget,
        region_index: int = 0,
    ) -> torch.Tensor:
        step_scale = budget.per_step[step_index] if 0 <= step_index < len(budget.per_step) else 0.0
        reg_scale = budget.per_region[region_index] if 0 <= region_index < len(budget.per_region) else 1.0
        return noise * (step_scale * reg_scale) ** 0.5
