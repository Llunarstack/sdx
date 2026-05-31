"""
Stateful **guidance session** for multi-step sampling (APG reverse momentum, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GuidanceSession:
    """Carries cross-step state through a denoise trajectory."""

    prev_apg_delta: Optional[torch.Tensor] = None
    apg_momentum_beta: float = 0.0
    step_index: int = 0

    def note_apg_delta(self, delta: torch.Tensor) -> None:
        self.prev_apg_delta = delta.detach()
        self.step_index += 1

    def reset(self) -> None:
        self.prev_apg_delta = None
        self.step_index = 0


@dataclass(slots=True)
class DynamicDitSchedule:
    """DyDiT-inspired timestep width schedule (training-free inference)."""

    enabled: bool = False
    early_width: float = 0.88
    late_width: float = 1.0
    power: float = 1.0

    def scale_at_progress(self, progress: float) -> float:
        """
        ``progress`` in [0,1]: 0 = start of denoise, 1 = end.

        Early steps use lower effective width (less CFG/model push when structure forms).
        """
        if not self.enabled:
            return 1.0
        p = float(max(0.0, min(1.0, progress)))
        # progress 0 -> early, progress 1 -> late
        t = p ** float(self.power)
        return float(self.early_width) + (float(self.late_width) - float(self.early_width)) * t


__all__ = ["DynamicDitSchedule", "GuidanceSession"]
