"""
Controlled surprise: inject structured noise at *specific* denoise steps, not globally.

Creativity without destroying prompt adherence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SerendipityCurve:
    """Per-step multiplicative noise scale (1.0 = no extra surprise)."""

    scales: Tuple[float, ...]
    peak_step: int
    dial: float  # user creativity 0..1


class SerendipityInjector:
    """
    Build a bell-shaped serendipity curve over denoise steps.

    Mid-schedule surprises explore layout; early/late stay faithful.
    """

    def __init__(self, num_steps: int = 28) -> None:
        self.num_steps = max(4, int(num_steps))

    def curve(self, dial: float = 0.3) -> SerendipityCurve:
        dial = float(max(0.0, min(1.0, dial)))
        n = self.num_steps
        mid = (n - 1) / 2.0
        scales: List[float] = []
        peak = 0
        peak_val = 0.0
        for i in range(n):
            # gaussian bump centered mid-schedule
            x = (i - mid) / max(1.0, mid)
            bump = torch.exp(torch.tensor(-0.5 * x * x)).item()
            scale = 1.0 + dial * 0.35 * bump
            scales.append(scale)
            if scale > peak_val:
                peak_val = scale
                peak = i
        return SerendipityCurve(scales=tuple(scales), peak_step=peak, dial=dial)

    def apply_to_noise(
        self,
        noise: torch.Tensor,
        step_index: int,
        curve: SerendipityCurve,
    ) -> torch.Tensor:
        if step_index < 0 or step_index >= len(curve.scales):
            return noise
        s = curve.scales[step_index]
        if abs(s - 1.0) < 1e-6:
            return noise
        return noise * s

    def regional_mask_boost(
        self,
        mask: torch.Tensor,
        step_index: int,
        curve: SerendipityCurve,
        *,
        background_only: bool = True,
    ) -> torch.Tensor:
        """
        Boost surprise in background (low mask) regions at serendipity peaks.

        ``mask`` in [0,1], 1 = foreground / prompt-locked.
        """
        s = curve.scales[step_index] if 0 <= step_index < len(curve.scales) else 1.0
        boost = max(0.0, s - 1.0)
        if boost <= 0:
            return mask
        if background_only:
            return mask * (1.0 - boost * 0.5 * (1.0 - mask))
        return mask
