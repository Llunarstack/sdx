"""
Per-step CFG selection via cheap latent heuristics (no extra model forwards).

Inspired by "Dynamic Classifier-Free Diffusion Guidance via Online Feedback"
(arXiv:2509.16131): pick the best CFG scale at each step from a small candidate set
using alignment / smoothness proxies in latent space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class LatentStepScore:
    cfg_scale: float
    alignment: float
    smoothness: float
    total: float


class DynamicCFGPicker:
    """
    Greedy CFG picker: score candidate scales using the current latent ``x``.

    ``alignment`` — penalize extreme latent magnitude (proxy for oversaturated CFG).
    ``smoothness`` — penalize high local variance (proxy for CFG artifacts).
    """

    def __init__(
        self,
        candidates: Sequence[float] = (4.0, 6.0, 7.5, 9.0, 11.0),
        *,
        alignment_weight: float = 0.4,
        smoothness_weight: float = 0.6,
    ) -> None:
        self.candidates = tuple(float(c) for c in candidates)
        self.alignment_weight = float(alignment_weight)
        self.smoothness_weight = float(smoothness_weight)

    def _score_latent(self, x: torch.Tensor, cfg: float) -> LatentStepScore:
        # Prefer moderate latent energy — very high norms correlate with CFG blow-up.
        energy = x.float().pow(2).mean().item()
        alignment = 1.0 / (1.0 + abs(energy - 1.0))

        # Total variation proxy: mean squared diff to local average.
        if x.ndim == 4 and x.shape[-1] > 1 and x.shape[-2] > 1:
            pooled = torch.nn.functional.avg_pool2d(x.float(), kernel_size=3, stride=1, padding=1)
            tv = (x.float() - pooled).pow(2).mean().item()
            smoothness = 1.0 / (1.0 + tv * 10.0)
        else:
            smoothness = 0.5

        # Slight preference for mid-range CFG (empirical sweet spot).
        cfg_prior = 1.0 - min(1.0, abs(cfg - 7.5) / 7.5) * 0.15
        total = (
            self.alignment_weight * alignment
            + self.smoothness_weight * smoothness
            + 0.15 * cfg_prior
        )
        return LatentStepScore(cfg_scale=cfg, alignment=alignment, smoothness=smoothness, total=total)

    def pick(self, x: torch.Tensor, *, default: float = 7.5) -> float:
        if not self.candidates:
            return float(default)
        best = max(self.candidates, key=lambda c: self._score_latent(x, c).total)
        return float(best)

    def schedule(
        self,
        x_trajectory: Sequence[torch.Tensor],
        *,
        default: float = 7.5,
    ) -> List[float]:
        """Build per-step CFG list aligned with ``x_trajectory`` length."""
        out: List[float] = []
        for x in x_trajectory:
            out.append(self.pick(x, default=default))
        return out
