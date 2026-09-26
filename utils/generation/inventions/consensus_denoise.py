"""CONSENSUS — Dual-trajectory agreement denoise for stills.

Idea never used as a stills default: run two seeds for K early steps, blend
latents where they agree (low |x_a-x_b|), keep diversity where they disagree.
Stabilizes structure without a second model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

__all__ = ["ConsensusConfig", "plan_consensus_seeds", "blend_consensus_latents"]


@dataclass
class ConsensusConfig:
    base_seed: int = 0
    twin_offset: int = 9973
    early_frac: float = 0.4  # blend only early
    agreement_temp: float = 0.15  # soft mask sharpness

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_consensus_seeds(cfg: ConsensusConfig | None = None) -> tuple[int, int]:
    c = cfg or ConsensusConfig()
    return int(c.base_seed), int(c.base_seed) + int(c.twin_offset)


def blend_consensus_latents(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    *,
    progress: float,
    cfg: ConsensusConfig | None = None,
) -> torch.Tensor:
    """
    Soft-agreement blend. When progress > early_frac, returns x_a unchanged.
    Mask = sigmoid(-(|xa-xb|/temp)); high agreement → average.
    """
    c = cfg or ConsensusConfig()
    p = float(progress)
    if p > float(c.early_frac):
        return x_a
    delta = (x_a.float() - x_b.float()).abs()
    # Per-spatial mean over channels
    if delta.ndim == 4:
        d = delta.mean(dim=1, keepdim=True)
    else:
        d = delta
    temp = max(float(c.agreement_temp), 1e-4)
    agree = torch.sigmoid(-(d / temp - 1.0))
    # agree≈1 → blend; agree≈0 → keep xa
    out = agree * 0.5 * (x_a.float() + x_b.float()) + (1.0 - agree) * x_a.float()
    return out.to(dtype=x_a.dtype)
