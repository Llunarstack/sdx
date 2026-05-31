"""
**CFG-Rejection** — early denoise path filtering via accumulated guidance score.

When generating multiple candidates, abort trajectories whose early CFG score
differences suggest low final quality (arXiv:2505.23343 scaffold).

Training-free; useful with ``--num > 1`` best-of-N.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class CFGRejectionTracker:
    """Track per-trajectory CFG score gaps during early denoise steps."""

    tau_steps: int = 4
    scores: List[float] = field(default_factory=list)

    def note(self, out_cond: torch.Tensor, out_uncond: torch.Tensor) -> float:
        """Instantaneous guidance gap (L2 norm of cond-uncond)."""
        delta = (out_cond - out_uncond).float()
        g = float(delta.pow(2).mean().sqrt().item())
        self.scores.append(g)
        return g

    def accumulated_early_score(self) -> float:
        if not self.scores:
            return 0.0
        use = self.scores[: max(1, int(self.tau_steps))]
        return float(sum(use))

    def should_reject(self, threshold: float, *, best_accum: Optional[float] = None) -> bool:
        acc = self.accumulated_early_score()
        if best_accum is None:
            return acc > float(threshold)
        return acc > float(best_accum) * 1.35 + float(threshold)


def pick_best_candidate_index(
    accum_scores: List[float],
    *,
    reject_threshold: float = 0.0,
) -> int:
    """Lower accumulated early CFG gap often correlates with better paths (heuristic)."""
    if not accum_scores:
        return 0
    ranked = sorted(range(len(accum_scores)), key=lambda i: accum_scores[i])
    for idx in ranked:
        if accum_scores[idx] <= float(reject_threshold) or reject_threshold <= 0:
            return idx
    return ranked[0]


__all__ = ["CFGRejectionTracker", "pick_best_candidate_index"]
