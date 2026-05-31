"""
Per-batch CFG gap probe for **CFG-Rejection** reranking (multi-sample).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from utils.superior.cfg_rejection import CFGRejectionTracker, pick_best_candidate_index


@dataclass
class GuidanceProbe:
    """Accumulate early-step CFG gaps per batch row."""

    tau_steps: int = 4
    trackers: List[CFGRejectionTracker] = field(default_factory=list)

    def ensure_batch(self, batch_size: int) -> None:
        while len(self.trackers) < int(batch_size):
            self.trackers.append(CFGRejectionTracker(tau_steps=int(self.tau_steps)))

    def note(self, out_cond: torch.Tensor, out_uncond: torch.Tensor, *, step: int) -> None:
        if int(step) >= int(self.tau_steps):
            return
        b = int(out_cond.shape[0])
        self.ensure_batch(b)
        for i in range(b):
            self.trackers[i].note(out_cond[i : i + 1], out_uncond[i : i + 1])

    def accumulated_scores(self) -> List[float]:
        return [t.accumulated_early_score() for t in self.trackers]

    def rerank_indices(self) -> List[int]:
        scores = self.accumulated_scores()
        if not scores:
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        return order

    def best_index(self) -> int:
        scores = self.accumulated_scores()
        return int(pick_best_candidate_index(scores))


__all__ = ["GuidanceProbe"]
