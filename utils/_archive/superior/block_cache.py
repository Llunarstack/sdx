"""
**Block-wise DiT cache** (BWCache / TeaCache-inspired, training-free).

Caches per-block **residuals** across denoising steps. When the step fingerprint
(timestep + latent stats) changes little, reuses cached block deltas instead of
re-running transformer blocks — typically 1.3–2× fewer block FLOPs at modest thresholds.

See ``docs/research/SUPERIOR_RESEARCH_2026.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(slots=True)
class BlockCacheConfig:
    rel_l1_threshold: float = 0.18
    """Accumulated relative L1 on step fingerprint before full recompute."""
    max_reuse_streak: int = 3
    recompute_every: int = 4
    """Force full block pass every N denoising steps (anti drift)."""
    enabled: bool = True
    cfg_split_fingerprint: bool = False
    """Fingerprint only the cond half of a 2×B CFG batch (Cache-DiT / DBCache)."""


@dataclass(slots=True)
class BlockCacheStats:
    total_forwards: int = 0
    block_skips: int = 0
    full_recomputes: int = 0


class BlockDiTCache:
    """
    Per-sample-loop block cache. Attach to DiT via ``forward(..., block_cache=self)``.

    TeaCache-style: accumulate relative L1 distance between consecutive step
    fingerprints; when below threshold, apply cached per-block residuals.
    """

    def __init__(self, config: Optional[BlockCacheConfig] = None) -> None:
        self.config = config or BlockCacheConfig()
        self.stats = BlockCacheStats()
        self._fingerprints: list[torch.Tensor] = []
        self._residuals: Dict[int, torch.Tensor] = {}
        self._reuse_streak = 0
        self._step_index = 0
        self._accumulated_dist = 0.0
        self._force_full = True

    def reset(self) -> None:
        self.stats = BlockCacheStats()
        self._fingerprints.clear()
        self._residuals.clear()
        self._reuse_streak = 0
        self._step_index = 0
        self._accumulated_dist = 0.0
        self._force_full = True

    def begin_forward(self, fingerprint: torch.Tensor) -> None:
        """Call once per DiT forward with a (B, D) or (D,) step fingerprint."""
        cfg = self.config
        self.stats.total_forwards += 1
        self._step_index += 1
        fp = fingerprint.detach().float().reshape(-1)
        if not cfg.enabled:
            self._force_full = True
            return
        if self._step_index % max(1, int(cfg.recompute_every)) == 0:
            self._force_full = True
            self._accumulated_dist = 0.0
            self._reuse_streak = 0
            self.stats.full_recomputes += 1
            self._fingerprints = [fp]
            return
        if not self._fingerprints:
            self._force_full = True
            self._fingerprints = [fp]
            return
        prev = self._fingerprints[-1]
        dist = (fp - prev).abs().mean() / (prev.abs().mean() + 1e-6)
        self._accumulated_dist += float(dist.item())
        self._fingerprints.append(fp)
        if (
            self._accumulated_dist < float(cfg.rel_l1_threshold)
            and self._reuse_streak < int(cfg.max_reuse_streak)
            and self._residuals
        ):
            self._force_full = False
            self._reuse_streak += 1
        else:
            self._force_full = True
            self._accumulated_dist = 0.0
            self._reuse_streak = 0
            self.stats.full_recomputes += 1

    def should_skip_block(self, block_index: int) -> bool:
        return not self._force_full and block_index in self._residuals

    def apply_residual(self, x: torch.Tensor, block_index: int) -> torch.Tensor:
        res = self._residuals[block_index]
        self.stats.block_skips += 1
        if res.shape == x.shape:
            return x + res
        return x + res.to(dtype=x.dtype, device=x.device)

    def note_block(self, block_index: int, x_before: torch.Tensor, x_after: torch.Tensor) -> None:
        self._residuals[block_index] = (x_after - x_before).detach()

    @staticmethod
    def fingerprint_from_tensors(
        t_emb: torch.Tensor,
        x: torch.Tensor,
        *,
        cfg_split: bool = False,
    ) -> torch.Tensor:
        """Build step fingerprint from timestep embedding + latent token mean."""
        te = t_emb
        xs = x
        if bool(cfg_split) and te.shape[0] >= 2 and te.shape[0] % 2 == 0:
            half = te.shape[0] // 2
            te = te[:half]
            xs = xs[:half]
        te = te.detach().float().reshape(te.shape[0], -1).mean(dim=0)
        xm = xs.detach().float().reshape(xs.shape[0], -1).mean(dim=0)
        return torch.cat([te, xm], dim=0)


__all__ = ["BlockCacheConfig", "BlockCacheStats", "BlockDiTCache"]
