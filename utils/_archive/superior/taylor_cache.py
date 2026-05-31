"""
**TaylorSeer** block feature forecasting (ICCV 2025).

Upgrades block cache from static reuse to Taylor-series prediction of per-block
residuals using finite-difference derivatives across anchor timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .block_cache import BlockCacheConfig, BlockDiTCache


@dataclass(slots=True)
class TaylorCacheConfig(BlockCacheConfig):
    use_taylor: bool = True
    cache_interval: int = 4
    """Steps between full block recomputes (N in TaylorSeer paper)."""
    max_order: int = 1
    """Taylor expansion order (0 = plain reuse, 1 = linear forecast)."""


def taylor_forecast_tensor(
    history: List[torch.Tensor],
    steps_since_anchor: int,
    *,
    interval: int = 4,
    max_order: int = 1,
) -> torch.Tensor:
    """
    Predict block residual at ``steps_since_anchor`` steps after last anchor.

    Uses finite differences on the last ``max_order+1`` anchor residuals::
        F_pred = F_0 + (dF / dt) * k + ...
    """
    if not history:
        raise ValueError("empty history")
    k = float(max(1, int(steps_since_anchor)))
    n = float(max(1, int(interval)))
    f0 = history[-1]
    if int(max_order) <= 0 or len(history) < 2:
        return f0
    f1 = history[-2]
    d1 = (f0 - f1) / n
    pred = f0 + d1 * k
    if int(max_order) >= 2 and len(history) >= 3:
        f2 = history[-3]
        d2 = (f0 - 2.0 * f1 + f2) / (n * n)
        pred = pred + 0.5 * d2 * (k * k)
    return pred


class TaylorBlockCache(BlockDiTCache):
    """Block cache with TaylorSeer forecast instead of zero-order reuse."""

    def __init__(self, config: Optional[TaylorCacheConfig] = None) -> None:
        self.taylor_config = config or TaylorCacheConfig()
        super().__init__(self.taylor_config)
        self._anchor_residuals: Dict[int, List[torch.Tensor]] = {}
        self._steps_since_anchor = 0

    def reset(self) -> None:
        super().reset()
        self._anchor_residuals.clear()
        self._steps_since_anchor = 0

    def begin_forward(self, fingerprint: torch.Tensor) -> None:
        was_full = self._force_full
        super().begin_forward(fingerprint)
        if self._force_full:
            self._steps_since_anchor = 0
        elif not was_full:
            self._steps_since_anchor += 1

    def note_block(self, block_index: int, x_before: torch.Tensor, x_after: torch.Tensor) -> None:
        super().note_block(block_index, x_before, x_after)
        if self._force_full:
            res = (x_after - x_before).detach()
            hist = self._anchor_residuals.setdefault(block_index, [])
            hist.append(res)
            keep = int(self.taylor_config.max_order) + 1
            if len(hist) > keep:
                del hist[:-keep]

    def apply_residual(self, x: torch.Tensor, block_index: int) -> torch.Tensor:
        cfg = self.taylor_config
        if cfg.use_taylor and block_index in self._anchor_residuals and self._steps_since_anchor > 0:
            hist = self._anchor_residuals[block_index]
            if hist:
                pred = taylor_forecast_tensor(
                    hist,
                    self._steps_since_anchor,
                    interval=int(cfg.cache_interval),
                    max_order=int(cfg.max_order),
                )
                self.stats.block_skips += 1
                if pred.shape == x.shape:
                    return x + pred
                return x + pred.to(dtype=x.dtype, device=x.device)
        return super().apply_residual(x, block_index)


__all__ = ["TaylorBlockCache", "TaylorCacheConfig", "taylor_forecast_tensor"]
