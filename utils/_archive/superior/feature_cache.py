"""
**Feature-cache policy** for DiT sampling (SpeCa / TeaCache-inspired).

When consecutive latent updates are small, reuse the previous model epsilon prediction
for one step — optional speed win; disable when quality-critical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(slots=True)
class FeatureCacheState:
    """Per-sequence cache for sample loop."""

    last_x: Optional[torch.Tensor] = None
    last_pred: Optional[torch.Tensor] = None
    reuse_count: int = 0
    skip_count: int = 0


@dataclass(slots=True)
class FeatureCacheConfig:
    delta_threshold: float = 0.012
    max_reuse_streak: int = 2
    enabled: bool = True


class FeatureCachePolicy:
    """Decide whether to reuse previous DiT output at step ``i``."""

    def __init__(self, config: Optional[FeatureCacheConfig] = None) -> None:
        self.config = config or FeatureCacheConfig()
        self.state = FeatureCacheState()

    def reset(self) -> None:
        self.state = FeatureCacheState()

    def maybe_reuse(
        self,
        x: torch.Tensor,
        fresh_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        If ``x`` changed little since last step, return cached ``last_pred`` instead of
        ``fresh_pred`` (caller still runs one forward — this is for optional external hooks).

        For inline use in sample_loop: call ``should_skip_forward`` before forward instead.
        """
        cfg = self.config
        st = self.state
        if not cfg.enabled or st.last_x is None or st.last_pred is None:
            st.last_x = x.detach()
            st.last_pred = fresh_pred.detach()
            return fresh_pred
        delta = (x - st.last_x).abs().mean()
        if (
            float(delta.item()) < cfg.delta_threshold
            and st.reuse_count < cfg.max_reuse_streak
            and st.last_pred.shape == fresh_pred.shape
        ):
            st.reuse_count += 1
            st.skip_count += 1
            return st.last_pred
        st.reuse_count = 0
        st.last_x = x.detach()
        st.last_pred = fresh_pred.detach()
        return fresh_pred

    def should_skip_forward(self, x: torch.Tensor) -> bool:
        """True if caller may reuse ``last_pred`` without running DiT."""
        cfg = self.config
        st = self.state
        if not cfg.enabled or st.last_x is None or st.last_pred is None:
            return False
        delta = (x - st.last_x).abs().mean()
        return float(delta.item()) < cfg.delta_threshold and st.reuse_count < cfg.max_reuse_streak

    def cached_prediction(self) -> torch.Tensor:
        """Return last cached prediction (must exist)."""
        if self.state.last_pred is None:
            raise RuntimeError("feature cache miss")
        self.state.reuse_count += 1
        self.state.skip_count += 1
        return self.state.last_pred

    def note_fresh(self, x: torch.Tensor, pred: torch.Tensor) -> None:
        self.state.reuse_count = 0
        self.state.last_x = x.detach()
        self.state.last_pred = pred.detach()


__all__ = ["FeatureCacheConfig", "FeatureCachePolicy", "FeatureCacheState"]
