"""
**Interval CFG** — disable classifier-free guidance in early/late denoise phases.

Kynkäänniemi et al. / CFG++ analyses: high-noise early steps and very low-noise
late steps often hurt or don't need CFG. Training-free schedule on progress ``p in [0,1]``
(0 = start/noisy, 1 = end/clean).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def should_apply_cfg(
    progress: float,
    *,
    skip_early_frac: float = 0.0,
    skip_late_frac: float = 0.0,
) -> bool:
    """
    Return False when CFG should be disabled at this denoise progress.

    ``skip_early_frac=0.15`` → no CFG for first 15% of steps.
    ``skip_late_frac=0.1`` → no CFG for last 10% of steps.
    """
    p = float(max(0.0, min(1.0, progress)))
    if float(skip_early_frac) > 0.0 and p < float(skip_early_frac):
        return False
    if float(skip_late_frac) > 0.0 and p > (1.0 - float(skip_late_frac)):
        return False
    return True


def interval_cfg_prediction(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    *,
    progress: float,
    skip_early_frac: float = 0.0,
    skip_late_frac: float = 0.0,
) -> torch.Tensor:
    """Return conditional-only prediction when outside the CFG interval."""
    if should_apply_cfg(progress, skip_early_frac=skip_early_frac, skip_late_frac=skip_late_frac):
        return None
    return out_cond


__all__ = ["interval_cfg_prediction", "should_apply_cfg"]
