"""
Lightweight **latent bridge** helpers (interpolation A → B), not full diffusion-bridge training.

Stochastic / VP-consistent bridges need schedule-aware noise and a dedicated objective — see
``docs/MODERN_DIFFUSION.md`` and theme ``diffusion_bridges`` in ``utils/architecture/architecture_map.py``.
"""

from __future__ import annotations

import torch


def linear_latent_interp(x_a: torch.Tensor, x_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Batch linear interpolation: ``(1 - t) * x_a + t * x_b``.

    ``t`` shape ``(B,)`` or broadcastable to ``x_a`` (e.g. ``(B, 1, 1, 1)``).
    """
    if x_a.shape != x_b.shape:
        raise ValueError("x_a and x_b must match shape")
    tt = t
    if tt.ndim == 1:
        tt = tt.reshape(-1, *([1] * (x_a.ndim - 1)))
    return (1.0 - tt) * x_a + tt * x_b


__all__ = ["linear_latent_interp"]
