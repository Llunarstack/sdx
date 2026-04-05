"""
CFG schedule helpers for diffusion sampling.

Motivation:
- Recent analyses show fixed CFG can over-steer early steps and collapse detail late.
- These helpers provide deterministic schedules that can be plugged into samplers.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _safe_progress(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    s = float(step_index) / float(max(total_steps - 1, 1))
    return max(0.0, min(1.0, s))


def cfg_scale_linear(
    base_cfg: float,
    step_index: int,
    total_steps: int,
    *,
    start_multiplier: float = 0.7,
    end_multiplier: float = 1.0,
) -> float:
    """
    Linearly ramp CFG from early to late steps.
    """
    p = _safe_progress(step_index, total_steps)
    m = float(start_multiplier) + (float(end_multiplier) - float(start_multiplier)) * p
    return float(base_cfg) * m


def cfg_scale_cosine_ramp(
    base_cfg: float,
    step_index: int,
    total_steps: int,
    *,
    min_multiplier: float = 0.65,
    max_multiplier: float = 1.0,
) -> float:
    """
    Smooth monotonic ramp with cosine easing.
    """
    p = _safe_progress(step_index, total_steps)
    e = 0.5 * (1.0 - math.cos(math.pi * p))  # 0->1
    m = float(min_multiplier) + (float(max_multiplier) - float(min_multiplier)) * e
    return float(base_cfg) * m


def cfg_scale_piecewise(
    base_cfg: float,
    step_index: int,
    total_steps: int,
    *,
    stage_multipliers: Iterable[float] = (0.75, 1.0, 0.9),
) -> float:
    """
    3-stage schedule: early / middle / late.
    """
    p = _safe_progress(step_index, total_steps)
    vals = list(stage_multipliers)
    if len(vals) != 3:
        raise ValueError("stage_multipliers must have exactly 3 values")
    if p < 1.0 / 3.0:
        m = float(vals[0])
    elif p < 2.0 / 3.0:
        m = float(vals[1])
    else:
        m = float(vals[2])
    return float(base_cfg) * m


def cfg_scale_snr_aware(
    base_cfg: float,
    alpha_cumprod_t: torch.Tensor,
    *,
    low_noise_multiplier: float = 0.9,
    high_noise_multiplier: float = 0.7,
) -> torch.Tensor:
    """
    Compute per-sample CFG using current alpha_cumprod:
      snr = a / (1-a)
    High noise (low snr): lower multiplier.
    Low noise (high snr): higher multiplier.
    """
    a = alpha_cumprod_t.to(dtype=torch.float32)
    snr = a / (1.0 - a + 1e-8)
    # Map log-SNR roughly to [0,1] via sigmoid.
    z = torch.sigmoid(torch.log(snr + 1e-8))
    m = float(high_noise_multiplier) + (float(low_noise_multiplier) - float(high_noise_multiplier)) * z
    return torch.full_like(m, float(base_cfg)) * m

