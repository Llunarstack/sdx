"""
Per-step classifier-free guidance schedules.

A fixed high CFG sharpens composition early but over-steers the late, low-noise steps and washes
out detail. Each ``cfg_scale_*`` helper returns ``base_cfg`` times a schedule multiplier, so the
overall strength can change without re-tuning the schedule shape. ``GaussianDiffusion.sample``
selects one via ``cfg_guidance_schedule`` ("linear", "cosine", "piecewise", "snr").
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def _progress(step_index: int, total_steps: int) -> float:
    """Sampling progress in ``[0, 1]``: 0 is the first (noisiest) step, 1 the last."""
    if total_steps <= 1:
        return 1.0
    return max(0.0, min(1.0, step_index / (total_steps - 1)))


def cfg_scale_linear(
    base_cfg: float,
    step_index: int,
    total_steps: int,
    *,
    start_multiplier: float = 0.7,
    end_multiplier: float = 1.0,
) -> float:
    """Linear ramp from ``start_multiplier`` to ``end_multiplier``."""
    p = _progress(step_index, total_steps)
    return float(base_cfg) * (start_multiplier + (end_multiplier - start_multiplier) * p)


def cfg_scale_cosine_ramp(
    base_cfg: float,
    step_index: int,
    total_steps: int,
    *,
    min_multiplier: float = 0.65,
    max_multiplier: float = 1.0,
) -> float:
    """Cosine ease-in/ease-out ramp; avoids the visible knee of a linear ramp."""
    e = 0.5 * (1.0 - math.cos(math.pi * _progress(step_index, total_steps)))
    return float(base_cfg) * (min_multiplier + (max_multiplier - min_multiplier) * e)


def cfg_scale_piecewise(
    base_cfg: float,
    step_index: int,
    total_steps: int,
    *,
    stage_multipliers: Iterable[float] = (0.75, 1.0, 0.9),
) -> float:
    """Three flat stages over the early / middle / late thirds of sampling.

    The default eases in, peaks while structure forms, then backs off so the final low-noise
    steps don't over-sharpen.
    """
    vals = [float(v) for v in stage_multipliers]
    if len(vals) != 3:
        raise ValueError("stage_multipliers must have exactly 3 values")
    p = _progress(step_index, total_steps)
    stage = 0 if p < 1.0 / 3.0 else 1 if p < 2.0 / 3.0 else 2
    return float(base_cfg) * vals[stage]


def cfg_scale_snr_aware_multiplier(
    alpha_cumprod: float,
    *,
    low_noise_multiplier: float = 0.9,
    high_noise_multiplier: float = 0.7,
) -> float:
    """Multiplier keyed on the timestep's noise level rather than its step index.

    Interpolates by ``sigmoid(log SNR)``, which for VP equals ``alpha_cumprod``: pure noise
    gives ``high_noise_multiplier``, a clean signal gives ``low_noise_multiplier``.
    """
    z = max(0.0, min(1.0, float(alpha_cumprod)))
    return high_noise_multiplier + (low_noise_multiplier - high_noise_multiplier) * z
