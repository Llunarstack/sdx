"""
CFG (classifier-free guidance) schedule helpers for diffusion sampling.

Why this module exists
----------------------
A single fixed guidance scale is a blunt instrument: a high CFG value that
sharpens composition early in sampling tends to over-steer and wash out fine
detail in the later, low-noise steps. These helpers instead vary the effective
guidance *per step* (or per timestep), so guidance can be strong where it helps
structure and gentler where it would hurt detail.

Shared convention
-----------------
Every ``cfg_scale_*`` function returns ``base_cfg`` multiplied by a schedule
``multiplier`` in roughly ``[0, 1]``. Callers pass the model's nominal CFG as
``base_cfg`` and use the returned value as the actual guidance for that step.
Keeping the multiplier separate from the base scale means a user can change the
overall strength (``base_cfg``) without re-tuning the schedule shape.

These are consumed by the samplers in ``diffusion/gaussian_diffusion.py`` and
selected via the ``cfg_guidance_schedule`` option ("linear", "cosine",
"piecewise", "snr").
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _safe_progress(step_index: int, total_steps: int) -> float:
    """Return sampling progress as a fraction in ``[0, 1]``.

    ``0.0`` is the first step (highest noise) and ``1.0`` is the last step
    (lowest noise). Guards the single-step case (``total_steps <= 1``) and
    clamps so out-of-range indices can't push the multiplier past its bounds.
    """
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
    """Linearly ramp guidance from ``start_multiplier`` to ``end_multiplier``.

    The safe default: start below 1.0 to avoid over-steering the noisy early
    steps, then rise to full strength as structure locks in. Predictable and
    easy to reason about; reach for ``cosine`` if the linear knee looks abrupt.
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
    """Same idea as ``cfg_scale_linear`` but with cosine (ease-in/ease-out) shaping.

    The cosine curve spends more steps near both endpoints and less time in the
    transition, which avoids the visible "kink" a linear ramp can produce. Good
    default when you want a smooth, monotonic increase in guidance.
    """
    p = _safe_progress(step_index, total_steps)
    e = 0.5 * (1.0 - math.cos(math.pi * p))  # cosine ease mapping p: 0->1 smoothly
    m = float(min_multiplier) + (float(max_multiplier) - float(min_multiplier)) * e
    return float(base_cfg) * m


def cfg_scale_piecewise(
    base_cfg: float,
    step_index: int,
    total_steps: int,
    *,
    stage_multipliers: Iterable[float] = (0.75, 1.0, 0.9),
) -> float:
    """Three flat stages (early / middle / late thirds of sampling).

    Use when you want explicit, independent control of each phase rather than a
    smooth curve. The default (0.75, 1.0, 0.9) eases in, peaks while structure
    forms, then backs off slightly so the final low-noise steps don't over-sharpen.
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


def cfg_scale_snr_aware_multiplier(
    alpha_cumprod: float,
    *,
    low_noise_multiplier: float = 0.9,
    high_noise_multiplier: float = 0.7,
) -> float:
    """Scalar (Python ``float``) SNR-aware multiplier for a single timestep.

    Unlike the step-index schedules above, this keys off the *actual* noise
    level at a timestep via ``alpha_cumprod`` (the cumulative product of alphas,
    i.e. how much signal survives). This is the per-step scalar used inside the
    samplers; ``cfg_scale_snr_aware`` is the batched tensor counterpart and uses
    the same logistic-on-log-SNR shaping.

    Intuition: high noise (low SNR) -> weaker guidance (``high_noise_multiplier``);
    low noise (high SNR) -> stronger guidance (``low_noise_multiplier``).

    Args:
        alpha_cumprod: Cumulative alpha for the timestep, in ``(0, 1)``.
        low_noise_multiplier: Multiplier applied as SNR -> infinity (clean signal).
        high_noise_multiplier: Multiplier applied as SNR -> 0 (pure noise).
    """
    # Clamp away from 0/1 so the SNR ratio and log can't blow up at the extremes.
    a = float(max(1e-9, min(1.0 - 1e-9, float(alpha_cumprod))))
    snr = a / (1.0 - a + 1e-8)
    # z in [0,1): 0 at high noise, ->1 at low noise. Equivalent to sigmoid(log SNR).
    z = snr / (1.0 + snr + 1e-8)
    ln = float(low_noise_multiplier)
    hn = float(high_noise_multiplier)
    # Interpolate between the two endpoints by how "clean" the step is.
    return hn + (ln - hn) * z


def cfg_scale_snr_aware(
    base_cfg: float,
    alpha_cumprod_t: torch.Tensor,
    *,
    low_noise_multiplier: float = 0.9,
    high_noise_multiplier: float = 0.7,
) -> torch.Tensor:
    """Batched, per-sample SNR-aware guidance (tensor counterpart of the scalar above).

    Computes one guidance value per element of ``alpha_cumprod_t`` so a batch can
    be at different timesteps. Returns ``base_cfg`` scaled by the SNR multiplier.

    High noise (low SNR) -> ``high_noise_multiplier``; low noise (high SNR) ->
    ``low_noise_multiplier``.
    """
    a = alpha_cumprod_t.to(dtype=torch.float32)
    snr = a / (1.0 - a + 1e-8)
    # Map log-SNR into [0,1]; matches the scalar ``cfg_scale_snr_aware_multiplier`` path.
    z = torch.sigmoid(torch.log(snr + 1e-8))
    m = float(high_noise_multiplier) + (float(low_noise_multiplier) - float(high_noise_multiplier)) * z
    return torch.full_like(m, float(base_cfg)) * m
