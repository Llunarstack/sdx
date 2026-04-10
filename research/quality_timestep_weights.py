"""
Sketches for **combining ViT (or proxy) scores with diffusion timestep structure**.

Use cases: test-time reweighting, auxiliary loss shaping, or curriculum-style gates.
All ops are scalar / tensor-safe without importing ViT modules.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch

Scalar = Union[float, torch.Tensor]


def timestep_confidence_weight(
    t_norm: Scalar,
    *,
    low_noise_boost: float = 1.15,
    high_noise_damp: float = 0.85,
) -> torch.Tensor:
    """
    Map normalized diffusion time ``t_norm`` in ``[0, 1]`` (0=clean, 1=noise) to a scalar weight.

    Down-weights very noisy steps and slightly up-weights mid/late denoising — a simple prior
    when ViT scores are noisy at high t.
    """
    t = torch.as_tensor(t_norm, dtype=torch.float32).clamp(0.0, 1.0)
    # Piecewise: more trust (weight>1) in mid band, less at extremes
    mid = 4.0 * t * (1.0 - t)
    w = low_noise_boost * (0.5 + 0.5 * mid) + (1.0 - mid) * high_noise_damp
    return w


def vit_score_timestep_gate(
    vit_score: Scalar,
    t_norm: Scalar,
    *,
    score_threshold: float = 0.5,
    reject_high_t: float = 0.92,
) -> torch.Tensor:
    """
    Return a multiplicative gate in ``(0, 1]`` from a ViT-like score in ``[0, 1]`` and ``t_norm``.

    If the sample looks weak (``vit_score < threshold``) and we are still very noisy
    (``t_norm > reject_high_t``), damp aggressively so downstream code can skip expensive work.
    """
    s = torch.as_tensor(vit_score, dtype=torch.float32).clamp(0.0, 1.0)
    t = torch.as_tensor(t_norm, dtype=torch.float32).clamp(0.0, 1.0)
    bad = (s < score_threshold) & (t > reject_high_t)
    return torch.where(bad, torch.tensor(0.25, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32))


def merge_vit_and_diffusion_weights(
    w_vit: Sequence[float],
    w_diff: Sequence[float],
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> list[float]:
    """
    Normalize two positive weight vectors and blend: ``(1-alpha)*norm(w_vit) + alpha*norm(w_diff)``.

    Handy for pick-best ensembles that mix CLIP/ViT with diffusion likelihood proxies.
    """
    if not w_vit or not w_diff or len(w_vit) != len(w_diff):
        raise ValueError("w_vit and w_diff must be same non-empty length")
    a = max(0.0, min(1.0, float(alpha)))
    v = [max(float(x), eps) for x in w_vit]
    d = [max(float(x), eps) for x in w_diff]
    sv = sum(v)
    sd = sum(d)
    v = [x / sv for x in v]
    d = [x / sd for x in d]
    return [(1.0 - a) * vi + a * di for vi, di in zip(v, d)]
