"""CADS-style condition annealing for sampling.

CADS (Condition-Annealed Diffusion Sampling, Sadat et al.) fights the diversity
collapse you get at high guidance: it adds noise to the *conditioning* (e.g. text
embeddings) early in sampling — when only coarse structure is being decided — and
anneals that noise away later so fine details stay faithful to the prompt. The net
effect is more varied outputs without losing prompt adherence.

``cads_noise_std`` gives the noise std for a step; ``apply_condition_noise`` applies
it. Consumed by ``gaussian_diffusion.py``'s sampler.
"""

from __future__ import annotations

import torch


def cads_noise_std(
    *,
    progress: float,
    base_strength: float = 0.0,
    min_strength: float = 0.0,
    power: float = 1.0,
) -> float:
    """
    CADS-like condition annealing schedule over denoise progress.
    Early (small progress): stronger condition noise.
    Late (large progress): weaker noise.
    """
    p = max(0.0, min(1.0, float(progress)))
    b = max(0.0, float(base_strength))
    m = max(0.0, float(min_strength))
    pw = max(1e-6, float(power))
    s = m + (b - m) * ((1.0 - p) ** pw)
    return max(0.0, float(s))


def apply_condition_noise(
    cond: torch.Tensor,
    *,
    std: float,
) -> torch.Tensor:
    """
    Add gaussian noise to condition tensor (e.g. text embeddings).
    """
    sig = max(0.0, float(std))
    if sig <= 0.0:
        return cond
    n = torch.randn_like(cond) * sig
    return cond + n
