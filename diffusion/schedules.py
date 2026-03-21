"""
VP (variance-preserving) **beta schedules** for Gaussian diffusion.

Exposes `get_beta_schedule` used by `GaussianDiffusion`. Schedules are chosen to
stay numerically stable (betas clipped) and match common training stacks:

- **linear** — classic DDPM (Ho et al.).
- **cosine** — Nichol & Dhariwal–style cosine *alpha_bar* (as in this repo historically).
- **sigmoid** — smooth ramp in logit space (sometimes easier optimization than hard linear ends).
- **squaredcos_cap_v2** — Improved-DDPM / diffusers-style squared cosine with 0.008 offset (alias: ``squared_cosine_v2``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "get_beta_schedule",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "sigmoid_beta_schedule",
    "squared_cosine_beta_schedule_v2",
]


def _clip_betas(beta: np.ndarray) -> np.ndarray:
    return np.clip(beta.astype(np.float64), 1e-4, 0.999)


def linear_beta_schedule(num_timesteps: int) -> np.ndarray:
    return np.linspace(0.0001, 0.02, int(num_timesteps), dtype=np.float64)


def cosine_beta_schedule(num_timesteps: int) -> np.ndarray:
    steps = np.arange(num_timesteps + 1, dtype=np.float64)
    alpha_bar = np.cos(((steps / num_timesteps) + 0.01) / 1.01 * np.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return _clip_betas(beta)


def sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> np.ndarray:
    """Smooth S-curve from ~``beta_start`` to ~``beta_end``."""
    x = np.linspace(-6.0, 6.0, int(num_timesteps), dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-x))
    beta = sig * (beta_end - beta_start) + beta_start
    return _clip_betas(beta)


def squared_cosine_beta_schedule_v2(num_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    """
    Squared-cosine schedule with small offset (Improved DDPM / diffusers ``squaredcos_cap_v2``).
    ``alpha_bar(t) = cos^2((t + 0.008) / 1.008 * pi / 2)``.
    """

    def alpha_bar_fn(t: float) -> float:
        return float(np.cos((t + 0.008) / 1.008 * np.pi * 0.5) ** 2)

    n = int(num_timesteps)
    betas: list[float] = []
    for i in range(n):
        t1 = i / n
        t2 = (i + 1) / n
        ab1 = alpha_bar_fn(t1)
        ab2 = alpha_bar_fn(t2)
        b = min(1.0 - ab2 / max(ab1, 1e-12), max_beta)
        betas.append(b)
    return _clip_betas(np.asarray(betas, dtype=np.float64))


def get_beta_schedule(schedule_name: str, num_timesteps: int) -> np.ndarray:
    """
    Return ``(num_timesteps,)`` beta array.

    ``schedule_name``: ``linear`` | ``cosine`` | ``sigmoid`` | ``squaredcos_cap_v2`` | ``squared_cosine_v2``.
    """
    name = str(schedule_name).lower().strip()
    if name == "linear":
        return linear_beta_schedule(num_timesteps)
    if name == "cosine":
        return cosine_beta_schedule(num_timesteps)
    if name == "sigmoid":
        return sigmoid_beta_schedule(num_timesteps)
    if name in ("squaredcos_cap_v2", "squared_cosine_v2"):
        return squared_cosine_beta_schedule_v2(num_timesteps)
    raise ValueError(
        f"Unknown beta schedule {schedule_name!r}; "
        "expected linear, cosine, sigmoid, or squaredcos_cap_v2"
    )
