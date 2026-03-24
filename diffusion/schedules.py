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


def _squared_cosine_beta_schedule_v2_numpy(num_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    """Vectorized reference (same math as the former scalar loop; fast without native)."""
    n = int(num_timesteps)
    if n < 1:
        return np.array([], dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    t1 = i / n
    t2 = (i + 1) / n
    ab1 = np.cos((t1 + 0.008) / 1.008 * np.pi * 0.5) ** 2
    ab2 = np.cos((t2 + 0.008) / 1.008 * np.pi * 0.5) ** 2
    b = np.minimum(1.0 - ab2 / np.maximum(ab1, 1e-12), float(max_beta))
    return _clip_betas(b.astype(np.float64))


def squared_cosine_beta_schedule_v2(num_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    """
    Squared-cosine schedule with small offset (Improved DDPM / diffusers ``squaredcos_cap_v2``).
    ``alpha_bar(t) = cos^2((t + 0.008) / 1.008 * pi / 2)``.

    Uses ``sdx_beta_schedules`` when built; otherwise vectorized NumPy.
    """
    n = int(num_timesteps)
    if n < 1:
        return np.array([], dtype=np.float64)
    try:
        from sdx_native.beta_schedules_native import squared_cosine_betas_v2_native

        got = squared_cosine_betas_v2_native(n, max_beta)
        if got is not None and got.shape == (n,):
            return got
    except ImportError:
        pass
    return _squared_cosine_beta_schedule_v2_numpy(n, max_beta)


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
