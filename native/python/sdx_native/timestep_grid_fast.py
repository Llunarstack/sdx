"""Integer timestep grids for diffusion sampling / training schedules (numpy)."""

from __future__ import annotations

import numpy as np


def linspace_int_timesteps(num_steps: int, t_max: int, t_min: int = 0) -> np.ndarray:
    """Inclusive-spaced integer steps from ``t_max`` down toward ``t_min`` (length ``num_steps``)."""
    if num_steps < 1:
        raise ValueError("num_steps >= 1")
    raw = np.linspace(t_max, t_min, num_steps)
    return np.round(raw).astype(np.int64).clip(t_min, t_max)


def uniform_subsample(total: int, num_steps: int) -> np.ndarray:
    """Indices ``0..total-1`` uniformly sampled (unique, sorted descending)."""
    if total < 1 or num_steps < 1:
        raise ValueError("total and num_steps must be >= 1")
    idx = np.linspace(total - 1, 0, num_steps)
    return np.unique(np.round(idx).astype(np.int64))


def append_zero_terminal(steps: np.ndarray) -> np.ndarray:
    """Ensure final timestep 0 if not present (common sampler convention)."""
    if steps.size == 0:
        return np.array([0], dtype=np.int64)
    if int(steps[-1]) == 0:
        return steps
    return np.append(steps.astype(np.int64), 0)


def cosine_spacing_int(num_steps: int, t_max: int, t_min: int = 0) -> np.ndarray:
    """More steps near high noise (cosine spacing in index space)."""
    if num_steps < 1:
        raise ValueError("num_steps >= 1")
    u = (1.0 - np.cos(np.linspace(0, np.pi / 2, num_steps)))  # 0..1 monotonic
    ts = t_max - u * (t_max - t_min)
    return np.round(ts).astype(np.int64).clip(t_min, t_max)


def match_dims(batch: int, steps_1d: np.ndarray) -> np.ndarray:
    """``(T,)`` → ``(batch, T)`` view-free repeat."""
    return np.broadcast_to(steps_1d, (batch, steps_1d.shape[0])).copy()
