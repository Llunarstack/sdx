"""
Pluggable **inference timestep** (index) schedules for VP diffusion.

Schedules return integer indices in [0, num_train_timesteps), **strictly decreasing**
(noisy → clean). They compose with any ``num_inference_steps`` and any update **solver**
(see ``GaussianDiffusion.sample_loop(..., solver=...)``).

Schedules only choose *which* discrete training timesteps to visit; they do not change
the noise schedule or the model.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

__all__ = [
    "INFERENCE_TIMESTEP_SCHEDULES",
    "build_inference_timesteps",
    "list_timestep_schedules",
    "register_timestep_schedule",
]


def _enforce_strict_descending(idx: np.ndarray, num_train: int) -> np.ndarray:
    idx = np.clip(np.asarray(idx, dtype=np.int64), 0, num_train - 1)
    if idx.size == 0:
        return np.array([num_train - 1, 0], dtype=np.int64)
    out: List[int] = [int(idx[0])]
    for j in range(1, len(idx)):
        v = int(idx[j])
        if v >= out[-1]:
            v = out[-1] - 1
        out.append(max(v, 0))
    dedup: List[int] = [out[0]]
    for v in out[1:]:
        if v < dedup[-1]:
            dedup.append(v)
    return np.array(dedup, dtype=np.int64)


def _resample_length_numpy(desc: np.ndarray, target_len: int, num_train: int) -> np.ndarray:
    """Interpolate a descending index path to ``target_len`` points (pure NumPy reference)."""
    if target_len <= 0:
        return np.array([0], dtype=np.int64)
    d = _enforce_strict_descending(desc, num_train)
    if d.size >= target_len:
        if d.size == target_len:
            return d
        pos = np.linspace(0, d.size - 1, target_len)
        idx = np.round(pos).astype(np.int64)
        return _enforce_strict_descending(d[idx], num_train)
    if d.size == 1:
        return (
            np.linspace(num_train - 1, 0, target_len, dtype=np.int64)
            if target_len > 1
            else np.array([num_train - 1], dtype=np.int64)
        )
    old_x = np.linspace(0.0, 1.0, d.size)
    new_x = np.linspace(0.0, 1.0, target_len)
    vals = np.interp(new_x, old_x, d.astype(np.float64))
    rounded = np.round(vals).astype(np.int64)
    return _enforce_strict_descending(rounded, num_train)


def _resample_length(desc: np.ndarray, target_len: int, num_train: int) -> np.ndarray:
    """Prefer native finalization when ``sdx_inference_timesteps`` is built; else NumPy."""
    if target_len <= 0:
        return np.array([0], dtype=np.int64)
    try:
        from sdx_native.inference_timesteps_native import finalize_inference_timesteps_native

        got = finalize_inference_timesteps_native(desc, target_len, num_train)
        if got is not None:
            return got
    except ImportError:
        pass
    return _resample_length_numpy(desc, target_len, num_train)


ScheduleFn = Callable[[int, int, np.ndarray], np.ndarray]

INFERENCE_TIMESTEP_SCHEDULES: Dict[str, ScheduleFn] = {}
_KARRAS_RHO_DEFAULT = 7.0


def register_timestep_schedule(name: str):
    def deco(fn: ScheduleFn) -> ScheduleFn:
        INFERENCE_TIMESTEP_SCHEDULES[name.lower()] = fn
        return fn

    return deco


@register_timestep_schedule("ddim")
def _schedule_ddim(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    step = max(1, num_train // max(num_infer, 1))
    return np.arange(0, num_train, step, dtype=np.int64)[::-1]


@register_timestep_schedule("euler")
def _schedule_euler(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    return np.linspace(0, num_train - 1, num_infer, dtype=np.int64)[::-1]


def _schedule_karras_rho_with_rho(
    num_train: int, num_infer: int, alpha_cumprod: np.ndarray, rho: float = _KARRAS_RHO_DEFAULT
) -> np.ndarray:
    """ρ spacing in sigma space (σ = sqrt((1-ᾱ)/ᾱ)); denser steps where σ changes quickly."""
    ac = np.asarray(alpha_cumprod, dtype=np.float64)
    ac = np.clip(ac, 1e-9, 1.0 - 1e-9)
    sig = np.sqrt((1.0 - ac) / ac)
    if num_infer <= 1:
        return np.array([num_train - 1], dtype=np.int64)
    rho = float(rho)
    ramp = np.linspace(0.0, 1.0, num_infer) ** rho
    sig_max = float(sig[-1])
    sig_min = float(sig[0])
    if sig_max <= sig_min + 1e-12:
        return _schedule_euler(num_train, num_infer, alpha_cumprod)
    sig_t = sig_max * (sig_min / sig_max) ** ramp
    return np.array([int(np.argmin(np.abs(sig - s))) for s in sig_t], dtype=np.int64)


@register_timestep_schedule("snr_uniform")
def _schedule_snr_uniform(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    """Approximately uniform spacing in log-SNR (log ᾱ - log(1-ᾱ))."""
    ac = np.asarray(alpha_cumprod, dtype=np.float64)
    ac = np.clip(ac, 1e-9, 1.0 - 1e-9)
    logsnr = np.log(ac) - np.log(1.0 - ac)
    if num_infer <= 1:
        return np.array([num_train - 1], dtype=np.int64)
    u = np.linspace(float(logsnr[-1]), float(logsnr[0]), num_infer)
    return np.array([int(np.argmin(np.abs(logsnr - ui))) for ui in u], dtype=np.int64)


@register_timestep_schedule("quad_cosine")
def _schedule_quad_cosine(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    """More visits near extreme noise and near clean (smooth U in continuous time)."""
    if num_infer <= 1:
        return np.array([num_train - 1], dtype=np.int64)
    u = np.linspace(0.0, 1.0, num_infer)
    w = 0.5 - 0.5 * np.cos(np.pi * u)
    t_float = w * (num_train - 1)
    return np.round(t_float).astype(np.int64)[::-1]


def build_inference_timesteps(
    name: str,
    num_train_timesteps: int,
    num_inference_steps: int,
    alpha_cumprod: np.ndarray,
    *,
    karras_rho: float = 7.0,
) -> np.ndarray:
    """
    :param name: registered schedule key (e.g. ``ddim``, ``karras_rho``).
    :param alpha_cumprod: (T,) training ᾱ (numpy).
    :return: (num_inference_steps,) int64 indices, high → low.
    """
    key = str(name).lower().strip()
    known = sorted(set(INFERENCE_TIMESTEP_SCHEDULES.keys()) | {"karras_rho"})
    if key == "karras_rho":
        raw = _schedule_karras_rho_with_rho(
            num_train_timesteps, num_inference_steps, alpha_cumprod, rho=float(karras_rho)
        )
    elif key in INFERENCE_TIMESTEP_SCHEDULES:
        fn = INFERENCE_TIMESTEP_SCHEDULES[key]
        raw = fn(num_train_timesteps, num_inference_steps, alpha_cumprod)
    else:
        raise ValueError(f"Unknown timestep schedule {name!r}. Choose one of: {', '.join(known)}")
    return _resample_length(raw, num_inference_steps, num_train_timesteps)


def list_timestep_schedules() -> List[str]:
    return sorted(set(INFERENCE_TIMESTEP_SCHEDULES.keys()) | {"karras_rho"})
