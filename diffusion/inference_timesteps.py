"""
Inference timestep (index) schedules for VP diffusion.

A schedule returns strictly decreasing (noisy -> clean) integer indices in ``[0, num_train_timesteps)``.
It only picks which training timesteps to visit, so it composes with any step count and any solver
(``GaussianDiffusion.sample_loop(..., solver=...)``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "INFERENCE_TIMESTEP_SCHEDULES",
    "build_inference_timesteps",
    "list_timestep_schedules",
    "register_timestep_schedule",
]

_INDICES_PREFIX = "indices:"
_KARRAS_RHO_DEFAULT = 7.0
# Schedules whose shape is warped by ``karras_rho``.
_RHO_SCHEDULES = frozenset({"karras_rho", "apex"})

EDM_SIGMA_MIN = 0.002
EDM_SIGMA_MAX = 80.0


def parse_indices_timestep_schedule(name: str) -> np.ndarray | None:
    """Explicit indices ``indices:999,800,...,0`` (noisy -> clean), or ``None`` for any other name.

    The prefix avoids clashing with the comma-separated training ``timestep_respacing`` strings.
    """
    raw = str(name).strip()
    if not raw.lower().startswith(_INDICES_PREFIX):
        return None
    body = raw[len(_INDICES_PREFIX) :].strip()
    parts = [p.strip() for p in body.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return np.asarray([int(float(p)) for p in parts], dtype=np.int64)
    except ValueError:
        raise ValueError(f"indices: schedule contains non-integers: {body!r}") from None


def _enforce_strict_descending(idx: np.ndarray, num_train: int) -> np.ndarray:
    """Clip to the index range and force strict descent, dropping steps once 0 is reached."""
    idx = np.clip(np.asarray(idx, dtype=np.int64), 0, num_train - 1)
    if idx.size == 0:
        return np.array([num_train - 1, 0], dtype=np.int64)
    out = [int(idx[0])]
    for v in idx[1:]:
        v = min(int(v), out[-1] - 1)
        if v < 0:
            break
        out.append(v)
    return np.array(out, dtype=np.int64)


def _resample_length_numpy(desc: np.ndarray, target_len: int, num_train: int) -> np.ndarray:
    """Resample a descending index path to ``target_len > 0`` points."""
    d = _enforce_strict_descending(desc, num_train)
    if d.size == target_len:
        return d
    if d.size > target_len:
        pos = np.round(np.linspace(0, d.size - 1, target_len)).astype(np.int64)
        return _enforce_strict_descending(d[pos], num_train)
    if d.size == 1:
        return np.linspace(num_train - 1, 0, target_len, dtype=np.int64)
    vals = np.interp(np.linspace(0.0, 1.0, target_len), np.linspace(0.0, 1.0, d.size), d.astype(np.float64))
    return _enforce_strict_descending(np.round(vals).astype(np.int64), num_train)


def _resample_length(desc: np.ndarray, target_len: int, num_train: int) -> np.ndarray:
    """Use the optional native ``sdx_inference_timesteps`` finalizer when built, else NumPy."""
    if target_len <= 0:
        return np.array([0], dtype=np.int64)
    try:
        from sdx_native.inference_timesteps_native import finalize_inference_timesteps_native
    except ImportError:
        pass
    else:
        got = finalize_inference_timesteps_native(desc, target_len, num_train)
        if got is not None:
            return got
    return _resample_length_numpy(desc, target_len, num_train)


ScheduleFn = Callable[..., np.ndarray]

INFERENCE_TIMESTEP_SCHEDULES: dict[str, ScheduleFn] = {}


def register_timestep_schedule(name: str):
    def deco(fn: ScheduleFn) -> ScheduleFn:
        INFERENCE_TIMESTEP_SCHEDULES[name.lower()] = fn
        return fn

    return deco


def _log_snr(alpha_cumprod: np.ndarray) -> np.ndarray:
    ac = np.clip(np.asarray(alpha_cumprod, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(ac) - np.log(1.0 - ac)


def _nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the entry of ``values`` closest to each target (first on ties)."""
    return np.abs(values[None, :] - np.asarray(targets)[:, None]).argmin(axis=1).astype(np.int64)


@register_timestep_schedule("ddim")
def _schedule_ddim(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    step = max(1, num_train // max(num_infer, 1))
    return np.arange(0, num_train, step, dtype=np.int64)[::-1]


@register_timestep_schedule("euler")
def _schedule_euler(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    return np.linspace(0, num_train - 1, num_infer, dtype=np.int64)[::-1]


def karras_sigmas(sigma_max: float, sigma_min: float, n: int, rho: float = _KARRAS_RHO_DEFAULT) -> np.ndarray:
    """Karras et al. 2022 (EDM, Eq. 5): ``n`` noise levels from ``sigma_max`` to ``sigma_min``, uniform in sigma**(1/rho)."""
    ramp = np.linspace(0.0, 1.0, max(1, int(n)))
    hi, lo = float(sigma_max) ** (1.0 / rho), float(sigma_min) ** (1.0 / rho)
    return (hi + ramp * (lo - hi)) ** rho


@register_timestep_schedule("karras_rho")
def _schedule_karras_rho(
    num_train: int, num_infer: int, alpha_cumprod: np.ndarray, rho: float = _KARRAS_RHO_DEFAULT
) -> np.ndarray:
    """EDM rho-spacing in sigma = sqrt((1-abar)/abar), each level snapped to the nearest training index (log-sigma)."""
    if num_infer <= 1:
        return np.array([num_train - 1], dtype=np.int64)
    ac = np.clip(np.asarray(alpha_cumprod, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    log_sig = 0.5 * (np.log1p(-ac) - np.log(ac))
    target = karras_sigmas(np.exp(log_sig[-1]), np.exp(log_sig[0]), num_infer, rho=float(rho))
    return np.round(np.interp(np.log(target), log_sig, np.arange(num_train))).astype(np.int64)


@register_timestep_schedule("snr_uniform")
def _schedule_snr_uniform(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    """Approximately uniform spacing in log-SNR."""
    if num_infer <= 1:
        return np.array([num_train - 1], dtype=np.int64)
    logsnr = _log_snr(alpha_cumprod)
    return _nearest_indices(logsnr, np.linspace(logsnr[-1], logsnr[0], num_infer))


@register_timestep_schedule("quad_cosine")
def _schedule_quad_cosine(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    """Denser near pure noise and near clean (cosine-warped uniform grid)."""
    if num_infer <= 1:
        return np.array([num_train - 1], dtype=np.int64)
    w = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, num_infer))
    return np.round(w * (num_train - 1)).astype(np.int64)[::-1]


def ays_progress_knots(variant: str) -> np.ndarray:
    """Normalized progress 0 (noise) -> 1 (clean) for Align-Your-Steps style grids."""
    if variant == "ays_dit":
        # Denser in the mid-SNR band, the hardest denoising region for the DiT.
        knots = [0.0, 0.04, 0.10, 0.18, 0.28, 0.38, 0.48, 0.58, 0.68, 0.78, 0.88, 0.94, 0.98, 1.0]
    else:
        knots = [0.0, 0.05, 0.12, 0.22, 0.35, 0.50, 0.65, 0.78, 0.88, 0.94, 0.98, 1.0]
    return np.array(knots, dtype=np.float64)


def _schedule_ays_family(num_train: int, num_infer: int, alpha_cumprod: np.ndarray, *, variant: str) -> np.ndarray:
    """Place the AYS progress knots on the log-SNR path and snap ``num_infer`` resampled points to indices."""
    if num_infer <= 1:
        return np.array([num_train - 1], dtype=np.int64)
    logsnr = _log_snr(alpha_cumprod)
    ls_clean, ls_noise = float(logsnr[0]), float(logsnr[-1])
    knots = ays_progress_knots(variant)
    target_ls = ls_noise + knots * (ls_clean - ls_noise)
    target = np.interp(np.linspace(0.0, 1.0, num_infer), np.linspace(0.0, 1.0, len(knots)), target_ls)
    idx = _nearest_indices(logsnr, target)
    return idx[::-1].copy() if idx[0] < idx[-1] else idx


@register_timestep_schedule("ays")
def _schedule_ays(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    return _schedule_ays_family(num_train, num_infer, alpha_cumprod, variant="ays")


@register_timestep_schedule("ays_dit")
def _schedule_ays_dit(num_train: int, num_infer: int, alpha_cumprod: np.ndarray) -> np.ndarray:
    return _schedule_ays_family(num_train, num_infer, alpha_cumprod, variant="ays_dit")


@register_timestep_schedule("apex")
def _schedule_apex(
    num_train: int, num_infer: int, alpha_cumprod: np.ndarray, rho: float = _KARRAS_RHO_DEFAULT
) -> np.ndarray:
    """APEX three-phase density on the lambda path (see ``diffusion.solvers.apex``)."""
    from diffusion.solvers.apex import build_apex_vp_indices

    return build_apex_vp_indices(num_train, num_infer, alpha_cumprod, rho=rho)


def build_inference_timesteps(
    name: str,
    num_train_timesteps: int,
    num_inference_steps: int,
    alpha_cumprod: np.ndarray,
    *,
    karras_rho: float = _KARRAS_RHO_DEFAULT,
) -> np.ndarray:
    """``(num_inference_steps,)`` int64 indices, noisy -> clean, for a registered schedule or ``indices:...``.

    ``karras_rho`` warps the ``karras_rho`` and ``apex`` schedules.
    """
    explicit = parse_indices_timestep_schedule(name)
    if explicit is not None:
        clipped = np.clip(explicit, 0, int(num_train_timesteps) - 1)
        return _resample_length(clipped, num_inference_steps, num_train_timesteps)

    key = str(name).strip().lower()
    fn = INFERENCE_TIMESTEP_SCHEDULES.get(key)
    if fn is None:
        raise ValueError(f"Unknown timestep schedule {name!r}. Choose one of: {', '.join(list_timestep_schedules())}")
    if key in _RHO_SCHEDULES:
        raw = fn(num_train_timesteps, num_inference_steps, alpha_cumprod, rho=float(karras_rho))
    else:
        raw = fn(num_train_timesteps, num_inference_steps, alpha_cumprod)
    return _resample_length(raw, num_inference_steps, num_train_timesteps)


def list_timestep_schedules() -> list[str]:
    return sorted(INFERENCE_TIMESTEP_SCHEDULES)
