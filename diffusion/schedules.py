"""VP beta schedules for ``GaussianDiffusion`` (betas clipped to ``[1e-4, 0.999]`` for stability).

- ``linear``: DDPM (Ho et al. 2020).
- ``cosine``: IDDPM cosine alpha_bar (Nichol & Dhariwal 2021).
- ``sigmoid``: smooth ramp in logit space.
- ``squaredcos_cap_v2`` (alias ``squared_cosine_v2``): the diffusers name for the same IDDPM cosine.

Any name with the ``_ztsnr`` suffix (e.g. ``linear_ztsnr``) is rescaled to zero terminal SNR
(:func:`rescale_zero_terminal_snr`).

Also: SNR / alpha_cumprod helpers, the SD3 / Flux resolution time shifts, DDIM-style respacing
(``space_timesteps``), and training-time timesteps: ``sample_training_timesteps`` (uniform, logit-normal,
high/low-noise) and ``get_timestep_loss_weight`` (min-SNR, min-SNR-soft, EDM, v, eps, unit; every scheme is a
weight on the x0 error converted to the space the model predicts in, Hang et al. 2023 Sec. 3.4).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

# ==================================================================================================
# snr_utils: NumPy SNR / alpha_cumprod helpers and the SD3 / Flux resolution time shifts.
#
# The schedule helpers use the optional Rust ``sdx_diffusion_math`` library when it is built.
# ==================================================================================================


def rust_diffusion_math(fn_name: str, *args: Any) -> NDArray[Any] | None:
    """Result of ``sdx_native.ffi.<fn_name>(*args)``, or ``None`` when the library is unavailable."""
    try:
        from sdx_native import ffi as diffusion_math_native

        return getattr(diffusion_math_native, fn_name)(*args)
    except (ImportError, OSError, RuntimeError):
        return None


def alpha_cumprod_from_betas(betas: NDArray[Any]) -> NDArray[Any]:
    """``alpha_cumprod[t] = prod_{s<=t} (1 - beta[s])``."""
    beta = np.asarray(betas, dtype=np.float64)
    result = rust_diffusion_math("maybe_alpha_cumprod_rust", beta)
    return result if result is not None else np.cumprod(1.0 - beta)


def snr_from_alpha_cumprod(alpha_cumprod: NDArray[Any]) -> NDArray[Any]:
    """``SNR(t) = alpha_bar[t] / (1 - alpha_bar[t])`` for VP diffusion."""
    ab = np.asarray(alpha_cumprod, dtype=np.float64)
    result = rust_diffusion_math("maybe_snr_from_alpha_cumprod_rust", ab)
    return result if result is not None else ab / (1.0 - ab + 1e-8)


def snr_from_betas(betas: NDArray[Any]) -> NDArray[Any]:
    return snr_from_alpha_cumprod(alpha_cumprod_from_betas(betas))


def time_shift(s, shift: float):
    """SD3 resolution shift of flow time ``s`` (``s = 1`` is noise): ``shift*s / (1 + (shift-1)*s)``.

    Esser et al. 2024 (SD3), Eq. 23. It divides the flow SNR ``((1-s)/s)^2`` by ``shift**2`` at every ``s``;
    Flux's ``time_shift(mu, 1, s)`` is the same map with ``shift = exp(mu)``. Floats, arrays and tensors.
    """
    return shift * s / (1.0 + (shift - 1.0) * s)


def inverse_time_shift(s, shift: float):
    """Inverse of :func:`time_shift`."""
    return s / (shift - (shift - 1.0) * s)


def sd3_resolution_shift(num_tokens: float, base_tokens: float, base_shift: float = 1.0) -> float:
    """Shift for ``num_tokens`` given a schedule tuned at ``base_tokens``: ``base_shift * sqrt(m / n)`` (SD3 Sec. 5.3.2).

    More pixels keep more signal at the same noise level, so larger canvases need proportionally more noise
    to destroy the same global structure.
    """
    return float(base_shift) * math.sqrt(float(num_tokens) / float(base_tokens))


def flux_resolution_shift(
    num_tokens: float,
    *,
    base_tokens: float = 256,
    max_tokens: float = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Flux dynamic shift: ``mu`` linear in the image sequence length (``base_shift`` at ``base_tokens``,
    ``max_shift`` at ``max_tokens``), ``shift = exp(mu)``."""
    slope = (max_shift - base_shift) / (max_tokens - base_tokens)
    return math.exp(base_shift + slope * (float(num_tokens) - base_tokens))


def vp_shifted_timestep(alpha_cumprod: NDArray[Any], t: int, shift: float) -> int:
    """VP timestep whose log-SNR is ``logSNR(t) - 2 log(shift)``: the SD3 resolution shift on a discrete VP
    schedule (the same SNR shift Hoogeboom et al. 2023 apply to the noise schedule of larger images)."""
    if shift == 1.0:
        return int(t)
    log_snr = np.log(snr_from_alpha_cumprod(alpha_cumprod))
    target = log_snr[int(t)] - 2.0 * math.log(float(shift))
    # log_snr decreases with t, so -log_snr is ascending.
    i = int(np.clip(np.searchsorted(-log_snr, -target), 1, len(log_snr) - 1))
    return i if abs(log_snr[i] - target) <= abs(log_snr[i - 1] - target) else i - 1


# ==================================================================================================
# schedules
# ==================================================================================================


def _clip_betas(beta: NDArray[Any]) -> NDArray[Any]:
    return np.clip(beta.astype(np.float64), 1e-4, 0.999)


def linear_beta_schedule(num_timesteps: int) -> NDArray[Any]:
    result = rust_diffusion_math("maybe_linear_beta_schedule_rust", int(num_timesteps))
    if result is not None:
        return result.astype(np.float64)
    return np.linspace(0.0001, 0.02, int(num_timesteps), dtype=np.float64)


def _squared_cosine_betas_numpy(num_timesteps: int, max_beta: float = 0.999) -> NDArray[Any]:
    """``beta_t = 1 - alpha_bar(t+1) / alpha_bar(t)`` with ``alpha_bar(t) = cos^2((t/T + 0.008) / 1.008 * pi/2)``."""
    n = int(num_timesteps)
    if n < 1:
        return np.array([], dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    ab1 = np.cos((i / n + 0.008) / 1.008 * np.pi * 0.5) ** 2
    ab2 = np.cos(((i + 1) / n + 0.008) / 1.008 * np.pi * 0.5) ** 2
    return _clip_betas(np.minimum(1.0 - ab2 / np.maximum(ab1, 1e-12), float(max_beta)))


def cosine_beta_schedule(num_timesteps: int) -> NDArray[Any]:
    """Nichol & Dhariwal (IDDPM) cosine schedule; same betas as ``squaredcos_cap_v2``."""
    result = rust_diffusion_math("maybe_cosine_beta_schedule_rust", int(num_timesteps))
    if result is not None:
        return _clip_betas(result)
    return _squared_cosine_betas_numpy(num_timesteps)


def sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> NDArray[Any]:
    """Smooth S-curve from ~``beta_start`` to ~``beta_end``."""
    x = np.linspace(-6.0, 6.0, int(num_timesteps), dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-x))
    return _clip_betas(sig * (beta_end - beta_start) + beta_start)


def squared_cosine_beta_schedule_v2(num_timesteps: int, max_beta: float = 0.999) -> NDArray[Any]:
    """Improved-DDPM / diffusers ``squaredcos_cap_v2``: Rust library, else the C++ one, else NumPy."""
    n = int(num_timesteps)
    if n < 1:
        return np.array([], dtype=np.float64)
    result = rust_diffusion_math("maybe_squaredcos_beta_schedule_v2_rust", n, float(max_beta))
    if result is not None and result.shape == (n,):
        return result
    try:
        from sdx_native.ffi import squared_cosine_betas_v2_native
    except ImportError:
        pass
    else:
        got = squared_cosine_betas_v2_native(n, max_beta)
        if got is not None and got.shape == (n,):
            return got
    return _squared_cosine_betas_numpy(n, max_beta)


ZTSNR_SUFFIX = "_ztsnr"
# Terminal alpha_bar kept just above 0 (at most diffusers' 2**-24, and below the previous step so the schedule stays
# monotone): the last step is pure noise for any practical purpose, while sigma = sqrt((1-abar)/abar) and log-SNR stay
# finite for the samplers.
_ZTSNR_TERMINAL_ALPHA_BAR = 2.0**-24


def rescale_zero_terminal_snr(betas: NDArray[Any]) -> NDArray[Any]:
    """Lin et al. 2024, "Common Diffusion Noise Schedules and Sample Steps are Flawed", Algorithm 1.

    Shift and scale ``sqrt(alpha_bar)`` so it keeps its first value and ends at 0: the model is trained on pure noise
    at ``t = T`` (no signal leak), which is what sampling starts from, so it can produce truly dark/bright images.
    Requires v- (or x0-) prediction; eps-prediction is undefined at zero SNR.
    """
    alphas_bar_sqrt = np.sqrt(np.cumprod(1.0 - np.asarray(betas, dtype=np.float64)))
    first, last = alphas_bar_sqrt[0], alphas_bar_sqrt[-1]
    alphas_bar_sqrt = (alphas_bar_sqrt - last) * first / (first - last)
    alphas_bar = alphas_bar_sqrt**2
    alphas_bar[-1] = min(_ZTSNR_TERMINAL_ALPHA_BAR, 0.5 * alphas_bar[-2])
    alphas = np.concatenate([alphas_bar[:1], alphas_bar[1:] / alphas_bar[:-1]])
    return 1.0 - alphas


def get_beta_schedule(schedule_name: str, num_timesteps: int) -> NDArray[Any]:
    """``(num_timesteps,)`` betas for ``linear`` | ``cosine`` | ``sigmoid`` | ``squaredcos_cap_v2`` |
    ``squared_cosine_v2``, optionally with the ``_ztsnr`` suffix (zero terminal SNR)."""
    name = str(schedule_name).lower().strip()
    if name.endswith(ZTSNR_SUFFIX):
        return rescale_zero_terminal_snr(get_beta_schedule(name[: -len(ZTSNR_SUFFIX)], num_timesteps))
    if name == "linear":
        return linear_beta_schedule(num_timesteps)
    if name == "cosine":
        return cosine_beta_schedule(num_timesteps)
    if name == "sigmoid":
        return sigmoid_beta_schedule(num_timesteps)
    if name in ("squaredcos_cap_v2", "squared_cosine_v2"):
        return squared_cosine_beta_schedule_v2(num_timesteps)
    raise ValueError(f"Unknown beta schedule {schedule_name!r}; expected linear, cosine, sigmoid, or squaredcos_cap_v2")


# ==================================================================================================
# respace: Timestep respacing: choose which of the training timesteps to visit when sampling.
#
# Ported from external/DiT/diffusion/schedules.py.
# ==================================================================================================


def space_timesteps(num_timesteps: int, section_counts: str | list[int]) -> NDArray[Any]:
    """Sorted unique timestep indices in ``[0, num_timesteps)``.

    ``section_counts`` is ``"ddimN"`` (an even integer stride giving exactly N steps, falling back
    to ``linspace`` when no stride fits), a comma string such as ``"10,15,20"``, or a list of ints:
    the number of steps to take in each equal-sized portion of the schedule.
    """
    if isinstance(section_counts, str):
        s = section_counts.strip().lower()
        if s.startswith("ddim"):
            n = int(s[4:].strip())
            t = int(num_timesteps)
            # len(range(0, t, stride)) == (t - 1) // stride + 1
            for stride in range(1, t):
                if (t - 1) // stride + 1 == n:
                    return np.arange(0, t, stride, dtype=np.int64)
            return np.linspace(0, t - 1, n, dtype=np.int64)
        section_counts = [int(x) for x in section_counts.split(",") if x.strip()]
    if not section_counts:
        return np.arange(num_timesteps, dtype=np.int64)
    size_per, extra = divmod(num_timesteps, len(section_counts))
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"Section has {size} steps but requested {section_count}; cannot divide.")
        frac_stride = 1.0 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return np.unique(np.array(all_steps, dtype=np.int64))


_EPS = 1e-12


def _x0_space_weight(lw: str, snr: torch.Tensor, gamma: float, sigma_data: float) -> torch.Tensor | None:
    if lw == "min_snr":
        return snr.clamp(max=gamma) if gamma > 0 else None
    if lw == "min_snr_soft":
        return snr * gamma / (snr + gamma) if gamma > 0 else None
    if lw == "edm":
        sigma2 = 1.0 / snr.clamp(min=_EPS)
        sd2 = float(sigma_data) ** 2
        return (sigma2 + sd2) / (sigma2 * sd2)
    if lw == "v":
        return snr + 1.0
    if lw == "eps":
        return snr
    if lw == "unit":
        return None
    raise ValueError(f"Unknown loss_weighting {lw!r}; use min_snr | min_snr_soft | edm | v | eps | unit")


def get_timestep_loss_weight(
    loss_weighting: str,
    *,
    snr: torch.Tensor | None,
    alpha_cumprod: torch.Tensor,
    min_snr_gamma: float,
    loss_weighting_sigma_data: float,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    """
    Per-batch-element weight ``(B,)`` multiplying the per-sample MSE of the model's prediction.

    ``snr`` is ``abar / (1 - abar)`` at the sampled timesteps (computed from ``alpha_cumprod`` when ``None``).
    ``prediction_type``: ``epsilon`` | ``v`` | ``x0`` (the space the per-sample MSE was taken in).
    """
    if snr is None:
        snr = alpha_cumprod / (1.0 - alpha_cumprod).clamp(min=_EPS)
    snr = snr.to(torch.float32)
    w = _x0_space_weight(str(loss_weighting).lower().strip(), snr, float(min_snr_gamma), loss_weighting_sigma_data)
    if w is None:
        return torch.ones_like(snr)
    pt = str(prediction_type).lower()
    if pt == "v":
        return w / (snr + 1.0)
    if pt == "x0":
        return w
    # eps: w/snr is 0/0 only where snr == 0 (zero-terminal SNR); its limit there is 1 for the min-SNR family.
    return torch.where(snr > 0, w / snr.clamp(min=_EPS), torch.ones_like(snr))


def sample_training_timesteps(
    batch_size: int,
    num_timesteps: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    mode: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """
    ``(batch_size,)`` step indices in ``[0, num_timesteps - 1]``.

    ``mode``: ``uniform`` (classic DDPM ``randint``); ``logit_normal`` (``u ~ sigmoid(N(logit_mean, logit_std))``,
    ``t = round(u * (T-1))``); ``high_noise`` (``u ~ Beta(2, 1)``, more heavily corrupted latents);
    ``low_noise`` (``u ~ Beta(1, 2)``, more fine-detail tail).
    """
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    T, B = int(num_timesteps), int(batch_size)
    m = str(mode or "uniform").strip().lower()
    if m == "uniform":
        return torch.randint(0, T, (B,), device=device, dtype=dtype)
    if m in ("logit_normal", "logit-normal", "sd3"):
        u = torch.sigmoid(torch.randn(B, device=device, dtype=torch.float32) * float(logit_std) + float(logit_mean))
    elif m in ("high_noise", "high-noise", "beta_high", "low_noise", "low-noise", "beta_low", "detail"):
        a, b = (2.0, 1.0) if m.startswith(("high", "beta_high")) else (1.0, 2.0)
        u = torch.distributions.Beta(torch.tensor(a, device=device), torch.tensor(b, device=device)).sample((B,))
    else:
        raise ValueError(f"Unknown timestep_sample_mode {mode!r}; use uniform | logit_normal | high_noise | low_noise")
    return (u * float(T - 1)).round().to(dtype=dtype)


__all__ = [
    "alpha_cumprod_from_betas",
    "flux_resolution_shift",
    "inverse_time_shift",
    "sd3_resolution_shift",
    "snr_from_alpha_cumprod",
    "snr_from_betas",
    "time_shift",
    "vp_shifted_timestep",
    "get_beta_schedule",
    "linear_beta_schedule",
    "rescale_zero_terminal_snr",
    "cosine_beta_schedule",
    "sigmoid_beta_schedule",
    "squared_cosine_beta_schedule_v2",
]
