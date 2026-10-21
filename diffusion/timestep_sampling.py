"""
Training-time timesteps for VP diffusion: which ``t`` to train on and how to weight the loss there.

**Sampling.** SD3 and rectified-flow pipelines draw a continuous ``u ~ sigmoid(N(mu, sigma))`` so training
spends capacity on the hard mid-noise band; ``sample_training_timesteps`` draws the discrete analogue (and
Beta-biased high/low-noise variants) as integer indices — ``q_sample`` and the schedule are unchanged.

**Weighting.** ``get_timestep_loss_weight`` defines every scheme as a weight on the **x0 (denoiser) error** and
converts it to the space the model predicts in, using ``||eps_err||^2 = snr * ||x0_err||^2`` and ``||v_err||^2 = (snr + 1) * ||x0_err||^2``:
divide by ``snr`` for epsilon, by ``snr + 1`` for v, nothing for x0 (Hang et al. 2023, "Efficient Diffusion
Training via Min-SNR Weighting Strategy", Sec. 3.4).

- ``min_snr``: x0 weight ``min(snr, gamma)``, i.e. ``min(snr, gamma)/snr`` for eps, ``min(snr, gamma)/(snr+1)`` for v.
- ``min_snr_soft``: x0 weight ``snr*gamma/(snr+gamma)`` (eps weight ``gamma/(snr+gamma)``, no hard kink).
- ``edm``: Karras et al. 2022 ``lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma*sigma_data)^2`` with the VE noise level
  ``sigma = sqrt((1-abar)/abar)``.
- ``v``: x0 weight ``snr + 1`` (plain v-MSE, EDM with ``sigma_data = 1``).
- ``eps``: x0 weight ``snr`` (plain eps-MSE).
- ``unit``: plain MSE on whatever the model predicts.
"""

from __future__ import annotations

import torch

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
