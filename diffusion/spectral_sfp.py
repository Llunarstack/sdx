"""
Spectral Flow Prediction (SFP): a frequency-weighted replacement for the spatial MSE of the denoising loss.

The forward process and the prediction target are unchanged; the squared error is measured as a weighted sum of
``|FFT(pred - target)|^2`` whose radial weights move from low frequencies at high noise (large ``t``) to high
frequencies at low noise. Used by both the VP (``GaussianDiffusion``) and rectified-flow objectives.
"""

from __future__ import annotations

import torch


def radial_normalized_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``(H, W)`` distance from the fftshifted spectrum centre, normalized to ``[0, 1]``."""
    ys = (torch.arange(h, device=device, dtype=dtype) - (h - 1) * 0.5) / max(h, 1)
    xs = (torch.arange(w, device=device, dtype=dtype) - (w - 1) * 0.5) / max(w, 1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(gy**2 + gx**2)
    return r / (r.max() + 1e-8)


def time_frequency_weights(
    h: int,
    w: int,
    t: torch.Tensor,
    num_timesteps: int,
    *,
    low_sigma: float = 0.22,
    high_sigma: float = 0.22,
    tau_power: float = 1.0,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``(B, 1, H, W)`` non-negative weights summing to 1 over the frequency axes for each batch row.

    ``t = 0`` is almost clean and weights high frequencies; ``t = num_timesteps - 1`` is almost pure noise and
    weights low frequencies.
    """
    r = radial_normalized_grid(h, w, device, dtype)
    w_low = torch.exp(-((r / (float(low_sigma) + eps)) ** 2))
    w_high = torch.exp(-(((1.0 - r) / (float(high_sigma) + eps)) ** 2))
    w_low = (w_low / (w_low.sum() + eps)).view(1, 1, h, w)
    w_high = (w_high / (w_high.sum() + eps)).view(1, 1, h, w)

    tau = (t.float() / max(num_timesteps - 1, 1)).clamp(0.0, 1.0)
    if tau_power != 1.0:
        tau = tau.pow(float(tau_power))
    tau_b = tau.view(-1, 1, 1, 1)
    wb = tau_b * w_low + (1.0 - tau_b) * w_high
    return wb / (wb.sum(dim=(-2, -1), keepdim=True) + eps)


def spectral_sfp_per_sample_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    num_timesteps: int,
    *,
    low_sigma: float = 0.22,
    high_sigma: float = 0.22,
    tau_power: float = 1.0,
) -> torch.Tensor:
    """Per-sample loss ``(B,)``: ``sum_w W(w, t) |FFT(err)(w)|^2`` averaged over channels.

    ``pred`` / ``target`` are ``(B, C, H, W)``; ``t`` holds ``(B,)`` integer indices (large ``t`` = noisy). With the
    orthonormal FFT (Parseval) and weights summing to 1, uniform weights give exactly the mean spatial MSE.
    """
    err = (pred - target).to(torch.float32)
    _, _, h, w = err.shape
    f = torch.fft.fftshift(torch.fft.fft2(err, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    power = f.real.pow(2) + f.imag.pow(2)
    wb = time_frequency_weights(
        h,
        w,
        t,
        num_timesteps,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        tau_power=tau_power,
        device=err.device,
        dtype=err.dtype,
    )
    return (power * wb).mean(dim=1).sum(dim=(-2, -1))
