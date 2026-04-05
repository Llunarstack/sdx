"""
Spectral Flow Prediction (SFP) — prototype **frequency-weighted** diffusion loss on VP-DDPM targets.

This is **not** continuous flow matching: the forward process is unchanged. We only replace
spatial MSE with a weighted sum of |FFT(pred - target)|² in the latent, where radial weights
shift from low frequencies at **high noise** (large t) toward high frequencies at **low t**.

See docs/MODERN_DIFFUSION.md (SFP subsection) for motivation and limits.
"""

from __future__ import annotations

import torch


def radial_normalized_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Distance from spectrum center after fftshift, normalized to ~[0, 1]. Shape (H, W)."""
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(((gy - cy) / max(h, 1)) ** 2 + ((gx - cx) / max(w, 1)) ** 2)
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
    """
    W(ω, t): (B, 1, H, W) non-negative weights summing to 1 over spatial freq axes per batch row.

    VP index convention: t=0 almost clean, t=num_timesteps-1 almost pure noise.
    Large t → emphasize low spatial frequencies; small t → emphasize high frequencies.
    """
    r = radial_normalized_grid(h, w, device, dtype)
    ls = float(low_sigma) + eps
    hs = float(high_sigma) + eps
    w_low = torch.exp(-((r / ls) ** 2))
    w_high = torch.exp(-(((1.0 - r) / hs) ** 2))
    w_low = w_low / (w_low.sum() + eps)
    w_high = w_high / (w_high.sum() + eps)

    tau = (t.float() / max(num_timesteps - 1, 1)).clamp(0.0, 1.0)
    if tau_power != 1.0:
        tau = tau.pow(float(tau_power))
    tau_b = tau.view(-1, 1, 1, 1)
    wb = tau_b * w_low.view(1, 1, h, w) + (1.0 - tau_b) * w_high.view(1, 1, h, w)
    wb = wb / (wb.sum(dim=(-2, -1), keepdim=True) + eps)
    return wb


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
    """
    Per-sample scalar loss (B,) matching the scale of mean spatial MSE when weights are uniform.

    pred, target: (B, C, H, W). t: (B,) integer diffusion indices.
    """
    err = (pred - target).to(torch.float32)
    b, c, h, w = err.shape
    f = torch.fft.fftshift(torch.fft.fft2(err, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    p = f.real.pow(2) + f.imag.pow(2)
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
    weighted = (p * wb).mean(dim=1)
    return weighted.mean(dim=(-2, -1))
