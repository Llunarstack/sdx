"""Training-free latent filters applied to the clean latent before decode.

- ``qsilk_micrograin_stabilize`` (QSilk micrograin, CADE 2.5 appendix): quantile clamp plus gated
  high-frequency reinjection for natural micro-texture at high resolution (``--qsilk-micrograin``).
- ``spectral_latent_lowfreq_blend``: pull toward the low-frequency FFT band for global coherence
  (``--spectral-coherence-latent``).
- ``highfreq_layout_prior``: unsharp-style spatial high-pass emphasis (``--domain-prior-latent``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _fft_highpass(x: torch.Tensor, cutoff_frac: float) -> torch.Tensor:
    """High band of ``(B, C, H, W)`` above a radial FFT cutoff (fraction of the max frequency radius)."""
    cf = float(min(0.45, max(0.03, cutoff_frac)))
    h, w = x.shape[-2:]
    spec = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(h, device=x.device).abs().view(h, 1)
    fx = torch.fft.rfftfreq(w, device=x.device).view(1, -1)
    rad = torch.sqrt(fy * fy + fx * fx)
    keep = (rad / (rad.max() + 1e-8) > cf).to(spec.dtype)
    return torch.fft.irfft2(spec * keep, s=(h, w), dim=(-2, -1), norm="ortho")


def qsilk_micrograin_stabilize(
    x: torch.Tensor,
    *,
    quantile: float = 0.995,
    detail_amount: float = 0.12,
    cutoff_frac: float = 0.2,
) -> torch.Tensor:
    """Clamp extreme latent values, then add a gated high-frequency detail residual (``detail_amount=0``: clamp only)."""
    if x.ndim != 4 or float(detail_amount) <= 0.0 and float(quantile) <= 0.0:
        return x
    q = float(max(0.9, min(0.9999, quantile)))
    flat = x.float().reshape(x.shape[0], -1)
    thr = torch.quantile(flat.abs(), q, dim=1, keepdim=True).view(-1, 1, 1, 1)
    y = x.clamp(-thr.to(dtype=x.dtype), thr.to(dtype=x.dtype))
    if float(detail_amount) <= 0.0:
        return y

    high = _fft_highpass(y.float(), cutoff_frac).to(dtype=x.dtype)
    edge = high.abs().mean(dim=(1, 2, 3), keepdim=True)
    gate = (edge / (edge.mean() + 1e-6)).clamp(0.25, 2.0)
    return y + float(detail_amount) * gate * high


def spectral_latent_lowfreq_blend(x: torch.Tensor, strength: float = 0.0, cutoff_frac: float = 0.15) -> torch.Tensor:
    """``(1 - s) * x + s * lowpass(x)``; ``strength`` 0 is identity (try 0.05-0.2)."""
    if strength <= 0:
        return x
    s = float(min(1.0, strength))
    return x - s * _fft_highpass(x.float(), cutoff_frac).to(dtype=x.dtype)


def highfreq_layout_prior(x: torch.Tensor, strength: float = 0.0, kernel_size: int = 3) -> torch.Tensor:
    """``x + strength * (x - box_blur(x))``; ``strength`` 0 is identity (try 0.02-0.08)."""
    if strength <= 0:
        return x
    k = max(3, int(kernel_size) | 1)
    return x + float(strength) * (x - F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2))


__all__ = ["highfreq_layout_prior", "qsilk_micrograin_stabilize", "spectral_latent_lowfreq_blend"]
