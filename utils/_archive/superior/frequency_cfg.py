"""
**Frequency-decoupled guidance (FDG)** on latent predictions.

Inspired by FDG (2025): apply different CFG strength to low vs high spatial frequency
components of the cond-uncond delta. Reduces oversaturation / halos at high CFG.

See ``docs/research/SUPERIOR_RESEARCH_2026.md``.
"""

from __future__ import annotations

import torch


def _split_freq(x: torch.Tensor, cutoff_frac: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``(B,C,H,W)`` into low/high via radial FFT mask."""
    if x.ndim != 4:
        return x, torch.zeros_like(x)
    cf = float(min(0.45, max(0.03, cutoff_frac)))
    b, c, h, w = x.shape
    X = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(h, device=x.device, dtype=torch.float32).abs().view(h, 1)
    fx = torch.fft.rfftfreq(w, device=x.device, dtype=torch.float32).abs().view(1, -1)
    rad = torch.sqrt(fy * fy + fx * fx)
    rad = rad / (rad.max() + 1e-8)
    mask = (rad <= cf).to(dtype=X.dtype)
    X_low = X * mask
    X_high = X * (1.0 - mask)
    low = torch.fft.irfft2(X_low, s=(h, w), dim=(-2, -1), norm="ortho").to(dtype=x.dtype)
    high = torch.fft.irfft2(X_high, s=(h, w), dim=(-2, -1), norm="ortho").to(dtype=x.dtype)
    return low, high


def frequency_decoupled_cfg_delta(
    delta: torch.Tensor,
    *,
    cfg_scale: float,
    low_cfg_scale: float | None = None,
    high_cfg_scale: float | None = None,
    cutoff_frac: float = 0.15,
) -> torch.Tensor:
    """
    Scale low/high frequency parts of CFG delta with separate guidance strengths.

    Defaults: low uses full ``cfg_scale``, high uses ``0.65 * cfg_scale`` (gentler detail push).
    """
    cs = float(cfg_scale)
    lc = float(low_cfg_scale if low_cfg_scale is not None else cs)
    hc = float(high_cfg_scale if high_cfg_scale is not None else cs * 0.65)
    d_low, d_high = _split_freq(delta, cutoff_frac=cutoff_frac)
    return lc * d_low + hc * d_high


def apply_fdg_cfg(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    *,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    fdg_strength: float = 1.0,
    cutoff_frac: float = 0.15,
) -> torch.Tensor:
    """CFG combine with optional FDG on the delta (``fdg_strength=0`` → standard CFG)."""
    if out_cond.shape != out_uncond.shape and out_cond.shape[1] > out_uncond.shape[1]:
        out_cond = out_cond[:, : out_uncond.shape[1]]
    delta = out_cond - out_uncond
    if cfg_rescale > 0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / cfg_rescale, 1.0)
    s = float(max(0.0, min(1.0, fdg_strength)))
    if s <= 0.0:
        return out_uncond + float(cfg_scale) * delta
    fdg_delta = frequency_decoupled_cfg_delta(delta, cfg_scale=float(cfg_scale), cutoff_frac=cutoff_frac)
    std_delta = float(cfg_scale) * delta
    mixed = (1.0 - s) * std_delta + s * fdg_delta
    return out_uncond + mixed


__all__ = ["apply_fdg_cfg", "frequency_decoupled_cfg_delta"]
