"""
**ZeResFDG** — Frequency-decoupled, Rescaled, and Zero-projected guidance (CADE 2.5).

Unifies three training-free sampler tweaks:
1. **Zero-projection** — remove CFG delta parallel to unconditional (CFGZero / APG-style).
2. **FDG** — separate low/high frequency guidance strengths.
3. **Energy rescaling** — match guided prediction norm to conditional branch.

Optional **spectral EMA** mode switches detail-seeking vs conservative FDG high-band scale.

Rychkovskiy et al., arXiv:2510.12954 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.superior.frequency_cfg import _split_freq, frequency_decoupled_cfg_delta


def zero_project_delta(
    delta: torch.Tensor,
    uncond_ref: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Remove component of ``delta`` parallel to ``uncond_ref`` (suppress uncond leakage).

    ``r = delta - proj_uncond(delta)`` — use ``delta`` built from cond-uncond residual.
    """
    denom = (uncond_ref * uncond_ref).sum(dim=tuple(range(1, uncond_ref.ndim)), keepdim=True) + eps
    coeff = (delta * uncond_ref).sum(dim=tuple(range(1, delta.ndim)), keepdim=True) / denom
    return delta - coeff * uncond_ref


def energy_rescale_guided(
    guided: torch.Tensor,
    cond_ref: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match per-sample RMS of ``guided`` to ``cond_ref`` (mitigate overexposure)."""
    dims = tuple(range(1, guided.ndim))
    g_norm = (guided * guided).sum(dim=dims, keepdim=True).sqrt() + eps
    c_norm = (cond_ref * cond_ref).sum(dim=dims, keepdim=True).sqrt() + eps
    return guided * (c_norm / g_norm)


@dataclass(slots=True)
class SpectralGuidanceEMA:
    """Track high-frequency energy ratio; hysteresis toggles detail-seeking FDG."""

    ema: float = 0.0
    alpha: float = 0.15
    detail_threshold: float = 0.22
    conservative_high_scale: float = 0.55
    detail_high_scale: float = 0.85
    detail_mode: bool = False

    def update(self, latent: torch.Tensor) -> float:
        """Update EMA from latent high-frequency energy fraction; return high FDG scale."""
        if latent.ndim != 4:
            return self.conservative_high_scale
        _, high = _split_freq(latent.float(), cutoff_frac=0.15)
        low_e = float(latent.float().pow(2).mean().item()) + 1e-8
        high_e = float(high.pow(2).mean().item())
        ratio = high_e / (low_e + high_e)
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * ratio
        if self.detail_mode:
            if self.ema < self.detail_threshold * 0.85:
                self.detail_mode = False
        elif self.ema > self.detail_threshold:
            self.detail_mode = True
        return self.detail_high_scale if self.detail_mode else self.conservative_high_scale


def apply_zeresfdg_cfg(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    *,
    cfg_scale: float,
    cfg_rescale: float = 0.7,
    fdg_cutoff_frac: float = 0.15,
    zero_project: bool = True,
    energy_rescale: bool = True,
    spectral_ema: SpectralGuidanceEMA | None = None,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    ZeResFDG-guided prediction from cond/uncond model outputs.

    ``strength`` blends with standard CFG (1 = full ZeResFDG).
    """
    if out_cond.shape != out_uncond.shape and out_cond.shape[1] > out_uncond.shape[1]:
        out_cond = out_cond[:, : out_uncond.shape[1]]

    delta = out_cond - out_uncond
    if float(cfg_rescale) > 0.0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / float(cfg_rescale), 1.0)

    if zero_project:
        delta = zero_project_delta(delta, out_uncond)

    high_scale = float(cfg_scale) * 0.65
    if spectral_ema is not None:
        high_scale = float(cfg_scale) * spectral_ema.update(out_cond)

    fdg_delta = frequency_decoupled_cfg_delta(
        delta,
        cfg_scale=float(cfg_scale),
        low_cfg_scale=float(cfg_scale),
        high_cfg_scale=high_scale,
        cutoff_frac=float(fdg_cutoff_frac),
    )
    guided = out_uncond + fdg_delta
    if energy_rescale:
        guided = energy_rescale_guided(guided, out_cond)

    s = float(max(0.0, min(1.0, strength)))
    if s >= 1.0:
        return guided
    std = out_uncond + float(cfg_scale) * (out_cond - out_uncond)
    return (1.0 - s) * std + s * guided


__all__ = [
    "SpectralGuidanceEMA",
    "apply_zeresfdg_cfg",
    "energy_rescale_guided",
    "zero_project_delta",
]
