"""SPECTRA — Spectral Phase-Entropy Controlled Token Routing (inference).

Idea: denoising difficulty is frequency-dependent. Early steps need coarse
structure (low-freq); late steps need microtexture (high-freq). SPECTRA
modulates CFG and optional HF residual boost by a spectral schedule keyed
to denoise progress — without a second network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["SpectraState", "spectra_cfg_scale", "spectra_hf_boost", "spectra_mix_latent"]


@dataclass
class SpectraState:
    """Progress in [0,1] noisy→clean."""

    structure_end: float = 0.35
    detail_start: float = 0.55
    cfg_structure: float = 1.15  # multiplier early
    cfg_detail: float = 0.92  # slightly lower late to avoid burn
    hf_boost_peak: float = 0.08


def spectra_cfg_scale(base_cfg: float, progress: float, state: SpectraState | None = None) -> float:
    st = state or SpectraState()
    p = float(max(0.0, min(1.0, progress)))
    if p < st.structure_end:
        m = st.cfg_structure
    elif p > st.detail_start:
        # Smooth ramp down
        t = (p - st.detail_start) / max(1e-6, 1.0 - st.detail_start)
        m = st.cfg_structure + (st.cfg_detail - st.cfg_structure) * t
    else:
        m = 1.0
    return float(base_cfg) * float(m)


def spectra_hf_boost(progress: float, state: SpectraState | None = None) -> float:
    """Peak mid-late for microtexture; near-zero early."""
    st = state or SpectraState()
    p = float(max(0.0, min(1.0, progress)))
    # Bell peaking at ~0.75
    peak = 0.75
    w = abs(p - peak)
    boost = st.hf_boost_peak * max(0.0, 1.0 - w / 0.35)
    if p < st.structure_end:
        boost *= 0.15
    return float(boost)


def spectra_mix_latent(x: torch.Tensor, progress: float, state: SpectraState | None = None) -> torch.Tensor:
    """
    Lightweight spectral mix: blend x with a high-pass version of itself late
    in denoising (detail phase). Pure tensor op; no model call.
    """
    boost = spectra_hf_boost(progress, state)
    if boost <= 1e-8 or not torch.is_tensor(x):
        return x
    # Simple high-pass via x - avg_pool
    if x.ndim != 4:
        return x
    blur = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    hp = x - blur
    return x + float(boost) * hp
