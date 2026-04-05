"""
Research-facing **hooks and stubs** aligned with 2025–2026 industry themes (DiT, layout-first,
inference intervention, glyph text, flow matching, reference/kontext).

This module does **not** replace training objectives: full **flow matching / rectified flow**
needs a different loss and integrator (see ``docs/MODERN_DIFFUSION.md``). It provides:

- **Dual-stage layout plan** — numbers for coarse-then-fine latent sampling (pair with
  ``sample.py --hires-fix`` or a custom two-call pipeline).
- **Glyph / byte-level text** — ``typing.Protocol`` + null encoder for future ByT5-style sidecars.
- **Self-correction rewind** — typed storage + no-op scorer (CLIP/object hooks are optional later).
- **Temporal harmonizer** — placeholder for video / multi-frame conditioning.
- **Spectral lowfreq blend** — FFT low-pass mix on latents (inference global-coherence knob).
- **CLIP alignment score** — optional real score in ``score_latent_prompt_alignment`` when given a VAE + model id.
- **Periodic CLIP monitor** — ``sample.py`` + ``sample_loop`` can boost CFG when mid-loop cosine is low (expensive).

Implemented in-repo today: **volatile CFG** in ``GaussianDiffusion.sample_loop``; **reference
image** tokens in ``sample.py``; **SFP** spectral training loss; **logit-normal** timestep sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dual-stage / layout-first (GLM-Image–style *workflow*, not a full second model)
# ---------------------------------------------------------------------------


@dataclass
class DualStageLatentPlan:
    """Coarse layout latent grid vs target fine grid (both in VAE latent units, i.e. px/8)."""

    layout_latent_hw: Tuple[int, int]
    target_latent_hw: Tuple[int, int]
    layout_steps: int
    detail_steps: int


def plan_dual_stage_latents(
    target_image_size_px: int,
    *,
    layout_scale_div: int = 2,
    layout_steps: int = 20,
    detail_steps: int = 18,
) -> DualStageLatentPlan:
    """
    Build a two-stage plan: first denoise at ``target // layout_scale_div`` latent resolution,
    then upscale + refine at full target (use second ``sample_loop`` with noise, or ``--hires-fix``).

    ``layout_scale_div`` must be >= 2 and divide the latent side evenly.
    """
    if layout_scale_div < 2:
        raise ValueError("layout_scale_div must be >= 2")
    t = int(target_image_size_px)
    if t < 16 or t % 8 != 0:
        raise ValueError("target_image_size_px must be >= 16 and a multiple of 8")
    fine_lat = t // 8
    if fine_lat % layout_scale_div != 0:
        raise ValueError("latent side must be divisible by layout_scale_div")
    coarse_lat = fine_lat // layout_scale_div
    return DualStageLatentPlan(
        layout_latent_hw=(coarse_lat, coarse_lat),
        target_latent_hw=(fine_lat, fine_lat),
        layout_steps=int(layout_steps),
        detail_steps=int(detail_steps),
    )


def apply_size_embed_to_model_kwargs(
    model_kwargs_cond: Dict,
    model_kwargs_uncond: Dict,
    *,
    cfg,
    cond_emb: torch.Tensor,
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> Tuple[Dict, Dict]:
    """Copy kwargs and set ``size_embed`` for PixArt-style models (no-op if ``size_embed_dim`` is 0)."""
    mc = dict(model_kwargs_cond)
    mu = dict(model_kwargs_uncond)
    if int(getattr(cfg, "size_embed_dim", 0) or 0) <= 0:
        return mc, mu
    bsz = cond_emb.shape[0]
    sz = torch.tensor([[float(latent_h), float(latent_w)]], device=device, dtype=torch.float32).expand(bsz, -1)
    mc["size_embed"] = sz
    mu["size_embed"] = sz
    return mc, mu


def spectral_latent_lowfreq_blend(
    x: torch.Tensor, strength: float = 0.0, cutoff_frac: float = 0.15
) -> torch.Tensor:
    """
    Blend latent with its **low-frequency** FFT reconstruction (global coherence heuristic).

    ``strength`` 0 = identity. Try 0.05–0.2. ``cutoff_frac`` is normalized radial freq in [0, ~0.45].
    Complements ``highfreq_layout_prior`` (spatial high-pass); this is **spectral** low-pass mix.
    """
    if strength <= 0:
        return x
    s = float(min(1.0, max(0.0, strength)))
    cf = float(min(0.45, max(0.03, cutoff_frac)))
    _b, _c, h, w = x.shape
    X = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(h, device=x.device, dtype=torch.float32).abs().view(h, 1)
    fx = torch.fft.rfftfreq(w, device=x.device, dtype=torch.float32).abs().view(1, -1)
    rad = torch.sqrt(fy * fy + fx * fx)
    rad = rad / (rad.max() + 1e-8)
    mask = (rad <= cf).to(dtype=X.dtype)
    X_low = X * mask
    low = torch.fft.irfft2(X_low, s=(h, w), dim=(-2, -1), norm="ortho").to(dtype=x.dtype)
    return (1.0 - s) * x + s * low


def highfreq_layout_prior(x: torch.Tensor, strength: float = 0.0, kernel_size: int = 3) -> torch.Tensor:
    """
    Cheap latent high-frequency emphasis (domain / structure prior). strength 0 = off; try 0.02–0.08.
    """
    if strength <= 0:
        return x
    k = int(kernel_size) | 1
    k = max(3, k)
    blur = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x + float(strength) * (x - blur)


# ---------------------------------------------------------------------------
# Glyph / byte-level text conditioning (ByT5-class encoders) — integration surface only
# ---------------------------------------------------------------------------


class GlyphEncoderProtocol(Protocol):
    """Future: byte- or glyph-level encoder (shapes) for text-in-image fidelity."""

    embed_dim: int

    def encode_utf8(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Return (B, L, D) tensor to fuse with T5/CLIP (model-specific wiring not done here)."""
        ...


class NullGlyphEncoder:
    """No-op encoder: returns zeros (keeps DiT API experiments from crashing)."""

    embed_dim: int = 64

    def encode_utf8(self, texts: List[str], device: torch.device) -> torch.Tensor:
        b = max(1, len(texts))
        return torch.zeros((b, 1, self.embed_dim), device=device, dtype=torch.float32)


class GlyphToCondProjector(torch.nn.Module):
    """
    Zero-init linear: maps glyph tokens (B, Lg, Dg) -> text dim (B, Lg, Dt) for residual add to T5 states.
    Training this projector is a separate exercise; at init it is a no-op.
    """

    def __init__(self, glyph_dim: int, text_dim: int):
        super().__init__()
        self.glyph_dim = int(glyph_dim)
        self.text_dim = int(text_dim)
        self.proj = torch.nn.Linear(self.glyph_dim, self.text_dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, glyph_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(glyph_tokens)


# ---------------------------------------------------------------------------
# Self-correction / rewind (stub scorer — extend with CLIP or object detector)
# ---------------------------------------------------------------------------


@dataclass
class RewindState:
    latent: torch.Tensor
    step_index: int
    timestep_value: int


def score_latent_prompt_alignment(
    latent: torch.Tensor,
    prompt: str,
    *,
    device: Optional[torch.device] = None,
    clip_model_id: str = "",
    vae: Any = None,
    latent_scale: float = 1.0,
    ae_type: str = "kl",
    rae_bridge: Any = None,
) -> float:
    """
    Alignment score in [0, 1] (higher = better). Default 0.5 when CLIP is not configured.

    Pass ``clip_model_id`` (e.g. ``openai/clip-vit-base-patch32``) plus ``vae`` / scale / ``ae_type``
    to run real CLIP cosine on a decoded preview; maps cosine [-1, 1] → [0, 1].
    """
    if not str(clip_model_id).strip() or vae is None:
        return 0.5
    try:
        from utils.generation.clip_alignment import latent_x0_clip_cosine

        dev = device if device is not None else latent.device
        c = latent_x0_clip_cosine(
            latent,
            prompt=prompt,
            model_id=str(clip_model_id),
            device=dev,
            vae=vae,
            latent_scale=float(latent_scale),
            ae_type=str(ae_type),
            rae_bridge=rae_bridge,
            batch_index=0,
        )
        return float(max(0.0, min(1.0, (float(c) + 1.0) * 0.5)))
    except Exception:
        return 0.5


def should_rewind(
    score: float,
    *,
    threshold: float = 0.35,
) -> bool:
    """If score drops below ``threshold``, a rewind policy *could* restore ``RewindState``."""
    return score < threshold


# ---------------------------------------------------------------------------
# Flow matching note (training not implemented in GaussianDiffusion)
# ---------------------------------------------------------------------------

FLOW_MATCHING_TRAINING_NOTE = (
    "Full flow matching / rectified-flow training is not VP-DDPM-compatible; "
    "see docs/MODERN_DIFFUSION.md and diffusion/timestep_sampling.py for discrete-time analogues."
)


# ---------------------------------------------------------------------------
# Video / temporal harmonizer placeholder (NVIDIA DiffusionHarmonizer-class ideas)
# ---------------------------------------------------------------------------


class TemporalHarmonizerStub:
    """Reserve API for frame-to-frame conditioning; optional blend with previous latent."""

    def condition_latent(
        self,
        latent_t: torch.Tensor,
        latent_prev: Optional[torch.Tensor] = None,
        *,
        alpha_prev: float = 0.0,
    ) -> torch.Tensor:
        if latent_prev is None or alpha_prev <= 0:
            return latent_t
        a = float(min(1.0, max(0.0, alpha_prev)))
        return (1.0 - a) * latent_t + a * latent_prev.to(dtype=latent_t.dtype, device=latent_t.device)


__all__ = [
    "DualStageLatentPlan",
    "FLOW_MATCHING_TRAINING_NOTE",
    "GlyphEncoderProtocol",
    "GlyphToCondProjector",
    "NullGlyphEncoder",
    "RewindState",
    "TemporalHarmonizerStub",
    "apply_size_embed_to_model_kwargs",
    "highfreq_layout_prior",
    "plan_dual_stage_latents",
    "score_latent_prompt_alignment",
    "should_rewind",
    "spectral_latent_lowfreq_blend",
]
