"""
Textual-inversion **embedding bridge** — load any TI format, retarget any width.

Handles the three formats in the wild (A1111 ``string_to_param`` .pt files,
safetensors ``emb_params``, SDXL dual ``clip_l``/``clip_g``) plus raw tensors.
Vectors trained for one encoder width are resampled to another with per-vector
norm preservation — approximate (encoder spaces differ), but it carries the
learned concept direction well enough to condition sdx's CLIP branch; exact
transfer would need a learned projector (see :class:`EmbeddingProjector`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.compat.asset_sniffer import load_state_dict

__all__ = [
    "EmbeddingAsset",
    "EmbeddingProjector",
    "encoder_hint_for_dim",
    "load_textual_inversion",
    "resize_vectors",
]

_DIM_HINTS: dict[int, str] = {
    768: "clip_l (SD1.5 / SDXL first encoder)",
    1024: "open_clip_h (SD2)",
    1280: "clip_g (SDXL second encoder)",
    2048: "sdxl concatenated clip_l+clip_g",
    4096: "t5_xxl (Flux / sdx)",
}


@dataclass(slots=True)
class EmbeddingAsset:
    """A loaded TI embedding: one (n_vectors, dim) tensor per encoder slot."""

    vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    source_format: str = "unknown"


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.ndim == 1 else t.reshape(-1, t.shape[-1])


def load_textual_inversion(path_or_state: str | Path | dict) -> EmbeddingAsset:
    """Load a textual-inversion embedding in any known layout."""
    state = load_state_dict(path_or_state)
    if "string_to_param" in state:
        params = state["string_to_param"]
        vecs = next(iter(params.values())) if isinstance(params, dict) else params
        return EmbeddingAsset({"clip": _as_2d(vecs).float()}, "a1111")
    if "emb_params" in state:
        return EmbeddingAsset({"clip": _as_2d(state["emb_params"]).float()}, "safetensors_ti")
    if "clip_l" in state or "clip_g" in state:
        out = {k: _as_2d(state[k]).float() for k in ("clip_l", "clip_g") if k in state}
        return EmbeddingAsset(out, "sdxl_dual")
    if len(state) == 1:
        t = next(iter(state.values()))
        if torch.is_tensor(t):
            return EmbeddingAsset({"clip": _as_2d(t).float()}, "raw_tensor")
    raise ValueError("Unrecognized textual-inversion layout")


def encoder_hint_for_dim(dim: int) -> str:
    """Human-readable guess at which encoder a vector width belongs to."""
    return _DIM_HINTS.get(int(dim), f"unknown ({dim}-d)")


def resize_vectors(vectors: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Resample (n, dim) vectors to (n, target_dim), preserving each vector's L2
    norm so downstream attention sees comparable magnitudes.
    """
    v = _as_2d(vectors).float()
    if v.shape[1] == int(target_dim):
        return v
    norms = v.norm(dim=1, keepdim=True)
    out = F.interpolate(v.unsqueeze(1), size=int(target_dim), mode="linear", align_corners=False).squeeze(1)
    out_norms = out.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return out * (norms / out_norms)


class EmbeddingProjector(nn.Module):
    """
    Learnable refinement on top of :func:`resize_vectors` — zero-init residual,
    so it starts as the plain resample and can be calibrated on a handful of
    (foreign embedding, sdx rendering) pairs when exact transfer matters.
    """

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.refine = nn.Linear(target_dim, target_dim)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        base = resize_vectors(vectors, self.target_dim).to(self.refine.weight.dtype)
        return base + self.refine(base)
