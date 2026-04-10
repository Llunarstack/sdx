"""
**Cross-tower agreement** sketches (e.g. DiT latent vs ViT patch embedding, teacher vs student).

Use as building blocks for auxiliary losses — not wired to any encoder here.
"""

from __future__ import annotations

import torch


def flatten_cosine_similarity(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between two same-shape tensors after flattening (scalar)."""
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shape")
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    return (af * bf).sum() / (af.norm() * bf.norm()).clamp_min(eps)


def agreement_loss_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error in flat space."""
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shape")
    return (a.float() - b.float()).pow(2).mean()


def agreement_loss_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``1 - cos(a,b)`` in ``[0, 2]`` (0 = identical direction up to scale)."""
    return 1.0 - flatten_cosine_similarity(a, b)
