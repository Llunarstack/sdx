"""
Universal adapter **math** — one ΔW for every LoRA-family algorithm.

Every adapter format ultimately encodes a weight delta ``ΔW`` for a base layer;
the formats differ only in factorization. Reconstructing ΔW is the
architecture-independent step of the universal bridge: once a foreign
adapter's deltas exist as plain matrices, they can be re-factorized (SVD) at
any rank and re-projected to any target layer shape.

Algorithms: LoRA/LoCon (``up @ down``), LoHa (Hadamard of two low-rank
products), LoKr (Kronecker product, optionally factorized), full-diff.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "delta_from_loha",
    "delta_from_lokr",
    "delta_from_lora",
    "project_delta",
    "svd_factorize",
]


def _alpha_scale(alpha: float | None, rank: int) -> float:
    return float(alpha) / max(1, int(rank)) if alpha is not None else 1.0


def delta_from_lora(down: torch.Tensor, up: torch.Tensor, *, alpha: float | None = None) -> torch.Tensor:
    """ΔW = up @ down · (α/rank). Conv adapters are flattened to 2D."""
    d = down.reshape(down.shape[0], -1).float()
    u = up.reshape(up.shape[0], -1).float()
    return (u @ d) * _alpha_scale(alpha, d.shape[0])


def delta_from_loha(
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    *,
    alpha: float | None = None,
) -> torch.Tensor:
    """LoHa: ΔW = (w1_a @ w1_b) ⊙ (w2_a @ w2_b) · (α/rank)."""
    m1 = w1_a.reshape(w1_a.shape[0], -1).float() @ w1_b.reshape(w1_b.shape[0], -1).float()
    m2 = w2_a.reshape(w2_a.shape[0], -1).float() @ w2_b.reshape(w2_b.shape[0], -1).float()
    return m1 * m2 * _alpha_scale(alpha, w1_b.shape[0])


def delta_from_lokr(
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    w1_a: torch.Tensor | None = None,
    w1_b: torch.Tensor | None = None,
    w2_a: torch.Tensor | None = None,
    w2_b: torch.Tensor | None = None,
    alpha: float | None = None,
) -> torch.Tensor:
    """LoKr: ΔW = kron(W1, W2) · (α/rank); either factor may itself be low-rank."""
    rank = 0
    if w1 is None and w1_a is not None and w1_b is not None:
        w1 = w1_a.float() @ w1_b.float()
        rank = int(w1_b.shape[0])
    if w2 is None and w2_a is not None and w2_b is not None:
        w2 = w2_a.reshape(w2_a.shape[0], -1).float() @ w2_b.reshape(w2_b.shape[0], -1).float()
        rank = rank or int(w2_b.shape[0])
    if w1 is None or w2 is None:
        raise ValueError("LoKr needs w1 and w2 (directly or as a_b factor pairs)")
    delta = torch.kron(w1.float(), w2.reshape(w2.shape[0], -1).float())
    return delta * _alpha_scale(alpha, rank or max(w1.shape))


def svd_factorize(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Re-factorize ΔW (out, in) into ``(down (r, in), up (out, r))`` with
    ``up @ down ≈ ΔW`` — the format :class:`models.lora.MultiLoRALinear` eats.
    """
    d = delta.float()
    r = max(1, min(int(rank), min(d.shape)))
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    root = s[:r].clamp(min=0).sqrt()
    up = u[:, :r] * root.unsqueeze(0)
    down = root.unsqueeze(1) * vh[:r, :]
    return down, up


def project_delta(delta: torch.Tensor, out_shape: tuple[int, int], *, rank: int = 32) -> torch.Tensor:
    """
    Heuristically project ΔW onto a layer of different shape.

    Factorizes at ``rank``, linearly resamples each factor along the resized
    dimension, and rescales the input factor by ``in/in_new`` so the dot
    product against activations keeps its magnitude. This preserves the
    *direction* of the adaptation, not its exact effect — cross-architecture
    transfer is approximate by nature; treat strength as a user-tunable dial.
    """
    out_t, in_t = int(out_shape[0]), int(out_shape[1])
    d = delta.float()
    if tuple(d.shape) == (out_t, in_t):
        return d
    down, up = svd_factorize(d, rank)
    if down.shape[1] != in_t:
        scale = down.shape[1] / float(in_t)
        down = F.interpolate(down.unsqueeze(1), size=in_t, mode="linear", align_corners=False).squeeze(1)
        down = down * scale
    if up.shape[0] != out_t:
        up = F.interpolate(up.T.unsqueeze(1), size=out_t, mode="linear", align_corners=False).squeeze(1).T
    return up @ down
