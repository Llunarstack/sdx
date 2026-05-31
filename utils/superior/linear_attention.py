"""
**SLA-inspired** linear attention fallback for marginal token blocks (inference).

Sparse-Linear Attention (arXiv:2509.24006): exact attention on critical blocks,
linear attention on marginal blocks. This module provides a cheap ``O(N)`` linear
attention path for experimentation without full SLA training.

Use via ``--linear-attn-fraction`` on supported attention wrappers (scaffold).
"""

from __future__ import annotations

import torch


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Linear attention: ``out = (Q @ (K^T V)) / (Q @ K^T 1)``.

    ``q,k,v``: (B, H, N, D).
    """
    k = k.float()
    v = v.float()
    q = q.float()
    kv = torch.einsum("bhnd,bhne->bhde", k, v)
    k_sum = k.sum(dim=2)
    num = torch.einsum("bhnd,bhde->bhne", q, kv)
    den = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1) + eps
    out = (num / den).to(dtype=v.dtype)
    return out


def hybrid_attention_fraction(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    linear_frac: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Blend standard softmax attention with linear attention.

    ``linear_frac=0`` → full softmax; ``linear_frac=1`` → full linear.
    """
    lf = float(max(0.0, min(1.0, linear_frac)))
    if lf <= 0.0:
        d = q.shape[-1]
        sc = scale if scale is not None else d**-0.5
        attn = torch.softmax(torch.einsum("bhnd,bhmd->bhnm", q * sc, k * sc), dim=-1)
        return torch.einsum("bhnm,bhme->bhne", attn, v)
    lin = linear_attention(q, k, v)
    if lf >= 1.0:
        return lin
    d = q.shape[-1]
    sc = scale if scale is not None else d**-0.5
    attn = torch.softmax(torch.einsum("bhnd,bhmd->bhnm", q * sc, k * sc), dim=-1)
    dense = torch.einsum("bhnm,bhme->bhne", attn, v)
    return (1.0 - lf) * dense + lf * lin


__all__ = ["hybrid_attention_fraction", "linear_attention"]
