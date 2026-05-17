"""
Next-generation ViT/DiT utility blocks.

Lightweight, opt-in modules that improve training stability and token routing
for deep vision transformers.  All residual-path modules are designed so that
a freshly initialised model behaves as (or close to) an identity mapping.

Modules
-------
LayerScale
    Per-channel residual scaling (CaiT-style).  Initialise ``init_value``
    near zero (e.g. ``1e-5``) for deep-network stability, or to ``1.0`` to
    start as a no-op.  The ``gamma`` parameter is always created so it remains
    learnable even when initialised to zero.

apply_topk_token_keep
    Soft top-k token gating that preserves tensor shape (no token dropping),
    making it safe to drop into existing DiT blocks without architecture
    changes.

References
----------
- CaiT (Touvron et al., 2021): https://arxiv.org/abs/2103.17239
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """
    Per-channel residual scaling (CaiT-style).

    Multiplies the residual branch output by a learnable per-channel scalar
    ``gamma``, initialised to ``init_value``.  Starting near zero (e.g.
    ``init_value=1e-5``) improves training stability for very deep networks
    (Touvron et al., 2021).

    Args:
        dim:        Feature dimension (number of channels).
        init_value: Initial value for ``gamma``.  Must be >= 0.
                    Set to ``1.0`` to start as a no-op.
                    Set to a small positive value (e.g. ``1e-5``) for deep
                    network stability.
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        if float(init_value) < 0.0:
            raise ValueError(f"LayerScale init_value must be >= 0, got {init_value!r}")
        self.gamma = nn.Parameter(
            torch.full((int(dim),), float(init_value))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gamma
        if g.dtype != x.dtype or g.device != x.device:
            g = g.to(dtype=x.dtype, device=x.device)
        return x * g


def apply_topk_token_keep(
    token_gate: torch.Tensor,
    score: torch.Tensor,
    *,
    keep_ratio: float,
    num_patch_tokens: int = 0,
    min_keep_value: float = 0.0,
) -> torch.Tensor:
    """
    Keep top-k patch tokens by `score` and softly suppress the rest.
    This preserves tensor shape (no token dropping), so it is low-risk in existing DiT blocks.
    """
    kr = float(keep_ratio)
    if kr >= 1.0:
        return token_gate
    if token_gate.ndim != 3 or score.ndim != 3:
        return token_gate
    b, n, _ = token_gate.shape
    p = int(num_patch_tokens) if int(num_patch_tokens) > 0 else n
    p = max(1, min(p, n))
    if p <= 0:
        return token_gate
    k = max(1, min(p, int(round(kr * p))))
    patch_score = score[:, :p, 0]
    idx = torch.topk(patch_score, k=k, dim=1, largest=True).indices  # (B, k)
    hard = torch.zeros((b, p), device=score.device, dtype=token_gate.dtype)
    hard.scatter_(1, idx, 1.0)
    hard = hard.unsqueeze(-1)  # (B, p, 1)
    out_patch = token_gate[:, :p, :] * hard + float(min_keep_value) * (1.0 - hard)
    if n == p:
        return out_patch
    tail = token_gate[:, p:, :]
    return torch.cat([out_patch, tail], dim=1)


__all__ = ["LayerScale", "apply_topk_token_keep"]
