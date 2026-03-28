"""
Next-gen ViT/DiT utility blocks.

These modules are lightweight and opt-in so old checkpoints remain compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Per-channel residual scaling (CaiT-style), useful for deep transformer stability."""

    def __init__(self, dim: int, init_value: float = 0.0):
        super().__init__()
        self.enabled = float(init_value) > 0.0
        if self.enabled:
            self.gamma = nn.Parameter(torch.full((int(dim),), float(init_value)))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma is None:
            return x
        return x * self.gamma.view(1, 1, -1).to(dtype=x.dtype, device=x.device)


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
    hard = hard.unsqueeze(-1)  # (B,p,1)
    out_patch = token_gate[:, :p, :] * hard + float(min_keep_value) * (1.0 - hard)
    if n == p:
        return out_patch
    tail = token_gate[:, p:, :]
    return torch.cat([out_patch, tail], dim=1)


__all__ = ["LayerScale", "apply_topk_token_keep"]
