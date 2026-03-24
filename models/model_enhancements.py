"""
Reusable model building blocks (norms, stochastic depth, conditioning).

Use across DiT-adjacent modules, multimodal fusion, and small MLP heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square layer normalization (no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        dtype = x.dtype
        v = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return (x * self.weight).to(dtype)


class DropPath(nn.Module):
    """Stochastic depth per sample (drop entire residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div(keep)
        return x * mask


class TokenFiLM(nn.Module):
    """
    FiLM from a global conditioning vector: ``x * (1 + gamma) + beta``.

    cond: (B, cond_dim) -> gamma, beta each (B, 1, dim) broadcast over tokens.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Linear(int(cond_dim), 2 * self.dim)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D), cond: (B, C)
        """
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


class SE1x1(nn.Module):
    """Squeeze–excitation style gating on channel vector (B, D) -> (B, D)."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        h = max(8, int(dim) // int(reduction))
        self.fc1 = nn.Linear(int(dim), h)
        self.fc2 = nn.Linear(h, int(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) or (B, N, D) — pool spatial/token if 3D
        if x.dim() == 3:
            z = x.mean(dim=1)
        else:
            z = x
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(z))))
        if x.dim() == 3:
            w = w.unsqueeze(1)
        return x * w


def apply_gate_residual(x: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x + gate * branch`` with gate broadcastable to branch."""
    return x + gate * branch


__all__ = [
    "RMSNorm",
    "DropPath",
    "TokenFiLM",
    "SE1x1",
    "apply_gate_residual",
]
