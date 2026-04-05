"""
Temperature-Adjusted Cross-modal Attention (TACA).

Fixes two known MM-DiT issues (arxiv 2506.07986):
  1. Cross-modal attention suppression from token imbalance (many image tokens vs few text tokens).
  2. Lack of timestep-aware attention weighting — early steps need coarse structure,
     late steps need fine text alignment.

Drop-in replacement for the cross-attention in DiTTextBlock / MM-DiT blocks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_enhancements import RMSNorm


class TACA(nn.Module):
    """
    Temperature-Adjusted Cross-modal Attention.

    Args:
        hidden_size: Image token dimension.
        text_dim: Text token dimension.
        num_heads: Number of attention heads.
        max_timesteps: Maximum diffusion timestep (for sinusoidal embedding).
        base_temp: Baseline softmax temperature (default 1.0).
        temp_range: How much temperature can vary across timesteps (additive).
        qk_norm: Apply RMSNorm to Q and K per head (SD3.5-style stability).
    """

    def __init__(
        self,
        hidden_size: int,
        text_dim: int,
        num_heads: int,
        max_timesteps: int = 1000,
        base_temp: float = 1.0,
        temp_range: float = 0.5,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.base_temp = float(base_temp)
        self.temp_range = float(temp_range)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(text_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(text_dim, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

        # Timestep-conditioned temperature: small MLP from sinusoidal embedding -> scalar delta
        t_emb_dim = 64
        self.t_embed = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 2, 1),
            nn.Tanh(),  # output in (-1, 1), scaled by temp_range
        )
        self._t_emb_dim = t_emb_dim
        self._max_t = max_timesteps

        # Token imbalance correction: learnable per-head scale applied to cross-modal logits
        self.imbalance_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

    # ------------------------------------------------------------------
    def _sinusoidal_t(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) int or float -> (B, t_emb_dim)."""
        d = self._t_emb_dim
        half = d // 2
        device = t.device
        freqs = torch.exp(
            -math.log(self._max_t) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.cos(), args.sin()], dim=-1)  # (B, d)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Image tokens (B, N, D).
            text_emb: Text tokens (B, L, T).
            timestep: Diffusion timestep (B,) int/float. If None, uses base_temp.
        Returns:
            (B, N, D) updated image tokens.
        """
        B, N, D = x.shape
        _, L, _ = text_emb.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (B, H, N, D) and (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute temperature
        if timestep is not None:
            t_emb = self._sinusoidal_t(timestep)          # (B, t_emb_dim)
            delta = self.t_embed(t_emb) * self.temp_range  # (B, 1)
            temp = (self.base_temp + delta).clamp(min=0.1).unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)
        else:
            temp = torch.tensor(self.base_temp, device=x.device, dtype=x.dtype)

        # Scaled dot-product with temperature and imbalance correction
        # imbalance_scale compensates for N >> L token count disparity
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, L)
        attn = attn * self.imbalance_scale / temp
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


__all__ = ["TACA"]
