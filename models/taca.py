"""Temperature-Adjusted Cross-modal Attention (TACA, arXiv:2506.07986).

Addresses two MM-DiT issues: text attention is suppressed by token imbalance (many image tokens, few text
tokens), and attention ignores the timestep although early steps need coarse structure and late steps fine
text alignment. Drop-in cross-attention for ``DiTBlockText`` and ``SuperiorViTBlock``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import memory_efficient_attention
from .dit import timestep_embedding
from .layout_attn_context import layout_cross_attn_bias


class TACA(nn.Module):
    """Cross-attention whose logits are scaled by a learned per-head imbalance factor and divided by a
    timestep-dependent temperature ``base_temp + temp_range * tanh(MLP(t))`` (clamped to >= 0.1)."""

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
        self.scale = self.head_dim**-0.5
        self.base_temp = float(base_temp)
        self.temp_range = float(temp_range)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(text_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(text_dim, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

        t_emb_dim = 64
        self.t_embed = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 2, 1),
            nn.Tanh(),
        )
        self._t_emb_dim = t_emb_dim
        self._max_t = max_timesteps
        self.imbalance_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        timestep: torch.Tensor | None = None,
        *,
        use_xformers: bool = True,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Image tokens ``(B, N, D)`` attend to text ``(B, L, text_dim)``; ``timestep=None`` uses ``base_temp``.

        ``return_weights`` also returns the ``(B, heads, N, L)`` attention probabilities.
        """
        B, N, D = x.shape
        L = text_emb.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if timestep is not None:
            t = timestep.expand(B) if timestep.numel() == 1 else timestep
            t_emb = timestep_embedding(t, self._t_emb_dim, max_period=self._max_t)
            delta = self.t_embed(t_emb.to(self.t_embed[0].weight.dtype)) * self.temp_range
            temp = (self.base_temp + delta).clamp(min=0.1).view(B, 1, 1, 1)
        else:
            temp = x.new_full((1, 1, 1, 1), self.base_temp)
        # Temperature and per-head imbalance scale the logits; folding sqrt into both Q and K
        # keeps the fused SDPA kernel. Layout here is (B, N, H, D).
        fac = (self.imbalance_scale.view(1, 1, self.num_heads, 1).clamp(min=1e-4) / temp).sqrt().to(q.dtype)
        q = q * fac
        k = k * fac

        bias = layout_cross_attn_bias(B, N, L, device=x.device, dtype=q.dtype)
        if return_weights:
            logits = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)) * self.scale
            if bias is not None:
                logits = logits + bias
            weights = logits.softmax(dim=-1)
            out = (weights @ v.transpose(1, 2)).transpose(1, 2).reshape(B, N, D)
            return self.out_proj(out), weights
        out = memory_efficient_attention(
            q, k, v, attn_mask=bias, scale=self.scale, use_xformers=use_xformers and bias is None
        )
        return self.out_proj(out.reshape(B, N, D))


__all__ = ["TACA"]
