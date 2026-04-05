"""
Linear Compressed Attention for Diffusion Transformers (arxiv 2503.16726).

Standard self-attention is O(N^2) in sequence length. At 1024x1024 with
patch size 2, N ≈ 262k — quadratic attention is infeasible.

This module provides two O(N) alternatives that can replace or augment
standard attention in DiT blocks:

1. LinearCompressedAttention:
   Compresses keys/values to a fixed-size context via learned pooling,
   then attends over the compressed context. Complexity: O(N * C) where
   C << N is the compressed context size.

2. LocalWindowAttention:
   Swin-style local window attention with optional global register tokens.
   Complexity: O(N * W^2) where W is the window size.

Both modules are drop-in replacements for nn.MultiheadAttention in DiT blocks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_enhancements import RMSNorm


class LinearCompressedAttention(nn.Module):
    """
    Compressed key-value attention: O(N * C) instead of O(N^2).

    Keys and values are pooled to `context_size` tokens via a learned
    linear projection before attention. Queries attend over the compressed
    context, preserving global receptive field at linear cost.

    Args:
        hidden_size: Token dimension.
        num_heads: Number of attention heads.
        context_size: Number of compressed KV tokens (C << N).
        qk_norm: Apply RMSNorm to Q/K per head.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int = 64,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.context_size = int(context_size)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learned compression: N tokens -> context_size tokens
        self.kv_compress = nn.Linear(hidden_size, context_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, D)
        Returns: (B, N, D)
        """
        B, N, D = x.shape
        C = self.context_size

        # Compress KV: (B, N, D) -> (B, C, D) via learned weighted sum
        # kv_compress: (D, C) applied as (B, N, D) @ (D, C) -> (B, N, C) then transpose
        compress_weights = self.kv_compress(x)  # (B, N, C)
        compress_weights = F.softmax(compress_weights, dim=1)  # (B, N, C) — soft pool over N
        x_compressed = torch.bmm(compress_weights.transpose(1, 2), x)  # (B, C, D)

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_compressed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_compressed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class LocalWindowAttention(nn.Module):
    """
    Local window self-attention (Swin-style) with optional global register tokens.

    Complexity: O(N * W^2) where W = window_size.
    Global registers attend to all windows, providing long-range context
    without the full O(N^2) cost.

    Args:
        hidden_size: Token dimension.
        num_heads: Number of attention heads.
        window_size: Spatial window size (tokens per side).
        num_global_tokens: Number of global tokens that attend everywhere (0 = disabled).
        qk_norm: Apply RMSNorm to Q/K per head.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 8,
        num_global_tokens: int = 4,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = int(window_size)
        self.num_global = int(num_global_tokens)
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

        if self.num_global > 0:
            self.global_tokens = nn.Parameter(torch.randn(1, self.num_global, hidden_size) * 0.02)

        nn.init.zeros_(self.out_proj.weight)

    def _window_partition(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, N, D) -> (B*nW, W*W, D) where nW = (h/ws)*(w/ws)."""
        B, N, D = x.shape
        ws = self.window_size
        x = x.reshape(B, h, w, D)
        # Pad to multiple of window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.reshape(B, hp // ws, ws, wp // ws, ws, D)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, D)
        return x, hp, wp

    def _window_unpartition(
        self, x: torch.Tensor, B: int, h: int, w: int, hp: int, wp: int
    ) -> torch.Tensor:
        """(B*nW, W*W, D) -> (B, h*w, D)."""
        ws = self.window_size
        D = x.shape[-1]
        x = x.reshape(B, hp // ws, wp // ws, ws, ws, D)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, hp, wp, D)
        return x[:, :h, :w, :].reshape(B, h * w, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) where N = h*w (square assumed).
        Returns: (B, N, D)
        """
        B, N, D = x.shape
        h = w = int(math.isqrt(N))
        assert h * w == N, "LocalWindowAttention requires square token grid"

        # Prepend global tokens
        if self.num_global > 0:
            g = self.global_tokens.expand(B, -1, -1)
            x_full = torch.cat([g, x], dim=1)  # (B, G+N, D)
        else:
            x_full = x

        # Partition into windows (local tokens only)
        x_local, hp, wp = self._window_partition(x, h, w)  # (B*nW, W^2, D)

        # QKV for local windows
        qkv_local = self.qkv(x_local)
        q_l, k_l, v_l = qkv_local.chunk(3, dim=-1)
        q_l = q_l.reshape(-1, self.window_size ** 2, self.num_heads, self.head_dim).transpose(1, 2)
        k_l = k_l.reshape(-1, self.window_size ** 2, self.num_heads, self.head_dim).transpose(1, 2)
        v_l = v_l.reshape(-1, self.window_size ** 2, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q_l = self.q_norm(q_l)
            k_l = self.k_norm(k_l)

        out_local = F.scaled_dot_product_attention(q_l, k_l, v_l, scale=self.scale)
        out_local = out_local.transpose(1, 2).reshape(-1, self.window_size ** 2, D)
        out_local = self._window_unpartition(out_local, B, h, w, hp, wp)  # (B, N, D)

        # Global tokens attend to all tokens
        if self.num_global > 0:
            qkv_g = self.qkv(x_full)
            q_g, k_g, v_g = qkv_g[:, :self.num_global].chunk(3, dim=-1)
            # Global queries attend to all keys/values
            _, k_all, v_all = qkv_g.chunk(3, dim=-1)
            q_g = q_g.reshape(B, self.num_global, self.num_heads, self.head_dim).transpose(1, 2)
            k_all = k_all.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_all = v_all.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            out_g = F.scaled_dot_product_attention(q_g, k_all, v_all, scale=self.scale)
            out_g = out_g.transpose(1, 2).reshape(B, self.num_global, D)
            # Distribute global context back to local tokens via addition
            out_local = out_local + out_g.mean(dim=1, keepdim=True)

        return self.out_proj(out_local)


__all__ = ["LinearCompressedAttention", "LocalWindowAttention"]
