"""Sub-quadratic attention for Diffusion Transformers.

``LinearCompressedAttention``
    Softly pools keys/values into ``context_size`` learned slots, then attends over them: O(N * C), C << N.

``LocalWindowAttention``
    Swin-style window attention plus optional global tokens that see the whole sequence: O(N * W^2).

Both map ``(B, N, D) -> (B, N, D)`` and are the ``use_linear_attn`` / ``use_window_attn`` self-attention
options of ``SuperiorViTBlock``.

``hybrid_attention_fraction`` blends softmax attention with kernel ``linear_attention`` (the SLA idea,
arXiv:2509.24006, without its trained block selection); ``--linear-attn-fraction`` routes every
``memory_efficient_attention`` call through it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Kernel linear attention over ``(B, H, N, D)``: ``phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1)``, O(N).

    ``phi = elu + 1`` (Katharopoulos et al. 2020) keeps every similarity positive, so the normaliser
    cannot cross zero the way raw ``Q K^T`` does.
    """
    q, k = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0
    kv = torch.einsum("bhnd,bhne->bhde", k, v.float())
    num = torch.einsum("bhnd,bhde->bhne", q, kv)
    den = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).unsqueeze(-1) + eps
    return (num / den).to(dtype=v.dtype)


def hybrid_attention_fraction(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    linear_frac: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """``(1 - f) * softmax attention + f * linear attention`` on ``(B, H, N, D)``; ``f`` is clamped to [0, 1]."""
    f = max(0.0, min(1.0, float(linear_frac)))
    if f >= 1.0:
        return linear_attention(q, k, v)
    dense = F.scaled_dot_product_attention(q, k, v, scale=scale)
    return dense if f <= 0.0 else (1.0 - f) * dense + f * linear_attention(q, k, v)


class LinearCompressedAttention(nn.Module):
    """Compressed key-value attention, O(N * C): every query attends over ``context_size`` soft-pooled slots."""

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
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.kv_compress = nn.Linear(hidden_size, context_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        # Not zero: this projection is not inside a gated residual, so zero-init would kill output and grads.
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        C = self.context_size

        # Soft-pool the N tokens into C learned slots: softmax over N of (B, N, C) scores.
        compress_weights = F.softmax(self.kv_compress(x), dim=1)
        x_compressed = torch.bmm(compress_weights.transpose(1, 2), x)

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_compressed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_compressed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


class LocalWindowAttention(nn.Module):
    """Swin-style self-attention inside ``window_size``-square windows of a square token grid.

    ``num_global_tokens`` learned tokens attend over the whole sequence; their mean is added to every patch
    output for long-range context. Grids that do not divide into windows are zero-padded, and the padding is
    masked out of the attention.
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
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

        if self.num_global > 0:
            self.global_tokens = nn.Parameter(torch.randn(1, self.num_global, hidden_size) * 0.02)

        # Zero-init is safe here: the caller adds this module as a residual (x + attention(x)).
        nn.init.zeros_(self.out_proj.weight)

    def _window_partition(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        """(B, h*w, C) -> (B*nW, W^2, C) zero-padded to whole windows, plus the padded h, w."""
        B, _, C = x.shape
        ws = self.window_size
        pad_h, pad_w = (ws - h % ws) % ws, (ws - w % ws) % ws
        x = F.pad(x.reshape(B, h, w, C), (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.reshape(B, hp // ws, ws, wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, ws * ws, C), hp, wp

    def _window_unpartition(self, x: torch.Tensor, B: int, h: int, w: int, hp: int, wp: int) -> torch.Tensor:
        """(B*nW, W^2, C) -> (B, h*w, C), dropping the padding."""
        ws = self.window_size
        C = x.shape[-1]
        x = x.reshape(B, hp // ws, wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, hp, wp, C)
        return x[:, :h, :w, :].reshape(B, h * w, C)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, N, D)`` with ``N`` a perfect square."""
        B, N, D = x.shape
        h = w = math.isqrt(N)
        if h * w != N:
            raise ValueError(f"LocalWindowAttention requires a square token grid; got N={N} (sqrt ≈ {N**0.5:.2f})")
        G = self.num_global

        x_full = torch.cat([self.global_tokens.expand(B, -1, -1), x], dim=1) if G > 0 else x
        qkv = self.qkv(x_full)
        windows, hp, wp = self._window_partition(qkv[:, G:], h, w)
        q_l, k_l, v_l = (self._heads(t) for t in windows.chunk(3, dim=-1))
        if self.q_norm is not None:
            q_l = self.q_norm(q_l)
            k_l = self.k_norm(k_l)

        key_mask = None
        if (hp, wp) != (h, w):
            valid, _, _ = self._window_partition(x.new_ones(B, N, 1), h, w)
            key_mask = valid.squeeze(-1).bool()[:, None, None, :]  # (B*nW, 1, 1, W^2)
        out_local = F.scaled_dot_product_attention(q_l, k_l, v_l, attn_mask=key_mask, scale=self.scale)
        out_local = out_local.transpose(1, 2).reshape(windows.shape[0], windows.shape[1], D)
        out_local = self._window_unpartition(out_local, B, h, w, hp, wp)

        if G > 0:
            q_full, k_full, v_full = qkv.chunk(3, dim=-1)
            q_g, k_all, v_all = self._heads(q_full[:, :G]), self._heads(k_full), self._heads(v_full)
            if self.q_norm is not None:
                q_g = self.q_norm(q_g)
                k_all = self.k_norm(k_all)
            out_g = F.scaled_dot_product_attention(q_g, k_all, v_all, scale=self.scale)
            out_local = out_local + out_g.transpose(1, 2).reshape(B, G, D).mean(dim=1, keepdim=True)

        return self.out_proj(out_local)


__all__ = ["LinearCompressedAttention", "LocalWindowAttention", "hybrid_attention_fraction", "linear_attention"]
