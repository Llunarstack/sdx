"""
Video generation backbone: a factorized spatiotemporal DiT (STDiT).

The video pipeline in ``pipelines/video`` is a rich *orchestration* layer (shot
planning, continuity, motion beats) but has no neural denoiser of its own. This
is that denoiser — the core that actually turns noise into video latents.

Design follows the STDiT pattern used by CogVideoX / Open-Sora: instead of one
expensive full-3D attention over every (frame, position) pair (which is O((T·N)^2)),
each block does two cheaper attentions —

  * **spatial**: tokens attend within their own frame (composition, layout), and
  * **temporal**: tokens at the *same spatial location* attend across frames
    (motion, consistency),

which is O(T·N^2 + N·T^2) and captures the same dependencies far more cheaply.
Conditioning (timestep + optional text/context) is injected with adaLN-Zero, so
every block starts as an identity and learns its modulation — the DiT recipe that
makes these models train stably. A ``causal_temporal`` flag gives autoregressive
frame generation (a frame only sees the past) for streaming / long video.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # x: (B, T, N, D); shift/scale: (B, 1, 1, D)
    return x * (1.0 + scale) + shift


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Standard sinusoidal timestep embedding, shape ``(B, dim)``."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FactorizedSpatioTemporalBlock(nn.Module):
    """One STDiT block: spatial attention, temporal attention, then MLP.

    Operates on tokens shaped ``(B, T, N, D)`` (T frames, N spatial tokens each).
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_m = nn.LayerNorm(dim, elementwise_affine=False)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # adaLN-Zero: 3 sublayers x (shift, scale, gate) = 9 modulation vectors.
        self.ada = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: Tensor, cond: Tensor, *, causal_temporal: bool = False) -> Tensor:
        b, t, n, d = x.shape
        mods = self.ada(cond).view(b, 9, 1, 1, d)
        sh_s, sc_s, g_s, sh_t, sc_t, g_t, sh_m, sc_m, g_m = mods.unbind(dim=1)

        # --- spatial: attend within each frame (over N tokens) ---
        h = _modulate(self.norm_s(x), sh_s, sc_s).reshape(b * t, n, d)
        h, _ = self.spatial_attn(h, h, h, need_weights=False)
        x = x + g_s * h.reshape(b, t, n, d)

        # --- temporal: attend across frames at each spatial location (over T) ---
        h = _modulate(self.norm_t(x), sh_t, sc_t).permute(0, 2, 1, 3).reshape(b * n, t, d)
        mask = None
        if causal_temporal:
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h, _ = self.temporal_attn(h, h, h, need_weights=False, attn_mask=mask)
        x = x + g_t * h.reshape(b, n, t, d).permute(0, 2, 1, 3)

        # --- MLP ---
        h = _modulate(self.norm_m(x), sh_m, sc_m)
        x = x + g_m * self.mlp(h)
        return x


class VideoDiT(nn.Module):
    """Compact factorized-spatiotemporal DiT over video latents.

    Input/output are video latents ``(B, C, T, H, W)``; internally they are
    patch-embedded per frame, processed by STDiT blocks, and unpatchified.
    """

    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        patch_size: int = 2,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.t_embed = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.context_proj = nn.Linear(context_dim, dim) if context_dim else None
        # Optional cross-attn to token sequences (image-DiT style); None keeps pooled-only path.
        self.context_cross = nn.MultiheadAttention(dim, num_heads, batch_first=True) if context_dim else None
        self.context_norm = nn.LayerNorm(dim) if context_dim else None
        self.blocks = nn.ModuleList([FactorizedSpatioTemporalBlock(dim, num_heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        self.proj_out = nn.Linear(dim, patch_size * patch_size * in_channels)
        # Lazily-sized positional tables (created on first forward for given T, N).
        # Register as None so later Parameter assignment is not shadowed in __dict__.
        self.register_parameter("_spatial_pos", None)
        self.register_parameter("_temporal_pos", None)

    def _pos(self, t: int, n: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if self._spatial_pos is None or self._spatial_pos.shape[0] != n:
            self.register_parameter("_spatial_pos", nn.Parameter(torch.randn(n, self.dim, device=device) * 0.02))
        if self._temporal_pos is None or self._temporal_pos.shape[0] != t:
            self.register_parameter("_temporal_pos", nn.Parameter(torch.randn(t, self.dim, device=device) * 0.02))
        return self._spatial_pos, self._temporal_pos

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        *,
        context: Tensor | None = None,
        causal_temporal: bool = False,
        first_frame_latent: Tensor | None = None,
    ) -> Tensor:
        """
        ``x``: ``(B, C, T, H, W)`` noisy latents.
        ``context``: pooled ``(B, context_dim)`` or sequence ``(B, L, context_dim)``.
        ``first_frame_latent``: optional I2V lock — replaces frame 0 of ``x`` before embed.
        """
        b, c, frames, h, w = x.shape
        p = self.patch_size
        if first_frame_latent is not None:
            x = x.clone()
            x[:, :, 0] = first_frame_latent
        # patch-embed each frame
        tokens = self.patch_embed(x.reshape(b * frames, c, h, w))  # (B*T, D, H', W')
        _, d, hp, wp = tokens.shape
        n = hp * wp
        tokens = tokens.flatten(2).transpose(1, 2).reshape(b, frames, n, d)  # (B, T, N, D)

        spatial_pos, temporal_pos = self._pos(frames, n, x.device)
        tokens = tokens + spatial_pos[None, None] + temporal_pos[None, :, None]

        cond = self.t_embed(timestep_embedding(t, d))
        ctx_tokens = None
        if context is not None and self.context_proj is not None:
            if context.ndim == 3:
                pooled = context.mean(dim=1)
                ctx_tokens = self.context_proj(context)
            else:
                pooled = context
            cond = cond + self.context_proj(pooled)

        for block in self.blocks:
            tokens = block(tokens, cond, causal_temporal=causal_temporal)
            if ctx_tokens is not None and self.context_cross is not None and self.context_norm is not None:
                # Flatten spatial into batch for cross-attn: (B*T, N, D) attend to (B, L, D)
                bt = b * frames
                flat = tokens.reshape(bt, n, d)
                # Repeat context per frame
                ctx_rep = ctx_tokens.unsqueeze(1).expand(b, frames, -1, d).reshape(bt, ctx_tokens.shape[1], d)
                attn_out, _ = self.context_cross(self.context_norm(flat), ctx_rep, ctx_rep, need_weights=False)
                tokens = (flat + attn_out).reshape(b, frames, n, d)

        sh, sc = self.ada_out(cond).view(b, 2, 1, 1, d).unbind(dim=1)
        tokens = _modulate(self.norm_out(tokens), sh, sc)
        out = self.proj_out(tokens)  # (B, T, N, p*p*C)
        # unpatchify
        out = out.reshape(b, frames, hp, wp, p, p, c)
        out = out.permute(0, 6, 1, 2, 4, 3, 5).reshape(b, c, frames, hp * p, wp * p)
        return out


def VideoDiT_B_2(**kwargs) -> VideoDiT:
    """Base VideoDiT preset (patch 2)."""
    return VideoDiT(dim=384, depth=12, num_heads=6, patch_size=2, **kwargs)


def VideoDiT_S_2(**kwargs) -> VideoDiT:
    """Small VideoDiT for smoke tests / short clips."""
    return VideoDiT(dim=256, depth=6, num_heads=4, patch_size=2, **kwargs)


VideoDiT_models = {
    "VideoDiT-B/2": VideoDiT_B_2,
    "VideoDiT-S/2": VideoDiT_S_2,
}
