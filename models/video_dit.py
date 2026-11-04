"""Video denoiser for ``pipelines/video``: a factorized spatiotemporal DiT (STDiT, as in CogVideoX / Open-Sora).

Instead of full 3D attention over every (frame, position) pair, O((T*N)^2), each block attends

  * **spatially** within each frame (composition, layout), then
  * **temporally** across frames at the same spatial location (motion, consistency),

which is O(T*N^2 + N*T^2). Timestep and optional text/context enter through adaLN-Zero. ``causal_temporal``
restricts each frame to past frames for autoregressive / streaming generation.

``PermanentVideoDiT`` adds slot memory for object permanence: K learnable slots read each frame's patch tokens,
write back into that frame and attend across time among themselves; ``permanence_consistency_loss``
penalizes slot features that jump between adjacent frames.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dit import sincos_pos_table, timestep_embedding


@functools.lru_cache(maxsize=16)
def _causal_mask(t: int, device: torch.device) -> Tensor:
    """``(t, t)`` bool mask, True above the diagonal: frame ``i`` cannot attend to later frames."""
    return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # x: (B, T, N, D); shift/scale: (B, 1, 1, D)
    return x * (1.0 + scale) + shift


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

        # Spatial: attend within each frame (over N tokens).
        h = _modulate(self.norm_s(x), sh_s, sc_s).reshape(b * t, n, d)
        h, _ = self.spatial_attn(h, h, h, need_weights=False)
        x = x + g_s * h.reshape(b, t, n, d)

        # Temporal: each spatial location attends across frames (over T).
        h = _modulate(self.norm_t(x), sh_t, sc_t).permute(0, 2, 1, 3).reshape(b * n, t, d)
        mask = _causal_mask(t, x.device) if causal_temporal else None
        h, _ = self.temporal_attn(h, h, h, need_weights=False, attn_mask=mask)
        x = x + g_t * h.reshape(b, n, t, d).permute(0, 2, 1, 3)
        return x + g_m * self.mlp(_modulate(self.norm_m(x), sh_m, sc_m))


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
        # Cross-attention to (B, L, context_dim) sequences; a pooled (B, context_dim) context only shifts ``cond``.
        self.context_cross = nn.MultiheadAttention(dim, num_heads, batch_first=True) if context_dim else None
        self.context_norm = nn.LayerNorm(dim) if context_dim else None
        self.blocks = nn.ModuleList([FactorizedSpatioTemporalBlock(dim, num_heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        self.proj_out = nn.Linear(dim, patch_size * patch_size * in_channels)

    def _embed(self, x: Tensor, t: Tensor, context: Tensor | None, first_frame_latent: Tensor | None):
        """Patch tokens ``(B, T, N, D)`` with positions, conditioning ``(B, D)``, context tokens (or None), grid."""
        b, c, frames, h, w = x.shape
        if first_frame_latent is not None:
            x = x.clone()
            x[:, :, 0] = first_frame_latent
        tokens = self.patch_embed(x.reshape(b * frames, c, h, w))
        _, d, hp, wp = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2).reshape(b, frames, hp * wp, d)
        # Fixed sin/cos positions: valid for any clip length and resolution, with nothing to train or lose on load.
        spatial = sincos_pos_table(d, hp, wp, 0, tokens.device, tokens.dtype)
        temporal = timestep_embedding(torch.arange(frames, device=x.device), d).to(tokens.dtype)
        tokens = tokens + spatial[:, None] + temporal[None, :, None]
        cond = self.t_embed(timestep_embedding(t, d))
        ctx_tokens = None
        if context is not None and self.context_proj is not None:
            pooled = context
            if context.ndim == 3:
                pooled = context.mean(dim=1)
                ctx_tokens = self.context_proj(context)
            cond = cond + self.context_proj(pooled)
        return tokens, cond, ctx_tokens, (hp, wp)

    def _attend_context(self, tokens: Tensor, ctx_tokens: Tensor | None) -> Tensor:
        """Every frame's tokens cross-attend to the ``(B, L, D)`` context sequence."""
        if ctx_tokens is None or self.context_cross is None:
            return tokens
        b, frames, n, d = tokens.shape
        flat = tokens.reshape(b * frames, n, d)
        ctx = ctx_tokens.unsqueeze(1).expand(b, frames, -1, d).reshape(b * frames, -1, d)
        attn_out, _ = self.context_cross(self.context_norm(flat), ctx, ctx, need_weights=False)
        return (flat + attn_out).reshape(b, frames, n, d)

    def _unpatchify(self, tokens: Tensor, cond: Tensor, hp: int, wp: int) -> Tensor:
        b, frames, _, d = tokens.shape
        p, c = self.patch_size, self.in_channels
        sh, sc = self.ada_out(cond).view(b, 2, 1, 1, d).unbind(dim=1)
        out = self.proj_out(_modulate(self.norm_out(tokens), sh, sc)).reshape(b, frames, hp, wp, p, p, c)
        return out.permute(0, 6, 1, 2, 4, 3, 5).reshape(b, c, frames, hp * p, wp * p)

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
        tokens, cond, ctx_tokens, (hp, wp) = self._embed(x, t, context, first_frame_latent)
        for block in self.blocks:
            tokens = self._attend_context(block(tokens, cond, causal_temporal=causal_temporal), ctx_tokens)
        return self._unpatchify(tokens, cond, hp, wp)


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


class SlotMemoryBank(nn.Module):
    """Discrete entity memory carried frame by frame through the clip."""

    def __init__(self, dim: int, num_slots: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.read = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.write = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """``tokens`` ``(B, T, N, D)`` -> updated tokens and temporally attended slot states ``(B, T, K, D)``."""
        b, t, _, d = tokens.shape
        slots = self.slots[None].expand(b, -1, -1)  # (B, K, D)
        slot_traj: list[Tensor] = []
        out_frames: list[Tensor] = []
        for ti in range(t):
            frame = tokens[:, ti]
            # Read: slots query the frame's patches; write: patches query the updated slots.
            read_out, _ = self.read(self.norm_s(slots), self.norm_q(frame), frame, need_weights=False)
            slots = slots + read_out
            slot_traj.append(slots)
            w_out, _ = self.write(self.norm_w(frame), self.norm_s(slots), slots, need_weights=False)
            out_frames.append(frame + w_out)
        # Each slot attends over its own trajectory; the per-frame mean of the result is a light residual.
        st = torch.stack(slot_traj, dim=2).reshape(b * self.num_slots, t, d)
        st = st + self.temporal(st, st, st, need_weights=False)[0]
        stacked = st.reshape(b, self.num_slots, t, d).permute(0, 2, 1, 3)
        tokens_out = torch.stack(out_frames, dim=1) + 0.15 * stacked.mean(dim=2, keepdim=True)
        return tokens_out, stacked


class IdentityTokenBank(nn.Module):
    """Phantom / VACE-style identity pathway: reference tokens outside the denoised latent that every frame
    cross-attends to (zero-initialized output projection)."""

    def __init__(self, dim: int, num_tokens: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: Tensor, identity: Tensor | None = None) -> Tensor:
        """``tokens`` ``(B, T, N, D)``; ``identity``: external refs ``(B, L, D)`` / ``(B, D)``, else the learned bank."""
        b, t, n, d = tokens.shape
        if identity is None:
            id_tok = self.tokens[None].expand(b, -1, -1)
        else:
            id_tok = identity.unsqueeze(1) if identity.ndim == 2 else identity
        bt = b * t
        flat = tokens.reshape(bt, n, d)
        id_rep = id_tok.unsqueeze(1).expand(b, t, -1, d).reshape(bt, id_tok.shape[1], d)
        out, _ = self.cross(self.norm(flat), id_rep, id_rep, need_weights=False)
        return (flat + self.proj(out)).reshape(b, t, n, d)


class PermanentVideoDiT(VideoDiT):
    """VideoDiT with slot-memory permanence + identity token binding."""

    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        patch_size: int = 2,
        context_dim: int | None = None,
        num_slots: int = 8,
        permanence_every: int = 2,
        num_id_tokens: int = 4,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            context_dim=context_dim,
        )
        self.num_slots = num_slots
        self.permanence_every = max(1, int(permanence_every))
        self.slot_banks = nn.ModuleList(
            [
                SlotMemoryBank(dim, num_slots=num_slots, num_heads=max(1, num_heads // 2))
                for _ in range(math.ceil(depth / self.permanence_every))
            ]
        )
        self.id_bank = IdentityTokenBank(dim, num_tokens=num_id_tokens, num_heads=max(1, num_heads // 2))

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        *,
        context: Tensor | None = None,
        identity: Tensor | None = None,
        causal_temporal: bool = False,
        first_frame_latent: Tensor | None = None,
        return_slots: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        tokens, cond, ctx_tokens, (hp, wp) = self._embed(x, t, context, first_frame_latent)
        tokens = self.id_bank(tokens, identity=identity)
        slot_states = None
        bank_i = 0
        for bi, block in enumerate(self.blocks):
            tokens = self._attend_context(block(tokens, cond, causal_temporal=causal_temporal), ctx_tokens)
            if (bi + 1) % self.permanence_every == 0 and bank_i < len(self.slot_banks):
                tokens, slot_states = self.slot_banks[bank_i](tokens)
                bank_i += 1
        out = self._unpatchify(tokens, cond, hp, wp)
        if not return_slots:
            return out
        if slot_states is None:
            b, frames, _, d = tokens.shape
            slot_states = tokens.new_zeros(b, frames, self.num_slots, d)
        return out, slot_states


def permanence_consistency_loss(slot_states: Tensor) -> Tensor:
    """Mean ``1 - cos`` between each slot's features in adjacent frames; ``slot_states`` is ``(B, T, K, D)``."""
    if slot_states.shape[1] < 2:
        return slot_states.new_zeros(())
    return (1.0 - F.cosine_similarity(slot_states[:, 1:], slot_states[:, :-1], dim=-1, eps=1e-12)).mean()


def PermanentVideoDiT_S_2(**kwargs) -> PermanentVideoDiT:
    return PermanentVideoDiT(dim=256, depth=6, num_heads=4, patch_size=2, num_slots=8, **kwargs)


def PermanentVideoDiT_B_2(**kwargs) -> PermanentVideoDiT:
    return PermanentVideoDiT(dim=384, depth=12, num_heads=6, patch_size=2, num_slots=12, **kwargs)


PermanentVideoDiT_models = {
    "PermanentVideoDiT-S/2": PermanentVideoDiT_S_2,
    "PermanentVideoDiT-B/2": PermanentVideoDiT_B_2,
}


__all__ = [
    "SlotMemoryBank",
    "IdentityTokenBank",
    "PermanentVideoDiT",
    "permanence_consistency_loss",
    "PermanentVideoDiT_S_2",
    "PermanentVideoDiT_B_2",
    "PermanentVideoDiT_models",
]
