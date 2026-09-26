"""
PermanentVideoDiT — STDiT + **slot memory** for object permanence.

Public video APIs denoise a spacetime volume with no discrete entity state.
We add K learnable *slots* that:

1. Cross-attend to each frame's patch tokens (read what's there),
2. Attend across time among themselves (entity memory),
3. Write back into every frame (force persistence),

so disappearing props must fight an explicit memory bank — not just hope
temporal attention remembers them. Trains with an optional permanence loss
that penalizes slot-feature collapse across adjacent frames.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .video_dit import (
    VideoDiT,
    _modulate,
    timestep_embedding,
)

__all__ = [
    "SlotMemoryBank",
    "IdentityTokenBank",
    "PermanentVideoDiT",
    "permanence_consistency_loss",
    "identity_consistency_loss",
    "PermanentVideoDiT_S_2",
    "PermanentVideoDiT_B_2",
    "PermanentVideoDiT_models",
]


class SlotMemoryBank(nn.Module):
    """Discrete entity memory shared across the clip."""

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
        """
        ``tokens``: (B, T, N, D)
        Returns updated tokens and slot states (B, T, K, D).
        """
        b, t, n, d = tokens.shape
        slots = self.slots[None].expand(b, -1, -1)  # (B, K, D)
        slot_traj: list[Tensor] = []
        out_frames: list[Tensor] = []
        for ti in range(t):
            frame = tokens[:, ti]  # (B, N, D)
            # Read: slots query frame patches
            s_norm = self.norm_s(slots)
            read_out, _ = self.read(s_norm, self.norm_q(frame), frame, need_weights=False)
            slots = slots + read_out
            slot_traj.append(slots)
            # Write: patches query slots
            w_out, _ = self.write(self.norm_w(frame), self.norm_s(slots), slots, need_weights=False)
            out_frames.append(frame + w_out)
        # Temporal bind slots across time
        stacked = torch.stack(slot_traj, dim=1)  # (B, T, K, D)
        bt = b * self.num_slots
        st = stacked.permute(0, 2, 1, 3).reshape(bt, t, d)  # (B*K, T, D)
        st2, _ = self.temporal(st, st, st, need_weights=False)
        st = st + st2
        stacked = st.reshape(b, self.num_slots, t, d).permute(0, 2, 1, 3)
        # Re-write with temporally-smoothed slots (last pass blend)
        tokens_out = torch.stack(out_frames, dim=1)
        # Light residual from final slot mean
        slot_mean = stacked.mean(dim=2, keepdim=True)  # (B, T, 1, D)
        tokens_out = tokens_out + 0.15 * slot_mean
        return tokens_out, stacked


class IdentityTokenBank(nn.Module):
    """
    Phantom / VACE-inspired identity pathway: reference tokens live outside the
    denoise latent and inject additive hints every frame (Element Binding in DiT).
    """

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
        """
        ``tokens``: (B, T, N, D)
        ``identity``: optional (B, L, D) external refs; else learnable bank.
        """
        b, t, n, d = tokens.shape
        if identity is None:
            id_tok = self.tokens[None].expand(b, -1, -1)
        else:
            id_tok = identity
            if id_tok.ndim == 2:
                id_tok = id_tok.unsqueeze(1)
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
        b, c, frames, h, w = x.shape
        p = self.patch_size
        if first_frame_latent is not None:
            x = x.clone()
            x[:, :, 0] = first_frame_latent
        tokens = self.patch_embed(x.reshape(b * frames, c, h, w))
        _, d, hp, wp = tokens.shape
        n = hp * wp
        tokens = tokens.flatten(2).transpose(1, 2).reshape(b, frames, n, d)
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

        # Bind identity before blocks (Phantom-style parallel pathway)
        tokens = self.id_bank(tokens, identity=identity)

        slot_states = None
        bank_i = 0
        for bi, block in enumerate(self.blocks):
            tokens = block(tokens, cond, causal_temporal=causal_temporal)
            if ctx_tokens is not None and self.context_cross is not None and self.context_norm is not None:
                bt = b * frames
                flat = tokens.reshape(bt, n, d)
                ctx_rep = ctx_tokens.unsqueeze(1).expand(b, frames, -1, d).reshape(bt, ctx_tokens.shape[1], d)
                attn_out, _ = self.context_cross(self.context_norm(flat), ctx_rep, ctx_rep, need_weights=False)
                tokens = (flat + attn_out).reshape(b, frames, n, d)
            if (bi + 1) % self.permanence_every == 0 and bank_i < len(self.slot_banks):
                tokens, slot_states = self.slot_banks[bank_i](tokens)
                bank_i += 1

        sh, sc = self.ada_out(cond).view(b, 2, 1, 1, d).unbind(dim=1)
        tokens = _modulate(self.norm_out(tokens), sh, sc)
        out = self.proj_out(tokens)
        out = out.reshape(b, frames, hp, wp, p, p, c)
        out = out.permute(0, 6, 1, 2, 4, 3, 5).reshape(b, c, frames, hp * p, wp * p)
        if return_slots:
            if slot_states is None:
                slot_states = torch.zeros(b, frames, self.num_slots, d, device=x.device)
            return out, slot_states
        return out


def permanence_consistency_loss(slot_states: Tensor) -> Tensor:
    """
    Penalize abrupt slot feature jumps between adjacent frames.

    ``slot_states``: (B, T, K, D). Returns scalar.
    """
    if slot_states.shape[1] < 2:
        return slot_states.new_zeros(())
    a = F.normalize(slot_states[:, 1:], dim=-1)
    b = F.normalize(slot_states[:, :-1], dim=-1)
    # 1 - cosine similarity → want small
    cos = (a * b).sum(dim=-1)
    return (1.0 - cos).mean()


def identity_consistency_loss(frame_tokens: Tensor) -> Tensor:
    """
    Penalize subject-band token drift across adjacent frames.

    ``frame_tokens``: (B, T, N, D) — use mean-pooled tokens as identity proxy.
    """
    if frame_tokens.shape[1] < 2:
        return frame_tokens.new_zeros(())
    pooled = frame_tokens.mean(dim=2)  # (B, T, D)
    a = F.normalize(pooled[:, 1:], dim=-1)
    b = F.normalize(pooled[:, :-1], dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def PermanentVideoDiT_S_2(**kwargs) -> PermanentVideoDiT:
    return PermanentVideoDiT(dim=256, depth=6, num_heads=4, patch_size=2, num_slots=8, **kwargs)


def PermanentVideoDiT_B_2(**kwargs) -> PermanentVideoDiT:
    return PermanentVideoDiT(dim=384, depth=12, num_heads=6, patch_size=2, num_slots=12, **kwargs)


PermanentVideoDiT_models = {
    "PermanentVideoDiT-S/2": PermanentVideoDiT_S_2,
    "PermanentVideoDiT-B/2": PermanentVideoDiT_B_2,
}
