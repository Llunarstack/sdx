"""PermanentVideoDiT: ``VideoDiT`` plus slot memory for object permanence.

K learnable slots read each frame's patch tokens (cross-attention), write back into that frame, and attend
across time among themselves, so an object that leaves the view keeps an explicit memory entry instead of
relying on temporal attention alone. ``permanence_consistency_loss`` penalizes slot features that jump
between adjacent frames.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .video_dit import VideoDiT

__all__ = [
    "SlotMemoryBank",
    "IdentityTokenBank",
    "PermanentVideoDiT",
    "permanence_consistency_loss",
    "PermanentVideoDiT_S_2",
    "PermanentVideoDiT_B_2",
    "PermanentVideoDiT_models",
]


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
