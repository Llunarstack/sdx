"""Opt-in extras for ``DiT_Text`` / ``SuperiorViT`` adding four inductive biases isotropic DiT lacks:

1. **Long encoder→decoder skips** (U-ViT, Hourglass, UDT, DDT) — high-frequency
   spatial detail that isotropic transformers forget over depth.
2. **Joint image+text self-attention** (MMDiT-lite) — image tokens attend over image and
   text keys in one softmax (the text rows are not updated).
3. **Mix-FFN** (Sana) — a depthwise conv next to the MLP so pores, edges, and
   fabric live in local neighborhoods, not only global attention.
4. **Time-gated high-frequency residual** (Hi-DiT) — extra texture only in the
   low-noise regime, where latent DiT otherwise goes waxy.

Every extra path is **zero-initialized**, so a loaded DiT checkpoint stays an
identity residual until training (or a ``HybridDiT-*`` run) moves the weights.

``SuperiorViT`` attaches the same extras after vanilla weight init. Its token
layout is prefix-first (registers + jumbo, then patches), so callers wrap
``after_block`` / ``after_all`` around the patch body only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import memory_efficient_attention


class EncoderDecoderSkip(nn.Module):
    """U-ViT long skip: decoder += Linear(encoder). Zero → identity."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, decoder: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
        return decoder + self.proj(encoder.to(dtype=decoder.dtype))


class MixFFN(nn.Module):
    """Sana-style Mix-FFN: GELU MLP + depthwise 3×3 for local texture."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.fc2 = nn.Linear(dim, dim)
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)

    def forward(self, patches: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, d = patches.shape
        if n != h * w or h < 2 or w < 2:
            return patches
        y = F.gelu(self.fc1(patches)).transpose(1, 2).reshape(b, d, h, w)
        return patches + self.fc2(self.dw(y).flatten(2).transpose(1, 2))


class JointMMDiTAttention(nn.Module):
    """SD3 / Flux MMDiT-lite: one self-attn over concatenated image and text tokens."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % max(1, num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // int(num_heads)
        self.scale = self.head_dim**-0.5
        self.norm = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, patches: torch.Tensor, text: torch.Tensor, *, use_xformers: bool = True) -> torch.Tensor:
        """Joint attention over ``[patches, text]``; only the patch rows are returned (text stays frozen)."""
        n = patches.shape[1]
        x = self.norm(torch.cat([patches, text.to(dtype=patches.dtype)], dim=1))
        b, length, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, length, 3, self.num_heads, self.head_dim).unbind(2)
        out = memory_efficient_attention(q, k, v, scale=self.scale, use_xformers=use_xformers)
        return patches + self.out_proj(out[:, :n].reshape(b, n, -1))


class TimeGatedHFResidual(nn.Module):
    """Hi-DiT-style depthwise HF residual gated by ``(1 - t / (T - 1))**2``: it fades in as noise -> 0."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)

    def forward(
        self,
        patches: torch.Tensor,
        h: int,
        w: int,
        t: torch.Tensor,
        *,
        num_timesteps: int = 1000,
    ) -> torch.Tensor:
        b, n, d = patches.shape
        if n != h * w or h < 2 or w < 2:
            return patches
        nt = max(1, int(num_timesteps) - 1)
        gate = (1.0 - (t.float() / float(nt)).clamp(0.0, 1.0)).pow(2)
        gate = gate.view(b, 1, 1, 1).to(dtype=patches.dtype)
        spat = patches.transpose(1, 2).reshape(b, d, h, w)
        spat = spat + gate * self.dw(spat)
        return spat.flatten(2).transpose(1, 2)


class HybridDiTExtras(nn.Module):
    """Bundled extras attached to ``DiT_Text`` / ``SuperiorViT`` after vanilla weight init.

    Per forward: ``begin_forward()``, then ``after_block`` after every block, then ``after_all``. U-ViT skips
    pair block ``depth - 1 - i`` with encoder block ``i`` (first ``depth // 2`` blocks); Mix-FFN runs in the
    second half; a joint MMDiT attention follows every ``mmdit_every_n``-th block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        *,
        uvit_skips: bool = False,
        hidit_hf: bool = False,
        mmdit_every_n: int = 0,
        mix_ffn: bool = False,
        num_timesteps: int = 1000,
    ):
        super().__init__()
        self.uvit_skips_on = bool(uvit_skips)
        self.hidit_on = bool(hidit_hf)
        self.mix_on = bool(mix_ffn)
        self.mmdit_every_n = int(mmdit_every_n or 0)
        self.depth = int(depth)
        self.half = int(depth) // 2
        self.num_timesteps = max(2, int(num_timesteps))
        self.skips = nn.ModuleList(
            [EncoderDecoderSkip(hidden_size) for _ in range(self.half)] if self.uvit_skips_on and self.half > 0 else []
        )
        n_joint = 0
        if self.mmdit_every_n > 0:
            n_joint = max(1, int(depth) // self.mmdit_every_n)
        self.joints = nn.ModuleList([JointMMDiTAttention(hidden_size, num_heads) for _ in range(n_joint)])
        self.mix = MixFFN(hidden_size) if self.mix_on else None
        self.hf = TimeGatedHFResidual(hidden_size) if self.hidit_on else None
        self._enc: list[torch.Tensor] = []

    def zero_init(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "zero_init"):
                m.zero_init()

    def begin_forward(self) -> None:
        self._enc = []

    def after_block(
        self,
        i: int,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        *,
        num_patches: int,
        h_patches: int,
        w_patches: int,
        use_xformers: bool,
    ) -> torch.Tensor:
        patches, rest = x[:, :num_patches], x[:, num_patches:]
        if self.uvit_skips_on:
            if i < self.half:
                self._enc.append(patches)
            elif i >= self.depth - self.half:
                enc_idx = self.depth - 1 - i
                patches = self.skips[enc_idx](patches, self._enc[enc_idx])
        if (
            self.mmdit_every_n > 0
            and (i + 1) % self.mmdit_every_n == 0
            and self.joints
            and text_emb is not None
            and text_emb.numel() > 0
        ):
            j = min((i + 1) // self.mmdit_every_n - 1, len(self.joints) - 1)
            patches = self.joints[j](patches, text_emb, use_xformers=use_xformers)
        if self.mix is not None and i >= self.half:
            patches = self.mix(patches, h_patches, w_patches)
        return _join(patches, rest)

    def after_all(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        num_patches: int,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        self._enc = []  # drop the skip activations as soon as the trunk is done
        if self.hf is None:
            return x
        patches, rest = x[:, :num_patches], x[:, num_patches:]
        patches = self.hf(patches, h_patches, w_patches, t, num_timesteps=self.num_timesteps)
        return _join(patches, rest)


def _join(patches: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    return patches if rest.shape[1] == 0 else torch.cat([patches, rest], dim=1)


HYBRID_EXTRAS_DEFAULTS = {"uvit_skips": True, "hidit_hf": True, "mix_ffn": True, "mmdit_every_n": 4}
HYBRID_DIT_DEFAULTS = {
    **HYBRID_EXTRAS_DEFAULTS,
    "qk_norm": True,
    "use_rope": True,
    "use_taca": True,
    "use_swiglu": True,
    "num_register_tokens": 4,
}


def _hybrid_dit_text(depth: int, hidden_size: int, num_heads: int, **kwargs):
    from .dit_text import DiT_Text

    return DiT_Text(
        depth=depth, hidden_size=hidden_size, patch_size=2, num_heads=num_heads, **{**HYBRID_DIT_DEFAULTS, **kwargs}
    )


def HybridDiT_S_2(**kwargs):
    return _hybrid_dit_text(12, 384, 6, **kwargs)


def HybridDiT_B_2(**kwargs):
    return _hybrid_dit_text(12, 768, 12, **kwargs)


def HybridDiT_L_2(**kwargs):
    return _hybrid_dit_text(24, 1024, 16, **kwargs)


def HybridDiT_XL_2(**kwargs):
    return _hybrid_dit_text(28, 1152, 16, **kwargs)


HybridDiT_models = {
    "HybridDiT-S/2": HybridDiT_S_2,
    "HybridDiT-B/2": HybridDiT_B_2,
    "HybridDiT-L/2": HybridDiT_L_2,
    "HybridDiT-XL/2": HybridDiT_XL_2,
}

__all__ = [
    "HYBRID_DIT_DEFAULTS",
    "HYBRID_EXTRAS_DEFAULTS",
    "EncoderDecoderSkip",
    "MixFFN",
    "JointMMDiTAttention",
    "TimeGatedHFResidual",
    "HybridDiTExtras",
    "HybridDiT_models",
    "HybridDiT_S_2",
    "HybridDiT_B_2",
    "HybridDiT_L_2",
    "HybridDiT_XL_2",
]
