"""Alternative DiT trunks.

``SuperiorViT``: a DiT built from post-DiT components (2D RoPE, register tokens, TACA cross-attention,
dynamic patch scheduling, QK-norm, LayerScale, a jumbo token, optional ``HybridDiTExtras``); same call
contract as ``DiT_Text``.

``EnhancedDiT``: class-conditional DiT whose blocks add layout, anatomy, text-rendering and consistency
modules, softmax-blended per block (legacy ``scripts/tools/enhanced.py`` training stack).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.layers import DropPath
from timm.models.vision_transformer import PatchEmbed

from .attention import TACA, LinearCompressedAttention, LocalWindowAttention, RoPE2D, apply_rope2d
from .dit import (
    DiTBlock,
    FinalLayer,
    LabelEmbedder,
    LayerScale,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    modulate,
    unpatchify,
    xavier_init_linears,
)
from .dit_text import HybridDiTExtras


class TimestepPatchScheduler:
    """Patch size per normalized timestep (1 = pure noise): coarse when noisy, fine when nearly clean.

    Early steps only settle global structure, so large patches there cut FLOPs (arXiv:2602.16968).
    """

    def __init__(
        self,
        fine_size: int = 4,
        base_size: int = 8,
        coarse_size: int = 16,
        high_thresh: float = 0.7,
        low_thresh: float = 0.3,
    ):
        self.fine_size = int(fine_size)
        self.base_size = int(base_size)
        self.coarse_size = int(coarse_size)
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)

    def get_patch_size(self, t_normalized: float) -> int:
        if t_normalized >= self.high_thresh:
            return self.coarse_size
        if t_normalized <= self.low_thresh:
            return self.fine_size
        return self.base_size


class DynamicPatchEmbed(nn.Module):
    """Patch embedding with one conv + learned position table per supported patch size."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 1152,
        patch_sizes: tuple[int, ...] = (4, 8, 16),
        img_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_sizes = tuple(patch_sizes)
        self.img_size = img_size
        self.projs = nn.ModuleDict(
            {str(p): nn.Conv2d(in_channels, hidden_size, kernel_size=p, stride=p, bias=False) for p in patch_sizes}
        )
        self.pos_embeds = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(1, (img_size // p) ** 2, hidden_size)) for p in patch_sizes}
        )
        for proj in self.projs.values():
            nn.init.xavier_uniform_(proj.weight.view(proj.weight.size(0), -1))
        for pe in self.pos_embeds.values():
            nn.init.trunc_normal_(pe, std=0.02)

    def forward(self, x: torch.Tensor, patch_size: int | None = None) -> tuple[torch.Tensor, int]:
        """``(B, C, H, W)`` -> tokens ``(B, N, D)`` and the patch size used (nearest supported; default middle)."""
        if patch_size is None:
            patch_size = self.patch_sizes[len(self.patch_sizes) // 2]
        if str(patch_size) not in self.projs:
            patch_size = min(self.patch_sizes, key=lambda p: abs(p - patch_size))
        tokens = self.projs[str(patch_size)](x)
        _, d, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        # Resample the learned table when the grid differs from init (shape, not just token count:
        # a 16x64 grid has as many tokens as 32x32 but a different layout).
        pe = self.pos_embeds[str(patch_size)]
        s_init = int(math.isqrt(pe.shape[1]))
        if (h, w) != (s_init, s_init):
            pe = pe.reshape(1, s_init, s_init, d).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
            pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, d)
        return tokens + pe, patch_size


class RegisterTokens(nn.Module):
    """Learnable register tokens prepended to the sequence (Darcet et al. 2023, arXiv:2309.16588).

    They absorb the global "artifact" information ViTs otherwise park in high-norm patch tokens.
    """

    def __init__(self, hidden_size: int, num_registers: int = 8, init_std: float = 0.02):
        super().__init__()
        self.num_registers = int(num_registers)
        self.registers = nn.Parameter(torch.randn(1, self.num_registers, int(hidden_size)) * float(init_std))

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.registers.expand(x.shape[0], -1, -1), x], dim=1)


class JumboToken(nn.Module):
    """One wide global token with its own wider FFN (Jumbo, arXiv:2502.15021): capacity without widening patches."""

    def __init__(self, hidden_size: int, jumbo_size: int = 0, ffn_expansion: int = 4):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.jumbo_size = int(jumbo_size) if jumbo_size > 0 else int(hidden_size) * 2
        self.up_proj = nn.Linear(self.hidden_size, self.jumbo_size, bias=False)
        self.down_proj = nn.Linear(self.jumbo_size, self.hidden_size, bias=False)
        h = self.jumbo_size * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(self.jumbo_size, h, bias=False),
            nn.GELU(),
            nn.Linear(h, self.jumbo_size, bias=False),
        )
        self.norm = nn.LayerNorm(self.jumbo_size)
        self.token = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.weight)

    def get_token(self, batch_size: int) -> torch.Tensor:
        return self.token.expand(batch_size, -1, -1)

    def process(self, jumbo: torch.Tensor) -> torch.Tensor:
        """Jumbo FFN after attention: ``(B, 1, D)`` -> ``(B, 1, D)``, residual in the wide space."""
        z = self.up_proj(jumbo)
        z = self.norm(z + self.ffn(z))
        return self.down_proj(z)


class AdaLNModulation(nn.Module):
    """AdaLN-Zero: produces 6 modulation params from a conditioning vector."""

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Returns (x_normed, gate_attn, shift_ffn, scale_ffn, gate_ffn)."""
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(6, dim=-1)
        return modulate(self.norm(x), shift_attn, scale_attn), gate_attn, shift_ffn, scale_ffn, gate_ffn


class SelfAttentionRoPE2D(nn.Module):
    """Self-attention with 2D RoPE on the patch tokens and optional QK-Norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.rope = RoPE2D(self.head_dim, base=rope_base)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        x: (B, N, D) with N = prefix (register/jumbo) tokens + height * width patch tokens.
        """
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        n_prefix = N - height * width
        if 0 <= n_prefix < N:
            cos, sin = self.rope.get_freqs(height, width, device=x.device, dtype=q.dtype)
            q_rot, k_rot = apply_rope2d(q[:, :, n_prefix:], k[:, :, n_prefix:], cos, sin)
            q = torch.cat([q[:, :, :n_prefix], q_rot], dim=2)
            k = torch.cat([k[:, :, :n_prefix], k_rot], dim=2)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward; the 8/3 expansion (rounded up to 64) matches a 4x GELU MLP's parameter count."""

    def __init__(self, hidden_size: int, expansion: float = 8 / 3):
        super().__init__()
        inner = (int(hidden_size * expansion) + 63) // 64 * 64
        self.gate_proj = nn.Linear(hidden_size, inner, bias=False)
        self.value_proj = nn.Linear(hidden_size, inner, bias=False)
        self.out_proj = nn.Linear(inner, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class SuperiorViTBlock(nn.Module):
    """AdaLN-Zero block: self-attention (2D RoPE, or linear / window attention), optional TACA
    cross-attention to text (``text_dim > 0``), SwiGLU or GELU FFN, LayerScale and stochastic depth."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        text_dim: int = 0,
        use_taca: bool = True,
        use_linear_attn: bool = False,
        use_window_attn: bool = False,
        use_swiglu: bool = True,
        layer_scale_init: float = 1e-4,
        drop_path: float = 0.0,
        qk_norm: bool = True,
        window_size: int = 8,
        linear_context_size: int = 64,
    ):
        super().__init__()
        self.use_taca = use_taca and text_dim > 0
        self.use_linear_attn = use_linear_attn
        self.use_window_attn = use_window_attn

        self.adaLN = AdaLNModulation(hidden_size, cond_dim)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if use_linear_attn:
            self.attn = LinearCompressedAttention(
                hidden_size, num_heads, context_size=linear_context_size, qk_norm=qk_norm
            )
        elif use_window_attn:
            self.attn = LocalWindowAttention(hidden_size, num_heads, window_size=window_size, qk_norm=qk_norm)
        else:
            self.attn = SelfAttentionRoPE2D(hidden_size, num_heads, qk_norm=qk_norm)

        if self.use_taca:
            self.cross_attn = TACA(hidden_size, text_dim, num_heads, qk_norm=qk_norm)
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.ffn = (
            SwiGLUFFN(hidden_size)
            if use_swiglu
            else nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )
        )
        self.ls_attn = LayerScale(hidden_size, layer_scale_init)
        self.ls_ffn = LayerScale(hidden_size, layer_scale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        height: int,
        width: int,
        text_emb: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: ``(B, N, D)``, prefix (register/jumbo) tokens first, then the ``height x width`` patches;
        ``c``: ``(B, cond_dim)``; ``timestep`` drives the TACA temperature."""
        x_normed, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaLN(x, c)
        if self.use_linear_attn or self.use_window_attn:
            attn_out = self.attn(x_normed)
        else:
            attn_out = self.attn(x_normed, height, width)
        x = x + self.drop_path(gate_attn.unsqueeze(1) * self.ls_attn(attn_out))

        if self.use_taca and text_emb is not None:
            x = x + self.drop_path(self.cross_attn(self.norm_cross(x), text_emb, timestep))

        x_ffn = modulate(self.norm2(x), shift_ffn, scale_ffn)
        return x + self.drop_path(gate_ffn.unsqueeze(1) * self.ls_ffn(self.ffn(x_ffn)))


class SuperiorViT(nn.Module):
    """SuperiorViT denoiser for latent diffusion.

    ``patch_size`` is the finest patch; ``use_dynamic_patch`` adds 2x and 4x patch sizes picked per step by
    ``TimestepPatchScheduler``. ``text_dim=0`` is class-conditional only, ``num_classes=0`` text-only;
    ``class_dropout_prob`` zeroes whole text sequences during training (CFG null condition).
    ``uvit_skips`` / ``hidit_hf`` / ``mix_ffn`` / ``mmdit_every_n`` toggle ``HybridDiTExtras``.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        text_dim: int = 768,
        num_classes: int = 0,
        learn_sigma: bool = True,
        class_dropout_prob: float = 0.0,
        num_registers: int = 8,
        use_jumbo: bool = True,
        use_dynamic_patch: bool = True,
        drop_path_rate: float = 0.1,
        use_swiglu: bool = True,
        use_taca: bool = True,
        qk_norm: bool = True,
        layer_scale_init: float = 1e-4,
        uvit_skips: bool = False,
        hidit_hf: bool = False,
        mix_ffn: bool = False,
        mmdit_every_n: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.class_dropout_prob = float(class_dropout_prob)
        self.num_registers = max(0, int(num_registers))
        self.use_jumbo = use_jumbo
        self.use_dynamic_patch = use_dynamic_patch
        self._grad_checkpointing = False

        if use_dynamic_patch:
            patch_sizes = (patch_size, patch_size * 2, patch_size * 4)
            self.patch_embed = DynamicPatchEmbed(
                in_channels=in_channels,
                hidden_size=hidden_size,
                patch_sizes=patch_sizes,
                img_size=input_size,
            )
            self.patch_scheduler = TimestepPatchScheduler(*patch_sizes)
        else:
            patch_sizes = (patch_size,)
            self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size, strict_img_size=False)

        if self.num_registers > 0:
            self.registers = RegisterTokens(hidden_size, self.num_registers)
        if use_jumbo:
            self.jumbo = JumboToken(hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.1) if num_classes > 0 else None
        self.text_proj = nn.Linear(text_dim, hidden_size, bias=False) if 0 < text_dim != hidden_size else None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                SuperiorViTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    cond_dim=hidden_size,
                    text_dim=hidden_size if text_dim > 0 else 0,
                    use_taca=use_taca,
                    use_swiglu=use_swiglu,
                    layer_scale_init=layer_scale_init,
                    drop_path=dpr[i],
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        # One output head per patch size so dynamic patching can unpatchify any active size.
        self.final_layer = nn.ModuleDict({str(p): FinalLayer(hidden_size, p, self.out_channels) for p in patch_sizes})

        self._init_weights()
        # Built after vanilla init so the zero-initialized extras leave base weights untouched.
        self.hybrid = None
        if uvit_skips or hidit_hf or mix_ffn or int(mmdit_every_n or 0) > 0:
            self.hybrid = HybridDiTExtras(
                hidden_size,
                num_heads,
                depth,
                uvit_skips=uvit_skips,
                hidit_hf=hidit_hf,
                mmdit_every_n=int(mmdit_every_n or 0),
                mix_ffn=mix_ffn,
            )

    def _init_weights(self):
        xavier_init_linears(self)
        # Restore the zero-inits that make every residual branch start as identity (AdaLN-Zero).
        for block in self.blocks:
            nn.init.zeros_(block.adaLN.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN.adaLN_modulation[-1].bias)
            if block.use_taca:
                nn.init.zeros_(block.cross_attn.out_proj.weight)
        for head in self.final_layer.values():
            nn.init.zeros_(head.adaLN_modulation[-1].weight)
            nn.init.zeros_(head.adaLN_modulation[-1].bias)
            nn.init.zeros_(head.linear.weight)
            nn.init.zeros_(head.linear.bias)
        if self.use_jumbo:
            nn.init.zeros_(self.jumbo.down_proj.weight)

    def enable_gradient_checkpointing(self) -> None:
        self._grad_checkpointing = True

    def _num_prefix_tokens(self) -> int:
        return self.num_registers + int(self.use_jumbo)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """``x`` ``(B, C, H, W)``, ``t`` ``(B,)``, optional text ``(B, L, text_dim)`` and labels ``(B,)`` ->
        ``(B, out_channels, H, W)``. ``**kwargs`` are sampler/trainer extras only ``DiT_Text`` consumes."""
        B, _, H, W = x.shape
        if self.use_dynamic_patch:
            t_med = float((t.float() / 1000.0).clamp(0, 1).median())
            tokens, active_patch = self.patch_embed(x, patch_size=self.patch_scheduler.get_patch_size(t_med))
        else:
            tokens, active_patch = self.patch_embed(x), self.patch_size
        h_tok, w_tok = H // active_patch, W // active_patch
        n_patch = h_tok * w_tok

        if self.use_jumbo:
            tokens = torch.cat([self.jumbo.get_token(B), tokens], dim=1)
        if self.num_registers > 0:
            tokens = self.registers.prepend(tokens)
        n_prefix = self._num_prefix_tokens()

        c = self.t_embedder(t)
        if y is not None and self.y_embedder is not None:
            c = c + self.y_embedder(y, self.training)
        text_emb = encoder_hidden_states
        if text_emb is not None and self.text_proj is not None:
            text_emb = self.text_proj(text_emb)
        if text_emb is not None and self.training and self.class_dropout_prob > 0:
            keep = torch.rand(B, 1, 1, device=text_emb.device) >= self.class_dropout_prob
            text_emb = text_emb * keep.to(text_emb.dtype)

        if self.hybrid is not None:
            self.hybrid.begin_forward()
        for i, block in enumerate(self.blocks):
            if self._grad_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, c, h_tok, w_tok, text_emb, t, use_reentrant=False
                )
            else:
                tokens = block(tokens, c, h_tok, w_tok, text_emb=text_emb, timestep=t)
            if self.use_jumbo:
                j = self.num_registers
                jumbo = tokens[:, j : j + 1]
                tokens = torch.cat([tokens[:, :j], jumbo + self.jumbo.process(jumbo), tokens[:, j + 1 :]], dim=1)
            if self.hybrid is not None:
                body = self.hybrid.after_block(
                    i,
                    tokens[:, n_prefix:],
                    text_emb,
                    num_patches=n_patch,
                    h_patches=h_tok,
                    w_patches=w_tok,
                    use_xformers=False,
                )
                tokens = torch.cat([tokens[:, :n_prefix], body], dim=1)

        patch_tokens = tokens[:, n_prefix:]
        if self.hybrid is not None:
            patch_tokens = self.hybrid.after_all(patch_tokens, t, num_patches=n_patch, h_patches=h_tok, w_patches=w_tok)

        p = active_patch
        out = self.final_layer[str(p)](patch_tokens, c)
        out = out.reshape(B, h_tok, w_tok, p, p, self.out_channels)
        return out.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, h_tok * p, w_tok * p)


def SuperiorViT_XL_2(**kwargs):
    return SuperiorViT(hidden_size=1152, depth=28, num_heads=16, patch_size=2, **kwargs)


def SuperiorViT_L_2(**kwargs):
    return SuperiorViT(hidden_size=1024, depth=24, num_heads=16, patch_size=2, **kwargs)


def SuperiorViT_B_2(**kwargs):
    return SuperiorViT(hidden_size=768, depth=12, num_heads=12, patch_size=2, **kwargs)


def SuperiorViT_S_2(**kwargs):
    return SuperiorViT(hidden_size=384, depth=12, num_heads=6, patch_size=2, **kwargs)


SuperiorViT_models = {
    "SuperiorViT-XL/2": SuperiorViT_XL_2,
    "SuperiorViT-L/2": SuperiorViT_L_2,
    "SuperiorViT-B/2": SuperiorViT_B_2,
    "SuperiorViT-S/2": SuperiorViT_S_2,
}


class SpatialControlModule(nn.Module):
    """Object layout ``(B, num_objects, 4)`` boxes -> attention toward each object's slot."""

    def __init__(self, hidden_size: int, num_objects: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_objects = num_objects
        self.object_pos_embed = nn.Parameter(torch.randn(num_objects, hidden_size) * 0.02)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 2),  # x, y, w, h
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.count_constraint = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, num_objects), nn.Sigmoid()
        )
        self.spatial_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

    def forward(self, x: torch.Tensor, spatial_layout: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``(B, N, D)``; ``spatial_layout``: ``(B, num_objects, 4)`` as (x, y, w, h)."""
        if spatial_layout is None:
            return x
        spatial_features = self.spatial_encoder(spatial_layout) + self.object_pos_embed.unsqueeze(0)
        # Count head gates each object's values, so slots it deems absent contribute nothing.
        count_weights = self.count_constraint(spatial_features.mean(dim=1))
        values = spatial_features * count_weights.unsqueeze(-1)
        x_attended, _ = self.spatial_attention(x, spatial_features, values)
        return x + 0.3 * x_attended


class AnatomyAwareAttention(nn.Module):
    """Self-attention + constraint MLP residual; inside ``anatomy_mask`` an extra attention over finger slots."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.anatomy_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        # Unused by forward; kept so EnhancedDiT checkpoints keep loading strictly.
        self.body_part_embeddings = nn.Parameter(torch.randn(10, hidden_size) * 0.02)
        self.anatomy_constraint = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.hand_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.finger_embeddings = nn.Parameter(torch.randn(5, hidden_size) * 0.02)

    def forward(self, x: torch.Tensor, anatomy_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``(B, N, D)``; ``anatomy_mask``: ``(B, M)`` region mask on a square grid (resized to ``N``)."""
        B, N, D = x.shape
        x_anatomy, _ = self.anatomy_attention(x, x, x)
        x = x + 0.4 * self.anatomy_constraint(x_anatomy)
        if anatomy_mask is None:
            return x
        x_hands, _ = self.hand_attention(x, x, x)
        finger_features = self.finger_embeddings.unsqueeze(0).expand(B, -1, -1)
        finger_attended, _ = self.hand_attention(x_hands, finger_features, finger_features)
        mask = anatomy_mask
        if mask.shape[1] != N:
            side, target = int(mask.shape[1] ** 0.5), int(N**0.5)
            if side != target:
                mask = F.interpolate(mask.view(B, 1, side, side), size=(target, target), mode="nearest").view(B, -1)
        return torch.where(mask.unsqueeze(-1) > 0.5, x + 0.3 * finger_attended, x)


class TextRenderingLayer(nn.Module):
    """Glyph-token cross-attention: text token ids (+ typography style, + 2D positions) attended by the image."""

    def __init__(self, hidden_size: int, vocab_size: int = 50000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.text_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.typography_embeddings = nn.Parameter(torch.randn(20, hidden_size) * 0.02)
        self.text_position_net = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),  # +2 for the (x, y) text position
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.char_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.text_quality_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size), nn.Dropout(0.1)
        )

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
        typography_style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``text_tokens``: ``(B, L)`` ids; ``text_positions``: ``(B, L, 2)``; ``typography_style``: ``(B,)`` ids."""
        if text_tokens is None:
            return x
        text_features = self.text_embeddings(text_tokens)
        if typography_style is not None:
            text_features = text_features + self.typography_embeddings[typography_style].unsqueeze(1)
        if text_positions is not None:
            text_features = self.text_position_net(torch.cat([text_features, text_positions], dim=-1))
        text_attended, _ = self.char_attention(text_features, text_features, text_features)
        text_enhanced = self.text_quality_net(text_attended)
        x_text_attended, _ = self.char_attention(x, text_enhanced, text_enhanced)
        return x + 0.5 * x_text_attended


class ConsistencyEmbedder(nn.Module):
    """Character / style id embeddings (+ learned memory rows) mixed into every token."""

    def __init__(self, hidden_size: int, max_characters: int = 100, max_styles: int = 50):
        super().__init__()
        self.hidden_size = hidden_size
        self.character_embeddings = nn.Embedding(max_characters, hidden_size)
        self.style_embeddings = nn.Embedding(max_styles, hidden_size)
        self.consistency_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # character + style
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.character_memory = nn.Parameter(torch.randn(max_characters, hidden_size) * 0.02)
        self.style_memory = nn.Parameter(torch.randn(max_styles, hidden_size) * 0.02)

    def forward(
        self, x: torch.Tensor, character_id: torch.Tensor | None = None, style_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``character_id`` / ``style_id``: ``(B,)`` ids; a missing one contributes zeros."""
        if character_id is None and style_id is None:
            return x
        B, N, D = x.shape
        char = style = x.new_zeros(B, D)
        if character_id is not None:
            char = self.character_embeddings(character_id) + self.character_memory[character_id]
        if style_id is not None:
            style = self.style_embeddings(style_id) + self.style_memory[style_id]
        consistency = self.consistency_net(torch.cat([char, style], dim=-1))
        return x + 0.3 * consistency.unsqueeze(1).expand(-1, N, -1)


class EnhancedDiTBlock(DiTBlock):
    """``DiTBlock`` followed by the four feature modules, mixed with softmax-normalised learned weights."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **kwargs)
        self.spatial_control = SpatialControlModule(hidden_size)
        self.anatomy_attention = AnatomyAwareAttention(hidden_size, num_heads)
        self.text_rendering = TextRenderingLayer(hidden_size)
        self.consistency_embedder = ConsistencyEmbedder(hidden_size)
        self.feature_blend = nn.Parameter(torch.ones(4) * 0.25)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        spatial_layout: torch.Tensor | None = None,
        anatomy_mask: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
        typography_style: torch.Tensor | None = None,
        character_id: torch.Tensor | None = None,
        style_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = super().forward(x, c)
        features = (
            self.spatial_control(x, spatial_layout),
            self.anatomy_attention(x, anatomy_mask),
            self.text_rendering(x, text_tokens, text_positions, typography_style),
            self.consistency_embedder(x, character_id, style_id),
        )
        return sum(w * f for w, f in zip(F.softmax(self.feature_blend, dim=0), features, strict=True))


class EnhancedDiT(nn.Module):
    """Class-conditional DiT built from ``EnhancedDiTBlock``; each feature module can be switched off."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        enable_spatial_control: bool = True,
        enable_anatomy_awareness: bool = True,
        enable_text_rendering: bool = True,
        enable_consistency: bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.enable_spatial_control = enable_spatial_control
        self.enable_anatomy_awareness = enable_anatomy_awareness
        self.enable_text_rendering = enable_text_rendering
        self.enable_consistency = enable_consistency
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [EnhancedDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._grad_checkpointing = False
        self.initialize_weights()

    def enable_gradient_checkpointing(self):
        self._grad_checkpointing = True

    def initialize_weights(self):
        """Base DiT init: sin/cos positions, Xavier patch embed, N(0, 0.02) embeddings, adaLN-Zero + zero output."""
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        spatial_layout: torch.Tensor | None = None,
        anatomy_mask: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
        typography_style: torch.Tensor | None = None,
        character_id: torch.Tensor | None = None,
        style_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: ``(B, C, H, W)`` latent, ``t`` / ``y``: ``(B,)``; feature inputs are shaped as in each module."""
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        block_kwargs = dict(
            spatial_layout=spatial_layout if self.enable_spatial_control else None,
            anatomy_mask=anatomy_mask if self.enable_anatomy_awareness else None,
            text_tokens=text_tokens if self.enable_text_rendering else None,
            text_positions=text_positions if self.enable_text_rendering else None,
            typography_style=typography_style if self.enable_text_rendering else None,
            character_id=character_id if self.enable_consistency else None,
            style_id=style_id if self.enable_consistency else None,
        )
        checkpointing = self._grad_checkpointing and self.training
        for block in self.blocks:
            if checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False, **block_kwargs)
            else:
                x = block(x, c, **block_kwargs)
        return unpatchify(self.final_layer(x, c), self.patch_size, self.out_channels, h, w)


def EnhancedDiT_XL_2(**kwargs):
    return EnhancedDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def EnhancedDiT_L_2(**kwargs):
    return EnhancedDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def EnhancedDiT_B_2(**kwargs):
    return EnhancedDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


EnhancedDiT_models: dict = {
    "EnhancedDiT-XL/2": EnhancedDiT_XL_2,
    "EnhancedDiT-L/2": EnhancedDiT_L_2,
    "EnhancedDiT-B/2": EnhancedDiT_B_2,
}


__all__ = [
    "SuperiorViT",
    "SuperiorViTBlock",
    "SuperiorViT_models",
    "SuperiorViT_XL_2",
    "SuperiorViT_L_2",
    "SuperiorViT_B_2",
    "SuperiorViT_S_2",
    "SelfAttentionRoPE2D",
    "SwiGLUFFN",
    "AdaLNModulation",
    "EnhancedDiT",
    "EnhancedDiTBlock",
    "EnhancedDiT_XL_2",
    "EnhancedDiT_L_2",
    "EnhancedDiT_B_2",
    "EnhancedDiT_models",
]
