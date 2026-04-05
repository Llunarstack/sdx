"""
SuperiorViT — ViT/DiT architecture that incorporates all 2024-2025 improvements
over the original Facebook DiT:

  1. 2D RoPE (FLUX/SD3-style) — better spatial reasoning, resolution extrapolation
  2. Register Tokens (DINOv2 follow-up) — cleaner attention, no artifact tokens
  3. TACA cross-attention — fixes token imbalance + timestep-aware text alignment
  4. Dynamic Patch Scheduling — coarse patches at high noise, fine at low noise
  5. Linear Compressed Attention — O(N*C) global attention for long sequences
  6. Local Window Attention — O(N*W^2) local attention with global registers
  7. QK-Norm (RMSNorm per head) — training stability at high resolution
  8. LayerScale — deep network stability (CaiT-style)
  9. Jumbo Token — wide global summary token for extra capacity
 10. AdaLN-Zero with timestep + text conditioning (DiT baseline, kept)

The block is modular — each feature can be toggled independently so you can
ablate or gradually adopt improvements without breaking existing checkpoints.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_patch import DynamicPatchEmbed, TimestepPatchScheduler
from .linear_attention import LinearCompressedAttention, LocalWindowAttention
from .model_enhancements import DropPath, RMSNorm
from .register_tokens import JumboToken, RegisterTokens
from .rope2d import RoPE2D
from .taca import TACA
from .vit_next_blocks import LayerScale


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# AdaLN conditioning (timestep + optional class/text pooled embedding)
# ---------------------------------------------------------------------------

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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns (x_normed, shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)."""
        params = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = params
        x_normed = modulate(self.norm(x), shift_a, scale_a)
        return x_normed, gate_a, shift_f, scale_f, gate_f


# ---------------------------------------------------------------------------
# Self-attention with 2D RoPE + QK-Norm
# ---------------------------------------------------------------------------

class SelfAttentionRoPE2D(nn.Module):
    """Self-attention with 2D RoPE and optional QK-Norm."""

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
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.rope = RoPE2D(self.head_dim, base=rope_base)

        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        x: (B, N, D) where N = height * width (+ any prepended register/jumbo tokens).
        height, width: spatial dims of the *patch* tokens (excluding registers).
        """
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,D)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply 2D RoPE only to the patch tokens (last height*width positions)
        n_patch = height * width
        n_prefix = N - n_patch  # register + jumbo tokens at front
        if n_patch > 0 and n_patch <= N:
            q_patch = q[:, :, n_prefix:, :]
            k_patch = k[:, :, n_prefix:, :]
            cos, sin = self.rope.get_freqs(height, width, device=x.device, dtype=x.dtype)
            from .rope2d import apply_rope2d
            # apply_rope2d expects (B, H, N, D)
            q_patch_rot, k_patch_rot = apply_rope2d(
                q_patch.transpose(1, 2),  # (B, N, H, D)
                k_patch.transpose(1, 2),
                cos, sin,
            )
            q[:, :, n_prefix:, :] = q_patch_rot.transpose(1, 2)
            k[:, :, n_prefix:, :] = k_patch_rot.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# FFN (SwiGLU — better than GELU for transformers)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: 2/3 of standard FFN params, better performance."""

    def __init__(self, hidden_size: int, expansion: float = 8 / 3):
        super().__init__()
        inner = int(hidden_size * expansion)
        # Round to multiple of 64 for efficiency
        inner = (inner + 63) // 64 * 64
        self.w1 = nn.Linear(hidden_size, inner, bias=False)
        self.w2 = nn.Linear(hidden_size, inner, bias=False)
        self.w3 = nn.Linear(inner, hidden_size, bias=False)
        nn.init.zeros_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# SuperiorViT Block
# ---------------------------------------------------------------------------

class SuperiorViTBlock(nn.Module):
    """
    Single transformer block with all improvements enabled by flags.

    Args:
        hidden_size: Token dimension.
        num_heads: Attention heads.
        cond_dim: Conditioning vector dimension (timestep + class/text pooled).
        text_dim: Text encoder output dimension (for cross-attention).
        use_rope2d: Enable 2D RoPE on self-attention.
        use_taca: Enable TACA cross-attention (requires text_dim).
        use_linear_attn: Replace self-attention with LinearCompressedAttention.
        use_window_attn: Replace self-attention with LocalWindowAttention.
        use_swiglu: Use SwiGLU FFN instead of GELU MLP.
        layer_scale_init: LayerScale init value (0 = disabled).
        drop_path: Stochastic depth probability.
        qk_norm: QK-Norm on self-attention heads.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        text_dim: int = 0,
        use_rope2d: bool = True,
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
        self.use_rope2d = use_rope2d
        self.use_linear_attn = use_linear_attn
        self.use_window_attn = use_window_attn

        # AdaLN conditioning
        self.adaLN = AdaLNModulation(hidden_size, cond_dim)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention (choose variant)
        if use_linear_attn:
            self.attn = LinearCompressedAttention(
                hidden_size, num_heads, context_size=linear_context_size, qk_norm=qk_norm
            )
        elif use_window_attn:
            self.attn = LocalWindowAttention(
                hidden_size, num_heads, window_size=window_size, qk_norm=qk_norm
            )
        else:
            self.attn = SelfAttentionRoPE2D(hidden_size, num_heads, qk_norm=qk_norm)

        # Cross-attention (TACA or skip)
        if self.use_taca:
            self.cross_attn = TACA(hidden_size, text_dim, num_heads, qk_norm=qk_norm)
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # FFN
        self.ffn = SwiGLUFFN(hidden_size) if use_swiglu else nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # LayerScale
        self.ls_attn = LayerScale(hidden_size, layer_scale_init)
        self.ls_ffn = LayerScale(hidden_size, layer_scale_init)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        height: int,
        width: int,
        text_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) patch tokens (may include prepended register/jumbo tokens).
            c: (B, cond_dim) conditioning vector.
            height, width: spatial dims of patch tokens (excluding prefix tokens).
            text_emb: (B, L, T) text tokens for cross-attention.
            timestep: (B,) diffusion timestep for TACA temperature.
        """
        # AdaLN modulation
        x_normed, gate_a, shift_f, scale_f, gate_f = self.adaLN(x, c)

        # Self-attention
        if self.use_linear_attn or self.use_window_attn:
            attn_out = self.attn(x_normed)
        else:
            attn_out = self.attn(x_normed, height, width)

        x = x + self.drop_path(gate_a.unsqueeze(1) * self.ls_attn(attn_out))

        # Cross-attention (TACA)
        if self.use_taca and text_emb is not None:
            x_cross = self.norm_cross(x)
            x = x + self.drop_path(self.cross_attn(x_cross, text_emb, timestep))

        # FFN
        x_ffn = modulate(self.norm2(x), shift_f, scale_f)
        x = x + self.drop_path(gate_f.unsqueeze(1) * self.ls_ffn(self.ffn(x_ffn)))

        return x


# ---------------------------------------------------------------------------
# SuperiorViT — full model
# ---------------------------------------------------------------------------

class SuperiorViT(nn.Module):
    """
    Full SuperiorViT model for latent diffusion.

    Improvements over Facebook DiT:
      - 2D RoPE instead of learned sinusoidal pos embed
      - Register tokens (cleaner attention, no artifact tokens)
      - TACA cross-attention (timestep-aware, token-imbalance corrected)
      - Dynamic patch scheduling (coarse->fine across timesteps)
      - SwiGLU FFN (better than GELU MLP)
      - QK-Norm (training stability)
      - LayerScale (deep network stability)
      - Jumbo global token (extra capacity)
      - AdaLN-Zero conditioning (kept from DiT)

    Args:
        input_size: Latent spatial size (e.g. 32 for 256px with 8x VAE).
        patch_size: Base patch size.
        in_channels: Latent channels.
        hidden_size: Transformer hidden dim.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        text_dim: Text encoder output dim (0 = class-conditional only).
        num_classes: Number of classes (0 = text-only).
        learn_sigma: Predict variance (doubles output channels).
        num_registers: Number of register tokens.
        use_jumbo: Enable jumbo global token.
        use_dynamic_patch: Enable dynamic patch scheduling.
        drop_path_rate: Max stochastic depth rate.
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
        num_registers: int = 8,
        use_jumbo: bool = True,
        use_dynamic_patch: bool = True,
        drop_path_rate: float = 0.1,
        use_swiglu: bool = True,
        use_taca: bool = True,
        use_rope2d: bool = True,
        qk_norm: bool = True,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_registers = num_registers
        self.use_jumbo = use_jumbo
        self.use_dynamic_patch = use_dynamic_patch

        # Patch embedding
        if use_dynamic_patch:
            self.patch_embed = DynamicPatchEmbed(
                in_channels=in_channels,
                hidden_size=hidden_size,
                patch_sizes=(patch_size, patch_size * 2, patch_size * 4),
                img_size=input_size,
            )
            self.patch_scheduler = TimestepPatchScheduler(
                fine_size=patch_size,
                base_size=patch_size * 2,
                coarse_size=patch_size * 4,
            )
        else:
            from timm.models.vision_transformer import PatchEmbed
            self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size)

        # Register tokens
        if num_registers > 0:
            self.registers = RegisterTokens(hidden_size, num_registers)

        # Jumbo token
        if use_jumbo:
            self.jumbo = JumboToken(hidden_size)

        # Timestep embedding
        from .dit import TimestepEmbedder
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Class embedding (optional)
        if num_classes > 0:
            from .dit import LabelEmbedder
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.1)
        else:
            self.y_embedder = None

        # Text projection (optional)
        if text_dim > 0 and text_dim != hidden_size:
            self.text_proj = nn.Linear(text_dim, hidden_size, bias=False)
        else:
            self.text_proj = None

        # Conditioning dim = hidden_size (timestep + class/text pooled)
        cond_dim = hidden_size

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SuperiorViTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                cond_dim=cond_dim,
                text_dim=hidden_size if text_dim > 0 else 0,
                use_rope2d=use_rope2d,
                use_taca=use_taca,
                use_swiglu=use_swiglu,
                layer_scale_init=layer_scale_init,
                drop_path=dpr[i],
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])

        # Final layer (AdaLN + linear out)
        from .dit import FinalLayer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

    def _get_num_prefix_tokens(self) -> int:
        n = 0
        if self.num_registers > 0:
            n += self.num_registers
        if self.use_jumbo:
            n += 1
        return n

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy latent.
            t: (B,) diffusion timestep.
            y: (B,) class labels (optional).
            text_emb: (B, L, T) text encoder output (optional).
        Returns:
            (B, C, H, W) predicted noise / velocity.
        """
        B, C, H, W = x.shape

        # Dynamic patch size selection
        if self.use_dynamic_patch:
            t_norm = (t.float() / 1000.0).clamp(0, 1)
            # Use the median timestep for the batch to pick patch size
            t_med = float(t_norm.median())
            active_patch = self.patch_scheduler.get_patch_size(t_med)
            tokens, active_patch = self.patch_embed(x, patch_size=active_patch)
        else:
            tokens = self.patch_embed(x)
            active_patch = self.patch_size

        h_tok = H // active_patch
        w_tok = W // active_patch

        # Prepend jumbo token
        if self.use_jumbo:
            jumbo_tok = self.jumbo.get_token(B)
            tokens = torch.cat([jumbo_tok, tokens], dim=1)

        # Prepend register tokens
        if self.num_registers > 0:
            tokens = self.registers.prepend(tokens)

        # Conditioning vector
        c = self.t_embedder(t)
        if y is not None and self.y_embedder is not None:
            c = c + self.y_embedder(y, self.training)
        if text_emb is not None and self.text_proj is not None:
            text_emb = self.text_proj(text_emb)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c, h_tok, w_tok, text_emb=text_emb, timestep=t)

        # Post-process jumbo token
        if self.use_jumbo:
            n_prefix = self.num_registers + 1  # registers + jumbo
            jumbo_out = tokens[:, self.num_registers:self.num_registers + 1, :]
            jumbo_out = self.jumbo.process(jumbo_out)
            tokens = torch.cat([
                tokens[:, :self.num_registers, :],
                jumbo_out,
                tokens[:, n_prefix:, :],
            ], dim=1)

        # Strip prefix tokens (registers + jumbo)
        n_prefix = self._get_num_prefix_tokens()
        patch_tokens = tokens[:, n_prefix:, :]  # (B, h_tok*w_tok, D)

        # Final layer
        out = self.final_layer(patch_tokens, c)  # (B, N, patch_size^2 * out_channels)

        # Unpatchify
        p = active_patch
        out = out.reshape(B, h_tok, w_tok, p, p, self.out_channels)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, H, W)
        return out


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

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
]
