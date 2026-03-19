# DiT Predecessor: transformer with more and better features than DiT.
# QK-norm (LLaMA/FLUX-style), SwiGLU MLP, AdaLN-Zero, deeper/wider default.
# Same text conditioning interface as DiT_Text (style, control, negative prompt).
import torch
import torch.nn as nn
import numpy as np
from .dit import FinalLayer, TimestepEmbedder, get_2d_sincos_pos_embed
from .attention import memory_efficient_attention, create_block_causal_mask_2d, SSMTokenMixer
from .controlnet import ControlNetEncoder
from .dit_text import TextEmbedder, CrossAttention
from .moe import MoEExperts, MoEProjection, MoERouter
from timm.models.vision_transformer import PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root mean square layer norm (faster, LLaMA/FLUX-style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SelfAttentionQKNorm(nn.Module):
    """Self-attention with QK normalization for stability and better scaling (LLaMA/FLUX-style)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.moe_out_proj = None
        if moe_num_experts and int(moe_num_experts) > 0:
            self.moe_out_proj = MoEProjection(
                hidden_size,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
            )
        self.dropout = nn.Dropout(dropout)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        use_xformers: bool = True,
        routing_context: torch.Tensor = None,
        router_override=None,
        report_aux_loss: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = memory_efficient_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scale, use_xformers=use_xformers
        )
        out = out.reshape(B, N, C)
        if self.moe_out_proj is not None:
            out = self.moe_out_proj(
                out,
                routing_context=routing_context,
                router_override=router_override,
                report_aux_loss=report_aux_loss,
            )
        else:
            out = self.out_proj(out)
        return self.dropout(out)


class SwiGLU(nn.Module):
    """SwiGLU MLP: gate * silu(up) then down (better capacity than plain GELU)."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.gate = nn.Linear(hidden_size, mlp_hidden)
        self.up = nn.Linear(hidden_size, mlp_hidden)
        self.down = nn.Linear(mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class CrossAttentionQKNorm(nn.Module):
    """Cross-attention (query=spatial, key/value=text) with QK normalization for stability (Z-Image/PixArt-sigma style)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        text_dim: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(text_dim, hidden_size)
        self.v_proj = nn.Linear(text_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.moe_out_proj = None
        if moe_num_experts and int(moe_num_experts) > 0:
            self.moe_out_proj = MoEProjection(
                hidden_size,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
            )
        self.dropout = nn.Dropout(dropout)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        use_xformers: bool = True,
        routing_context: torch.Tensor = None,
        router_override=None,
        report_aux_loss: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = text_emb.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = memory_efficient_attention(
            q, k, v, attn_mask=None, scale=self.scale, use_xformers=use_xformers
        )
        out = out.reshape(B, N, C)
        if self.moe_out_proj is not None:
            out = self.moe_out_proj(
                out,
                routing_context=routing_context,
                router_override=router_override,
                report_aux_loss=report_aux_loss,
            )
        else:
            out = self.out_proj(out)
        return self.dropout(out)


class DiTBlockSupreme(nn.Module):
    """Supreme block: RMSNorm + QK-norm self-attn + QK-norm cross-attn + SwiGLU + AdaLN-Zero."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        text_dim: int,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        use_ssm: bool = False,
        ssm_kernel_size: int = 7,
        token_routing_enabled: bool = False,
        token_routing_strength: float = 1.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        if use_ssm:
            if moe_num_experts is not None and int(moe_num_experts) > 0:
                raise ValueError("use_ssm enabled but moe_num_experts>0: disable MoE or disable SSM for now.")
            self.attn = SSMTokenMixer(hidden_size, kernel_size=ssm_kernel_size, dropout=0.0)
        else:
            self.attn = SelfAttentionQKNorm(
                hidden_size,
                num_heads,
                dropout=0.0,
                moe_num_experts=int(moe_num_experts),
                moe_top_k=int(moe_top_k),
            )
        self.norm_cross = RMSNorm(hidden_size, eps=eps)
        self.cross_attn = CrossAttentionQKNorm(
            hidden_size,
            num_heads,
            text_dim,
            dropout=0.0,
            eps=eps,
            moe_num_experts=int(moe_num_experts),
            moe_top_k=int(moe_top_k),
        )
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        if moe_num_experts and int(moe_num_experts) > 0:
            self.moe_router = MoERouter(
                hidden_size,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
            )
            experts = nn.ModuleList([SwiGLU(hidden_size, mlp_ratio=mlp_ratio) for _ in range(int(moe_num_experts))])
            self.mlp = MoEExperts(hidden_size, experts, top_k=int(moe_top_k))
        else:
            self.moe_router = None
            self.mlp = SwiGLU(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.token_routing_enabled = bool(token_routing_enabled)
        self.token_routing_strength = float(token_routing_strength)
        if self.token_routing_enabled:
            s = max(0.0, min(1.0, self.token_routing_strength))
            self.token_routing_strength = s
            self.token_router = nn.Sequential(
                RMSNorm(hidden_size, eps=eps),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.token_router = None

    def forward(
        self,
        x,
        c,
        text_emb,
        attn_mask=None,
        use_xformers=True,
        num_patch_tokens: int = 0,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        router_override = self.moe_router if self.moe_router is not None else None
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        token_gate = 1.0
        if self.token_routing_enabled and self.token_router is not None:
            token_gate = torch.sigmoid(self.token_router(x_norm))  # (B, N, 1)
            if num_patch_tokens and x.shape[1] > num_patch_tokens:
                token_gate[:, num_patch_tokens:, :] = 1.0
            s = self.token_routing_strength
            token_gate = ((1.0 - s) + s * token_gate).clamp(0.0, 1.0)
        attn_out = self.attn(
            x_norm,
            attn_mask=attn_mask,
            use_xformers=use_xformers,
            routing_context=c,
            router_override=router_override,
            report_aux_loss=False,
        )
        x = x + gate_msa.unsqueeze(1) * token_gate * attn_out
        x = x + token_gate * self.cross_attn(
            self.norm_cross(x),
            text_emb,
            use_xformers=use_xformers,
            routing_context=c,
            router_override=router_override,
            report_aux_loss=False,
        )
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if isinstance(self.mlp, MoEExperts):
            x = x + gate_mlp.unsqueeze(1) * token_gate * self.mlp(
                mlp_in,
                routing_context=c,
                router_override=router_override,
                report_aux_loss=True,
            )
        else:
            x = x + gate_mlp.unsqueeze(1) * token_gate * self.mlp(mlp_in)
        return x


class DiTBlockPredecessor(nn.Module):
    """Predecessor block: QK-norm self-attn, cross-attn to text, SwiGLU MLP, AdaLN-Zero."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        text_dim: int,
        mlp_ratio: float = 4.0,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        use_ssm: bool = False,
        ssm_kernel_size: int = 7,
        # ViT-Gen features
        token_routing_enabled: bool = False,
        token_routing_strength: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_ssm:
            if moe_num_experts is not None and int(moe_num_experts) > 0:
                raise ValueError("use_ssm enabled but moe_num_experts>0: disable MoE or disable SSM for now.")
            self.attn = SSMTokenMixer(hidden_size, kernel_size=ssm_kernel_size, dropout=0.0)
        else:
            self.attn = SelfAttentionQKNorm(
                hidden_size,
                num_heads,
                dropout=0.0,
                moe_num_experts=int(moe_num_experts),
                moe_top_k=int(moe_top_k),
            )
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads,
            text_dim,
            moe_num_experts=int(moe_num_experts),
            moe_top_k=int(moe_top_k),
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if moe_num_experts and int(moe_num_experts) > 0:
            self.moe_router = MoERouter(
                hidden_size,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
            )
            experts = nn.ModuleList([SwiGLU(hidden_size, mlp_ratio=mlp_ratio) for _ in range(int(moe_num_experts))])
            self.mlp = MoEExperts(hidden_size, experts, top_k=int(moe_top_k))
        else:
            self.moe_router = None
            self.mlp = SwiGLU(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.token_routing_enabled = bool(token_routing_enabled)
        self.token_routing_strength = float(token_routing_strength)
        if self.token_routing_enabled:
            s = max(0.0, min(1.0, self.token_routing_strength))
            self.token_routing_strength = s
            self.token_router = nn.Sequential(
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.token_router = None

    def forward(
        self,
        x,
        c,
        text_emb,
        attn_mask=None,
        use_xformers=True,
        num_patch_tokens: int = 0,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        router_override = self.moe_router if self.moe_router is not None else None
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        token_gate = 1.0
        if self.token_routing_enabled and self.token_router is not None:
            token_gate = torch.sigmoid(self.token_router(x_norm))  # (B, N, 1)
            if num_patch_tokens and x.shape[1] > num_patch_tokens:
                token_gate[:, num_patch_tokens:, :] = 1.0
            s = self.token_routing_strength
            token_gate = ((1.0 - s) + s * token_gate).clamp(0.0, 1.0)
        attn_out = self.attn(
            x_norm,
            attn_mask=attn_mask,
            use_xformers=use_xformers,
            routing_context=c,
            router_override=router_override,
            report_aux_loss=False,
        )
        x = x + gate_msa.unsqueeze(1) * token_gate * attn_out
        x = x + token_gate * self.cross_attn(
            self.norm_cross(x),
            text_emb,
            use_xformers=use_xformers,
            routing_context=c,
            router_override=router_override,
            report_aux_loss=False,
        )
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if isinstance(self.mlp, MoEExperts):
            x = x + gate_mlp.unsqueeze(1) * token_gate * self.mlp(
                mlp_in,
                routing_context=c,
                router_override=router_override,
                report_aux_loss=True,
            )
        else:
            x = x + gate_mlp.unsqueeze(1) * token_gate * self.mlp(mlp_in)
        return x


class DiT_Predecessor_Text(nn.Module):
    """DiT's predecessor: more and better features — QK-norm, SwiGLU, AdaLN-Zero, deeper/wider.
    Same conditioning as DiT_Text: text, style, control, negative prompt. No reference image."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1280,
        depth=32,
        num_heads=20,
        mlp_ratio=4.0,
        text_dim=4096,
        max_text_len=300,
        class_dropout_prob=0.1,
        learn_sigma=True,
        num_ar_blocks=0,
        use_xformers=True,
        style_embed_dim=0,
        control_cond_dim=0,
        creativity_embed_dim=0,
        size_embed_dim=0,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        # REPA (Representation Alignment)
        repa_out_dim: int = 0,
        repa_projector_hidden_dim: int = 0,
        # ViT-Gen / hybrid SSM swap:
        ssm_every_n: int = 0,
        ssm_kernel_size: int = 7,
        # ViT-Gen features (register tokens + soft token routing)
        num_register_tokens: int = 0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        kv_merge_factor: int = 1,
        token_routing_enabled: bool = False,
        token_routing_strength: float = 1.0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.text_dim = text_dim
        self._grad_checkpointing = False
        self.num_ar_blocks = num_ar_blocks
        self.use_xformers = use_xformers
        self.style_embed_dim = style_embed_dim
        self.control_cond_dim = control_cond_dim
        self.creativity_embed_dim = creativity_embed_dim
        self._moe_num_experts = int(moe_num_experts) if moe_num_experts is not None else 0
        self._moe_top_k = int(moe_top_k) if moe_top_k is not None else 2

        # REPA projector: maps DiT hidden tokens -> vision encoder embedding dim.
        self.repa_out_dim = int(repa_out_dim) if repa_out_dim is not None else 0
        self.repa_projector_hidden_dim = int(repa_projector_hidden_dim) if repa_projector_hidden_dim is not None else 0
        self.repa_head = None
        if self.repa_out_dim > 0:
            if self.repa_projector_hidden_dim > 0:
                self.repa_head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, self.repa_projector_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.repa_projector_hidden_dim, self.repa_out_dim),
                )
            else:
                self.repa_head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, self.repa_out_dim),
                )
        self._repa_projected = None

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_embedder = TextEmbedder(text_dim, hidden_size, class_dropout_prob)
        if style_embed_dim > 0:
            self.style_proj = nn.Linear(style_embed_dim, hidden_size)
        else:
            self.style_proj = None
        if creativity_embed_dim > 0:
            self.creativity_proj = nn.Sequential(
                nn.Linear(1, max(16, creativity_embed_dim // 2)),
                nn.SiLU(),
                nn.Linear(max(16, creativity_embed_dim // 2), hidden_size),
            )
        else:
            self.creativity_proj = None
        if control_cond_dim > 0:
            self.control_encoder = ControlNetEncoder(
                control_size=input_size, patch_size=patch_size, in_channels=3, hidden_size=hidden_size
            )
        else:
            self.control_encoder = None
        # ViT-Gen feature flags
        self.num_patches = int(self.x_embedder.num_patches)
        self.num_register_tokens = int(num_register_tokens)
        self.use_rope = bool(use_rope)
        self.kv_merge_factor = int(kv_merge_factor)  # not yet applied in predecessor models
        self.rope_base = float(rope_base)  # not yet applied in predecessor models
        self.token_routing_enabled = bool(token_routing_enabled)
        self.token_routing_strength = float(token_routing_strength)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # Learnable register tokens (scratchpad) appended to the patch token stream.
        self.register_tokens = None
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, hidden_size))

        self.register_buffer("_ar_mask", None)
        if num_ar_blocks > 0:
            p = int(self.num_patches ** 0.5)
            self._ar_mask = create_block_causal_mask_2d(p, p, num_ar_blocks)

        # Blocks receive text_emb from text_embedder (already projected to hidden_size), not raw encoder_hidden_states
        blocks = []
        for i in range(depth):
            use_ssm = bool(ssm_every_n and int(ssm_every_n) > 0 and (i % int(ssm_every_n) == 0))
            blocks.append(
                DiTBlockPredecessor(
                    hidden_size,
                    num_heads,
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    moe_num_experts=self._moe_num_experts,
                    moe_top_k=self._moe_top_k,
                    use_ssm=use_ssm,
                    ssm_kernel_size=ssm_kernel_size,
                    token_routing_enabled=self.token_routing_enabled,
                    token_routing_strength=self.token_routing_strength,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self._moe_layers = [m for m in self.modules() if hasattr(m, "last_aux_loss")]
        self.moe_aux_loss = None

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.text_embedder.proj.weight, std=0.02)
        if self.style_proj is not None:
            nn.init.normal_(self.style_proj.weight, std=0.02)
            nn.init.zeros_(self.style_proj.bias)
        if self.control_encoder is not None:
            nn.init.xavier_uniform_(self.control_encoder.proj.weight.view(self.control_encoder.proj.weight.shape[0], -1))
            nn.init.zeros_(self.control_encoder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], c, h * p, h * p))

    def enable_gradient_checkpointing(self):
        self._grad_checkpointing = True

    def forward(
        self,
        x,
        t,
        encoder_hidden_states=None,
        encoder_hidden_states_negative=None,
        negative_prompt_weight=0.5,
        style_embedding=None,
        style_strength=0.7,
        control_image=None,
        control_scale=1.0,
        conditioning_scale=1.0,
        **kwargs,
    ):
        self._repa_projected = None
        if encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get("y_embed")
        assert encoder_hidden_states is not None, "encoder_hidden_states required"
        if getattr(self, "_moe_layers", None):
            self.moe_aux_loss = None
            for m in self._moe_layers:
                m.last_aux_loss = None
        x_patches = self.x_embedder(x)
        if not self.use_rope:
            x_patches = x_patches + self.pos_embed
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(x_patches.shape[0], -1, -1)
            x = torch.cat([x_patches, reg], dim=1)
        else:
            x = x_patches
        if control_image is not None and self.control_encoder is not None and control_scale > 0:
            control_feat = self.control_encoder(control_image)
            if control_feat.shape[1] == x.shape[1]:
                x = x + control_scale * control_feat
        t_emb = self.t_embedder(t)
        text_emb = self.text_embedder(encoder_hidden_states, train=self.training)
        if conditioning_scale != 1.0:
            text_emb = text_emb * conditioning_scale
        token_weights = kwargs.get("token_weights")
        if token_weights is not None:
            w = token_weights.to(text_emb.device).to(text_emb.dtype)
            if w.dim() == 1:
                w = w.view(1, -1, 1)
            text_emb = text_emb * w
        if encoder_hidden_states_negative is not None and negative_prompt_weight > 0:
            neg_emb = self.text_embedder(encoder_hidden_states_negative, train=False)
            text_emb = text_emb - negative_prompt_weight * neg_emb
        if style_embedding is not None and self.style_proj is not None and style_strength > 0:
            style_emb = self.style_proj(style_embedding)
            if style_emb.dim() == 2:
                style_emb = style_emb.unsqueeze(1).expand(-1, text_emb.size(1), -1)
            text_emb = text_emb + style_strength * style_emb
        c = t_emb
        creativity = kwargs.get("creativity")
        if creativity is not None and self.creativity_proj is not None:
            cre = self.creativity_proj(creativity.unsqueeze(1).to(c.dtype))
            c = c + cre
        attn_mask = getattr(self, "_ar_mask", None)
        for block in self.blocks:
            if self._grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, text_emb, attn_mask,
                    use_reentrant=False,
                    num_patch_tokens=self.num_patches,
                )
            else:
                x = block(
                    x,
                    c,
                    text_emb,
                    attn_mask=attn_mask,
                    use_xformers=self.use_xformers,
                    num_patch_tokens=self.num_patches,
                )
        # REPA: expose projected token features (pool over patches) before final output mapping.
        x_out = x[:, : self.num_patches, :] if x.shape[1] > self.num_patches else x
        if self.repa_head is not None:
            pooled = x_out.mean(dim=1)  # (B, hidden_size)
            self._repa_projected = self.repa_head(pooled)  # (B, repa_out_dim)
        x = self.final_layer(x_out, c)
        if getattr(self, "_moe_layers", None):
            aux_vals = [m.last_aux_loss for m in self._moe_layers if m.last_aux_loss is not None]
            self.moe_aux_loss = sum(aux_vals) if aux_vals else None
        return self.unpatchify(x)


class DiT_Supreme_Text(nn.Module):
    """Supreme DiT: RMSNorm + QK-norm (self & cross) + SwiGLU + AdaLN-Zero + optional SizeEmbedder.
    Best practices from Z-Image, PixArt-sigma, Lumina Next-DiT. Same conditioning API as DiT_Text."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        text_dim=4096,
        max_text_len=300,
        class_dropout_prob=0.1,
        learn_sigma=True,
        num_ar_blocks=0,
        use_xformers=True,
        style_embed_dim=0,
        control_cond_dim=0,
        creativity_embed_dim=0,
        size_embed_dim=0,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        # REPA (Representation Alignment)
        repa_out_dim: int = 0,
        repa_projector_hidden_dim: int = 0,
        # ViT-Gen / hybrid SSM swap:
        ssm_every_n: int = 0,
        ssm_kernel_size: int = 7,
        # ViT-Gen features (register tokens + token routing; rope/kv merge not yet applied here)
        num_register_tokens: int = 0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        kv_merge_factor: int = 1,
        token_routing_enabled: bool = False,
        token_routing_strength: float = 1.0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.text_dim = text_dim
        self._grad_checkpointing = False
        self.num_ar_blocks = num_ar_blocks
        self.use_xformers = use_xformers
        self.style_embed_dim = style_embed_dim
        self.control_cond_dim = control_cond_dim
        self.creativity_embed_dim = creativity_embed_dim
        self.size_embed_dim = size_embed_dim
        self._moe_num_experts = int(moe_num_experts) if moe_num_experts is not None else 0
        self._moe_top_k = int(moe_top_k) if moe_top_k is not None else 2

        # REPA projector: maps DiT hidden tokens -> vision encoder embedding dim.
        self.repa_out_dim = int(repa_out_dim) if repa_out_dim is not None else 0
        self.repa_projector_hidden_dim = int(repa_projector_hidden_dim) if repa_projector_hidden_dim is not None else 0
        self.repa_head = None
        if self.repa_out_dim > 0:
            if self.repa_projector_hidden_dim > 0:
                self.repa_head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, self.repa_projector_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.repa_projector_hidden_dim, self.repa_out_dim),
                )
            else:
                self.repa_head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, self.repa_out_dim),
                )
        self._repa_projected = None

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_embedder = TextEmbedder(text_dim, hidden_size, class_dropout_prob)
        if style_embed_dim > 0:
            self.style_proj = nn.Linear(style_embed_dim, hidden_size)
        else:
            self.style_proj = None
        if creativity_embed_dim > 0:
            self.creativity_proj = nn.Sequential(
                nn.Linear(1, max(16, creativity_embed_dim // 2)),
                nn.SiLU(),
                nn.Linear(max(16, creativity_embed_dim // 2), hidden_size),
            )
        else:
            self.creativity_proj = None
        if size_embed_dim > 0:
            from .pixart_blocks import SizeEmbedder
            self.size_embedder = SizeEmbedder(size_embed_dim, concat_dims=False)
            self.size_proj = nn.Linear(size_embed_dim, hidden_size)
        else:
            self.size_embedder = None
            self.size_proj = None
        if control_cond_dim > 0:
            self.control_encoder = ControlNetEncoder(
                control_size=input_size, patch_size=patch_size, in_channels=3, hidden_size=hidden_size
            )
        else:
            self.control_encoder = None
        # ViT-Gen feature flags
        self.num_patches = int(self.x_embedder.num_patches)
        self.num_register_tokens = int(num_register_tokens)
        self.use_rope = bool(use_rope)
        self.kv_merge_factor = int(kv_merge_factor)  # not yet applied in supreme models
        self.rope_base = float(rope_base)  # not yet applied in supreme models
        self.token_routing_enabled = bool(token_routing_enabled)
        self.token_routing_strength = float(token_routing_strength)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.register_tokens = None
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, hidden_size))

        self.register_buffer("_ar_mask", None)
        if num_ar_blocks > 0:
            p = int(self.num_patches ** 0.5)
            self._ar_mask = create_block_causal_mask_2d(p, p, num_ar_blocks)

        blocks = []
        for i in range(depth):
            use_ssm = bool(ssm_every_n and int(ssm_every_n) > 0 and (i % int(ssm_every_n) == 0))
            blocks.append(
                DiTBlockSupreme(
                    hidden_size,
                    num_heads,
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    moe_num_experts=self._moe_num_experts,
                    moe_top_k=self._moe_top_k,
                    use_ssm=use_ssm,
                    ssm_kernel_size=ssm_kernel_size,
                    token_routing_enabled=self.token_routing_enabled,
                    token_routing_strength=self.token_routing_strength,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.use_xformers = use_xformers
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self._moe_layers = [m for m in self.modules() if hasattr(m, "last_aux_loss")]
        self.moe_aux_loss = None

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.text_embedder.proj.weight, std=0.02)
        if self.style_proj is not None:
            nn.init.normal_(self.style_proj.weight, std=0.02)
            nn.init.zeros_(self.style_proj.bias)
        if self.control_encoder is not None:
            nn.init.xavier_uniform_(self.control_encoder.proj.weight.view(self.control_encoder.proj.weight.shape[0], -1))
            nn.init.zeros_(self.control_encoder.proj.bias)
        if self.size_proj is not None:
            nn.init.normal_(self.size_proj.weight, std=0.02)
            nn.init.zeros_(self.size_proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], c, h * p, h * p))

    def enable_gradient_checkpointing(self):
        self._grad_checkpointing = True

    def forward(
        self,
        x,
        t,
        encoder_hidden_states=None,
        encoder_hidden_states_negative=None,
        negative_prompt_weight=0.5,
        style_embedding=None,
        style_strength=0.7,
        control_image=None,
        control_scale=1.0,
        conditioning_scale=1.0,
        return_attn=False,
        **kwargs,
    ):
        self._repa_projected = None
        if encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get("y_embed")
        assert encoder_hidden_states is not None, "encoder_hidden_states required"
        if getattr(self, "_moe_layers", None):
            self.moe_aux_loss = None
            for m in self._moe_layers:
                m.last_aux_loss = None
        x_patches = self.x_embedder(x)
        if not self.use_rope:
            x_patches = x_patches + self.pos_embed
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(x_patches.shape[0], -1, -1)
            x = torch.cat([x_patches, reg], dim=1)
        else:
            x = x_patches
        if control_image is not None and self.control_encoder is not None and control_scale > 0:
            control_feat = self.control_encoder(control_image)
            if control_feat.shape[1] == x.shape[1]:
                x = x + control_scale * control_feat
        t_emb = self.t_embedder(t)
        text_emb = self.text_embedder(encoder_hidden_states, train=self.training)
        if conditioning_scale != 1.0:
            text_emb = text_emb * conditioning_scale
        token_weights = kwargs.get("token_weights")
        if token_weights is not None:
            w = token_weights.to(text_emb.device).to(text_emb.dtype)
            if w.dim() == 1:
                w = w.view(1, -1, 1)
            text_emb = text_emb * w
        if encoder_hidden_states_negative is not None and negative_prompt_weight > 0:
            neg_emb = self.text_embedder(encoder_hidden_states_negative, train=False)
            text_emb = text_emb - negative_prompt_weight * neg_emb
        if style_embedding is not None and self.style_proj is not None and style_strength > 0:
            style_emb = self.style_proj(style_embedding)
            if style_emb.dim() == 2:
                style_emb = style_emb.unsqueeze(1).expand(-1, text_emb.size(1), -1)
            text_emb = text_emb + style_strength * style_emb
        c = t_emb
        size_embed = kwargs.get("size_embed")
        if size_embed is not None and self.size_embedder is not None and self.size_proj is not None:
            bs = x.shape[0]
            size_emb = self.size_embedder(size_embed, bs)
            c = c + self.size_proj(size_emb)
        creativity = kwargs.get("creativity")
        if creativity is not None and self.creativity_proj is not None:
            cre = self.creativity_proj(creativity.unsqueeze(1).to(c.dtype))
            c = c + cre
        attn_mask = getattr(self, "_ar_mask", None)
        for block in self.blocks:
            if self._grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, text_emb, attn_mask,
                    use_reentrant=False,
                    num_patch_tokens=self.num_patches,
                )
            else:
                x = block(
                    x,
                    c,
                    text_emb,
                    attn_mask=attn_mask,
                    use_xformers=self.use_xformers,
                    num_patch_tokens=self.num_patches,
                )
        # REPA: expose projected token features (pool over patches) before final output mapping.
        x_out = x[:, : self.num_patches, :] if x.shape[1] > self.num_patches else x
        if self.repa_head is not None:
            pooled = x_out.mean(dim=1)  # (B, hidden_size)
            self._repa_projected = self.repa_head(pooled)  # (B, repa_out_dim)
        x = self.final_layer(x_out, c)
        if getattr(self, "_moe_layers", None):
            aux_vals = [m.last_aux_loss for m in self._moe_layers if m.last_aux_loss is not None]
            self.moe_aux_loss = sum(aux_vals) if aux_vals else None
        return self.unpatchify(x)


def DiT_P_2_Text(**kwargs):
    """Predecessor: depth=32, hidden=1280, heads=20 (deeper/wider than DiT-XL)."""
    return DiT_Predecessor_Text(
        depth=32,
        hidden_size=1280,
        patch_size=2,
        num_heads=20,
        mlp_ratio=4.0,
        **kwargs,
    )


def DiT_P_L_2_Text(**kwargs):
    """Predecessor medium: depth=28, hidden=1152 (same size as DiT-XL but with QK-norm/SwiGLU/AdaLN-Zero)."""
    return DiT_Predecessor_Text(
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )


def DiT_Supreme_2_Text(**kwargs):
    """Supreme: depth=28, hidden=1152 (XL-sized, RMSNorm + QK-norm self+cross + SwiGLU + AdaLN-Zero)."""
    return DiT_Supreme_Text(
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )


def DiT_Supreme_L_2_Text(**kwargs):
    """Supreme Large: depth=32, hidden=1280 (deeper/wider, same recipe)."""
    return DiT_Supreme_Text(
        depth=32,
        hidden_size=1280,
        patch_size=2,
        num_heads=20,
        mlp_ratio=4.0,
        **kwargs,
    )
