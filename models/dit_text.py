# DiT with text conditioning: styles, ControlNet, ref, negative prompt.
# Blends style + control + prompt without sloppy output.
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, PatchEmbed

from .attention import (
    SelfAttention,
    SSMTokenMixer,
    create_block_causal_mask_2d,
    memory_efficient_attention,
)
from .controlnet import ControlNetEncoder, encode_control_stack, inject_control_tokens
from .dit import FinalLayer, TimestepEmbedder, get_2d_sincos_pos_embed
from .model_enhancements import DropPath
from .moe import MoEFeedForward, MoERouter
from .pixart_blocks import SizeEmbedder, ZeroInitPatchChannelGate
from .vit_next_blocks import LayerScale, apply_topk_token_keep


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttention(nn.Module):
    """Cross-attention (query=spatial, key/value=text) with xformers when available, optional QK-norm."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        text_dim,
        dropout=0.0,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(text_dim, hidden_size)
        self.v_proj = nn.Linear(text_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.moe_out_proj = None
        if moe_num_experts and int(moe_num_experts) > 0:
            from .moe import MoEProjection

            self.moe_out_proj = MoEProjection(
                hidden_size,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
            )
        self.dropout = nn.Dropout(dropout)
        # QK-norm for cross-attention (stabilises text-image alignment at high resolutions).
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else None

    def forward(
        self,
        x,
        text_emb,
        use_xformers=True,
        return_weights=False,
        routing_context=None,
        router_override=None,
        report_aux_loss: bool = False,
    ):
        B, N, C = x.shape
        _, L, _ = text_emb.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        # QK-norm: normalise per head before attention.
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if return_weights:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
            attn_weights = attn.softmax(dim=-1)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).reshape(B, N, C)
            if self.moe_out_proj is not None:
                out = self.moe_out_proj(
                    out,
                    routing_context=routing_context,
                    router_override=router_override,
                    report_aux_loss=report_aux_loss,
                )
            else:
                out = self.out_proj(out)
            return self.dropout(out), attn_weights
        out = memory_efficient_attention(q, k, v, attn_mask=None, scale=self.scale, use_xformers=use_xformers)
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


class DiTBlockText(nn.Module):
    """DiT block: self-attn (xformers + optional block AR mask) + cross-attn to text."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        text_dim,
        mlp_ratio=4.0,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        use_ssm: bool = False,
        ssm_kernel_size: int = 7,
        # ViT-Gen features
        use_rope: bool = False,
        rope_base: float = 10000.0,
        kv_merge_factor: int = 1,
        token_routing_enabled: bool = False,
        token_routing_strength: float = 1.0,
        token_keep_ratio: float = 1.0,
        token_keep_min_value: float = 0.0,
        drop_path: float = 0.0,
        layerscale_init: float = 0.0,
        # QK-norm (SD3.5-style training stability)
        qk_norm: bool = False,
        # Prompt adherence options
        prompt_reinject_every_n: int = 0,
        prompt_reinject_alpha: float = 0.0,
        prompt_reinject_decay: float = 1.0,
        prompt_timestep_schedule_enabled: bool = False,
        prompt_early_scale: float = 1.1,
        prompt_late_scale: float = 1.0,
        prompt_schedule_num_timesteps: int = 1000,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_ssm:
            # Current SSM mixer is a simple token mixer and doesn't implement MoE projections.
            # Keep it compatible by disallowing MoE in the self-attention path.
            if moe_num_experts is not None and int(moe_num_experts) > 0:
                raise ValueError("use_ssm enabled but moe_num_experts>0: disable MoE or disable SSM for now.")
            self.attn = SSMTokenMixer(hidden_size, kernel_size=ssm_kernel_size, dropout=0.0)
        else:
            self.attn = SelfAttention(
                hidden_size,
                num_heads,
                dropout=0.0,
                moe_num_experts=int(moe_num_experts) if moe_num_experts is not None else 0,
                moe_top_k=int(moe_top_k) if moe_top_k is not None else 2,
                use_rope=use_rope,
                rope_base=rope_base,
                kv_merge_factor=kv_merge_factor,
                qk_norm=qk_norm,
            )
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads,
            text_dim,
            moe_num_experts=int(moe_num_experts) if moe_num_experts is not None else 0,
            moe_top_k=int(moe_top_k) if moe_top_k is not None else 2,
            qk_norm=qk_norm,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        if moe_num_experts and int(moe_num_experts) > 0:
            self.moe_router = MoERouter(
                hidden_size,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
            )
            self.mlp = MoEFeedForward(
                hidden_size,
                mlp_hidden,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
                dropout=0.0,
                act_layer=nn.GELU,
            )
        else:
            self.moe_router = None
            self.mlp = Mlp(hidden_size, hidden_features=mlp_hidden, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # Cross-scale token routing (soft gating per token; does not reduce compute graph yet).
        self.token_routing_enabled = bool(token_routing_enabled)
        self.token_routing_strength = float(token_routing_strength)
        self.prompt_reinject_every_n = int(prompt_reinject_every_n)
        self.prompt_reinject_alpha = float(prompt_reinject_alpha)
        self.prompt_reinject_decay = float(prompt_reinject_decay)
        self.prompt_timestep_schedule_enabled = bool(prompt_timestep_schedule_enabled)
        self.prompt_early_scale = float(prompt_early_scale)
        self.prompt_late_scale = float(prompt_late_scale)
        self.prompt_schedule_num_timesteps = max(2, int(prompt_schedule_num_timesteps))
        self.token_gate = None
        self.token_keep_ratio = max(0.05, min(1.0, float(token_keep_ratio)))
        self.token_keep_min_value = max(0.0, min(1.0, float(token_keep_min_value)))
        self.drop_path = DropPath(float(max(0.0, drop_path))) if float(drop_path) > 0 else nn.Identity()
        self.ls_attn = LayerScale(hidden_size, float(layerscale_init))
        self.ls_cross = LayerScale(hidden_size, float(layerscale_init))
        self.ls_mlp = LayerScale(hidden_size, float(layerscale_init))
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
        return_attn=False,
        num_patch_tokens: int = 0,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        token_gate = None
        if self.token_routing_enabled:
            token_gate = torch.sigmoid(self.token_router(x_norm))  # (B, N, 1)
            if num_patch_tokens and x.shape[1] > num_patch_tokens:
                # Never suppress register/scratchpad tokens.
                token_gate[:, num_patch_tokens:, :] = 1.0
            s = self.token_routing_strength
            token_gate = ((1.0 - s) + s * token_gate).clamp(0.0, 1.0)
            if self.token_keep_ratio < 1.0:
                token_gate = apply_topk_token_keep(
                    token_gate,
                    token_gate,
                    keep_ratio=self.token_keep_ratio,
                    num_patch_tokens=num_patch_tokens,
                    min_keep_value=self.token_keep_min_value,
                )
        else:
            token_gate = 1.0

        router_override = self.moe_router if self.moe_router is not None else None
        attn_out = self.attn(
            x_norm,
            attn_mask=attn_mask,
            use_xformers=use_xformers,
            routing_context=c,
            router_override=router_override,
            report_aux_loss=False,
            num_patch_tokens=num_patch_tokens if int(num_patch_tokens) > 0 else None,
        )

        x = x + self.drop_path(self.ls_attn(gate_msa.unsqueeze(1) * token_gate * attn_out))
        cross_in = self.norm_cross(x)
        # MLP input is computed after cross-attention modifies `x`.
        if return_attn:
            cross_out, attn_weights = self.cross_attn(
                cross_in,
                text_emb,
                use_xformers=False,
                return_weights=True,
                routing_context=c,
                router_override=router_override,
                report_aux_loss=False,
            )
            x = x + self.drop_path(self.ls_cross(token_gate * cross_out))
            mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
            if isinstance(self.mlp, MoEFeedForward):
                x = x + self.drop_path(self.ls_mlp(gate_mlp.unsqueeze(1) * token_gate * self.mlp(
                    mlp_in,
                    routing_context=c,
                    router_override=router_override,
                    report_aux_loss=True,
                )))
            else:
                x = x + self.drop_path(self.ls_mlp(gate_mlp.unsqueeze(1) * token_gate * self.mlp(mlp_in)))
            return x, attn_weights
        x = x + self.drop_path(self.ls_cross(token_gate * self.cross_attn(
            cross_in,
            text_emb,
            use_xformers=use_xformers,
            routing_context=c,
            router_override=router_override,
            report_aux_loss=False,
        )))
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if isinstance(self.mlp, MoEFeedForward):
            x = x + self.drop_path(self.ls_mlp(gate_mlp.unsqueeze(1) * token_gate * self.mlp(
                mlp_in,
                routing_context=c,
                router_override=router_override,
                report_aux_loss=True,
            )))
        else:
            x = x + self.drop_path(self.ls_mlp(gate_mlp.unsqueeze(1) * token_gate * self.mlp(mlp_in)))
        return x


class TextEmbedder(nn.Module):
    """Project text embeddings to hidden_size; optional dropout for CFG."""

    def __init__(self, text_dim, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_size)
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

    def forward(self, encoder_hidden_states, train=True, force_drop_ids=None):
        # encoder_hidden_states: (B, L, text_dim)
        out = self.proj(encoder_hidden_states)
        if train and self.dropout_prob > 0 and force_drop_ids is None:
            if torch.rand(1, device=out.device).item() < self.dropout_prob:
                out = torch.zeros_like(out, device=out.device, dtype=out.dtype)
        elif force_drop_ids is not None:
            out = torch.where(force_drop_ids.unsqueeze(1).unsqueeze(2), torch.zeros_like(out), out)
        return out


class DiT_Text(nn.Module):
    """DiT + text (PixArt-style). Negative prompt, styles, ControlNet, optional reference_tokens (CLIP→extra cross-attn), block AR, xformers."""

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
        ar_block_order: str = "raster",
        use_xformers=True,
        style_embed_dim=0,
        control_cond_dim=0,
        control_num_types: int = 0,
        creativity_embed_dim=0,
        size_embed_dim=0,
        patch_se: bool = False,
        patch_se_reduction: int = 8,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        # REPA (Representation Alignment)
        repa_out_dim: int = 0,
        repa_projector_hidden_dim: int = 0,
        # Which block index to hook for REPA alignment (-1 = after all blocks, default).
        # E.g. repa_layer=14 on a 28-block model hooks the middle layer where
        # semantic structure is most developed (per REPA paper findings).
        repa_layer: int = -1,
        # ViT-Gen / SSM swap:
        ssm_every_n: int = 0,
        ssm_kernel_size: int = 7,
        # ViT-Gen features
        num_register_tokens: int = 0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        kv_merge_factor: int = 1,
        token_routing_enabled: bool = False,
        token_routing_strength: float = 1.0,
        token_keep_ratio: float = 1.0,
        token_keep_min_value: float = 0.0,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 0.0,
        # QK-norm (SD3.5-style): normalise Q and K per head for training stability.
        # Recommended when training at high resolution or with large batch sizes.
        qk_norm: bool = False,
        # Prompt adherence (must match get_dit_build_kwargs)
        prompt_reinject_every_n: int = 0,
        prompt_reinject_alpha: float = 0.0,
        prompt_reinject_decay: float = 1.0,
        prompt_timestep_schedule_enabled: bool = False,
        prompt_early_scale: float = 1.1,
        prompt_late_scale: float = 1.0,
        prompt_schedule_num_timesteps: int = 1000,
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
        self.ar_block_order = str(ar_block_order or "raster")
        self.use_xformers = use_xformers
        self.style_embed_dim = style_embed_dim
        self.control_cond_dim = control_cond_dim
        self.control_num_types = int(max(0, control_num_types))
        self.creativity_embed_dim = creativity_embed_dim
        self.size_embed_dim = int(size_embed_dim) if size_embed_dim is not None else 0
        self._moe_num_experts = int(moe_num_experts) if moe_num_experts is not None else 0
        self._moe_top_k = int(moe_top_k) if moe_top_k is not None else 2

        # ViT-Gen feature flags
        self.num_register_tokens = int(num_register_tokens)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.kv_merge_factor = int(kv_merge_factor)
        self.token_routing_enabled = bool(token_routing_enabled)
        self.token_routing_strength = float(token_routing_strength)
        self.token_keep_ratio = float(token_keep_ratio)
        self.token_keep_min_value = float(token_keep_min_value)
        self.drop_path_rate = float(drop_path_rate)
        self.layerscale_init = float(layerscale_init)
        self.qk_norm = bool(qk_norm)
        self.prompt_reinject_every_n = int(prompt_reinject_every_n)
        self.prompt_reinject_alpha = float(prompt_reinject_alpha)
        self.prompt_reinject_decay = float(prompt_reinject_decay)
        self.prompt_timestep_schedule_enabled = bool(prompt_timestep_schedule_enabled)
        self.prompt_early_scale = float(prompt_early_scale)
        self.prompt_late_scale = float(prompt_late_scale)
        self.prompt_schedule_num_timesteps = max(2, int(prompt_schedule_num_timesteps))

        # REPA projector: maps DiT hidden tokens -> vision encoder embedding dim.
        self.repa_out_dim = int(repa_out_dim) if repa_out_dim is not None else 0
        self.repa_projector_hidden_dim = int(repa_projector_hidden_dim) if repa_projector_hidden_dim is not None else 0
        # repa_layer: which block index to hook for REPA (-1 = after all blocks).
        self.repa_layer = int(repa_layer) if repa_layer is not None else -1
        # Remove the stale init attribute if it was set.
        if hasattr(self, "_repa_layer_init"):
            del self._repa_layer_init
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
        if self.size_embed_dim > 0:
            self.size_embedder = SizeEmbedder(self.size_embed_dim, concat_dims=False)
            self.size_proj = nn.Linear(self.size_embed_dim, hidden_size)
        else:
            self.size_embedder = None
            self.size_proj = None
        self.patch_se = None
        if patch_se:
            self.patch_se = ZeroInitPatchChannelGate(hidden_size, reduction=max(2, int(patch_se_reduction)))
        if control_cond_dim > 0:
            self.control_encoder = ControlNetEncoder(
                control_size=input_size,
                patch_size=patch_size,
                in_channels=3,
                hidden_size=hidden_size,
                num_control_types=self.control_num_types,
            )
        else:
            self.control_encoder = None
        num_patches = self.x_embedder.num_patches
        self.num_patches = int(num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Learnable register tokens (scratchpad) appended to the patch token stream.
        self.register_tokens = None
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, hidden_size))

        # Block-wise AR mask (ACDiT-style): causal over spatial blocks
        self.register_buffer("_ar_mask", None)
        if num_ar_blocks > 0:
            p = int(num_patches**0.5)
            self._ar_mask = create_block_causal_mask_2d(p, p, num_ar_blocks, block_order=self.ar_block_order)

        # Blocks receive text_emb from text_embedder (already projected to hidden_size), not raw encoder_hidden_states
        blocks = []
        for i in range(depth):
            use_ssm = bool(ssm_every_n and int(ssm_every_n) > 0 and (i % int(ssm_every_n) == 0))
            dp = self.drop_path_rate * (float(i) / float(max(depth - 1, 1)))
            blocks.append(
                DiTBlockText(
                    hidden_size,
                    num_heads,
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    moe_num_experts=self._moe_num_experts,
                    moe_top_k=self._moe_top_k,
                    use_ssm=use_ssm,
                    ssm_kernel_size=ssm_kernel_size,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                    kv_merge_factor=self.kv_merge_factor,
                    token_routing_enabled=self.token_routing_enabled,
                    token_routing_strength=self.token_routing_strength,
                    token_keep_ratio=self.token_keep_ratio,
                    token_keep_min_value=self.token_keep_min_value,
                    drop_path=dp,
                    layerscale_init=self.layerscale_init,
                    qk_norm=self.qk_norm,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        # Collect MoE modules that expose last_aux_loss (FFN MoE + projection MoE).
        self._moe_layers = [m for m in self.modules() if hasattr(m, "last_aux_loss")]
        self.moe_aux_loss = None

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.text_embedder.proj.weight, std=0.02)
        if self.style_proj is not None:
            nn.init.normal_(self.style_proj.weight, std=0.02)
            nn.init.zeros_(self.style_proj.bias)
        if self.size_proj is not None:
            nn.init.normal_(self.size_proj.weight, std=0.02)
            nn.init.zeros_(self.size_proj.bias)
        if self.control_encoder is not None:
            nn.init.xavier_uniform_(
                self.control_encoder.proj.weight.view(self.control_encoder.proj.weight.shape[0], -1)
            )
            nn.init.zeros_(self.control_encoder.proj.bias)
            if getattr(self.control_encoder, "type_embed", None) is not None:
                nn.init.normal_(self.control_encoder.type_embed.weight, std=0.02)
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
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

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
        control_type=None,
        control_scale=1.0,
        conditioning_scale=1.0,
        creativity=None,
        return_attn=False,
        **kwargs,
    ):
        # Clear aux loss each forward to avoid stale values.
        if getattr(self, "_moe_layers", None):
            self.moe_aux_loss = None
            for m in self._moe_layers:
                m.last_aux_loss = None
        self._repa_projected = None
        # x: (N, C, H, W), t: (N,), encoder_hidden_states: (N, L, text_dim)
        if encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get("y_embed")
        assert encoder_hidden_states is not None, "encoder_hidden_states required"
        _b, _c_in, h_lat, w_lat = x.shape[0], x.shape[1], int(x.shape[2]), int(x.shape[3])
        x_patches = self.x_embedder(x)
        if not self.use_rope:
            x_patches = x_patches + self.pos_embed
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(x_patches.shape[0], -1, -1)
            x = torch.cat([x_patches, reg], dim=1)
        else:
            x = x_patches
        # ControlNet: inject into patch tokens only (keeps register tokens intact).
        if control_image is not None and self.control_encoder is not None:
            control_feat = encode_control_stack(
                self.control_encoder,
                control_image,
                target_hw=(h_lat, w_lat),
                control_type=control_type,
                control_scale=control_scale,
            )
            x = inject_control_tokens(
                x,
                control_feat,
                control_scale=1.0,
                patch_tokens=self.num_patches,
            )
        if self.patch_se is not None:
            x = self.patch_se(x)
        t_emb = self.t_embedder(t)
        text_emb = self.text_embedder(encoder_hidden_states, train=self.training)
        if conditioning_scale != 1.0:
            text_emb = text_emb * conditioning_scale
        # Optional timestep-aware text scaling:
        # high-noise steps get stronger text pull, later steps relax to preserve texture/detail.
        if bool(kwargs.get("prompt_timestep_schedule_enabled", self.prompt_timestep_schedule_enabled)):
            early = float(kwargs.get("prompt_early_scale", self.prompt_early_scale))
            late = float(kwargs.get("prompt_late_scale", self.prompt_late_scale))
            nt = float(kwargs.get("prompt_schedule_num_timesteps", self.prompt_schedule_num_timesteps))
            tau = (t.to(text_emb.device).float() / max(1.0, nt - 1.0)).clamp(0.0, 1.0).view(-1, 1, 1)
            ts = late + (early - late) * tau
            text_emb = text_emb * ts.to(dtype=text_emb.dtype)
        # IMPROVEMENTS 2.2: per-token emphasis (word) -> 1.2, [word] -> 0.8
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
        # IP-Adapter-style: optional image-derived tokens (same dim as text_emb after TextEmbedder).
        ref_tok = kwargs.get("reference_tokens")
        if ref_tok is not None:
            rs = float(kwargs.get("reference_scale", 1.0) or 0.0)
            if rs > 0.0:
                rt = ref_tok.to(device=text_emb.device, dtype=text_emb.dtype)
                if rt.shape[0] != text_emb.shape[0]:
                    if rt.shape[0] == 1:
                        rt = rt.expand(text_emb.shape[0], -1, -1)
                    else:
                        raise ValueError(
                            "reference_tokens batch must match encoder_hidden_states batch "
                            f"({rt.shape[0]} vs {text_emb.shape[0]})"
                        )
                text_emb = torch.cat([text_emb, rt * rs], dim=1)
        c = t_emb
        size_embed = kwargs.get("size_embed")
        if size_embed is None and self.size_embedder is not None and self.size_proj is not None:
            # Default: latent grid (H, W) from input x before patchify (supports non-square later).
            size_embed = torch.stack(
                [
                    torch.full((_b,), float(h_lat), device=x.device, dtype=torch.float32),
                    torch.full((_b,), float(w_lat), device=x.device, dtype=torch.float32),
                ],
                dim=1,
            )
        if size_embed is not None and self.size_embedder is not None and self.size_proj is not None:
            bs = x.shape[0]
            size_emb = self.size_embedder(size_embed.to(device=x.device, dtype=torch.float32), bs)
            c = c + self.size_proj(size_emb.to(dtype=c.dtype))
        creativity = kwargs.get("creativity")
        if creativity is not None and self.creativity_proj is not None:
            # creativity: (B,) in [0, 1]; add to conditioning
            cre = self.creativity_proj(creativity.unsqueeze(1).to(c.dtype))
            c = c + cre
        attn_mask = getattr(self, "_ar_mask", None)
        captured_attn = None
        base_text_emb = text_emb
        reinject_every = int(kwargs.get("prompt_reinject_every_n", self.prompt_reinject_every_n))
        reinject_alpha = float(kwargs.get("prompt_reinject_alpha", self.prompt_reinject_alpha))
        reinject_decay = float(kwargs.get("prompt_reinject_decay", self.prompt_reinject_decay))
        for i, block in enumerate(self.blocks):
            text_emb_i = text_emb
            if reinject_every > 0 and reinject_alpha > 0.0 and i > 0 and (i % reinject_every == 0):
                k = max(0, i // reinject_every - 1)
                a = reinject_alpha * (reinject_decay**k)
                text_emb_i = text_emb + float(a) * base_text_emb
            if return_attn and i == 0:
                if self._grad_checkpointing and self.training:
                    raise NotImplementedError("return_attn with grad_checkpointing")
                x, captured_attn = block(
                    x,
                    c,
                    text_emb_i,
                    attn_mask=attn_mask,
                    use_xformers=self.use_xformers,
                    return_attn=True,
                    num_patch_tokens=self.num_patches,
                )
            elif self._grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    c,
                    text_emb_i,
                    attn_mask,
                    use_reentrant=False,
                    num_patch_tokens=self.num_patches,
                )
            else:
                x = block(
                    x,
                    c,
                    text_emb_i,
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
        x = self.unpatchify(x)
        if getattr(self, "_moe_layers", None):
            aux_vals = [m.last_aux_loss for m in self._moe_layers if m.last_aux_loss is not None]
            self.moe_aux_loss = sum(aux_vals) if aux_vals else None
        if return_attn:
            return x, captured_attn
        return x


def DiT_XL_2_Text(**kwargs):
    return DiT_Text(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_L_2_Text(**kwargs):
    return DiT_Text(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_B_2_Text(**kwargs):
    return DiT_Text(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


DiT_models_text = {
    "DiT-XL/2-Text": DiT_XL_2_Text,
    "DiT-L/2-Text": DiT_L_2_Text,
    "DiT-B/2-Text": DiT_B_2_Text,
}
