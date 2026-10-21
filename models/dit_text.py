"""DiT with text conditioning — the text-to-image denoiser (PixArt-style).

Where the base ``models/dit.py`` conditions on a single class label (added into the
timestep vector), this variant conditions on a **sequence of text embeddings** (from
T5/CLIP) via **cross-attention** in every block: spatial patch tokens are the query,
the text tokens are the keys/values.

On top of that:
  - classifier-free guidance via text dropout (see ``TextEmbedder``) and negative prompts,
  - ControlNet-style structural control and reference-image tokens (extra cross-attn),
  - style conditioning, MoE feed-forwards, and optional block-autoregressive masking,
  - opt-in TACA cross-attention, SwiGLU MLPs and ``HybridDiTExtras`` (U-ViT skips, MMDiT
    joint attention, Mix-FFN, Hi-DiT HF residual) — see ``models/hybrid_dit.py``.

``DiT_Predecessor_Text`` / ``DiT_Supreme_Text`` (``DiT-P*`` / ``DiT-Supreme*``) are the same trunk with
QK-norm, full-width SwiGLU and (Supreme) RMSNorm blocks.

Reuses ``TimestepEmbedder``, ``FinalLayer`` and the sin/cos position embeddings from
``models/dit.py`` so the timestep/output machinery stays identical to the base model.
"""

import functools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from timm.models.vision_transformer import Mlp, PatchEmbed

from .ar_masks_extended import create_block_causal_mask_2d
from .attention import SelfAttention, SSMTokenMixer, memory_efficient_attention
from .controlnet import ControlNetEncoder, encode_control_stack, inject_control_tokens
from .dit import (
    FinalLayer,
    LayerScale,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    grid_pos_embed,
    modulate,
    timestep_embedding,
    unpatchify,
    xavier_init_linears,
)
from .hybrid_dit import HybridDiTExtras
from .layout_attn_context import layout_cross_attn_bias
from .moe import MoEExperts, MoEFeedForward, MoEProjection, MoERouter
from .taca import TACA


@functools.lru_cache(maxsize=8)
def _block_causal_mask(hp: int, wp: int, num_ar_blocks: int, order: str, device, dtype) -> torch.Tensor:
    return create_block_causal_mask_2d(hp, wp, num_ar_blocks, block_order=order).to(device=device, dtype=dtype)


def grid_ar_mask(model: nn.Module, hp: int, wp: int) -> torch.Tensor | None:
    """Block-AR self-attention mask for an ``hp x wp`` patch grid, ``None`` when AR is off.

    The native grid uses the model's ``_ar_mask`` buffer; any other grid gets the same block order built
    for its own shape.
    """
    mask = model._ar_mask
    if mask is None or (hp, wp) == tuple(model.x_embedder.grid_size):
        return mask
    return _block_causal_mask(hp, wp, int(model.num_ar_blocks), str(model.ar_block_order), mask.device, mask.dtype)


def _keep_topk_tokens(gate: torch.Tensor, keep_ratio: float, num_patch_tokens: int, min_keep_value: float):
    """Keep the ``keep_ratio`` highest-gated patch tokens and set the rest to ``min_keep_value`` (shape kept)."""
    b, n, _ = gate.shape
    p = max(1, min(int(num_patch_tokens) or n, n))
    k = max(1, min(p, round(keep_ratio * p)))
    idx = torch.topk(gate[:, :p, 0], k=k, dim=1).indices
    hard = torch.zeros((b, p), device=gate.device, dtype=gate.dtype).scatter_(1, idx, 1.0).unsqueeze(-1)
    out = gate[:, :p] * hard + float(min_keep_value) * (1.0 - hard)
    return out if n == p else torch.cat([out, gate[:, p:]], dim=1)


class SizeEmbedder(nn.Module):
    """PixArt size conditioning: sinusoidal MLP embedding of each size scalar (e.g. latent H, W).

    ``concat_dims=False`` averages the per-dimension embeddings into one ``hidden_size`` vector.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, concat_dims: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self.concat_dims = concat_dims

    def forward(self, s: torch.Tensor, batch_size: int) -> torch.Tensor:
        """``s``: ``(B, num_dims)`` or ``(num_dims,)``; a smaller batch is tiled to ``batch_size``."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if s.shape[0] != batch_size:
            s = s.repeat((batch_size + s.shape[0] - 1) // s.shape[0], 1)[:batch_size]
        b, num_dims = s.shape
        s_emb = self.mlp(timestep_embedding(s.reshape(-1), self.frequency_embedding_size))
        s_emb = s_emb.reshape(b, num_dims * self.hidden_size)
        if not self.concat_dims:
            s_emb = s_emb.mean(dim=1, keepdim=True).expand(-1, self.hidden_size)
        return s_emb


class ZeroInitPatchChannelGate(nn.Module):
    """SE-style channel gate on patch tokens, ``x + x * tanh(g)``; ``g`` is zero at init so it starts as identity."""

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        mid = max(dim // max(1, reduction), 32)
        self.norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.fc1 = nn.Linear(dim, mid)
        self.fc2 = nn.Linear(mid, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.fc2(F.gelu(self.fc1(self.norm(x.mean(dim=1)))))
        return x + x * torch.tanh(g).unsqueeze(1)


class CrossAttention(nn.Module):
    """Cross-attention (query=spatial, key/value=text) with xformers when available, optional QK-norm.

    Honors the Dense Diffusion layout bias set via ``models.layout_attn_context``.
    """

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
        qk_norm_eps: float | None = None,
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
            self.moe_out_proj = MoEProjection(hidden_size, num_experts=int(moe_num_experts), top_k=int(moe_top_k))
        self.dropout = nn.Dropout(dropout)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None

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
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        bias = layout_cross_attn_bias(B, N, L, device=x.device, dtype=q.dtype)
        attn_weights = None
        if return_weights:
            logits = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)) * self.scale
            if bias is not None:
                logits = logits + bias
            attn_weights = logits.softmax(dim=-1)
            out = (attn_weights @ v.transpose(1, 2)).transpose(1, 2)
        else:
            out = memory_efficient_attention(
                q, k, v, attn_mask=bias, scale=self.scale, use_xformers=use_xformers and bias is None
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
        out = self.dropout(out)
        return (out, attn_weights) if return_weights else out


class SwiGLUMlp(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.gate = nn.Linear(hidden_size, mlp_hidden)
        self.up = nn.Linear(hidden_size, mlp_hidden)
        self.down = nn.Linear(mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DiTBlockText(nn.Module):
    """adaLN-Zero block: self-attn (block-AR mask, RoPE, PAG), cross-attn to text (or TACA), FFN.

    ``norm="rms"`` swaps the non-affine LayerNorms for affine RMSNorm. ``ffn`` is ``"gelu"``, ``"swiglu"``
    (2/3 width, same parameter count as the GELU MLP) or ``"swiglu_wide"`` (full ``mlp_ratio``; MoE experts
    become SwiGLU too). ``cross_qk_norm`` defaults to ``qk_norm``.
    """

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
        use_rope: bool = False,
        rope_base: float = 10000.0,
        kv_merge_factor: int = 1,
        token_routing_enabled: bool = False,
        token_routing_strength: float = 1.0,
        token_keep_ratio: float = 1.0,
        token_keep_min_value: float = 0.0,
        drop_path: float = 0.0,
        layerscale_init: float = 0.0,
        qk_norm: bool = False,
        cross_qk_norm: bool | None = None,
        qk_norm_eps: float | None = None,
        norm: str = "layer",
        use_taca: bool = False,
        ffn: str = "gelu",
    ):
        super().__init__()
        n_experts = int(moe_num_experts or 0)
        top_k = int(moe_top_k)

        def make_norm() -> nn.Module:
            if norm == "rms":
                return nn.RMSNorm(hidden_size, eps=1e-6)
            return nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.norm1 = make_norm()
        if use_ssm:
            if n_experts > 0:
                raise ValueError("use_ssm enabled but moe_num_experts>0: disable MoE or disable SSM for now.")
            self.attn = SSMTokenMixer(hidden_size, kernel_size=ssm_kernel_size, dropout=0.0)
        else:
            self.attn = SelfAttention(
                hidden_size,
                num_heads,
                moe_num_experts=n_experts,
                moe_top_k=top_k,
                use_rope=use_rope,
                rope_base=rope_base,
                kv_merge_factor=kv_merge_factor,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
            )
        self.norm_cross = make_norm()
        cross_qk = qk_norm if cross_qk_norm is None else cross_qk_norm
        self.use_taca = bool(use_taca)
        if self.use_taca:
            self.cross_attn = TACA(hidden_size, text_dim, num_heads, qk_norm=cross_qk)
        else:
            self.cross_attn = CrossAttention(
                hidden_size,
                num_heads,
                text_dim,
                moe_num_experts=n_experts,
                moe_top_k=top_k,
                qk_norm=cross_qk,
                qk_norm_eps=qk_norm_eps,
            )
        self.norm2 = make_norm()
        self.moe_router = None
        if n_experts > 0:
            self.moe_router = MoERouter(hidden_size, num_experts=n_experts, top_k=top_k)
            if ffn == "swiglu_wide":
                experts = nn.ModuleList([SwiGLUMlp(hidden_size, mlp_ratio=mlp_ratio) for _ in range(n_experts)])
                self.mlp = MoEExperts(hidden_size, experts, top_k=top_k)
            else:
                self.mlp = MoEFeedForward(
                    hidden_size,
                    int(hidden_size * mlp_ratio),
                    num_experts=n_experts,
                    top_k=top_k,
                    dropout=0.0,
                    act_layer=nn.GELU,
                )
        elif ffn == "swiglu":
            self.mlp = SwiGLUMlp(hidden_size, mlp_ratio=mlp_ratio * 2 / 3)
        elif ffn == "swiglu_wide":
            self.mlp = SwiGLUMlp(hidden_size, mlp_ratio=mlp_ratio)
        else:
            self.mlp = Mlp(hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # Soft per-token gate (does not shrink the compute graph); register tokens are never gated.
        self.token_routing_enabled = bool(token_routing_enabled)
        self.token_routing_strength = max(0.0, min(1.0, float(token_routing_strength)))
        self.token_keep_ratio = max(0.05, min(1.0, float(token_keep_ratio)))
        self.token_keep_min_value = max(0.0, min(1.0, float(token_keep_min_value)))
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0 else nn.Identity()
        self.ls_attn = LayerScale(hidden_size, float(layerscale_init))
        self.ls_cross = LayerScale(hidden_size, float(layerscale_init))
        self.ls_mlp = LayerScale(hidden_size, float(layerscale_init))
        self.token_router = None
        if self.token_routing_enabled:
            self.token_router = nn.Sequential(make_norm(), nn.Linear(hidden_size, 1))

    def forward(
        self,
        x,
        c,
        text_emb,
        attn_mask=None,
        use_xformers=True,
        return_attn=False,
        num_patch_tokens: int = 0,
        timestep=None,
        identity_attn_from: int | None = None,
        grid_hw: tuple[int, int] | None = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        token_gate = 1.0
        if self.token_router is not None:
            token_gate = torch.sigmoid(self.token_router(x_norm))
            if num_patch_tokens and x.shape[1] > num_patch_tokens:
                # Not in-place: sigmoid saves its output for backward.
                registers = torch.ones_like(token_gate[:, num_patch_tokens:])
                token_gate = torch.cat([token_gate[:, :num_patch_tokens], registers], dim=1)
            s = self.token_routing_strength
            token_gate = ((1.0 - s) + s * token_gate).clamp(0.0, 1.0)
            if self.token_keep_ratio < 1.0:
                token_gate = _keep_topk_tokens(
                    token_gate, self.token_keep_ratio, num_patch_tokens, self.token_keep_min_value
                )

        router_override = self.moe_router
        attn_kw = {}  # SSMTokenMixer takes neither grid_hw nor identity_from.
        if isinstance(self.attn, SelfAttention):
            attn_kw["grid_hw"] = grid_hw
            if identity_attn_from is not None:
                attn_kw["identity_from"] = identity_attn_from
        attn_out = self.attn(
            x_norm,
            attn_mask=attn_mask,
            use_xformers=use_xformers,
            routing_context=c,
            router_override=router_override,
            report_aux_loss=False,
            num_patch_tokens=num_patch_tokens if int(num_patch_tokens) > 0 else None,
            **attn_kw,
        )
        x = x + self.drop_path(self.ls_attn(gate_msa.unsqueeze(1) * token_gate * attn_out))

        cross_in = self.norm_cross(x)
        if self.use_taca:
            cross_out = self.cross_attn(
                cross_in, text_emb, timestep, use_xformers=use_xformers, return_weights=return_attn
            )
        else:
            cross_out = self.cross_attn(
                cross_in,
                text_emb,
                use_xformers=use_xformers,
                return_weights=return_attn,
                routing_context=c,
                router_override=router_override,
                report_aux_loss=False,
            )
        attn_weights = None
        if return_attn:
            cross_out, attn_weights = cross_out
        x = x + self.drop_path(self.ls_cross(token_gate * cross_out))

        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if self.moe_router is not None:
            mlp_out = self.mlp(mlp_in, routing_context=c, router_override=router_override, report_aux_loss=True)
        else:
            mlp_out = self.mlp(mlp_in)
        x = x + self.drop_path(self.ls_mlp(gate_mlp.unsqueeze(1) * token_gate * mlp_out))
        return (x, attn_weights) if return_attn else x


class TextEmbedder(nn.Module):
    """Project text embeddings to hidden_size; optional per-sample caption dropout for CFG.

    The in-model null is a zero vector, which differs from the empty-prompt encoding sample.py uses as the CFG
    uncond; train.py therefore drops captions to "" before encoding and builds models with ``dropout_prob=0``.
    """

    def __init__(self, text_dim, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_size)
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

    def forward(self, encoder_hidden_states, train=True, force_drop_ids=None):
        out = self.proj(encoder_hidden_states)
        if force_drop_ids is None and train and self.dropout_prob > 0:
            force_drop_ids = torch.rand(out.shape[0], device=out.device) < self.dropout_prob
        if force_drop_ids is not None:
            out = torch.where(force_drop_ids.view(-1, 1, 1), torch.zeros_like(out), out)
        return out


class DiT_Text(nn.Module):
    """Text-conditioned DiT (see module docstring). Optional ``forward`` kwargs: ``token_weights``,
    ``reference_tokens``/``reference_scale``, ``size_embed``, ``block_cache``, ``skip_blocks``,
    ``pag_blocks``/``pag_from`` and per-call ``prompt_*`` overrides."""

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
        repa_out_dim: int = 0,
        repa_projector_hidden_dim: int = 0,
        ssm_every_n: int = 0,
        ssm_kernel_size: int = 7,
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
        qk_norm: bool = False,
        prompt_reinject_every_n: int = 0,
        prompt_reinject_alpha: float = 0.0,
        prompt_reinject_decay: float = 1.0,
        prompt_timestep_schedule_enabled: bool = False,
        prompt_early_scale: float = 1.1,
        prompt_late_scale: float = 1.0,
        prompt_schedule_num_timesteps: int = 1000,
        use_taca: bool = False,
        use_swiglu: bool = False,
        uvit_skips: bool = False,
        hidit_hf: bool = False,
        mix_ffn: bool = False,
        mmdit_every_n: int = 0,
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
        self.size_embed_dim = int(size_embed_dim or 0)
        self._moe_num_experts = int(moe_num_experts or 0)
        self._moe_top_k = 2 if moe_top_k is None else int(moe_top_k)
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
        self.use_taca = bool(use_taca)
        self.use_swiglu = bool(use_swiglu)

        # REPA: projects pooled hidden tokens onto a frozen vision encoder's embedding space.
        self.repa_out_dim = int(repa_out_dim or 0)
        self.repa_projector_hidden_dim = int(repa_projector_hidden_dim or 0)
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

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_embedder = TextEmbedder(text_dim, hidden_size, class_dropout_prob)
        self.style_proj = nn.Linear(style_embed_dim, hidden_size) if style_embed_dim > 0 else None
        self.creativity_proj = None
        if creativity_embed_dim > 0:
            mid = max(16, creativity_embed_dim // 2)
            self.creativity_proj = nn.Sequential(nn.Linear(1, mid), nn.SiLU(), nn.Linear(mid, hidden_size))
        self.size_embedder = None
        self.size_proj = None
        if self.size_embed_dim > 0:
            self.size_embedder = SizeEmbedder(self.size_embed_dim, concat_dims=False)
            self.size_proj = nn.Linear(self.size_embed_dim, hidden_size)
        self.patch_se = None
        if patch_se:
            self.patch_se = ZeroInitPatchChannelGate(hidden_size, reduction=max(2, int(patch_se_reduction)))
        self.control_encoder = None
        if control_cond_dim > 0:
            self.control_encoder = ControlNetEncoder(
                control_size=input_size,
                patch_size=patch_size,
                in_channels=3,
                hidden_size=hidden_size,
                num_control_types=self.control_num_types,
            )
        num_patches = self.x_embedder.num_patches
        self.num_patches = int(num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Learnable register tokens (scratchpad) appended after the patch tokens.
        self.register_tokens = None
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, hidden_size))

        # Block-wise AR mask (ACDiT-style): causal over spatial blocks
        self.register_buffer("_ar_mask", None)
        if num_ar_blocks > 0:
            p = int(num_patches**0.5)
            self._ar_mask = create_block_causal_mask_2d(p, p, num_ar_blocks, block_order=self.ar_block_order)

        block_options = self._block_options()
        # Blocks see text_emb after TextEmbedder (already hidden_size), not raw encoder states.
        self.blocks = nn.ModuleList(
            DiTBlockText(
                hidden_size,
                num_heads,
                hidden_size,
                mlp_ratio=mlp_ratio,
                moe_num_experts=self._moe_num_experts,
                moe_top_k=self._moe_top_k,
                use_ssm=bool(ssm_every_n and int(ssm_every_n) > 0 and i % int(ssm_every_n) == 0),
                ssm_kernel_size=ssm_kernel_size,
                use_rope=self.use_rope,
                rope_base=self.rope_base,
                kv_merge_factor=self.kv_merge_factor,
                token_routing_enabled=self.token_routing_enabled,
                token_routing_strength=self.token_routing_strength,
                token_keep_ratio=self.token_keep_ratio,
                token_keep_min_value=self.token_keep_min_value,
                drop_path=self.drop_path_rate * i / max(depth - 1, 1),
                layerscale_init=self.layerscale_init,
                **block_options,
            )
            for i in range(depth)
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
                num_timesteps=self.prompt_schedule_num_timesteps,
            )
        self.initialize_weights()
        self._moe_layers = [m for m in self.modules() if hasattr(m, "last_aux_loss")]
        self.moe_aux_loss = None

    def _block_options(self) -> dict:
        """Per-block architecture knobs; the DiT-P / Supreme subclasses override them."""
        return {"qk_norm": self.qk_norm, "use_taca": self.use_taca, "ffn": "swiglu" if self.use_swiglu else "gelu"}

    def initialize_weights(self):
        xavier_init_linears(self)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.text_embedder.proj.weight, std=0.02)
        for proj in (self.style_proj, self.size_proj):
            if proj is not None:
                nn.init.normal_(proj.weight, std=0.02)
                nn.init.zeros_(proj.bias)
        if self.control_encoder is not None:
            nn.init.xavier_uniform_(
                self.control_encoder.proj.weight.view(self.control_encoder.proj.weight.shape[0], -1)
            )
            nn.init.zeros_(self.control_encoder.proj.bias)
            if self.control_encoder.type_embed is not None:
                nn.init.normal_(self.control_encoder.type_embed.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if block.use_taca:
                nn.init.zeros_(block.cross_attn.out_proj.weight)
        if self.hybrid is not None:
            self.hybrid.zero_init()
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def enable_gradient_checkpointing(self):
        self._grad_checkpointing = True

    def _text_conditioning(
        self,
        encoder_hidden_states,
        t,
        encoder_hidden_states_negative,
        negative_prompt_weight,
        style_embedding,
        style_strength,
        conditioning_scale,
        kwargs,
    ) -> torch.Tensor:
        """Text tokens for cross-attention: projection, scaling, emphasis, negative / style mixing, reference tokens."""
        text_emb = self.text_embedder(encoder_hidden_states, train=self.training)
        if conditioning_scale != 1.0:
            text_emb = text_emb * conditioning_scale
        # Timestep-aware text scale: stronger prompt pull at high noise, relaxed late to keep texture.
        if bool(kwargs.get("prompt_timestep_schedule_enabled", self.prompt_timestep_schedule_enabled)):
            early = float(kwargs.get("prompt_early_scale", self.prompt_early_scale))
            late = float(kwargs.get("prompt_late_scale", self.prompt_late_scale))
            nt = float(kwargs.get("prompt_schedule_num_timesteps", self.prompt_schedule_num_timesteps))
            tau = (t.to(text_emb.device).float() / max(1.0, nt - 1.0)).clamp(0.0, 1.0).view(-1, 1, 1)
            text_emb = text_emb * (late + (early - late) * tau).to(dtype=text_emb.dtype)
        # Per-token emphasis: (word) -> 1.2, [word] -> 0.8.
        token_weights = kwargs.get("token_weights")
        if token_weights is not None:
            w = token_weights.to(device=text_emb.device, dtype=text_emb.dtype)
            text_emb = text_emb * (w.view(1, -1, 1) if w.dim() == 1 else w)
        if encoder_hidden_states_negative is not None and negative_prompt_weight > 0:
            neg_emb = self.text_embedder(encoder_hidden_states_negative, train=False)
            text_emb = text_emb - negative_prompt_weight * neg_emb
        if style_embedding is not None and self.style_proj is not None and style_strength > 0:
            style_emb = self.style_proj(style_embedding)
            if style_emb.dim() == 2:
                style_emb = style_emb.unsqueeze(1).expand(-1, text_emb.size(1), -1)
            text_emb = text_emb + style_strength * style_emb
        # IP-Adapter-style image tokens (already hidden_size) appended to the text sequence.
        ref = kwargs.get("reference_tokens")
        ref_scale = float(kwargs.get("reference_scale", 1.0) or 0.0)
        if ref is not None and ref_scale > 0.0:
            ref = ref.to(device=text_emb.device, dtype=text_emb.dtype)
            if ref.shape[0] != text_emb.shape[0]:
                if ref.shape[0] != 1:
                    raise ValueError(
                        "reference_tokens batch must match encoder_hidden_states batch "
                        f"({ref.shape[0]} vs {text_emb.shape[0]})"
                    )
                ref = ref.expand(text_emb.shape[0], -1, -1)
            text_emb = torch.cat([text_emb, ref * ref_scale], dim=1)
        return text_emb

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
        if self._moe_layers:
            self.moe_aux_loss = None
            for m in self._moe_layers:
                m.last_aux_loss = None
        self._repa_projected = None
        if encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get("y_embed")
        if encoder_hidden_states is None:
            raise ValueError("DiT_Text.forward requires encoder_hidden_states (or y_embed).")
        b, h_lat, w_lat = x.shape[0], int(x.shape[2]), int(x.shape[3])
        hp, wp = h_lat // self.patch_size, w_lat // self.patch_size
        n_patch = hp * wp
        x = self.x_embedder(x)
        if not self.use_rope:
            x = x + grid_pos_embed(self.pos_embed, self.x_embedder.grid_size[0], hp, wp)
        if self.register_tokens is not None:
            x = torch.cat([x, self.register_tokens.expand(b, -1, -1)], dim=1)
        # ControlNet features go into the patch tokens only (registers stay intact).
        if control_image is not None and self.control_encoder is not None:
            control_feat = encode_control_stack(
                self.control_encoder,
                control_image,
                target_hw=(h_lat, w_lat),
                control_type=control_type,
                control_scale=control_scale,
            )
            x = inject_control_tokens(x, control_feat, patch_tokens=n_patch)
        if self.patch_se is not None:
            x = self.patch_se(x)
        t_emb = self.t_embedder(t)
        text_emb = self._text_conditioning(
            encoder_hidden_states,
            t,
            encoder_hidden_states_negative,
            negative_prompt_weight,
            style_embedding,
            style_strength,
            conditioning_scale,
            kwargs,
        )
        c = t_emb
        if self.size_embedder is not None:
            size_embed = kwargs.get("size_embed")
            if size_embed is None:
                size_embed = torch.tensor([[float(h_lat), float(w_lat)]], device=x.device).expand(b, -1)
            size_emb = self.size_embedder(size_embed.to(device=x.device, dtype=torch.float32), b)
            c = c + self.size_proj(size_emb.to(dtype=c.dtype))
        if creativity is not None and self.creativity_proj is not None:
            c = c + self.creativity_proj(creativity.unsqueeze(1).to(c.dtype))
        attn_mask = grid_ar_mask(self, hp, wp)
        captured_attn = None
        base_text_emb = text_emb
        reinject_every = int(kwargs.get("prompt_reinject_every_n", self.prompt_reinject_every_n))
        reinject_alpha = float(kwargs.get("prompt_reinject_alpha", self.prompt_reinject_alpha))
        reinject_decay = float(kwargs.get("prompt_reinject_decay", self.prompt_reinject_decay))
        block_cache = kwargs.get("block_cache")
        skip_blocks = set(kwargs.get("skip_blocks") or ())
        # Perturbed-Attention Guidance: rows ``pag_from:`` get identity self-attention in ``pag_blocks``.
        pag_blocks = set(kwargs.get("pag_blocks") or ())
        pag_from = kwargs.get("pag_from")
        if block_cache is not None:
            cfg_split = bool(kwargs.get("dbc_separate_cfg", block_cache.config.cfg_split_fingerprint))
            block_cache.begin_forward(block_cache.fingerprint_from_tensors(t_emb, x[:, :n_patch], cfg_split=cfg_split))
        if self.hybrid is not None:
            self.hybrid.begin_forward()
        checkpointing = self._grad_checkpointing and self.training
        for i, block in enumerate(self.blocks):
            text_emb_i = text_emb
            if reinject_every > 0 and reinject_alpha > 0.0 and i > 0 and i % reinject_every == 0:
                a = reinject_alpha * reinject_decay ** max(0, i // reinject_every - 1)
                text_emb_i = text_emb + float(a) * base_text_emb
            if i in skip_blocks:
                pass
            elif block_cache is not None and block_cache.should_skip_block(i):
                x = block_cache.apply_residual(x, i)
            else:
                x_before = x
                capture = return_attn and i == 0
                block_kw = dict(
                    attn_mask=attn_mask,
                    use_xformers=self.use_xformers,
                    num_patch_tokens=n_patch,
                    timestep=t,
                    identity_attn_from=pag_from if i in pag_blocks else None,
                    grid_hw=(hp, wp),
                )
                if checkpointing and not capture:
                    x = torch.utils.checkpoint.checkpoint(block, x, c, text_emb_i, use_reentrant=False, **block_kw)
                else:
                    if capture and checkpointing:
                        warnings.warn(
                            "return_attn=True with grad_checkpointing: block 0 runs without checkpointing "
                            "to capture attention weights.",
                            UserWarning,
                            stacklevel=2,
                        )
                    x = block(x, c, text_emb_i, return_attn=capture, **block_kw)
                    if capture:
                        x, captured_attn = x
                if block_cache is not None:
                    block_cache.note_block(i, x_before, x)
            if self.hybrid is not None:
                x = self.hybrid.after_block(
                    i,
                    x,
                    text_emb_i,
                    num_patches=n_patch,
                    h_patches=hp,
                    w_patches=wp,
                    use_xformers=self.use_xformers,
                )
        if self.hybrid is not None:
            x = self.hybrid.after_all(x, t, num_patches=n_patch, h_patches=hp, w_patches=wp)
        x_out = x[:, :n_patch, :]
        if self.repa_head is not None:
            self._repa_projected = self.repa_head(x_out.mean(dim=1))
        x = unpatchify(self.final_layer(x_out, c), self.patch_size, self.out_channels, hp, wp)
        if self._moe_layers:
            aux_vals = [m.last_aux_loss for m in self._moe_layers if m.last_aux_loss is not None]
            self.moe_aux_loss = sum(aux_vals) if aux_vals else None
        return (x, captured_attn) if return_attn else x


class DiT_Predecessor_Text(DiT_Text):
    """``DiT_Text`` trunk with QK-norm self-attention and full-width SwiGLU (SwiGLU experts under MoE).

    Positions are sin/cos only: these blocks never applied RoPE, KV pooling or TACA, so those flags are
    ignored to keep trained checkpoints valid. ``qk_norm`` adds QK-norm to the cross-attention.
    """

    def __init__(self, *, hidden_size=1280, depth=32, num_heads=20, **kwargs):
        kwargs.update(use_rope=False, kv_merge_factor=1, use_taca=False, use_swiglu=False)
        super().__init__(hidden_size=hidden_size, depth=depth, num_heads=num_heads, **kwargs)

    def _block_options(self) -> dict:
        return {"qk_norm": True, "cross_qk_norm": self.qk_norm, "qk_norm_eps": 1e-6, "ffn": "swiglu_wide"}


class DiT_Supreme_Text(DiT_Predecessor_Text):
    """Predecessor blocks with affine RMSNorm everywhere and QK-norm on the cross-attention too."""

    def __init__(self, *, hidden_size=1152, depth=28, num_heads=16, **kwargs):
        super().__init__(hidden_size=hidden_size, depth=depth, num_heads=num_heads, **kwargs)

    def _block_options(self) -> dict:
        return {**super()._block_options(), "cross_qk_norm": True, "norm": "rms"}


def DiT_XL_2_Text(**kwargs):
    return DiT_Text(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_L_2_Text(**kwargs):
    return DiT_Text(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_B_2_Text(**kwargs):
    return DiT_Text(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_P_2_Text(**kwargs):
    return DiT_Predecessor_Text(depth=32, hidden_size=1280, patch_size=2, num_heads=20, **kwargs)


def DiT_P_L_2_Text(**kwargs):
    return DiT_Predecessor_Text(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_Supreme_2_Text(**kwargs):
    return DiT_Supreme_Text(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_Supreme_L_2_Text(**kwargs):
    return DiT_Supreme_Text(depth=32, hidden_size=1280, patch_size=2, num_heads=20, **kwargs)


DiT_models_text = {
    "DiT-XL/2-Text": DiT_XL_2_Text,
    "DiT-L/2-Text": DiT_L_2_Text,
    "DiT-B/2-Text": DiT_B_2_Text,
    "DiT-P/2-Text": DiT_P_2_Text,
    "DiT-P-L/2-Text": DiT_P_L_2_Text,
    "DiT-Supreme/2-Text": DiT_Supreme_2_Text,
    "DiT-Supreme-L/2-Text": DiT_Supreme_L_2_Text,
}
