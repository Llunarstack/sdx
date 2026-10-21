"""Attention primitives: xformers / SDPA memory_efficient_attention (optional SLA linear blend), the DiT
self-attention (QK-norm, 2D RoPE, KV pooling, PAG identity rows) and the SSM token mixer."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_attention import hybrid_attention_fraction
from .moe import MoEProjection
from .rope2d import RoPE2D, apply_rope2d

try:
    import xformers.ops as xops

    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False
_XFORMERS_DISABLED = False  # set after the first runtime failure (version/CUDA mismatch)
_LINEAR_ATTN_FRAC: float = 0.0


def set_linear_attention_fraction(frac: float) -> None:
    """Global blend for SLA-style linear attention in ``memory_efficient_attention``."""
    global _LINEAR_ATTN_FRAC
    _LINEAR_ATTN_FRAC = float(max(0.0, min(1.0, frac)))


def memory_efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    use_xformers: bool = True,
    linear_attn_frac: float | None = None,
) -> torch.Tensor:
    """Attention on ``(B, N, H, D)`` tensors, returning ``(B, N, H, D)``; other ranks go straight to SDPA.

    ``attn_mask``: additive ``(N, N)`` / ``(B, N, N)`` tensor or an xformers ``AttentionBias``. Unmasked CUDA
    calls blend in linear attention when a linear fraction is set; xformers is tried before SDPA.
    """
    if q.dim() != 4:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    lf = float(linear_attn_frac) if linear_attn_frac is not None else _LINEAR_ATTN_FRAC
    if lf > 0.0 and attn_mask is None and q.is_cuda:
        out = hybrid_attention_fraction(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), linear_frac=lf, scale=scale
        )
        return out.transpose(1, 2)

    # Tensor masks go to SDPA, which broadcasts them; xformers wants an aligned, full (B, H, N, N) bias and a
    # failure here would disable it for every later (unmasked) call too.
    global _XFORMERS_DISABLED
    if use_xformers and _XFORMERS_AVAILABLE and not _XFORMERS_DISABLED and q.is_cuda and not torch.is_tensor(attn_mask):
        try:
            return xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=scale)
        except Exception as exc:  # any xformers failure: fall back to SDPA for the rest of the run
            _XFORMERS_DISABLED = True
            warnings.warn(f"xformers attention failed ({exc!r}); using PyTorch SDPA.", RuntimeWarning, stacklevel=2)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
    return out.transpose(1, 2)


class SelfAttention(nn.Module):
    """Self-attention with xformers/SDPA, optional causal/block-causal mask, and optional QK-norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        kv_merge_factor: int = 1,
        qk_norm: bool = False,
        qk_norm_eps: float | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.moe_out_proj = None
        if moe_num_experts > 0:
            self.moe_out_proj = MoEProjection(hidden_size, num_experts=int(moe_num_experts), top_k=int(moe_top_k))
        self.dropout = nn.Dropout(dropout)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.rope = RoPE2D(self.head_dim, base=self.rope_base) if self.use_rope else None
        self.kv_merge_factor = int(kv_merge_factor)
        # QK-norm (SD3.5-style): per-head RMSNorm on Q and K for training stability.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_xformers: bool = True,
        routing_context: torch.Tensor | None = None,
        router_override=None,
        report_aux_loss: bool = False,
        num_patch_tokens: int | None = None,
        identity_from: int | None = None,
        grid_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """``identity_from=i``: batch rows ``i:`` use an identity attention map, so their output is the
        value projection (Perturbed-Attention Guidance, Ahn et al. 2024).

        ``grid_hw``: patch rows/cols of the first ``rows*cols`` tokens (any trailing tokens are registers).
        Needed by ``use_rope`` (axial 2D RoPE, FLUX/SD3 layout) and by ``kv_merge_factor`` pooling; without
        it a square patch grid of ``num_patch_tokens`` is assumed.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        v_tokens = v
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if grid_hw is None and num_patch_tokens:
            side = int(round(int(num_patch_tokens) ** 0.5))
            grid_hw = (side, side) if side * side == int(num_patch_tokens) else None

        # Rotate before KV pooling so each pooled key averages keys that carry their true positions.
        if self.use_rope:
            if grid_hw is None:
                raise ValueError("use_rope needs grid_hw (patch rows, cols) for 2D RoPE.")
            n_rot = int(grid_hw[0]) * int(grid_hw[1])
            cos, sin = self.rope.get_freqs(int(grid_hw[0]), int(grid_hw[1]), device=x.device)
            q_rot, k_rot = apply_rope2d(q[:, :n_rot], k[:, :n_rot], cos, sin, input_format="BNHD")
            q = torch.cat([q_rot, q[:, n_rot:]], dim=1)
            k = torch.cat([k_rot, k[:, n_rot:]], dim=1)

        # KV pooling: keys/values are mean-pooled over the patch grid; queries keep full resolution.
        if self.kv_merge_factor > 1:
            if attn_mask is not None:
                raise ValueError("kv_merge_factor>1 is not supported with attn_mask in this implementation.")
            if grid_hw is None:
                raise ValueError("kv_merge_factor>1 requires grid_hw or a square num_patch_tokens.")
            f = int(self.kv_merge_factor)
            gh, gw = int(grid_hw[0]), int(grid_hw[1])
            N_patch = gh * gw
            if N_patch <= 0 or N_patch > N:
                raise ValueError(f"Invalid patch grid {gh}x{gw} for N={N}.")

            def _pool(t: torch.Tensor) -> torch.Tensor:
                # f x f mean pool over the patch grid; edge windows of a non-divisible grid average what they cover.
                g = t[:, :N_patch].reshape(B, gh, gw, C).permute(0, 3, 1, 2)
                g = F.avg_pool2d(g, f, f, ceil_mode=True)
                return g.flatten(2).transpose(1, 2).reshape(B, -1, self.num_heads, self.head_dim)

            k = torch.cat([_pool(k), k[:, N_patch:]], dim=1)
            v = torch.cat([_pool(v), v[:, N_patch:]], dim=1)

        if identity_from is None:
            out = memory_efficient_attention(q, k, v, attn_mask=attn_mask, scale=self.scale, use_xformers=use_xformers)
        else:
            p = int(identity_from)
            out = v_tokens
            if p > 0:
                mask = attn_mask[:p] if torch.is_tensor(attn_mask) and attn_mask.dim() == 3 else attn_mask
                attended = memory_efficient_attention(
                    q[:p], k[:p], v[:p], attn_mask=mask, scale=self.scale, use_xformers=use_xformers
                )
                out = torch.cat([attended, v_tokens[p:]], dim=0)
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


class SSMTokenMixer(nn.Module):
    """Depthwise-conv token mixer, an O(N) stand-in for self-attention in hybrid SSM-Transformer stacks.

    Not Mamba: one depthwise conv along the token axis between two projections. It ignores attention masks
    (so block-AR masking does not apply) and accepts ``SelfAttention``'s keyword arguments only to stay
    call-compatible inside DiT blocks.
    """

    def __init__(self, hidden_size: int, *, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = max(3, int(kernel_size) | 1)  # odd, >= 3
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.dwconv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=hidden_size,
        )
        self.act = nn.GELU()
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **_attention_kwargs) -> torch.Tensor:
        y = self.dwconv(self.in_proj(x).transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.out_proj(self.act(y)))
