# Memory-efficient attention: xformers with fallback to PyTorch SDPA or manual.
# Supports causal/block-causal masks for AR.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .moe import MoEProjection

_XFORMERS_AVAILABLE = False
try:
    import xformers.ops as xops
    _XFORMERS_AVAILABLE = True
except ImportError:
    pass


def _qkv_format(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Ensure (B, N, H, D) for xformers or SDPA."""
    # Input often (B, H, N, D) -> (B, N, H, D)
    if q.dim() == 4 and q.size(1) != q.size(2):
        pass  # assume (B, H, N, D)
    return q, k, v


def memory_efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_xformers: bool = True,
) -> torch.Tensor:
    """
    q, k, v: (B, N, H, D) or (B, H, N, D). Returns (B, N, H, D).
    attn_mask: optional (N, N) or (B, N, N) or xformers.AttentionBias (e.g. BlockDiagonalCausalMask).
    """
    # Normalize to (B, N, H, D) for consistency
    if q.dim() != 4:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
    B, N, H, D = q.shape
    if scale is None:
        scale = D ** -0.5

    if use_xformers and _XFORMERS_AVAILABLE and q.is_cuda:
        try:
            # xformers wants (B, N, H, D)
            out = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_mask,
                scale=scale,
            )
            return out
        except Exception:
            pass

    # Fallback: (B, N, H, D) -> (B, H, N, D) for PyTorch SDPA
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
    return out.transpose(1, 2)


def _apply_rope_1d(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply 1D RoPE on last-dim of q/k.
    x: (B, N, H, D), positions: (N,) integer.
    """
    D = x.shape[-1]
    D_even = D - (D % 2)
    if D_even <= 0:
        return x

    x_even = x[..., 0:D_even:2]
    x_odd = x[..., 1:D_even:2]
    dim_half = D_even // 2

    positions = positions.to(device=x.device, dtype=x_even.dtype)
    idx = torch.arange(dim_half, device=x.device, dtype=x_even.dtype)
    inv_freq = torch.pow(torch.tensor(base, device=x.device, dtype=x_even.dtype), -2.0 * idx / max(1, D_even))

    angles = positions[:, None] * inv_freq[None, :]  # (N, dim_half)
    cos = torch.cos(angles)[None, :, None, :]       # (1, N, 1, dim_half)
    sin = torch.sin(angles)[None, :, None, :]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.zeros_like(x)
    out[..., 0:D_even:2] = out_even
    out[..., 1:D_even:2] = out_odd
    if D_even < D:
        out[..., D_even:] = x[..., D_even:]
    return out


def create_block_causal_mask(num_patches_per_side: int, num_ar_blocks: int) -> Optional[torch.Tensor]:
    """
    Create block-causal mask for AR: patch grid is divided into num_ar_blocks x num_ar_blocks
    blocks; block i can only attend to blocks 0..i (causal within and across blocks).
    num_patches_per_side = sqrt(num_patches). Returns (N, N) float mask, -inf where masked.
    """
    if num_ar_blocks <= 0:
        return None
    N = num_patches_per_side * num_patches_per_side
    # Block index for each patch: row and col in block grid
    p = num_patches_per_side
    b = num_ar_blocks
    block_h = (p + b - 1) // b
    mask = torch.zeros(N, N, dtype=torch.float32)
    for i in range(N):
        ri, ci = i // p, i % p
        bi = (ri // block_h) * b + (ci * b // p) if p >= b else (ri * b // p) * b + (ci * b // p)
        bi = min(bi, b * b - 1)
        for j in range(N):
            rj, cj = j // p, j % p
            bj = (rj // block_h) * b + (cj * b // p) if p >= b else (rj * b // p) * b + (cj * b // p)
            bj = min(bj, b * b - 1)
            if bj > bi:
                mask[i, j] = float("-inf")
    return mask


def create_block_causal_mask_2d(h: int, w: int, num_ar_blocks: int) -> torch.Tensor:
    """
    h, w = patches per side. num_ar_blocks = blocks per dim (e.g. 2 -> 2x2 grid).
    Returns (N, N) mask: -inf where not allowed. Raster block order + causal within block.
    """
    n = h * w
    if num_ar_blocks <= 0:
        return torch.zeros(n, n, dtype=torch.float32)
    block_h = max(1, (h + num_ar_blocks - 1) // num_ar_blocks)
    block_w = max(1, (w + num_ar_blocks - 1) // num_ar_blocks)
    mask = torch.zeros(n, n, dtype=torch.float32)
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            bi = i // block_h
            bj = j // block_w
            block_idx = bi * num_ar_blocks + bj
            for i2 in range(h):
                for j2 in range(w):
                    idx2 = i2 * w + j2
                    bi2 = i2 // block_h
                    bj2 = j2 // block_w
                    block_idx2 = bi2 * num_ar_blocks + bj2
                    if block_idx2 > block_idx:
                        mask[idx, idx2] = float("-inf")
                    elif block_idx2 == block_idx and idx2 > idx:
                        mask[idx, idx2] = float("-inf")
    return mask


class SelfAttention(nn.Module):
    """Self-attention with xformers/SDPA and optional causal/block-causal mask."""

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
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.kv_merge_factor = int(kv_merge_factor)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        use_xformers: bool = True,
        routing_context: Optional[torch.Tensor] = None,
        router_override=None,
        report_aux_loss: bool = False,
        num_patch_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Hierarchical Patch Merging 2.0: kv pooling for keys/values only.
        # This keeps query resolution intact but reduces KV length for attention.
        if self.kv_merge_factor and int(self.kv_merge_factor) > 1:
            if attn_mask is not None:
                raise ValueError("kv_merge_factor>1 is not supported with attn_mask in this implementation.")
            if num_patch_tokens is None:
                raise ValueError("kv_merge_factor>1 requires num_patch_tokens.")
            f = int(self.kv_merge_factor)
            N_patch = int(num_patch_tokens)
            if N_patch <= 0 or N_patch > N:
                raise ValueError(f"Invalid num_patch_tokens={N_patch} for N={N}.")
            p = int(round(N_patch ** 0.5))
            if p * p != N_patch:
                raise ValueError(f"num_patch_tokens={N_patch} is not a perfect square.")
            if p % f != 0:
                raise ValueError(f"kv_merge_factor={f} must divide num_patches_per_side={p}.")
            N_reg = N - N_patch

            # Pool patch tokens into a coarser grid: (B, p, p, H, D) -> (B, p/f, p/f, H, D)
            k_patch = k[:, :N_patch].reshape(B, p, p, self.num_heads, self.head_dim)
            v_patch = v[:, :N_patch].reshape(B, p, p, self.num_heads, self.head_dim)
            k_patch = k_patch.reshape(B, p // f, f, p // f, f, self.num_heads, self.head_dim).mean(dim=(2, 4))
            v_patch = v_patch.reshape(B, p // f, f, p // f, f, self.num_heads, self.head_dim).mean(dim=(2, 4))
            k_patch = k_patch.reshape(B, (p // f) * (p // f), self.num_heads, self.head_dim)
            v_patch = v_patch.reshape(B, (p // f) * (p // f), self.num_heads, self.head_dim)

            if N_reg > 0:
                k_reg = k[:, N_patch:]
                v_reg = v[:, N_patch:]
                k = torch.cat([k_patch, k_reg], dim=1)
                v = torch.cat([v_patch, v_reg], dim=1)
            else:
                k = k_patch
                v = v_patch

        # RoPE: apply 1D rotary embeddings to self-attention q/k by token index.
        if self.use_rope:
            pos_q = torch.arange(N, device=x.device, dtype=torch.long)
            q = _apply_rope_1d(q, pos_q, base=self.rope_base)
            pos_k = torch.arange(k.shape[1], device=x.device, dtype=torch.long)
            k = _apply_rope_1d(k, pos_k, base=self.rope_base)

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


class SSMTokenMixer(nn.Module):
    """
    Lightweight SSM-ish token mixer (drop-in replacement for self-attention output).

    Complexity is close to O(N) for the mixing operation (depthwise conv over token axis).
    This is NOT full Mamba, but it provides the same integration point in DiT blocks so
    you can explore "hybrid SSM-Transformer" behavior without extra dependencies.
    """

    def __init__(self, hidden_size: int, *, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        kernel_size = int(kernel_size)
        kernel_size = max(3, kernel_size | 1)  # force odd >= 3
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        # Depthwise conv over token sequence length N (B, C, N).
        self.dwconv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=hidden_size,
            bias=True,
        )
        self.act = nn.GELU()
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        use_xformers: bool = True,
        routing_context: Optional[torch.Tensor] = None,
        router_override=None,
        report_aux_loss: bool = False,
        num_patch_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        # x: (B, N, C)
        y = self.in_proj(x)
        y = y.transpose(1, 2)  # (B, C, N)
        y = self.dwconv(y)
        y = y.transpose(1, 2)  # (B, N, C)
        y = self.act(y)
        y = self.out_proj(y)
        return self.dropout(y)
