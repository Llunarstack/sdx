"""Attention for the DiTs.

- ``memory_efficient_attention``: xformers / SDPA on ``(B, N, H, D)``, with an optional SLA-style linear blend.
- ``SelfAttention`` (QK-norm, 2D RoPE, KV pooling, PAG identity rows) and ``SSMTokenMixer``.
- Sub-quadratic attention: ``LinearCompressedAttention`` (O(N*C)), ``LocalWindowAttention`` (O(N*W^2)),
  ``linear_attention`` / ``hybrid_attention_fraction``.
- ``TACA``: temperature-adjusted cross-modal attention (arXiv:2506.07986).
- 2D RoPE (FLUX / SD3 convention): ``build_2d_rope_freqs``, ``apply_rope2d``, ``RoPE2D``.
- Masks and biases: block-causal AR masks (``create_block_causal_mask_2d``) and the context-local Dense
  Diffusion layout bias (``set_layout_attn`` / ``layout_cross_attn_bias``).
"""

from __future__ import annotations

import math
import warnings
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import MoEProjection, timestep_embedding

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
            cos, sin = self.rope.get_freqs(int(grid_hw[0]), int(grid_hw[1]), device=x.device, dtype=q.dtype)
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


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Kernel linear attention over ``(B, H, N, D)``: ``phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1)``, O(N).

    ``phi = elu + 1`` (Katharopoulos et al. 2020) keeps every similarity positive, so the normaliser
    cannot cross zero the way raw ``Q K^T`` does.
    """
    q, k = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0
    kv = torch.einsum("bhnd,bhne->bhde", k, v.float())
    num = torch.einsum("bhnd,bhde->bhne", q, kv)
    den = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).unsqueeze(-1) + eps
    return (num / den).to(dtype=v.dtype)


def hybrid_attention_fraction(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    linear_frac: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """``(1 - f) * softmax attention + f * linear attention`` on ``(B, H, N, D)``; ``f`` is clamped to [0, 1]."""
    f = max(0.0, min(1.0, float(linear_frac)))
    if f >= 1.0:
        return linear_attention(q, k, v)
    dense = F.scaled_dot_product_attention(q, k, v, scale=scale)
    return dense if f <= 0.0 else (1.0 - f) * dense + f * linear_attention(q, k, v)


class LinearCompressedAttention(nn.Module):
    """Compressed key-value attention, O(N * C): every query attends over ``context_size`` soft-pooled slots."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int = 64,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.context_size = int(context_size)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.kv_compress = nn.Linear(hidden_size, context_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        # Not zero: this projection is not inside a gated residual, so zero-init would kill output and grads.
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        C = self.context_size

        # Soft-pool the N tokens into C learned slots: softmax over N of (B, N, C) scores.
        compress_weights = F.softmax(self.kv_compress(x), dim=1)
        x_compressed = torch.bmm(compress_weights.transpose(1, 2), x)

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_compressed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_compressed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


class LocalWindowAttention(nn.Module):
    """Swin-style self-attention inside ``window_size``-square windows of a square token grid.

    ``num_global_tokens`` learned tokens attend over the whole sequence; their mean is added to every patch
    output for long-range context. Grids that do not divide into windows are zero-padded, and the padding is
    masked out of the attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 8,
        num_global_tokens: int = 4,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = int(window_size)
        self.num_global = int(num_global_tokens)
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

        if self.num_global > 0:
            self.global_tokens = nn.Parameter(torch.randn(1, self.num_global, hidden_size) * 0.02)

        # Zero-init is safe here: the caller adds this module as a residual (x + attention(x)).
        nn.init.zeros_(self.out_proj.weight)

    def _window_partition(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        """(B, h*w, C) -> (B*nW, W^2, C) zero-padded to whole windows, plus the padded h, w."""
        B, _, C = x.shape
        ws = self.window_size
        pad_h, pad_w = (ws - h % ws) % ws, (ws - w % ws) % ws
        x = F.pad(x.reshape(B, h, w, C), (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.reshape(B, hp // ws, ws, wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, ws * ws, C), hp, wp

    def _window_unpartition(self, x: torch.Tensor, B: int, h: int, w: int, hp: int, wp: int) -> torch.Tensor:
        """(B*nW, W^2, C) -> (B, h*w, C), dropping the padding."""
        ws = self.window_size
        C = x.shape[-1]
        x = x.reshape(B, hp // ws, wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, hp, wp, C)
        return x[:, :h, :w, :].reshape(B, h * w, C)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, N, D)`` with ``N`` a perfect square."""
        B, N, D = x.shape
        h = w = math.isqrt(N)
        if h * w != N:
            raise ValueError(f"LocalWindowAttention requires a square token grid; got N={N} (sqrt ≈ {N**0.5:.2f})")
        G = self.num_global

        x_full = torch.cat([self.global_tokens.expand(B, -1, -1), x], dim=1) if G > 0 else x
        qkv = self.qkv(x_full)
        windows, hp, wp = self._window_partition(qkv[:, G:], h, w)
        q_l, k_l, v_l = (self._heads(t) for t in windows.chunk(3, dim=-1))
        if self.q_norm is not None:
            q_l = self.q_norm(q_l)
            k_l = self.k_norm(k_l)

        key_mask = None
        if (hp, wp) != (h, w):
            valid, _, _ = self._window_partition(x.new_ones(B, N, 1), h, w)
            key_mask = valid.squeeze(-1).bool()[:, None, None, :]  # (B*nW, 1, 1, W^2)
        out_local = F.scaled_dot_product_attention(q_l, k_l, v_l, attn_mask=key_mask, scale=self.scale)
        out_local = out_local.transpose(1, 2).reshape(windows.shape[0], windows.shape[1], D)
        out_local = self._window_unpartition(out_local, B, h, w, hp, wp)

        if G > 0:
            q_full, k_full, v_full = qkv.chunk(3, dim=-1)
            q_g, k_all, v_all = self._heads(q_full[:, :G]), self._heads(k_full), self._heads(v_full)
            if self.q_norm is not None:
                q_g = self.q_norm(q_g)
                k_all = self.k_norm(k_all)
            out_g = F.scaled_dot_product_attention(q_g, k_all, v_all, scale=self.scale)
            out_local = out_local + out_g.transpose(1, 2).reshape(B, G, D).mean(dim=1, keepdim=True)

        return self.out_proj(out_local)


def build_2d_rope_freqs(
    height: int,
    width: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)``, each ``(height * width, head_dim)`` in raster order; ``head_dim`` must divide by 4.

    Angles are computed in float32 and only the result is cast to ``dtype``: half-precision angles of a few
    dozen radians are off by a large fraction of a turn.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
    half = head_dim // 2

    def _freqs_1d(length: int) -> torch.Tensor:
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
        return torch.cat([freqs, freqs], dim=-1)  # (length, half)

    row = _freqs_1d(height).unsqueeze(1).expand(height, width, half)
    col = _freqs_1d(width).unsqueeze(0).expand(height, width, half)
    freqs_2d = torch.cat([row, col], dim=-1).reshape(height * width, head_dim)
    return freqs_2d.cos().to(dtype), freqs_2d.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    input_format: str = "BHND",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` / ``k`` by ``(N, D)`` ``cos`` / ``sin`` tables; returns tensors shaped like the inputs.

    ``input_format`` is ``"BHND"`` (batch, heads, tokens, head_dim) or ``"BNHD"``; it is explicit because the
    two layouts cannot be told apart by shape when N == H.
    """
    if input_format.upper() == "BNHD":
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (N, 1, D): broadcast over heads
    cos = cos.to(dtype=q.dtype, device=q.device)
    sin = sin.to(dtype=q.dtype, device=q.device)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RoPE2D(torch.nn.Module):
    """2D RoPE that caches its ``(cos, sin)`` tables per grid size, device and dtype.

    ``q, k = RoPE2D(head_dim=64)(q, k, height=32, width=32)``
    """

    _MAX_CACHE: int = 256

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        self._cache: dict = {}

    def get_freqs(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # str(device) so torch.device("cuda:0") and torch.device("cuda", 0) share an entry.
        key = (height, width, str(device), dtype)
        if key not in self._cache:
            self._cache[key] = build_2d_rope_freqs(height, width, self.head_dim, self.base, device=device, dtype=dtype)
            if len(self._cache) > self._MAX_CACHE:
                self._cache.pop(next(iter(self._cache)))  # evict the oldest entry
        return self._cache[key]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        height: int,
        width: int,
        input_format: str = "BHND",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.get_freqs(height, width, device=q.device, dtype=q.dtype)
        return apply_rope2d(q, k, cos, sin, input_format=input_format)


def _morton_code(bi: int, bj: int, bits: int) -> int:
    z = 0
    for bit in range(bits):
        z |= ((bi >> bit) & 1) << (2 * bit)
        z |= ((bj >> bit) & 1) << (2 * bit + 1)
    return z


def _spiral_cells(num_ar_blocks: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = num_ar_blocks - 1, num_ar_blocks - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            out.append((top, c))
        top += 1
        for r in range(top, bottom + 1):
            out.append((r, right))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                out.append((bottom, c))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                out.append((r, left))
            left += 1
    return out


def block_visit_order(num_ar_blocks: int, block_order: str = "raster") -> list[tuple[int, int]]:
    """Macro-block ``(row, col)`` cells in visit order."""
    order = str(block_order or "raster").strip().lower()
    n = num_ar_blocks
    cells = [(bi, bj) for bi in range(n) for bj in range(n)]
    if order in ("raster", "row", "row_major", "default"):
        return cells
    if order in ("z", "z-order", "zorder", "morton"):
        bits = max(1, (n - 1).bit_length())
        return sorted(cells, key=lambda c: _morton_code(c[0], c[1], bits))
    if order in ("snake", "boustrophedon"):
        return sorted(cells, key=lambda c: c[0] * n + (c[1] if c[0] % 2 == 0 else n - 1 - c[1]))
    if order in ("spiral", "snail"):
        return _spiral_cells(n)
    raise ValueError(f"Unknown block_order {block_order!r}; use raster | zorder | snake | spiral")


def create_block_causal_mask_2d(
    h: int,
    w: int,
    num_ar_blocks: int,
    *,
    block_order: str = "raster",
) -> torch.Tensor:
    """Additive ``(h*w, h*w)`` mask for an ``h x w`` patch grid: 0 = attend, -inf = blocked.

    ``num_ar_blocks`` is macro-blocks per side (2 -> 2x2); ``<= 0`` returns an all-zero (full attention) mask.
    """
    n = h * w
    if num_ar_blocks <= 0:
        return torch.zeros(n, n, dtype=torch.float32)
    lut = torch.zeros(num_ar_blocks, num_ar_blocks, dtype=torch.long)
    for rank, (bi, bj) in enumerate(block_visit_order(num_ar_blocks, block_order)):
        lut[bi, bj] = rank
    bh = max(1, -(-h // num_ar_blocks))
    bw = max(1, -(-w // num_ar_blocks))
    rows = torch.arange(h).repeat_interleave(w) // bh
    cols = torch.arange(w).repeat(h) // bw
    rank = lut[rows, cols]
    idx = torch.arange(n)
    blocked = (rank[None, :] > rank[:, None]) | ((rank[None, :] == rank[:, None]) & (idx[None, :] > idx[:, None]))
    return torch.zeros(n, n, dtype=torch.float32).masked_fill(blocked, float("-inf"))


def ar_mask_sparsity_stats(mask: torch.Tensor) -> tuple[float, int]:
    """Fraction of allowed pairs (finite mask), and N for an N×N mask."""
    if mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be square (N, N)")
    ok = torch.isfinite(mask) & (mask > -1e18)
    return float(ok.float().mean().item()), int(mask.shape[0])


_layout_attn_ctx: ContextVar[dict[str, Any] | None] = ContextVar("sdx_layout_attn", default=None)


def set_layout_attn(
    *,
    plan: Any | None = None,
    region_masks: torch.Tensor | None = None,
    step_index: int = 0,
    cond_batch_size: int | None = None,
    num_patch_tokens: int | None = None,
) -> None:
    """Activate layout bias for subsequent CrossAttention forwards."""
    if plan is None or region_masks is None:
        _layout_attn_ctx.set(None)
        return
    _layout_attn_ctx.set(
        {
            "plan": plan,
            "region_masks": region_masks,
            "step_index": int(step_index),
            "cond_batch_size": cond_batch_size,
            "num_patch_tokens": num_patch_tokens,
        }
    )


def clear_layout_attn() -> None:
    _layout_attn_ctx.set(None)


def layout_cross_attn_bias(
    batch: int, n_query: int, n_text: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    """Additive ``(B, 1, N, L)`` cross-attention bias for the active layout step, else ``None``."""
    ctx = _layout_attn_ctx.get()
    if ctx is None:
        return None
    from frontier.layout import bias_cross_attention

    zeros = torch.zeros(batch, 1, n_query, n_text, device=device, dtype=dtype)
    bias = bias_cross_attention(zeros, **ctx)
    return None if bias is zeros else bias


class TACA(nn.Module):
    """Cross-attention whose logits are scaled by a learned per-head imbalance factor and divided by a
    timestep-dependent temperature ``base_temp + temp_range * tanh(MLP(t))`` (clamped to >= 0.1)."""

    def __init__(
        self,
        hidden_size: int,
        text_dim: int,
        num_heads: int,
        max_timesteps: int = 1000,
        base_temp: float = 1.0,
        temp_range: float = 0.5,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.base_temp = float(base_temp)
        self.temp_range = float(temp_range)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(text_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(text_dim, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

        t_emb_dim = 64
        self.t_embed = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 2, 1),
            nn.Tanh(),
        )
        self._t_emb_dim = t_emb_dim
        self._max_t = max_timesteps
        self.imbalance_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        timestep: torch.Tensor | None = None,
        *,
        use_xformers: bool = True,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Image tokens ``(B, N, D)`` attend to text ``(B, L, text_dim)``; ``timestep=None`` uses ``base_temp``.

        ``return_weights`` also returns the ``(B, heads, N, L)`` attention probabilities.
        """
        B, N, D = x.shape
        L = text_emb.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(text_emb).reshape(B, L, self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if timestep is not None:
            t = timestep.expand(B) if timestep.numel() == 1 else timestep
            t_emb = timestep_embedding(t, self._t_emb_dim, max_period=self._max_t)
            delta = self.t_embed(t_emb.to(self.t_embed[0].weight.dtype)) * self.temp_range
            temp = (self.base_temp + delta).clamp(min=0.1).view(B, 1, 1, 1)
        else:
            temp = x.new_full((1, 1, 1, 1), self.base_temp)
        # Temperature and per-head imbalance scale the logits; folding sqrt into both Q and K
        # keeps the fused SDPA kernel. Layout here is (B, N, H, D).
        fac = (self.imbalance_scale.view(1, 1, self.num_heads, 1).clamp(min=1e-4) / temp).sqrt().to(q.dtype)
        q = q * fac
        k = k * fac

        bias = layout_cross_attn_bias(B, N, L, device=x.device, dtype=q.dtype)
        if return_weights:
            logits = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)) * self.scale
            if bias is not None:
                logits = logits + bias
            weights = logits.softmax(dim=-1)
            out = (weights @ v.transpose(1, 2)).transpose(1, 2).reshape(B, N, D)
            return self.out_proj(out), weights
        out = memory_efficient_attention(
            q, k, v, attn_mask=bias, scale=self.scale, use_xformers=use_xformers and bias is None
        )
        return self.out_proj(out.reshape(B, N, D))


__all__ = [
    "LinearCompressedAttention",
    "LocalWindowAttention",
    "hybrid_attention_fraction",
    "linear_attention",
    "RoPE2D",
    "build_2d_rope_freqs",
    "apply_rope2d",
    "set_layout_attn",
    "clear_layout_attn",
    "layout_cross_attn_bias",
    "TACA",
]
