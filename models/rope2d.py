"""
2D Rotary Position Embeddings (RoPE-2D) for Vision Transformers.

Used in FLUX, SD3, and ViT-5 to replace learned or sinusoidal 2D position
embeddings. Key advantages over 1D RoPE or learned embeddings:
  - Extrapolates to unseen resolutions without fine-tuning
  - Encodes relative 2D spatial relationships directly in attention
  - No extra parameters

Implementation follows the FLUX / SD3 convention:
  - Split head_dim into two halves: first half encodes row (y), second encodes col (x)
  - Apply standard 1D RoPE independently to each half

References:
  - RoPE for ViT: arxiv 2403.13298
  - FLUX architecture: Black Forest Labs
  - SD3: arxiv 2403.03206
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def build_2d_rope_freqs(
    height: int,
    width: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (cos, sin) frequency tensors for 2D RoPE.

    Returns:
        cos: (H*W, head_dim)
        sin: (H*W, head_dim)
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    half = head_dim // 2  # half for rows, half for cols

    # 1D frequencies for each axis
    def _freqs_1d(length: int, dim: int) -> torch.Tensor:
        d_half = dim // 2
        theta = 1.0 / (base ** (torch.arange(0, d_half, 2, device=device, dtype=dtype) / d_half))
        pos = torch.arange(length, device=device, dtype=dtype)
        freqs = torch.outer(pos, theta)  # (length, d_half/2)
        return torch.cat([freqs, freqs], dim=-1)  # (length, d_half)

    row_freqs = _freqs_1d(height, half)  # (H, half)
    col_freqs = _freqs_1d(width, half)   # (W, half)

    # Broadcast to (H, W, half) then flatten
    row_freqs = row_freqs.unsqueeze(1).expand(height, width, half)  # (H, W, half)
    col_freqs = col_freqs.unsqueeze(0).expand(height, width, half)  # (H, W, half)

    freqs_2d = torch.cat([row_freqs, col_freqs], dim=-1)  # (H, W, head_dim)
    freqs_2d = freqs_2d.reshape(height * width, head_dim)  # (N, head_dim)

    return freqs_2d.cos(), freqs_2d.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting into two halves."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2D RoPE to query and key tensors.

    Args:
        q: (B, H, N, D) or (B, N, H, D)
        k: same shape as q
        cos: (N, D) precomputed cosines
        sin: (N, D) precomputed sines
    Returns:
        q_rot, k_rot: same shape as inputs
    """
    # Normalize to (B, H, N, D)
    transposed = False
    if q.dim() == 4 and q.shape[2] != q.shape[1]:
        # Assume (B, N, H, D) -> (B, H, N, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        transposed = True

    # cos/sin: (N, D) -> (1, 1, N, D)
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin

    if transposed:
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

    return q_rot, k_rot


class RoPE2D(torch.nn.Module):
    """
    Stateful 2D RoPE module that caches frequency tensors per resolution.

    Usage:
        rope = RoPE2D(head_dim=64)
        cos, sin = rope.get_freqs(height=32, width=32, device=x.device)
        q, k = rope.apply(q, k, cos, sin)
    """

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, device, dtype)
        if key not in self._cache:
            cos, sin = build_2d_rope_freqs(
                height, width, self.head_dim, self.base, device=device, dtype=dtype
            )
            self._cache[key] = (cos, sin)
        return self._cache[key]

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.get_freqs(height, width, device=q.device, dtype=q.dtype)
        return apply_rope2d(q, k, cos, sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply(q, k, height, width)


__all__ = ["RoPE2D", "build_2d_rope_freqs", "apply_rope2d"]
