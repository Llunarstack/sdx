"""2D rotary position embeddings (RoPE-2D), FLUX / SD3 convention.

The head dim is split in half: the first half carries a 1D RoPE over the row (y), the second over the column
(x). Each 1D RoPE uses ``dim // 2`` inverse frequencies ``1 / base**(2i / dim)``, duplicated for the
``rotate_half`` layout. Relative 2D offsets are encoded directly in attention, with no parameters, and the
encoding extrapolates to unseen resolutions.

References: RoPE for ViT (arXiv:2403.13298), SD3 (arXiv:2403.03206), FLUX (Black Forest Labs).
"""

from __future__ import annotations

import torch


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


__all__ = ["RoPE2D", "build_2d_rope_freqs", "apply_rope2d"]
