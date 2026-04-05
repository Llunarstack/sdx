"""
NF4 (NormalFloat 4-bit) block quantization — same level table as QLoRA / bitsandbytes-style NF4.

Use for **research** and optional CUDA dequant; full GPU NF4 **training** still needs
autograd integration (e.g. bitsandbytes, torchao). This module is the numeric core.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch


def _ensure_native_python_path() -> None:
    """So ``sdx_native.*`` resolves when only repo root is on ``sys.path`` (e.g. ``train.py``)."""
    root = Path(__file__).resolve().parents[2]
    np = root / "native" / "python"
    if np.is_dir() and str(np) not in sys.path:
        sys.path.insert(0, str(np))

# Block-normalized NF4 levels (symmetric, percentile-based; then scaled to [-1, 1] envelope).
NF4_LEVELS: Tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07993701049613953,
    0.16089616703987122,
    0.2467532904291153,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

def quantize_nf4_block(w: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-block NF4 quantization.

    Returns:
        codes_uint8: ceil(n/2) bytes — two 4-bit codes per byte (low nibble first pair element).
        absmax: float32 tensor of shape (num_blocks,)
        padded_len: original flattened length before padding to even for packing
    """
    w = w.detach().float().reshape(-1)
    n = int(w.numel())
    pad = (block_size - (n % block_size)) % block_size
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    w_flat = w
    nb = w_flat.numel() // block_size
    blocks = w_flat.view(nb, block_size)
    absmax = blocks.abs().amax(dim=1).clamp(min=1e-8)
    norm = blocks / absmax.unsqueeze(1)
    levels = torch.tensor(NF4_LEVELS, dtype=torch.float32, device=norm.device)
    diff = (norm.unsqueeze(-1) - levels).abs()
    idx = diff.argmin(dim=-1).to(torch.uint8)
    idx_flat = idx.reshape(-1)
    if idx_flat.numel() % 2 == 1:
        idx_flat = torch.nn.functional.pad(idx_flat, (0, 1))
    lo = idx_flat[0::2] & 0x0F
    hi = idx_flat[1::2] & 0x0F
    packed = (hi << 4) | lo
    return packed.to(torch.uint8), absmax.float(), torch.tensor(n, dtype=torch.int64)


def dequantize_nf4_block_auto(
    codes_uint8: torch.Tensor,
    absmax: torch.Tensor,
    *,
    block_size: int = 64,
    original_numel: int,
    prefer_cuda: bool = True,
) -> torch.Tensor:
    """
    Like ``dequantize_nf4_block`` but tries ``sdx_cuda_nf4`` (host buffers) when built.

    Returns tensor on ``absmax.device``. Falls back to pure PyTorch if the DLL is missing.
    """
    if prefer_cuda:
        try:
            import numpy as np

            _ensure_native_python_path()
            from sdx_native.nf4_dequant_native import maybe_nf4_dequant_cuda

            n_blocks = int(absmax.numel())
            n_weights = n_blocks * block_size
            packed = codes_uint8.detach().cpu().numpy()
            am = absmax.detach().cpu().numpy().astype(np.float32, copy=False)
            arr = maybe_nf4_dequant_cuda(packed, am, block_size=block_size, n_weights=n_weights)
            if arr is not None:
                return torch.from_numpy(arr[: int(original_numel)]).to(
                    device=absmax.device, dtype=torch.float32
                )
        except Exception:
            pass
    return dequantize_nf4_block(
        codes_uint8, absmax, block_size=block_size, original_numel=original_numel
    )


def dequantize_nf4_block(
    codes_uint8: torch.Tensor,
    absmax: torch.Tensor,
    *,
    block_size: int = 64,
    original_numel: int,
) -> torch.Tensor:
    """Inverse of ``quantize_nf4_block`` (ignores padding tail from packing)."""
    device = absmax.device
    levels = torch.tensor(NF4_LEVELS, dtype=torch.float32, device=device)
    lo = codes_uint8 & 0x0F
    hi = (codes_uint8 >> 4) & 0x0F
    idx = torch.stack([lo, hi], dim=1).reshape(-1)
    n_blocks = absmax.numel()
    n_weights = n_blocks * block_size
    idx = idx[:n_weights]
    vals = levels[idx.long()]
    blocks = vals.view(n_blocks, block_size) * absmax.unsqueeze(1)
    out = blocks.reshape(-1)[:original_numel]
    return out
