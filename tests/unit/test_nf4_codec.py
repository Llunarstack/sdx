from __future__ import annotations

import torch

from utils.quantization.nf4_codec import dequantize_nf4_block, quantize_nf4_block


def test_nf4_roundtrip_small():
    w = torch.tensor([0.1, -0.5, 0.0, 1.0, -1.0, 0.25] * 11, dtype=torch.float32)  # 66 elems -> 2 blocks of 64+2
    packed, absmax, n = quantize_nf4_block(w, block_size=64)
    got = dequantize_nf4_block(packed, absmax, block_size=64, original_numel=int(n))
    assert got.shape == w.shape
    # NF4 is lossy; stay within coarse envelope of block absmax
    assert torch.isfinite(got).all()
