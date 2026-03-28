import torch
import pytest

from models.vit_next_blocks import LayerScale, apply_topk_token_keep


def test_layerscale_identity_when_disabled():
    ls = LayerScale(dim=8, init_value=0.0)
    x = torch.randn(2, 4, 8)
    y = ls(x)
    assert torch.allclose(x, y)


def test_layerscale_scales_channels_when_enabled():
    ls = LayerScale(dim=4, init_value=0.5)
    x = torch.ones(1, 2, 4)
    y = ls(x)
    assert torch.allclose(y, torch.full_like(x, 0.5))


def test_apply_topk_token_keep_keeps_top_patch_tokens():
    gate = torch.ones(1, 6, 1)
    # First 4 are patch tokens, last 2 are register tokens.
    score = torch.tensor([[[0.1], [0.9], [0.8], [0.2], [0.0], [0.0]]], dtype=torch.float32)
    out = apply_topk_token_keep(gate, score, keep_ratio=0.5, num_patch_tokens=4, min_keep_value=0.1)
    # top2 of first 4 are indices 1 and 2 -> gate remains 1.0
    assert float(out[0, 1, 0]) == 1.0
    assert float(out[0, 2, 0]) == 1.0
    # others in patch area suppressed to floor
    assert float(out[0, 0, 0]) == pytest.approx(0.1, rel=0.0, abs=1e-6)
    assert float(out[0, 3, 0]) == pytest.approx(0.1, rel=0.0, abs=1e-6)
    # register tokens untouched
    assert float(out[0, 4, 0]) == 1.0
    assert float(out[0, 5, 0]) == 1.0
