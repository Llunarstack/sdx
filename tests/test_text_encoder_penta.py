"""Unit tests for penta / triple text encoder fusion (no HF weights required)."""

from __future__ import annotations

import torch
from utils.modeling.text_encoder_bundle import (
    PentaTextFusion,
    TripleTextFusion,
    load_fusion_from_state_dict,
)


def test_triple_fusion_appends_two_tokens():
    fusion = TripleTextFusion(768, 1280, out_dim=4096)
    t5 = torch.randn(2, 16, 4096)
    clip_l = torch.randn(2, 768)
    clip_bg = torch.randn(2, 1280)
    out = fusion(t5, clip_l, clip_bg)
    assert out.shape == (2, 18, 4096)
    assert fusion.extra_token_count == 2


def test_penta_fusion_appends_four_tokens():
    fusion = PentaTextFusion(768, 1280, 1024, 768, out_dim=4096)
    t5 = torch.randn(2, 16, 4096)
    clip_l = torch.randn(2, 768)
    clip_bg = torch.randn(2, 1280)
    clip_h = torch.randn(2, 1024)
    clip_long = torch.randn(2, 768)
    out = fusion(t5, clip_l, clip_bg, clip_h, clip_long)
    assert out.shape == (2, 20, 4096)
    assert fusion.extra_token_count == 4


def test_load_fusion_from_state_dict_detects_penta():
    fusion = PentaTextFusion(64, 96, 128, 64, out_dim=32)
    sd = fusion.state_dict()
    loaded = load_fusion_from_state_dict(sd, torch.device("cpu"))
    assert isinstance(loaded, PentaTextFusion)
    assert loaded.extra_token_count == 4


def test_load_fusion_from_state_dict_detects_triple():
    fusion = TripleTextFusion(64, 96, out_dim=32)
    sd = fusion.state_dict()
    loaded = load_fusion_from_state_dict(sd, torch.device("cpu"))
    assert isinstance(loaded, TripleTextFusion)
    assert loaded.extra_token_count == 2
