from __future__ import annotations

import torch
from utils.modeling.text_encoder_bundle import TripleTextFusion


def test_triple_fusion_shapes() -> None:
    f = TripleTextFusion(768, 1280, out_dim=4096)
    t5 = torch.randn(2, 16, 4096)
    cl = torch.randn(2, 768)
    bg = torch.randn(2, 1280)
    y = f(t5, cl, bg)
    assert y.shape == (2, 18, 4096)
