from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_vit_model_forward_smoke() -> None:
    from ViT.dataset import text_feature_vector
    from ViT.model import build_vit_model

    model = build_vit_model(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        text_feat_dim=8,
        hidden_dim=64,
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    tf = torch.stack([text_feature_vector("1girl, masterpiece"), text_feature_vector("blurry, low quality")], dim=0)
    with torch.no_grad():
        out = model(x, tf)
    assert out["quality_logit"].shape == (2,)
    assert out["adherence_score"].shape == (2,)
    assert out["embedding"].shape[0] == 2

