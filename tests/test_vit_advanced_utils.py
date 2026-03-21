from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_pairwise_ranking_loss_prefers_correct_order() -> None:
    from ViT.losses import pairwise_ranking_loss

    target = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    pred_good = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32)
    pred_bad = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    lg = pairwise_ranking_loss(pred_good, target)
    lb = pairwise_ranking_loss(pred_bad, target)
    assert float(lg) < float(lb)


def test_tta_predict_shapes() -> None:
    from ViT.dataset import text_feature_vector
    from ViT.model import build_vit_model
    from ViT.tta import tta_predict

    model = build_vit_model(model_name="vit_tiny_patch16_224", pretrained=False, text_feat_dim=8, hidden_dim=64)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    tf = torch.stack([text_feature_vector("1girl"), text_feature_vector("1boy")], dim=0)
    with torch.no_grad():
        out = tta_predict(model, x, tf)
    assert out["quality_logit"].shape == (2,)
    assert out["adherence_score"].shape == (2,)
    assert out["embedding"].shape[0] == 2
