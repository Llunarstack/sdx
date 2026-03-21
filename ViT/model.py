from __future__ import annotations

from typing import Dict

import timm
import torch
import torch.nn as nn


class ViTQualityAdherenceModel(nn.Module):
    """
    ViT backbone with two heads:
    - quality_logit: binary quality classification (good/bad)
    - adherence_score: regression in [0,1] for prompt adherence
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        text_feat_dim: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = int(self.backbone.num_features)

        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.adherence_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, images: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        img_feat = self.backbone(images)
        txt_feat = self.text_proj(text_features)
        fused = self.fuse(torch.cat([img_feat, txt_feat], dim=1))
        quality_logit = self.quality_head(fused).squeeze(1)
        adherence_score = self.adherence_head(fused).squeeze(1)
        return {
            "quality_logit": quality_logit,
            "adherence_score": adherence_score,
            "embedding": fused,
        }


def build_vit_model(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    text_feat_dim: int = 8,
    hidden_dim: int = 256,
) -> ViTQualityAdherenceModel:
    return ViTQualityAdherenceModel(
        model_name=model_name,
        pretrained=pretrained,
        text_feat_dim=text_feat_dim,
        hidden_dim=hidden_dim,
    )
