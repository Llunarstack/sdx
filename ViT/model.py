from __future__ import annotations

from typing import Dict, Optional

import timm
import torch
import torch.nn as nn

from utils.ar_dit_vit import AR_COND_DIM, default_unknown_ar_batch


class ViTQualityAdherenceModel(nn.Module):
    """
    ViT backbone with two heads:
    - quality_logit: binary quality classification (good/bad)
    - adherence_score: regression in [0,1] for prompt adherence

    Optional **DiT AR regime** conditioning (``use_ar_conditioning``): concatenates a 4-D one-hot
    (full DiT / AR 2×2 / AR 4×4 / unknown) so scores match how the **generator** was trained.
    See ``utils/ar_dit_vit.py`` and ``docs/AR.md``.

    This is a **discriminative** scorer (dataset QA / ranking), not the DiT generator.
    Stronger timm backbones: ``ViT/backbone_presets.py``, ``ViT/EXCELLENCE_VS_DIT.md``.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        text_feat_dim: int = 8,
        hidden_dim: int = 256,
        *,
        use_ar_conditioning: bool = True,
        ar_cond_dim: int = AR_COND_DIM,
    ):
        super().__init__()
        self.use_ar_conditioning = bool(use_ar_conditioning)
        self.ar_cond_dim = int(ar_cond_dim) if self.use_ar_conditioning else 0
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = int(self.backbone.num_features)

        cond_in = int(text_feat_dim) + self.ar_cond_dim
        self.text_proj = nn.Sequential(
            nn.Linear(cond_in, hidden_dim),
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

    def forward(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        ar_conditioning: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        img_feat = self.backbone(images)
        if self.use_ar_conditioning:
            if ar_conditioning is None:
                ar_conditioning = default_unknown_ar_batch(
                    images.shape[0], images.device, dtype=text_features.dtype
                )
            x_txt = torch.cat([text_features, ar_conditioning], dim=-1)
        else:
            x_txt = text_features
        txt_feat = self.text_proj(x_txt)
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
    *,
    use_ar_conditioning: bool = True,
    ar_cond_dim: int = AR_COND_DIM,
) -> ViTQualityAdherenceModel:
    return ViTQualityAdherenceModel(
        model_name=model_name,
        pretrained=pretrained,
        text_feat_dim=text_feat_dim,
        hidden_dim=hidden_dim,
        use_ar_conditioning=use_ar_conditioning,
        ar_cond_dim=ar_cond_dim,
    )
