"""

Also holds what used to be: config, losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.architecture.ar_block_conditioning import AR_COND_DIM, default_unknown_ar_batch

# ==================================================================================================
# model
# ==================================================================================================


class ViTQualityAdherenceModel(nn.Module):
    """
    ViT backbone with two heads:
    - quality_logit: binary quality classification (good/bad)
    - adherence_score: regression in [0,1] for prompt adherence

    Optional DiT AR regime conditioning (``use_ar_conditioning``): concatenates a 4-D one-hot
    (full DiT / AR 2x2 / AR 4x4 / unknown) so scores match how the generator was trained.
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
        fuse_dropout: float = 0.1,
        text_proj_dropout: float = 0.0,
        backbone_grad_checkpointing: bool = False,
        timm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.use_ar_conditioning = bool(use_ar_conditioning)
        self.ar_cond_dim = int(ar_cond_dim) if self.use_ar_conditioning else 0
        self.text_proj_dropout = float(text_proj_dropout)
        tk: dict[str, Any] = dict(timm_kwargs or {})
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            **tk,
        )
        if backbone_grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        feat_dim = int(self.backbone.num_features)
        cond_in = int(text_feat_dim) + self.ar_cond_dim
        self.text_proj = nn.Sequential(
            nn.Linear(cond_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        fd = float(fuse_dropout)
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(fd),
        )
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.adherence_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        ar_conditioning: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        img_feat = self.backbone(images)
        if self.use_ar_conditioning:
            if ar_conditioning is None:
                ar_conditioning = default_unknown_ar_batch(images.shape[0], images.device, dtype=text_features.dtype)
            x_txt = torch.cat([text_features, ar_conditioning], dim=-1)
        else:
            x_txt = text_features
        txt_feat = self.text_proj(x_txt)
        if self.text_proj_dropout > 0 and self.training:
            txt_feat = F.dropout(txt_feat, p=self.text_proj_dropout, training=True)
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
    fuse_dropout: float = 0.1,
    text_proj_dropout: float = 0.0,
    backbone_grad_checkpointing: bool = False,
    timm_kwargs: dict[str, Any] | None = None,
) -> ViTQualityAdherenceModel:
    return ViTQualityAdherenceModel(
        model_name=model_name,
        pretrained=pretrained,
        text_feat_dim=text_feat_dim,
        hidden_dim=hidden_dim,
        use_ar_conditioning=use_ar_conditioning,
        ar_cond_dim=ar_cond_dim,
        fuse_dropout=fuse_dropout,
        text_proj_dropout=text_proj_dropout,
        backbone_grad_checkpointing=backbone_grad_checkpointing,
        timm_kwargs=timm_kwargs,
    )


# ==================================================================================================
# config: config
# ==================================================================================================


@dataclass(slots=True)
class ViTConfig:
    # Data
    manifest_jsonl: str = ""
    image_root: str = ""
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    training_augment: bool = False

    # Model
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    # Optional extra kwargs for timm.create_model (e.g. qk_norm, reg tokens on supported archs).
    timm_kwargs: dict | None = None
    text_feat_dim: int = 8
    # "handcrafted" = legacy 8-D caption stats; "clip" = frozen CLIP text embeddings (see text_encoder.py).
    text_embed_mode: str = "handcrafted"
    text_encoder_model_id: str = "openai/clip-vit-base-patch32"
    hidden_dim: int = 256
    fuse_dropout: float = 0.1
    text_proj_dropout: float = 0.0
    backbone_grad_checkpointing: bool = False
    # DiT block-AR regime (0/2/4) as extra ViT conditioner; see utils/architecture/ar_block_conditioning.py
    use_ar_conditioning: bool = True
    ar_cond_dim: int = 4

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    quality_loss_weight: float = 1.0
    adherence_loss_weight: float = 1.0
    # 0 = plain BCE; >0 focal gamma (imbalanced quality labels)
    focal_loss_gamma: float = 0.0
    focal_loss_alpha: float | None = None
    # If True, SmoothL1 on adherence instead of MSE (outlier-robust)
    adherence_smooth_l1: bool = False
    adherence_smooth_l1_beta: float = 0.1

    # Runtime
    device: str = "cuda"
    seed: int = 42
    out_dir: str = "vit_runs"


# ==================================================================================================
# losses: losses
# ==================================================================================================


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    """
    Focal loss for imbalanced binary quality labels (down-weights easy examples).
    """
    gamma = float(gamma)
    if gamma <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    with torch.no_grad():
        prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal = (1.0 - p_t.clamp(min=1e-6, max=1.0 - 1e-6)) ** gamma * bce
    if alpha is not None:
        a = float(alpha)
        w = a * targets + (1.0 - a) * (1.0 - targets)
        focal = focal * w
    return focal.mean()


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.15,
    min_target_gap: float = 0.05,
) -> torch.Tensor:
    """
    Pairwise ranking for quality/adherence ordering.
    If target[i] > target[j] by at least min_target_gap, enforce pred[i] > pred[j] + margin.
    """
    b = int(pred.shape[0])
    if b < 2:
        return pred.new_zeros(())

    pred_i = pred.unsqueeze(1)
    pred_j = pred.unsqueeze(0)
    target_i = target.unsqueeze(1)
    target_j = target.unsqueeze(0)
    gap = target_i - target_j
    mask = gap > float(min_target_gap)
    if not mask.any():
        return pred.new_zeros(())

    dif = pred_i - pred_j
    losses = F.relu(float(margin) - dif)
    return losses[mask].mean()
