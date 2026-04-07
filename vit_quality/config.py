from __future__ import annotations

from dataclasses import dataclass


@dataclass
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

