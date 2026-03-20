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

    # Model
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    text_feat_dim: int = 8
    hidden_dim: int = 256

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    quality_loss_weight: float = 1.0
    adherence_loss_weight: float = 1.0

    # Runtime
    device: str = "cuda"
    seed: int = 42
    out_dir: str = "vit_runs"

