"""Load ViT quality/adherence checkpoints for reuse in ``infer.py`` and tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from ViT.model import ViTQualityAdherenceModel, build_vit_model


def load_vit_quality_checkpoint(
    ckpt_path: str | Path,
    *,
    use_ema: bool = False,
) -> Tuple[ViTQualityAdherenceModel, Dict[str, Any]]:
    """
    Load a ViT ``best.pt``-style checkpoint and return ``(model, config_dict)`` on **CPU**.

    Call ``model.to(device)`` after loading. Raises if checkpoint missing or incompatible.
    """
    path = Path(ckpt_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}

    model_name = cfg.get("model_name", "vit_base_patch16_224")
    text_feat_dim = int(cfg.get("text_feat_dim", 8))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    fuse_dropout = float(cfg.get("fuse_dropout", 0.1))
    text_proj_dropout = float(cfg.get("text_proj_dropout", 0.0))
    backbone_grad_checkpointing = bool(cfg.get("backbone_grad_checkpointing", False))
    # Older checkpoints omit these → full bidirectional DiT path (no AR side-info).
    use_ar_conditioning = bool(cfg.get("use_ar_conditioning", False))
    ar_cond_dim = int(cfg.get("ar_cond_dim", 4))

    model = build_vit_model(
        model_name=str(model_name),
        pretrained=False,
        text_feat_dim=text_feat_dim,
        hidden_dim=hidden_dim,
        use_ar_conditioning=use_ar_conditioning,
        ar_cond_dim=ar_cond_dim,
        fuse_dropout=fuse_dropout,
        text_proj_dropout=text_proj_dropout,
        backbone_grad_checkpointing=backbone_grad_checkpointing,
    )
    if use_ema and bool(ckpt.get("ema_state_dict")):
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["state_dict"], strict=True)

    model.eval()
    return model, cfg


def vit_model_parameter_report(model: ViTQualityAdherenceModel) -> Dict[str, int]:
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
