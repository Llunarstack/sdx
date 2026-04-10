"""Load ViT quality/adherence checkpoints for reuse in ``infer.py`` and tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from vit_quality.model import ViTQualityAdherenceModel, build_vit_model


def peek_vit_quality_config(ckpt_path: str | Path) -> Dict[str, Any]:
    """
    Load only the embedded ``config`` dict from a ViT quality checkpoint (no model build).

    Used for book-pipeline preflight and DiT/ViT AR alignment checks. Returns ``{}`` if missing
    or unreadable.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get("config") or {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def load_vit_quality_checkpoint(
    ckpt_path: str | Path,
    *,
    use_ema: bool = False,
) -> Tuple[ViTQualityAdherenceModel, Dict[str, Any]]:
    """
    Load a ViT ``best.pt``-style checkpoint and return ``(model, config_dict)`` on CPU.
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
    # Older checkpoints omit these -> full bidirectional DiT path (no AR side-info).
    use_ar_conditioning = bool(cfg.get("use_ar_conditioning", False))
    ar_cond_dim = int(cfg.get("ar_cond_dim", 4))
    timm_kw = cfg.get("timm_kwargs")
    if timm_kw is not None and not isinstance(timm_kw, dict):
        timm_kw = None

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
        timm_kwargs=timm_kw,
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

