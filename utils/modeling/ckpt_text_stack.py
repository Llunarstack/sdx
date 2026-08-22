"""
Read ``text_encoder_mode`` from a DiT checkpoint without loading weights.
"""

from __future__ import annotations

from pathlib import Path


def text_encoder_mode_from_checkpoint(ckpt_path: str) -> str:
    """Return ``t5``, ``triple``, or ``penta`` from embedded ``TrainConfig`` (default ``t5``)."""
    import torch

    p = Path(str(ckpt_path).strip())
    if not p.is_file():
        return "t5"
    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception:
        return "t5"
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if cfg is None:
        return "t5"
    mode = str(getattr(cfg, "text_encoder_mode", "t5") or "t5").lower()
    if mode in ("triple", "penta"):
        return mode
    return "t5"


def text_encoder_mode_label(mode: str) -> str:
    """Human-readable stack description."""
    m = str(mode or "t5").lower()
    if m == "penta":
        return "T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L"
    if m == "triple":
        return "T5 + CLIP-L + CLIP-bigG"
    return "T5 only"


def fusion_present_in_checkpoint(ckpt_path: str) -> bool:
    import torch

    p = Path(str(ckpt_path).strip())
    if not p.is_file():
        return False
    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    fusion = ckpt.get("text_encoder_fusion")
    return isinstance(fusion, dict) and bool(fusion)


__all__ = [
    "fusion_present_in_checkpoint",
    "text_encoder_mode_from_checkpoint",
    "text_encoder_mode_label",
]
