"""Tests for checkpoint text encoder mode introspection."""

from __future__ import annotations

import torch
from config.train_config import TrainConfig
from utils.modeling.ckpt_text_stack import (
    text_encoder_mode_from_checkpoint,
    text_encoder_mode_label,
)


def test_text_encoder_mode_label():
    assert "LongCLIP" in text_encoder_mode_label("penta")
    assert "CLIP-L" in text_encoder_mode_label("triple")


def test_text_encoder_mode_from_checkpoint(tmp_path):
    cfg = TrainConfig(text_encoder_mode="penta")
    ckpt = tmp_path / "mini.pt"
    torch.save({"config": cfg, "model": {}}, str(ckpt))
    assert text_encoder_mode_from_checkpoint(str(ckpt)) == "penta"
