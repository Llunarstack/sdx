"""Tests for text encoder stack readiness and checkpoint introspection."""

import torch
from config.train_config import TrainConfig
from utils.modeling.text_encoder_stack import (
    PENTA_CATALOG,
    TRIPLE_CATALOG,
    catalog_for_mode,
    checkpoint_text_stack,
    stack_status,
    stack_status_lines,
    text_encoder_mode_label,
)


def test_catalog_for_mode():
    assert catalog_for_mode("penta") == PENTA_CATALOG
    assert catalog_for_mode("triple") == TRIPLE_CATALOG
    assert catalog_for_mode("t5") == ("T5-XXL",)


def test_stack_status_returns_slots():
    st = stack_status("penta")
    assert st.mode == "penta"
    assert len(st.slots) == 5
    assert {s.name for s in st.slots} == set(PENTA_CATALOG)


def test_stack_status_lines_markdown():
    lines = stack_status_lines("triple")
    assert any("triple" in ln for ln in lines)
    assert any("CLIP-ViT-L-14" in ln for ln in lines)


def test_text_encoder_mode_label():
    assert "LongCLIP-L" in text_encoder_mode_label("penta")
    assert "CLIP-ViT-L-14" in text_encoder_mode_label("triple")
    assert text_encoder_mode_label("t5") == "T5-XXL"


def test_checkpoint_text_stack(tmp_path):
    ckpt = tmp_path / "mini.pt"
    torch.save(
        {"config": TrainConfig(text_encoder_mode="penta"), "model": {}, "text_encoder_fusion": {"w": 1}}, str(ckpt)
    )
    assert checkpoint_text_stack(str(ckpt)) == ("penta", True)
    assert checkpoint_text_stack(str(tmp_path / "missing.pt")) == ("t5", False)
