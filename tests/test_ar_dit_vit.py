"""Tests for DiT AR regime ↔ ViT conditioning bridge."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_normalize_num_ar_blocks() -> None:
    from utils.architecture.ar_dit_vit import normalize_num_ar_blocks

    assert normalize_num_ar_blocks(0) == 0
    assert normalize_num_ar_blocks(2) == 2
    assert normalize_num_ar_blocks(4) == 4
    assert normalize_num_ar_blocks(None) == -1
    assert normalize_num_ar_blocks("x") == -1
    assert normalize_num_ar_blocks(3) == -1
    assert normalize_num_ar_blocks("2") == 2
    assert normalize_num_ar_blocks(" 4 ") == 4


def test_parse_num_ar_blocks_from_row() -> None:
    from utils.architecture.ar_dit_vit import parse_num_ar_blocks_from_row

    assert parse_num_ar_blocks_from_row({}) == -1
    assert parse_num_ar_blocks_from_row({"num_ar_blocks": 2}) == 2
    assert parse_num_ar_blocks_from_row({"dit_num_ar_blocks": 4}) == 4
    assert parse_num_ar_blocks_from_row({"ar_blocks": 0}) == 0
    assert parse_num_ar_blocks_from_row({"num_ar_blocks": 2, "dit_num_ar_blocks": 4}) == 2
    assert parse_num_ar_blocks_from_row({"generator_num_ar_blocks": 4}) == 4
    assert parse_num_ar_blocks_from_row({"dit_config": {"num_ar_blocks": 2}}) == 2
    assert parse_num_ar_blocks_from_row({"train_config": {"dit_num_ar_blocks": 0}}) == 0
    assert parse_num_ar_blocks_from_row({"model_config": {"ar_blocks": 4}}) == 4
    assert parse_num_ar_blocks_from_row({"dit_config": {"train_config": {"num_ar_blocks": 2}}}) == 2


def test_ar_regime_label_and_tag_row() -> None:
    from utils.architecture.ar_dit_vit import ar_regime_label, tag_manifest_row_ar

    assert ar_regime_label(0) == "full_bidirectional"
    assert ar_regime_label(2) == "block_ar_2x2"
    assert ar_regime_label(4) == "block_ar_4x4"
    assert ar_regime_label(-1) == "unknown"
    assert ar_regime_label(99) == "unknown"

    row = {"image_path": "a.jpg", "caption": "x"}
    t = tag_manifest_row_ar(row, 2)
    assert t["num_ar_blocks"] == 2 and t["dit_num_ar_blocks"] == 2 and t["ar_regime"] == "block_ar_2x2"

    t2 = tag_manifest_row_ar(t, 4, overwrite=False)
    assert t2["num_ar_blocks"] == 2

    t3 = tag_manifest_row_ar(t, 4, overwrite=True)
    assert t3["num_ar_blocks"] == 4 and t3["ar_regime"] == "block_ar_4x4"


def test_read_num_ar_blocks_from_checkpoint() -> None:
    from utils.architecture.ar_dit_vit import read_num_ar_blocks_from_checkpoint

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    try:
        torch.save({"config": {"num_ar_blocks": 2}}, path)
        assert read_num_ar_blocks_from_checkpoint(path) == 2
        torch.save({"num_ar_blocks": 4}, path)
        assert read_num_ar_blocks_from_checkpoint(path) == 4
    finally:
        path.unlink(missing_ok=True)


def test_ar_conditioning_one_hot() -> None:
    from utils.architecture.ar_dit_vit import AR_COND_DIM, ar_conditioning_vector, batch_ar_conditioning

    assert ar_conditioning_vector(0).tolist() == [1, 0, 0, 0]
    assert ar_conditioning_vector(2).tolist() == [0, 1, 0, 0]
    assert ar_conditioning_vector(4).tolist() == [0, 0, 1, 0]
    assert ar_conditioning_vector(-1).tolist() == [0, 0, 0, 1]

    b = batch_ar_conditioning([0, -1], dtype=torch.float32)
    assert b.shape == (2, AR_COND_DIM)
    assert b[0, 0] == 1 and b[1, 3] == 1
