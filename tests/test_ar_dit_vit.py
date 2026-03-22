"""Tests for DiT AR regime ↔ ViT conditioning bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_normalize_num_ar_blocks() -> None:
    from utils.ar_dit_vit import normalize_num_ar_blocks

    assert normalize_num_ar_blocks(0) == 0
    assert normalize_num_ar_blocks(2) == 2
    assert normalize_num_ar_blocks(4) == 4
    assert normalize_num_ar_blocks(None) == -1
    assert normalize_num_ar_blocks("x") == -1
    assert normalize_num_ar_blocks(3) == -1


def test_parse_num_ar_blocks_from_row() -> None:
    from utils.ar_dit_vit import parse_num_ar_blocks_from_row

    assert parse_num_ar_blocks_from_row({}) == -1
    assert parse_num_ar_blocks_from_row({"num_ar_blocks": 2}) == 2
    assert parse_num_ar_blocks_from_row({"dit_num_ar_blocks": 4}) == 4
    assert parse_num_ar_blocks_from_row({"ar_blocks": 0}) == 0
    assert parse_num_ar_blocks_from_row({"num_ar_blocks": 2, "dit_num_ar_blocks": 4}) == 2


def test_ar_conditioning_one_hot() -> None:
    from utils.ar_dit_vit import AR_COND_DIM, ar_conditioning_vector, batch_ar_conditioning

    assert ar_conditioning_vector(0).tolist() == [1, 0, 0, 0]
    assert ar_conditioning_vector(2).tolist() == [0, 1, 0, 0]
    assert ar_conditioning_vector(4).tolist() == [0, 0, 1, 0]
    assert ar_conditioning_vector(-1).tolist() == [0, 0, 0, 1]

    b = batch_ar_conditioning([0, -1], dtype=torch.float32)
    assert b.shape == (2, AR_COND_DIM)
    assert b[0, 0] == 1 and b[1, 3] == 1
