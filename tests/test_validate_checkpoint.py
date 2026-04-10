"""Tests for ``utils.training.error_handling.validate_checkpoint``."""

from __future__ import annotations

from pathlib import Path

import torch
from utils.training.error_handling import validate_checkpoint


def test_validate_checkpoint_ema_only(tmp_path: Path) -> None:
    p = tmp_path / "ema.pt"
    torch.save({"ema": {"a": 1}}, p)
    assert validate_checkpoint(str(p)) is True


def test_validate_checkpoint_config_and_model(tmp_path: Path) -> None:
    p = tmp_path / "full.pt"
    torch.save({"config": {}, "model": {}}, p)
    assert validate_checkpoint(str(p)) is True


def test_validate_checkpoint_state_dict_only(tmp_path: Path) -> None:
    p = tmp_path / "weights.pt"
    torch.save({"state_dict": {}}, p)
    assert validate_checkpoint(str(p)) is False


def test_validate_checkpoint_missing_path(tmp_path: Path) -> None:
    assert validate_checkpoint(str(tmp_path / "nope.pt")) is False


def test_validate_checkpoint_non_dict_payload(tmp_path: Path) -> None:
    p = tmp_path / "tensor.pt"
    torch.save(torch.zeros(2), p)
    assert validate_checkpoint(str(p)) is False
