"""Tests for ``utils.training.error_handling`` helpers."""

from __future__ import annotations

import torch
from utils.training.error_handling import get_model_info


def test_get_model_info_linear() -> None:
    m = torch.nn.Linear(4, 2, bias=True)
    info = get_model_info(m)
    assert info["total_parameters"] == 10  # 4*2 + 2
    assert info["trainable_parameters"] == 10
    assert "model_size_mb" in info
    assert info["device"] == torch.device("cpu")
