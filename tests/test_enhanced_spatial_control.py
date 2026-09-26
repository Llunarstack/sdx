"""Tests for EnhancedDiT ``SpatialControlModule`` count-modulated layout attention."""

from __future__ import annotations

import torch
import torch.nn as nn
from models.enhanced_dit import SpatialControlModule


class _FixedCount(nn.Module):
    def __init__(self, value: float, num_objects: int) -> None:
        super().__init__()
        self.value = value
        self.num_objects = num_objects

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled.new_full((pooled.shape[0], self.num_objects), self.value)


def test_spatial_control_preserves_shape() -> None:
    torch.manual_seed(0)
    mod = SpatialControlModule(hidden_size=32, num_objects=4)
    x = torch.randn(2, 16, 32)
    layout = torch.rand(2, 4, 4)
    out = mod(x, layout)
    assert out.shape == (2, 16, 32)


def test_count_constraint_modulates_layout_attention() -> None:
    torch.manual_seed(0)
    mod = SpatialControlModule(hidden_size=32, num_objects=4)
    x = torch.randn(2, 16, 32)
    layout = torch.rand(2, 4, 4)

    mod.count_constraint = _FixedCount(0.0, num_objects=4)
    out_zero = mod(x, layout)
    mod.count_constraint = _FixedCount(1.0, num_objects=4)
    out_one = mod(x, layout)

    assert out_zero.shape == (2, 16, 32)
    assert out_one.shape == (2, 16, 32)
    assert not torch.allclose(out_zero, out_one)
