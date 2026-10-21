"""Tests for EnhancedDiT: count-modulated layout attention and gradient checkpointing."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from models.enhanced_dit import EnhancedDiT, SpatialControlModule


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


def test_gradient_checkpointing_recomputes_each_block() -> None:
    torch.manual_seed(0)
    plain = EnhancedDiT(input_size=8, hidden_size=32, depth=3, num_heads=4, num_classes=4, class_dropout_prob=0.0)
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ckpt = copy.deepcopy(plain)
    ckpt.enable_gradient_checkpointing()
    x, t, y = torch.randn(2, 4, 8, 8), torch.tensor([10, 500]), torch.tensor([1, 2])
    for m in (plain, ckpt):
        m(x, t, y).pow(2).mean().backward()
    for a, b in zip(plain.parameters(), ckpt.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)
