"""Smoke tests for models.model_enhancements."""

from __future__ import annotations

import torch

from models.model_enhancements import DropPath, RMSNorm, SE1x1, TokenFiLM, apply_gate_residual


def test_rmsnorm_shape() -> None:
    m = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    y = m(x)
    assert y.shape == x.shape


def test_drop_path_identity_when_eval() -> None:
    dp = DropPath(0.5)
    x = torch.ones(2, 4)
    dp.eval()
    assert torch.allclose(dp(x), x)


def test_token_film() -> None:
    m = TokenFiLM(dim=16, cond_dim=8)
    x = torch.randn(2, 5, 16)
    c = torch.randn(2, 8)
    y = m(x, c)
    assert y.shape == x.shape


def test_se1x1() -> None:
    m = SE1x1(32, reduction=4)
    x = torch.randn(2, 7, 32)
    y = m(x)
    assert y.shape == x.shape


def test_apply_gate_residual() -> None:
    x = torch.ones(2, 3)
    b = torch.ones(2, 3) * 2
    y = apply_gate_residual(x, b, torch.tensor(0.5))
    assert torch.allclose(y, torch.ones(2, 3) * 2.0)
