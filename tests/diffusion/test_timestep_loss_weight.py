"""Tests for ``diffusion/losses/timestep_loss_weight.py`` (shim: ``diffusion/timestep_loss_weight``)."""

from __future__ import annotations

import torch
from diffusion.timestep_loss_weight import get_timestep_loss_weight


def test_min_snr_matches_manual() -> None:
    snr = torch.tensor([0.5, 5.0, 50.0])
    alpha = torch.tensor([0.9, 0.5, 0.1])
    gamma = 5.0
    w = get_timestep_loss_weight(
        "min_snr",
        snr=snr,
        alpha_cumprod=alpha,
        min_snr_gamma=gamma,
        loss_weighting_sigma_data=0.5,
    )
    manual = torch.clamp(snr, max=gamma) / (snr + 1e-8)
    assert torch.allclose(w, manual)


def test_min_snr_soft_formula() -> None:
    snr = torch.tensor([1.0, 10.0])
    alpha = torch.ones_like(snr)
    gamma = 5.0
    w = get_timestep_loss_weight(
        "min_snr_soft",
        snr=snr,
        alpha_cumprod=alpha,
        min_snr_gamma=gamma,
        loss_weighting_sigma_data=0.5,
    )
    expected = gamma / (snr + gamma + 1e-8)
    assert torch.allclose(w, expected)


def test_min_snr_gamma_zero_is_ones() -> None:
    snr = torch.tensor([1.0, 2.0])
    alpha = torch.tensor([0.5, 0.5])
    w = get_timestep_loss_weight(
        "min_snr",
        snr=snr,
        alpha_cumprod=alpha,
        min_snr_gamma=0.0,
        loss_weighting_sigma_data=0.5,
    )
    assert (w == 1.0).all()


def test_edm_delegates() -> None:
    alpha = torch.tensor([0.25, 0.64])
    w = get_timestep_loss_weight(
        "edm",
        snr=None,
        alpha_cumprod=alpha,
        min_snr_gamma=5.0,
        loss_weighting_sigma_data=0.5,
    )
    assert w.shape == (2,)
    assert (w > 0).all()


def test_exported_from_package() -> None:
    from diffusion import get_timestep_loss_weight as exported

    snr = torch.tensor([2.0])
    alpha = torch.tensor([0.5])
    w = exported(
        "min_snr_soft",
        snr=snr,
        alpha_cumprod=alpha,
        min_snr_gamma=3.0,
        loss_weighting_sigma_data=0.5,
    )
    assert w.shape == (1,)
