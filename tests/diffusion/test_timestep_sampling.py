"""Tests for ``diffusion/timestep_sampling.py``."""

from __future__ import annotations

import pytest
import torch
from diffusion.timestep_sampling import sample_training_timesteps


@pytest.mark.parametrize("T", [1, 10, 1000])
@pytest.mark.parametrize("B", [1, 8, 32])
def test_uniform_shape_and_range(T: int, B: int) -> None:
    device = torch.device("cpu")
    t = sample_training_timesteps(B, T, device=device, mode="uniform")
    assert t.shape == (B,)
    assert t.dtype == torch.long
    assert (t >= 0).all() and (t < T).all()


def test_logit_normal_range() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    T = 1000
    B = 4096
    t = sample_training_timesteps(B, T, device=device, mode="logit_normal", logit_mean=0.0, logit_std=1.0)
    assert t.shape == (B,)
    assert (t >= 0).all() and (t <= T - 1).all()


def test_high_noise_biased_high_t() -> None:
    """Beta(2,1) should put more mass near u=1 → large discrete t."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    T = 1000
    B = 8000
    t = sample_training_timesteps(B, T, device=device, mode="high_noise")
    assert (t >= 0).all() and (t <= T - 1).all()
    mid = T // 2
    assert (t >= mid).float().mean() > 0.55


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        sample_training_timesteps(4, 100, device=torch.device("cpu"), mode="not_a_mode")


def test_zero_timesteps_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_training_timesteps(1, 0, device=torch.device("cpu"))


def test_exported_from_diffusion_package() -> None:
    from diffusion import sample_training_timesteps as exported

    t = exported(2, 50, device=torch.device("cpu"), mode="uniform")
    assert t.shape == (2,)
