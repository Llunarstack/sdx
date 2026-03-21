"""Tests for diffusion/schedules.py."""

from __future__ import annotations

import numpy as np
import pytest

from diffusion.schedules import (
    cosine_beta_schedule,
    get_beta_schedule,
    linear_beta_schedule,
    sigmoid_beta_schedule,
    squared_cosine_beta_schedule_v2,
)
from diffusion.snr_utils import alpha_cumprod_from_betas, snr_from_betas


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_all_schedules_shape_and_valid(n: int) -> None:
    for name in ("linear", "cosine", "sigmoid", "squaredcos_cap_v2"):
        b = get_beta_schedule(name, n)
        assert b.shape == (n,)
        assert np.isfinite(b).all()
        assert (b >= 1e-4).all() and (b <= 0.999).all()


def test_linear_matches_historical_endpoints() -> None:
    b = linear_beta_schedule(1000)
    assert np.isclose(b[0], 0.0001)
    assert np.isclose(b[-1], 0.02)


def test_alpha_cumprod_monotone_decreasing() -> None:
    """Cumulative alpha_bar should decrease toward high noise."""
    for fn in (cosine_beta_schedule, sigmoid_beta_schedule, squared_cosine_beta_schedule_v2):
        beta = fn(500)
        ab = alpha_cumprod_from_betas(beta)
        assert ab[0] > ab[-1]
        assert (ab[:-1] >= ab[1:]).all()


def test_unknown_schedule_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_beta_schedule("not_a_schedule", 100)


def test_snr_from_betas_positive() -> None:
    b = get_beta_schedule("cosine", 200)
    snr = snr_from_betas(b)
    assert (snr > 0).all() and np.isfinite(snr).all()


def test_create_diffusion_accepts_new_schedules() -> None:
    from diffusion import create_diffusion

    for bs in ("sigmoid", "squaredcos_cap_v2"):
        d = create_diffusion(num_timesteps=100, beta_schedule=bs)
        assert d.beta.shape[0] == 100
        assert hasattr(d, "snr")
