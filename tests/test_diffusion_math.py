"""
Tests for diffusion math: schedules, SNR, alpha_cumprod, loss weights.

These are the numerically critical paths — wrong values here silently corrupt
training without any obvious error. Run with:

    pytest tests/test_diffusion_math.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

class TestBetaSchedules:
    def test_linear_length(self):
        from diffusion.schedules import linear_beta_schedule
        b = linear_beta_schedule(1000)
        assert b.shape == (1000,)

    def test_linear_monotone_increasing(self):
        from diffusion.schedules import linear_beta_schedule
        b = linear_beta_schedule(1000)
        assert np.all(np.diff(b) >= 0), "linear betas must be non-decreasing"

    def test_linear_range(self):
        from diffusion.schedules import linear_beta_schedule
        b = linear_beta_schedule(1000)
        assert b[0] >= 1e-5
        assert b[-1] <= 0.999

    def test_cosine_length(self):
        from diffusion.schedules import cosine_beta_schedule
        b = cosine_beta_schedule(1000)
        assert b.shape == (1000,)

    def test_cosine_range(self):
        from diffusion.schedules import cosine_beta_schedule
        b = cosine_beta_schedule(1000)
        assert np.all(b > 0)
        assert np.all(b < 1)

    def test_squaredcos_v2_length(self):
        from diffusion.schedules import squared_cosine_beta_schedule_v2
        b = squared_cosine_beta_schedule_v2(1000)
        assert b.shape == (1000,)

    def test_squaredcos_v2_range(self):
        from diffusion.schedules import squared_cosine_beta_schedule_v2
        b = squared_cosine_beta_schedule_v2(1000)
        assert np.all(b > 0)
        assert np.all(b <= 0.999)

    def test_get_beta_schedule_dispatch(self):
        from diffusion.schedules import get_beta_schedule
        for name in ("linear", "cosine", "squaredcos_cap_v2"):
            b = get_beta_schedule(name, 100)
            assert b.shape == (100,), f"failed for {name}"

    def test_get_beta_schedule_unknown_raises(self):
        from diffusion.schedules import get_beta_schedule
        with pytest.raises(ValueError, match="Unknown beta schedule"):
            get_beta_schedule("bogus_schedule", 100)

    def test_sigmoid_length_and_range(self):
        from diffusion.schedules import sigmoid_beta_schedule
        b = sigmoid_beta_schedule(500)
        assert b.shape == (500,)
        assert np.all(b > 0) and np.all(b < 1)


# ---------------------------------------------------------------------------
# SNR / alpha_cumprod
# ---------------------------------------------------------------------------

class TestSnrUtils:
    def test_alpha_cumprod_monotone_decreasing(self):
        from diffusion.schedules import linear_beta_schedule
        from diffusion.snr_utils import alpha_cumprod_from_betas
        betas = linear_beta_schedule(1000)
        ac = alpha_cumprod_from_betas(betas)
        assert ac.shape == (1000,)
        assert np.all(np.diff(ac) < 0), "alpha_cumprod must be strictly decreasing"

    def test_alpha_cumprod_bounds(self):
        from diffusion.schedules import linear_beta_schedule
        from diffusion.snr_utils import alpha_cumprod_from_betas
        betas = linear_beta_schedule(1000)
        ac = alpha_cumprod_from_betas(betas)
        assert ac[0] < 1.0
        assert ac[-1] > 0.0

    def test_snr_positive(self):
        from diffusion.schedules import linear_beta_schedule
        from diffusion.snr_utils import snr_from_betas
        betas = linear_beta_schedule(1000)
        snr = snr_from_betas(betas)
        assert np.all(snr >= 0)

    def test_snr_monotone_decreasing(self):
        from diffusion.schedules import linear_beta_schedule
        from diffusion.snr_utils import snr_from_betas
        betas = linear_beta_schedule(1000)
        snr = snr_from_betas(betas)
        assert np.all(np.diff(snr) < 0), "SNR must decrease as noise increases"

    def test_snr_from_alpha_cumprod_matches_manual(self):
        from diffusion.schedules import linear_beta_schedule
        from diffusion.snr_utils import alpha_cumprod_from_betas, snr_from_alpha_cumprod
        betas = linear_beta_schedule(100)
        ac = alpha_cumprod_from_betas(betas)
        snr = snr_from_alpha_cumprod(ac)
        expected = ac / (1.0 - ac + 1e-8)
        np.testing.assert_allclose(snr, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# GaussianDiffusion construction
# ---------------------------------------------------------------------------

class TestGaussianDiffusion:
    def test_create_diffusion_default(self):
        from diffusion import create_diffusion
        d = create_diffusion(num_timesteps=1000)
        assert d.num_timesteps == 1000
        assert d.alpha_cumprod.shape == (1000,)
        assert d.snr.shape == (1000,)

    def test_alpha_cumprod_tensor_monotone(self):
        from diffusion import create_diffusion
        d = create_diffusion(num_timesteps=1000)
        ac = d.alpha_cumprod.numpy()
        # Non-increasing; tail may be flat zero under float32.
        assert np.all(np.diff(ac) <= 0)
        assert ac[0] > ac[-1]

    def test_sqrt_tensors_positive(self):
        from diffusion import create_diffusion
        d = create_diffusion(num_timesteps=1000)
        assert (d.sqrt_alpha_cumprod >= 0).all()
        assert (d.sqrt_one_minus_alpha_cumprod >= 0).all()

    def test_q_sample_shape(self):
        from diffusion import create_diffusion
        d = create_diffusion(num_timesteps=1000)
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        xt = d.q_sample(x, t)
        assert xt.shape == x.shape

    def test_q_sample_noise_offset(self):
        """Noise offset should not change shape."""
        from diffusion import create_diffusion
        d = create_diffusion(num_timesteps=1000)
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        xt = d.q_sample(x, t, noise_offset=0.1)
        assert xt.shape == x.shape

    def test_prediction_types(self):
        from diffusion import create_diffusion
        for pt in ("epsilon", "v", "x0"):
            d = create_diffusion(num_timesteps=100, prediction_type=pt)
            assert d.prediction_type == pt

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "squaredcos_cap_v2"])
    def test_beta_schedules(self, schedule):
        from diffusion import create_diffusion
        d = create_diffusion(num_timesteps=100, beta_schedule=schedule)
        assert d.num_timesteps == 100


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------

class TestLossWeights:
    def _make_tensors(self, T=100, device="cpu"):
        from diffusion.schedules import linear_beta_schedule
        from diffusion.snr_utils import alpha_cumprod_from_betas, snr_from_alpha_cumprod
        betas = linear_beta_schedule(T)
        ac = torch.from_numpy(alpha_cumprod_from_betas(betas)).float().to(device)
        snr = torch.from_numpy(snr_from_alpha_cumprod(ac.numpy())).float().to(device)
        # Sample a batch of random timesteps
        t = torch.randint(0, T, (8,))
        return ac[t], snr[t]

    def test_min_snr_shape(self):
        from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight
        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight("min_snr", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
        assert w.shape == (8,)
        assert (w > 0).all()

    def test_min_snr_capped(self):
        """min_snr weights must be <= 1 when gamma=5 (cap at SNR=5)."""
        from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight
        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight("min_snr", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
        assert (w <= 1.0 + 1e-5).all(), "min_snr weights should be <= 1"

    def test_unit_weight_ones(self):
        from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight
        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight("unit", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
        torch.testing.assert_close(w, torch.ones_like(w))

    def test_min_snr_soft_shape(self):
        from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight
        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight("min_snr_soft", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
        assert w.shape == (8,)
        assert (w > 0).all()

    def test_edm_weight_positive(self):
        from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight
        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight("edm", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
        assert (w > 0).all()


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------

class TestTimestepSampling:
    def test_uniform_range(self):
        from diffusion.timestep_sampling import sample_training_timesteps
        t = sample_training_timesteps(256, 1000, device=torch.device("cpu"), mode="uniform")
        assert t.shape == (256,)
        assert t.min() >= 0
        assert t.max() < 1000

    def test_logit_normal_range(self):
        from diffusion.timestep_sampling import sample_training_timesteps
        t = sample_training_timesteps(256, 1000, device=torch.device("cpu"), mode="logit_normal")
        assert t.min() >= 0
        assert t.max() < 1000

    def test_high_noise_bias(self):
        """high_noise mode should have mean > 500 (biased toward high t)."""
        from diffusion.timestep_sampling import sample_training_timesteps
        t = sample_training_timesteps(2048, 1000, device=torch.device("cpu"), mode="high_noise")
        assert t.float().mean() > 500, "high_noise should bias toward large t"

    def test_unknown_mode_raises(self):
        from diffusion.timestep_sampling import sample_training_timesteps
        with pytest.raises(ValueError, match="Unknown timestep_sample_mode"):
            sample_training_timesteps(4, 1000, device=torch.device("cpu"), mode="bogus")

    def test_invalid_num_timesteps_raises(self):
        from diffusion.timestep_sampling import sample_training_timesteps
        with pytest.raises(ValueError):
            sample_training_timesteps(4, 0, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------

class TestFlowMatching:
    def test_velocity_target_shape(self):
        """flow_matching_per_sample_losses returns (B,) per-sample losses."""
        from diffusion.flow_matching import flow_matching_per_sample_losses

        class _IdentityModel(torch.nn.Module):
            def forward(self, x, t, **kw):
                return torch.zeros_like(x)

        model = _IdentityModel()
        x0 = torch.randn(4, 4, 8, 8)
        eps = torch.randn(4, 4, 8, 8)
        losses = flow_matching_per_sample_losses(model, x0, eps, num_timesteps=100, model_kwargs={})
        assert losses.shape == (4,)
        assert (losses >= 0).all()

    def test_shape_mismatch_raises(self):
        from diffusion.flow_matching import flow_matching_per_sample_losses

        class _M(torch.nn.Module):
            def forward(self, x, t, **kw):
                return x

        with pytest.raises(ValueError, match="must match shape"):
            flow_matching_per_sample_losses(
                _M(),
                torch.randn(2, 4, 8, 8),
                torch.randn(2, 4, 4, 4),
                num_timesteps=100,
                model_kwargs={},
            )
