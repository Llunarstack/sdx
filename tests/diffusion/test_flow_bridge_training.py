"""Smoke tests for flow-matching loss, bridge shuffle, speculative CFG helper."""

import torch

from diffusion import create_diffusion
from diffusion.bridge_training import bridge_aux_vp_loss, shuffle_pair_latents
from diffusion.flow_matching import flow_matching_per_sample_losses
from utils.generation.speculative_denoise import speculative_cfg_prediction


def test_shuffle_pair_latents_shape():
    x = torch.randn(4, 2, 8, 8)
    y = shuffle_pair_latents(x)
    assert y.shape == x.shape


def test_flow_matching_per_sample_losses_shape():
    class _M(torch.nn.Module):
        def forward(self, xt, t, **kwargs):
            return xt

    m = _M()
    x0 = torch.randn(2, 4, 16, 16)
    eps = torch.randn_like(x0)
    L = flow_matching_per_sample_losses(m, x0, eps, 32, {})
    assert L.shape == (2,)


def test_bridge_aux_vp_loss_scalar():
    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=16,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    diffusion._to_device(torch.device("cpu"))

    class _M(torch.nn.Module):
        def forward(self, xt, t, **kwargs):
            return torch.zeros_like(xt)

    m = _M()
    lat = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 16, (2,), dtype=torch.long)
    loss = bridge_aux_vp_loss(
        diffusion,
        m,
        lat,
        t,
        {},
        mix_lambda=0.2,
        noise=None,
        noise_offset=0.0,
        min_snr_gamma=0.0,
        loss_weighting="unit",
        loss_weighting_sigma_data=0.5,
        use_spectral_sfp_loss=False,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_flow_sampler_recovers_linear_bridge():
    """Ideal velocity field v = eps - x0; Euler integration should land near x0."""
    T = 24
    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=T,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    diffusion._to_device(torch.device("cpu"))
    x0 = torch.randn(1, 4, 8, 8)
    eps = torch.randn_like(x0)

    class _IdealV(torch.nn.Module):
        def forward(self, xt, t, **kwargs):
            return eps - x0

    out = diffusion.sample_loop(
        _IdealV(),
        (1, 4, 8, 8),
        flow_matching_sample=True,
        flow_solver="euler",
        num_inference_steps=48,
        cfg_scale=1.0,
        device="cpu",
        dtype=torch.float32,
        flow_init_noise=eps.clone(),
    )
    err = (out - x0).float().abs().mean()
    assert err < 0.05


def test_speculative_cfg_prediction_runs():
    class _M(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return x * 0.1

    m = _M()
    x = torch.randn(1, 4, 8, 8)
    tb = torch.zeros(1, dtype=torch.long)
    mkc = {"encoder_hidden_states": torch.randn(1, 8, 64)}
    mku = {"encoder_hidden_states": torch.randn(1, 8, 64)}
    out = speculative_cfg_prediction(
        m,
        x,
        tb,
        model_kwargs_cond=mkc,
        model_kwargs_uncond=mku,
        cfg_scale=5.0,
        draft_cfg_scale=2.0,
        cfg_rescale=0.0,
        close_thresh=1.0,
        blend_on_close=0.5,
    )
    assert out.shape == x.shape
