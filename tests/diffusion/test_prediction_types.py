"""GaussianDiffusion prediction_type: epsilon, v, x0 — training target and x0/noise conversion."""

import torch

from diffusion import create_diffusion


def test_x0_training_target_matches_x_start():
    diff = create_diffusion(num_timesteps=100, prediction_type="x0", beta_schedule="linear")
    device = torch.device("cpu")
    B, C, H, W = 2, 4, 8, 8
    x_start = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 100, (B,), device=device)
    noise = torch.randn_like(x_start)

    def model(x, tt, **kw):
        return x_start

    out = diff.training_losses(model, x_start, t, noise=noise, min_snr_gamma=0.0, loss_weighting="unit")
    assert out["loss"].item() == 0.0


def test_predict_x0_and_noise_x0_roundtrip():
    diff = create_diffusion(num_timesteps=100, prediction_type="x0", beta_schedule="linear")
    diff._to_device(torch.device("cpu"))
    B, C, H, W = 1, 4, 8, 8
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([50], dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = diff.q_sample(x0, t, noise=noise)
    x0_hat, eps_hat = diff._predict_x0_and_noise(x0, x_t, t)
    assert torch.allclose(x0_hat, x0)
    assert torch.allclose(eps_hat, noise, atol=1e-4, rtol=1e-3)


def test_x0_plus_spectral_sfp_loss_matches_x_start():
    """x0 prediction + SFP: loss is FFT-weighted (pred - x0); zero when pred == x_start."""
    diff = create_diffusion(num_timesteps=100, prediction_type="x0", beta_schedule="linear")
    device = torch.device("cpu")
    B, C, H, W = 2, 4, 8, 8
    x_start = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 100, (B,), device=device)
    noise = torch.randn_like(x_start)

    def model(x, tt, **kw):
        return x_start

    out = diff.training_losses(
        model,
        x_start,
        t,
        noise=noise,
        min_snr_gamma=0.0,
        loss_weighting="unit",
        use_spectral_sfp_loss=True,
    )
    assert out["loss"].item() == 0.0


def test_epsilon_predict_recover_x0():
    diff = create_diffusion(num_timesteps=100, prediction_type="epsilon", beta_schedule="linear")
    diff._to_device(torch.device("cpu"))
    B, C, H, W = 1, 4, 8, 8
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([50], dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = diff.q_sample(x0, t, noise=noise)
    x0_hat, eps_hat = diff._predict_x0_and_noise(noise, x_t, t)
    assert torch.allclose(eps_hat, noise)
    assert torch.allclose(x0_hat, x0, atol=1e-4, rtol=1e-3)
