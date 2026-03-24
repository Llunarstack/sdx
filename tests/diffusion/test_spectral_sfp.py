"""Spectral SFP: time-frequency weights and per-sample loss."""

import torch

from diffusion.spectral_sfp import spectral_sfp_per_sample_loss, time_frequency_weights


def test_time_frequency_weights_sum_to_one():
    h, w = 16, 16
    t = torch.tensor([0, 500, 999])
    b = 3
    num_t = 1000
    device = torch.device("cpu")
    dtype = torch.float32
    wb = time_frequency_weights(h, w, t, num_t, device=device, dtype=dtype)
    assert wb.shape == (b, 1, h, w)
    s = wb.sum(dim=(-2, -1))
    assert torch.allclose(s, torch.ones(b, 1, device=device, dtype=dtype), atol=1e-4)


def test_spectral_loss_positive_and_zero_when_pred_matches():
    b, c, h, w = 2, 4, 8, 8
    pred = torch.randn(b, c, h, w)
    target = pred.clone()
    t = torch.randint(0, 1000, (b,))
    loss = spectral_sfp_per_sample_loss(pred, target, t, 1000)
    assert loss.shape == (b,)
    assert (loss < 1e-6).all()
