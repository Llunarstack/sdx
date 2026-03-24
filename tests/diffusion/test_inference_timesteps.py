"""Inference timestep schedules: length, monotonicity, registered names."""

import numpy as np
import torch

from diffusion import build_inference_timesteps, create_diffusion, list_timestep_schedules
from diffusion.inference_timesteps import _enforce_strict_descending


def test_list_includes_karras_and_ddim():
    names = list_timestep_schedules()
    assert "ddim" in names and "euler" in names and "karras_rho" in names


def test_build_descending_length():
    T = 1000
    ac = np.linspace(0.9999, 1e-4, T, dtype=np.float64)
    ac = np.clip(ac, 1e-6, 1 - 1e-6)
    for name in ("ddim", "euler", "karras_rho", "snr_uniform", "quad_cosine"):
        steps = build_inference_timesteps(name, T, 25, ac, karras_rho=7.0)
        assert steps.shape == (25,)
        assert steps[0] >= steps[-1]
        d = np.diff(steps.astype(np.int64))
        assert (d <= 0).all(), name


def test_set_timesteps_and_sample_loop_smoke():
    diff = create_diffusion(num_timesteps=50, beta_schedule="linear", prediction_type="epsilon")
    diff.set_timesteps(10, timestep_schedule="karras_rho", karras_rho=5.0)
    assert diff.timesteps.shape[0] == 10

    class Dummy(torch.nn.Module):
        def forward(self, x, t, **kw):
            return torch.zeros_like(x)

    m = Dummy()
    x0 = diff.sample_loop(
        m,
        (1, 4, 4, 4),
        model_kwargs_cond={},
        model_kwargs_uncond=None,
        cfg_scale=1.0,
        num_inference_steps=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        scheduler="snr_uniform",
        solver="heun",
    )
    assert x0 is not None and x0.shape == (1, 4, 4, 4)


def test_enforce_strict_descending():
    x = np.array([99, 99, 50, 50, 0], dtype=np.int64)
    y = _enforce_strict_descending(x, 100)
    assert (np.diff(y) <= 0).all()
    assert y[0] <= 99
