"""Volatile CFG path in sample_loop (no crash, shape preserved)."""

import torch

from diffusion import create_diffusion


def test_sample_loop_volatile_cfg_boost_runs():
    diff = create_diffusion(num_timesteps=50, beta_schedule="linear", prediction_type="epsilon")

    class Dummy(torch.nn.Module):
        def forward(self, x, t, **kw):
            return torch.zeros_like(x)

    m = Dummy()
    x0 = diff.sample_loop(
        m,
        (1, 4, 4, 4),
        model_kwargs_cond={},
        model_kwargs_uncond={"encoder_hidden_states": torch.zeros(1, 2, 8)},
        cfg_scale=2.0,
        num_inference_steps=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        volatile_cfg_boost=0.12,
        volatile_cfg_quantile=0.6,
        volatile_cfg_window=4,
    )
    assert x0 is not None and x0.shape == (1, 4, 4, 4)
