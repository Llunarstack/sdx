"""Periodic CLIP-style monitor in sample_loop (mock score, no transformers)."""

import torch

from diffusion import create_diffusion


def test_sample_loop_periodic_alignment_cfg_boost_runs():
    diff = create_diffusion(num_timesteps=40, beta_schedule="linear", prediction_type="epsilon")

    class Dummy(torch.nn.Module):
        def forward(self, x, t, **kw):
            return torch.zeros_like(x)

    calls = []

    def _fn(step_i: int, x0p: torch.Tensor) -> float:
        calls.append(step_i)
        return -1.0  # always below threshold -> CFG boost

    m = Dummy()
    x0 = diff.sample_loop(
        m,
        (1, 4, 4, 4),
        model_kwargs_cond={},
        model_kwargs_uncond={"encoder_hidden_states": torch.zeros(1, 2, 8)},
        cfg_scale=2.0,
        num_inference_steps=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
        periodic_alignment_interval=2,
        periodic_alignment_threshold=0.5,
        periodic_alignment_cfg_boost=0.1,
        periodic_alignment_fn=_fn,
    )
    assert x0 is not None and x0.shape == (1, 4, 4, 4)
    assert len(calls) >= 1
