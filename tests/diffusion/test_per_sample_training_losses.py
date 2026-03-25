"""``per_sample_training_losses`` matches batch mean of ``training_losses``."""

import torch

from diffusion import create_diffusion


def test_per_sample_matches_training_mean():
    diff = create_diffusion(num_timesteps=20, beta_schedule="linear", prediction_type="epsilon")

    class Dummy(torch.nn.Module):
        def forward(self, x, t, **kw):
            return torch.zeros_like(x)

    m = Dummy()
    b = 3
    x = torch.randn(b, 4, 4, 4)
    t = torch.randint(0, 20, (b,), dtype=torch.long)
    noise = torch.randn_like(x)
    mk = {"encoder_hidden_states": torch.zeros(b, 2, 8)}

    ps = diff.per_sample_training_losses(
        m, x, t, model_kwargs=mk, noise=noise, refinement_prob=0.0, refinement_max_t=0
    )
    assert ps.shape == (b,)
    agg = diff.training_losses(
        m, x, t, model_kwargs=mk, noise=noise, refinement_prob=0.0, refinement_max_t=0
    )["loss"]
    assert torch.allclose(ps.mean(), agg, rtol=1e-5, atol=1e-5)
