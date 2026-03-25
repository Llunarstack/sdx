import torch

from diffusion.latent_bridge import linear_latent_interp


def test_linear_latent_interp_endpoints():
    a = torch.zeros(2, 4, 2, 2)
    b = torch.ones(2, 4, 2, 2)
    t0 = torch.zeros(2)
    t1 = torch.ones(2)
    assert torch.allclose(linear_latent_interp(a, b, t0), a)
    assert torch.allclose(linear_latent_interp(a, b, t1), b)
