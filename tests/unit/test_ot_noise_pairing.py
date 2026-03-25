import torch

from utils.training.ot_noise_pairing import pair_noise_to_latents, sinkhorn_plan


def test_sinkhorn_plan_rows_sum():
    B = 4
    c = torch.rand(B, B, dtype=torch.float32) * 2.0
    p = sinkhorn_plan(c, reg=0.1, n_iters=60)
    assert p.shape == (B, B)
    rs = p.sum(dim=1)
    cs = p.sum(dim=0)
    assert torch.allclose(rs, torch.full_like(rs, 1.0 / B), rtol=0.05, atol=0.05)
    assert torch.allclose(cs, torch.full_like(cs, 1.0 / B), rtol=0.05, atol=0.05)


def test_pair_noise_soft_shape():
    x = torch.randn(3, 4, 8, 8)
    n = torch.randn(3, 4, 8, 8)
    out = pair_noise_to_latents(x, n, reg=0.08, n_iters=30, mode="soft")
    assert out.shape == n.shape
