import torch

from utils.training.diffusion_dpo_loss import dpo_preference_loss, neg_mse_as_logp


def test_dpo_preference_loss_finite():
    lw = torch.tensor(0.0, requires_grad=True)
    ll = torch.tensor(-1.0, requires_grad=True)
    rw = torch.tensor(0.0)
    rl = torch.tensor(-1.0)
    loss = dpo_preference_loss(lw, ll, rw, rl, beta=1.0)
    assert loss.ndim == 0
    loss.backward()
    assert lw.grad is not None


def test_neg_mse_as_logp():
    p = torch.zeros(1, 4, 2, 2)
    t = torch.ones(1, 4, 2, 2)
    v = neg_mse_as_logp(p, t)
    assert v.ndim == 0 and v.item() < 0
