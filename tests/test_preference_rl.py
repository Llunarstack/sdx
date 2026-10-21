"""Preference-RL weighting (GRPO / GRPO-Guard / SRPO) and the Flow-GRPO SDE rollout step."""

from __future__ import annotations

import pytest
import torch
from utils.training.flow_grpo import flow_sde_step, grpo_weighted_loss
from utils.training.grpo import GRPOGuardConfig, grpo_guard_weighted_loss


def _grad_per_sample(loss_fn) -> torch.Tensor:
    per = torch.ones(4, requires_grad=True)
    loss_fn(per).backward()
    return per.grad


def test_grpo_fits_high_reward_rollouts_hardest():
    adv = torch.tensor([2.0, 0.5, -0.5, -2.0])
    g = _grad_per_sample(lambda per: grpo_weighted_loss(per, adv))
    assert torch.all(g[:-1] > g[1:])


def test_grpo_needs_the_group():
    with pytest.raises(ValueError):
        grpo_weighted_loss(torch.ones(1), torch.ones(1))


def test_grpo_guard_weights_increase_with_advantage():
    adv = torch.tensor([2.0, 0.5, -0.5, -2.0])
    cfg = GRPOGuardConfig(grad_reweight=False)
    g = _grad_per_sample(lambda per: grpo_guard_weighted_loss(per, adv, torch.full((4,), 0.5), config=cfg))
    assert torch.all(g[:-1] >= g[1:]) and g[0] > g[-1]


def test_flow_sde_keeps_the_flow_marginals():
    """With the exact velocity of 1-D Gaussian data, the SDE must still land on the data distribution."""
    torch.manual_seed(0)
    data_std = 0.5

    def velocity(x, s):  # E[eps - x0 | x_s] for x0 ~ N(0, data_std^2)
        var = (1 - s) ** 2 * data_std**2 + s**2
        return x * (s - (1 - s) * data_std**2) / var

    x = torch.randn(20000)
    grid = torch.linspace(1.0, 0.0, 201).tolist()
    for s_cur, s_next in zip(grid[:-1], grid[1:]):
        v = velocity(x, s_cur)
        x = flow_sde_step(x, v, s_cur, s_next, noise_level=0.7) if s_next > 0 else x - s_cur * v
    assert float(x.std()) == pytest.approx(data_std, rel=0.05)
