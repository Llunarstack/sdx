"""Wave 7: TaylorSeer, DenseGRPO, Rectified-CFG++, LCM sampling hooks."""

from __future__ import annotations

import torch
from utils.generation.rectified_cfgpp import rectified_cfgpp_combine
from utils.superior.taylor_cache import TaylorBlockCache, TaylorCacheConfig, taylor_forecast_tensor
from utils.training.dense_grpo import (
    dense_reward_gains,
    estimate_x0_from_xt_flow,
    reward_aware_sde_scale,
    step_advantages_from_gains,
)


def test_taylor_forecast_linear() -> None:
    h0 = torch.ones(2, 2)
    h1 = h0 + 0.5
    pred = taylor_forecast_tensor([h0, h1], steps_since_anchor=2, interval=4, max_order=1)
    assert pred.shape == h0.shape
    assert float(pred.mean()) > float(h1.mean())


def test_taylor_block_cache_apply() -> None:
    cache = TaylorBlockCache(TaylorCacheConfig(rel_l1_threshold=0.5, recompute_every=8))
    fp = torch.tensor([1.0, 2.0])
    cache.begin_forward(fp)
    x = torch.zeros(1, 4, 2, 2)
    cache.note_block(0, x, x + 1.0)
    cache.begin_forward(fp + 0.001)
    cache.begin_forward(fp + 0.002)
    out = cache.apply_residual(x, 0)
    assert out.shape == x.shape


def test_rectified_cfgpp_clamps_delta() -> None:
    cond = torch.randn(1, 4, 8, 8) * 10
    uncond = torch.zeros(1, 4, 8, 8)
    std = uncond + 7.5 * (cond - uncond)
    rc = rectified_cfgpp_combine(cond, uncond, cfg_scale=7.5, tangent_norm=0.5)
    assert not torch.allclose(std, rc)


def test_reward_aware_sde_peak_mid() -> None:
    mid = float(reward_aware_sde_scale(0.5, base=0.35))
    end = float(reward_aware_sde_scale(0.05, base=0.35))
    assert mid > end


def test_dense_reward_gains() -> None:
    gains = dense_reward_gains([0.8], [[0.2, 0.5, 0.8]])
    assert gains[0][0] == 0.2
    assert abs(gains[0][2] - 0.3) < 1e-6


def test_step_advantages_from_gains() -> None:
    adv = step_advantages_from_gains([0.1, 0.5, 0.2, 0.3])
    assert adv.numel() == 4


def test_flow_x0_estimate() -> None:
    x_t = torch.tensor([[[[1.0]]]])
    v = torch.tensor([[[[0.5]]]])
    t = torch.tensor([0.5])
    x0 = estimate_x0_from_xt_flow(x_t, v, t)
    assert float(x0.item()) == 0.75
