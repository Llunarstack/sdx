"""Wave 8: APG guidance, Flash-GRPO, BranchGRPO."""

from __future__ import annotations

import torch
from utils.generation.apg_guidance import apg_cfg_combine, decompose_parallel_orthogonal
from utils.generation.cfg_batched import combine_cfg_outputs
from utils.training.branch_grpo import (
    BranchGRPOConfig,
    enumerate_branch_paths,
    fuse_branch_rewards,
    prefix_reuse_factor,
)
from utils.training.flash_grpo import (
    iso_temporal_group_advantages,
    rectify_policy_gradient,
    sde_discretization_lambda,
)


def test_apg_removes_parallel_oversaturation() -> None:
    cond = torch.tensor([[[[2.0, 1.0]]]])
    uncond = torch.tensor([[[[0.5, 0.5]]]])
    std = uncond + 7.5 * (cond - uncond)
    apg = apg_cfg_combine(cond, uncond, cfg_scale=7.5, parallel_eta=0.0)
    assert not torch.allclose(std, apg)
    delta = cond - uncond
    par, orth = decompose_parallel_orthogonal(delta, cond)
    assert torch.allclose(par + orth, delta)


def test_combine_cfg_apg_path() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    out = combine_cfg_outputs(cond, uncond, uncond, cfg_scale=5.0, apg_parallel_eta=0.0)
    assert out.shape == cond.shape


def test_sde_lambda_peaks_mid() -> None:
    mid = float(sde_discretization_lambda(0.5))
    edge = float(sde_discretization_lambda(0.02))
    assert mid > edge


def test_iso_temporal_advantages() -> None:
    rewards = torch.tensor([0.9, 0.8, 0.2, 0.1])
    t_idx = torch.tensor([10, 10, 20, 20])
    adv = iso_temporal_group_advantages(rewards, t_idx, num_timesteps=50)
    assert adv[0] > adv[1]
    assert adv[2] > adv[3]


def test_rectify_policy_gradient() -> None:
    loss = torch.tensor(1.0)
    rect = rectify_policy_gradient(loss, 0.5)
    assert float(rect) > float(loss)


def test_branch_paths_and_fusion() -> None:
    paths = enumerate_branch_paths(20, branch_factor=2, split_fractions=(0.35, 0.65))
    assert len(paths) == 4
    fused = fuse_branch_rewards([0.3, 0.9, 0.4, 0.8], mode="max")
    assert fused == 0.9
    save = prefix_reuse_factor(2, 2, 20)
    assert 0.0 <= save <= 1.0


def test_branch_config_defaults() -> None:
    cfg = BranchGRPOConfig()
    assert cfg.branch_factor >= 2
