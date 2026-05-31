"""Wave 12: CFG++, interval CFG, GRPO-Guard, CFG-Rejection, branch rollout."""

from __future__ import annotations

import torch
from utils.generation.cfg_interval import should_apply_cfg
from utils.generation.cfg_pp import cfg_pp_combine, cfg_scale_to_pp_lambda
from utils.generation.guidance_stack import combine_guided_prediction
from utils.superior.cfg_rejection import CFGRejectionTracker, pick_best_candidate_index
from utils.training.branch_grpo import branch_rollout_flow_samples, enumerate_branch_paths
from utils.training.grpo_guard import GRPOGuardConfig, grpo_guard_weighted_loss, ratio_norm_advantages


def test_cfg_pp_differs_from_standard_cfg() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    pp = cfg_pp_combine(cond, uncond, cfg_lambda=0.6)
    std = uncond + 7.0 * (cond - uncond)
    assert not torch.allclose(pp, std)


def test_cfg_scale_to_pp_lambda() -> None:
    assert 0.0 < cfg_scale_to_pp_lambda(7.5) < 1.0


def test_interval_cfg_skips_early() -> None:
    assert not should_apply_cfg(0.05, skip_early_frac=0.15)
    assert should_apply_cfg(0.5, skip_early_frac=0.15)


def test_combine_guided_cfg_pp() -> None:
    cond = torch.randn(1, 4, 4, 4)
    uncond = torch.zeros_like(cond)
    out = combine_guided_prediction(
        cond,
        uncond,
        uncond,
        cfg_scale=7.0,
        cfg_pp_lambda=0.55,
        zeresfdg_strength=0.0,
    )
    assert out.shape == cond.shape


def test_interval_skips_in_stack() -> None:
    cond = torch.ones(1, 2, 2, 2)
    uncond = torch.zeros(1, 2, 2, 2)
    out = combine_guided_prediction(
        cond,
        uncond,
        uncond,
        cfg_scale=7.0,
        cfg_skip_early_frac=0.5,
        sample_step=0,
        total_steps=10,
    )
    assert torch.allclose(out, cond)


def test_grpo_guard_weights() -> None:
    adv = torch.tensor([1.0, -1.0, 0.2])
    normed = ratio_norm_advantages(adv)
    assert normed.numel() == 3


def test_grpo_guard_loss() -> None:
    loss = torch.tensor([0.5, 0.8])
    adv = torch.tensor([1.0, -0.5])
    t = torch.tensor([0.3, 0.7])
    total = grpo_guard_weighted_loss(loss, adv, t, config=GRPOGuardConfig())
    assert float(total.item()) > 0.0


def test_cfg_rejection_tracker() -> None:
    tr = CFGRejectionTracker(tau_steps=2)
    c = torch.randn(1, 4, 4, 4)
    u = torch.zeros_like(c)
    tr.note(c, u)
    tr.note(c * 2, u)
    assert tr.accumulated_early_score() > 0.0


def test_pick_best_candidate() -> None:
    assert pick_best_candidate_index([0.9, 0.2, 0.5]) == 1


def test_branch_rollout_paths() -> None:
    calls: list[int] = []

    def _roll() -> torch.Tensor:
        calls.append(1)
        return torch.zeros(1, 2, 2, 2)

    outs = branch_rollout_flow_samples(_roll, num_paths=4, steps=8, base_seed=0)
    assert len(outs) == 4
    assert len(calls) == 4
    assert len(enumerate_branch_paths(8)) >= 1
