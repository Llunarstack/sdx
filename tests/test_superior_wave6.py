"""Wave 6: block cache, flow-GRPO, consistency/LADD distill, hard-negative flywheel."""

from __future__ import annotations

import torch
from utils.superior.block_cache import BlockCacheConfig, BlockDiTCache
from utils.superior.hard_negative import benchmark_sample_args_for_negatives, mine_hard_negatives
from utils.training.flow_grpo import group_relative_advantages, grpo_weighted_loss, reference_kl_penalty


def test_block_cache_skip_and_apply() -> None:
    cache = BlockDiTCache(BlockCacheConfig(rel_l1_threshold=0.5, recompute_every=99))
    fp1 = torch.tensor([1.0, 2.0, 3.0])
    fp2 = torch.tensor([1.01, 2.01, 3.01])
    cache.begin_forward(fp1)
    x = torch.zeros(1, 4, 2, 2)
    cache.note_block(0, x, x + 1.0)
    cache.begin_forward(fp2)
    assert cache.should_skip_block(0)
    out = cache.apply_residual(x, 0)
    assert torch.allclose(out, x + 1.0)


def test_grpo_advantages_zero_mean() -> None:
    r = torch.tensor([0.2, 0.5, 0.9, 0.3])
    adv = group_relative_advantages(r)
    assert abs(float(adv.mean())) < 0.01


def test_grpo_weighted_loss() -> None:
    loss = torch.tensor([1.0, 2.0, 3.0])
    adv = torch.tensor([1.0, 0.0, -1.0])
    w = grpo_weighted_loss(loss, adv)
    assert torch.isfinite(w)


def test_reference_kl_penalty() -> None:
    a = torch.randn(1, 4, 4, 4)
    b = a + 0.1
    p = reference_kl_penalty(a, b, coef=0.1)
    assert float(p) > 0.0


def test_hard_negative_benchmark_args() -> None:
    bundle = mine_hard_negatives([{"composite": 0.3, "edge_sharpness": 20.0}])
    args = benchmark_sample_args_for_negatives(bundle)
    assert "--negative-prompt" in args
    assert args[1]
