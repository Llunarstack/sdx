"""Wave 10: TP-GRPO, DyDiT dynamic width, APG momentum, linear attention."""

from __future__ import annotations

import torch
from utils.generation.guidance_session import DynamicDitSchedule, GuidanceSession
from utils.generation.guidance_stack import combine_guided_prediction
from utils.superior.dynamic_dit import apply_dynamic_width, spatial_token_importance, timestep_dynamic_width
from utils.superior.linear_attention import hybrid_attention_fraction, linear_attention
from utils.training.turning_point_grpo import (
    TurningPointGRPOConfig,
    detect_turning_point_indices,
    incremental_rewards_from_trajectory,
    tp_grpo_step_weights,
)


def test_incremental_rewards() -> None:
    inc = incremental_rewards_from_trajectory([0.2, 0.5, 0.9])
    assert inc == [0.2, 0.3, 0.4]


def test_turning_point_detection() -> None:
    inc = [0.1, -0.2, -0.1, 0.3]
    tps = detect_turning_point_indices(inc)
    assert 0 in tps


def test_tp_grpo_weights_boost_turning_point() -> None:
    traj = [0.1, 0.3, 0.55, 0.5, 0.9]
    w = tp_grpo_step_weights(traj, 0.9, config=TurningPointGRPOConfig(long_term_weight=0.5))
    assert len(w) == len(traj)
    assert max(w) > min(w)


def test_timestep_dynamic_width_monotone() -> None:
    early = timestep_dynamic_width(0.0, early=0.8, late=1.0)
    late = timestep_dynamic_width(1.0, early=0.8, late=1.0)
    assert early < late


def test_apply_dynamic_width() -> None:
    out = torch.ones(1, 4, 2, 2)
    scaled = apply_dynamic_width(out, 0.0, early=0.5)
    assert float(scaled.mean()) == 0.5


def test_spatial_importance_shape() -> None:
    x = torch.randn(2, 4, 8, 8)
    imp = spatial_token_importance(x)
    assert imp.shape == (2, 1, 8, 8)


def test_guidance_session_momentum() -> None:
    sess = GuidanceSession(apg_momentum_beta=0.3)
    d = torch.ones(1, 2, 2, 2)
    sess.note_apg_delta(d)
    assert sess.prev_apg_delta is not None
    assert sess.step_index == 1


def test_combine_apg_with_session() -> None:
    cond = torch.randn(1, 4, 4, 4)
    uncond = torch.randn(1, 4, 4, 4)
    sess = GuidanceSession(apg_momentum_beta=0.25)
    o1 = combine_guided_prediction(cond, uncond, uncond, cfg_scale=5.0, apg_parallel_eta=0.0, guidance_session=sess)
    o2 = combine_guided_prediction(cond, uncond, uncond, cfg_scale=5.0, apg_parallel_eta=0.0, guidance_session=sess)
    assert o1.shape == o2.shape


def test_linear_attention_runs() -> None:
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = linear_attention(q, k, v)
    assert out.shape == v.shape


def test_hybrid_attention_blend() -> None:
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = hybrid_attention_fraction(q, k, v, linear_frac=0.5)
    assert out.shape == v.shape


def test_dynamic_dit_schedule() -> None:
    sched = DynamicDitSchedule(enabled=True, early_width=0.85, late_width=1.0)
    assert sched.scale_at_progress(0.0) < sched.scale_at_progress(1.0)
