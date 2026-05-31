"""
**TurningPoint-GRPO (TP-GRPO)** — step incremental rewards + turning-point long-term credit.

Alleviates sparse terminal rewards in flow-based GRPO by:
1. Incremental step rewards ``ΔR_t = R_t - R_{t-1}``
2. Turning points where ``sign(ΔR_t) != sign(ΔR_{t+1})`` get aggregated long-term bonus

Tong et al., arXiv:2602.06422 (2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from utils.training.dense_grpo import dense_reward_gains
from utils.training.flow_grpo import group_relative_advantages


@dataclass(slots=True)
class TurningPointGRPOConfig:
    long_term_weight: float = 0.5
    """Weight on aggregated reward assigned at turning-point steps."""
    use_incremental_only: bool = False
    """If True, skip turning-point bonus (ablation)."""


def incremental_rewards_from_trajectory(step_rewards: Sequence[float]) -> List[float]:
    """``ΔR_t`` from absolute per-step rewards ``R_t``."""
    prev = 0.0
    out: List[float] = []
    for r in step_rewards:
        rv = float(r)
        out.append(rv - prev)
        prev = rv
    return out


def detect_turning_point_indices(incremental: Sequence[float]) -> List[int]:
    """
    Turning point at step ``t`` when ``sign(ΔR_t) != sign(ΔR_{t+1})`` (non-zero signs).
    """
    inc = [float(x) for x in incremental]
    if len(inc) < 2:
        return []
    tps: List[int] = []
    for t in range(len(inc) - 1):
        a, b = inc[t], inc[t + 1]
        if a == 0.0 or b == 0.0:
            continue
        if (a > 0) != (b > 0):
            tps.append(t)
    return tps


def long_term_aggregate_reward(
    incremental: Sequence[float],
    terminal_reward: float,
) -> float:
    """Simple long-term signal: terminal minus mean incremental."""
    inc = list(incremental)
    if not inc:
        return float(terminal_reward)
    return float(terminal_reward) - sum(inc) / len(inc)


def tp_grpo_step_weights(
    step_rewards: Sequence[float],
    terminal_reward: float,
    *,
    config: TurningPointGRPOConfig | None = None,
) -> List[float]:
    """
    Per-step training weights combining incremental + turning-point long-term bonus.

    Returns one scalar weight per denoise step (same length as ``step_rewards``).
    """
    cfg = config or TurningPointGRPOConfig()
    inc = incremental_rewards_from_trajectory(step_rewards)
    if not inc:
        return [float(terminal_reward)]

    weights = list(inc)
    if not cfg.use_incremental_only:
        lt = long_term_aggregate_reward(inc, terminal_reward)
        for t in detect_turning_point_indices(inc):
            weights[t] = weights[t] + float(cfg.long_term_weight) * lt
    return weights


def tp_grpo_advantages_from_trajectories(
    trajectories: Sequence[Sequence[float]],
    terminal_rewards: Sequence[float],
    *,
    config: TurningPointGRPOConfig | None = None,
) -> torch.Tensor:
    """
    Flatten per-trajectory step weights → group-relative advantages.

    ``trajectories[g]`` = list of step-wise absolute rewards for sample g.
    """
    cfg = config or TurningPointGRPOConfig()
    flat: List[float] = []
    for traj, term in zip(trajectories, terminal_rewards):
        w = tp_grpo_step_weights(traj, float(term), config=cfg)
        flat.extend(w)
    if not flat:
        return torch.zeros(0)
    return group_relative_advantages(torch.tensor(flat, dtype=torch.float32))


def tp_grpo_from_dense_gains(
    terminal_rewards: Sequence[float],
    step_rewards: Sequence[Sequence[float]],
    *,
    config: TurningPointGRPOConfig | None = None,
) -> torch.Tensor:
    """Bridge DenseGRPO-style step reward tables into TP-GRPO advantages."""
    gains = dense_reward_gains(list(terminal_rewards), list(step_rewards))
    trajectories = [[float(r) for r in row] for row in gains]
    # Reconstruct pseudo absolute rewards from gains for TP detection
    abs_trajs: List[List[float]] = []
    for row in trajectories:
        acc = 0.0
        abs_row: List[float] = []
        for g in row:
            acc += float(g)
            abs_row.append(acc)
        abs_trajs.append(abs_row)
    return tp_grpo_advantages_from_trajectories(abs_trajs, terminal_rewards, config=config)


__all__ = [
    "TurningPointGRPOConfig",
    "detect_turning_point_indices",
    "incremental_rewards_from_trajectory",
    "long_term_aggregate_reward",
    "tp_grpo_advantages_from_trajectories",
    "tp_grpo_from_dense_gains",
    "tp_grpo_step_weights",
]
