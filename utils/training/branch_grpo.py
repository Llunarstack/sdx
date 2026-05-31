"""
**BranchGRPO** scaffold — tree-structured rollouts for diffusion GRPO (2025).

At scheduled split steps, trajectories branch into multiple children while reusing shared
prefix computation. Depth-wise reward fusion aggregates leaf scores back to branches.

Inspired by BranchGRPO (structured branching in diffusion models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch


@dataclass(slots=True)
class BranchGRPOConfig:
    branch_factor: int = 2
    """Children per split node."""
    split_step_fractions: Tuple[float, ...] = (0.35, 0.65)
    """Fractions along denoise trajectory where branching occurs."""
    fuse: str = "mean"
    """Leaf reward fusion: mean | max | min."""


@dataclass(slots=True)
class BranchNode:
    node_id: str
    parent_id: str
    depth: int
    split_at_step: int
    reward: float = 0.0
    children: List[str] = field(default_factory=list)


def split_steps_from_fractions(total_steps: int, fractions: Sequence[float]) -> List[int]:
    """Map fractional positions to integer step indices in ``[0, total_steps)``."""
    n = max(1, int(total_steps))
    out: List[int] = []
    for f in fractions:
        idx = int(round(float(f) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def enumerate_branch_paths(
    total_steps: int,
    *,
    branch_factor: int = 2,
    split_fractions: Sequence[float] = (0.35, 0.65),
) -> List[Tuple[int, ...]]:
    """
    Return leaf paths as tuples of branch choices (0..branch_factor-1) per split.

    Does not run the model — planning helper for trainers.
    """
    splits = split_steps_from_fractions(total_steps, split_fractions)
    if not splits:
        return [(0,)]
    k = max(1, int(branch_factor))
    paths: List[Tuple[int, ...]] = [(0,)]
    for _ in splits:
        paths = [p + (b,) for p in paths for b in range(k)]
    return paths


def fuse_branch_rewards(
    leaf_rewards: Sequence[float],
    *,
    mode: str = "mean",
) -> float:
    """Aggregate rewards from branch leaves (depth-wise fusion at root)."""
    if not leaf_rewards:
        return 0.0
    vals = [float(r) for r in leaf_rewards]
    m = str(mode).lower().strip()
    if m == "max":
        return max(vals)
    if m == "min":
        return min(vals)
    return sum(vals) / len(vals)


def branch_relative_advantages(
    path_rewards: torch.Tensor,
    path_ids: Sequence[str],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GRPO-style advantages over branch leaves sharing the same prompt.

    ``path_rewards``: (P,), one reward per leaf path.
    """
    r = path_rewards.detach().float().view(-1)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    mu = r.mean()
    sd = r.std(unbiased=False)
    return ((r - mu) / (sd + eps)).clamp(-3.0, 3.0)


def branch_rollout_flow_samples(
    rollout_fn,
    *,
    num_paths: int,
    steps: int,
    branch_factor: int = 2,
    split_fractions: Sequence[float] = (0.35, 0.65),
    base_seed: int = 0,
) -> List[torch.Tensor]:
    """
    Run multiple rollout paths with distinct seeds (BranchGRPO scaffold).

    Full prefix-sharing would require hooking inside ``sample_loop``; this helper
    reuses the planning API and runs independent short rollouts per leaf path.
    """
    paths = enumerate_branch_paths(
        int(steps),
        branch_factor=int(branch_factor),
        split_fractions=split_fractions,
    )
    n = max(1, int(num_paths))
    paths = paths[:n] if len(paths) >= n else paths + [paths[-1]] * (n - len(paths))
    latents: List[torch.Tensor] = []
    for i, _path in enumerate(paths):
        if base_seed >= 0:
            torch.manual_seed(int(base_seed) + i)
        latents.append(rollout_fn())
    return latents


def prefix_reuse_factor(num_splits: int, branch_factor: int, total_steps: int) -> float:
    """
    Rough compute savings vs independent rollouts (0..1).

    Shared prefix length increases with earlier splits.
    """
    splits = split_steps_from_fractions(total_steps, (0.35, 0.65)[: max(0, num_splits)])
    if not splits:
        return 0.0
    k = max(1, int(branch_factor))
    independent = k ** len(splits) * total_steps
    shared = sum(splits) + sum(total_steps - s for s in splits) * k
    return float(max(0.0, 1.0 - shared / max(independent, 1)))


__all__ = [
    "BranchGRPOConfig",
    "BranchNode",
    "branch_relative_advantages",
    "branch_rollout_flow_samples",
    "enumerate_branch_paths",
    "fuse_branch_rewards",
    "prefix_reuse_factor",
    "split_steps_from_fractions",
]
