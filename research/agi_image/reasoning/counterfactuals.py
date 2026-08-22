from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BranchPoint:
    """Where a plan forked (lighting, entity count, camera)."""

    description: str
    active_branch_id: str


@dataclass(slots=True)
class CounterfactualBranch:
    """Sibling render intent under the same high-level goal."""

    branch_id: str
    delta_prompt: str
    parent_branch_id: str | None = None
    expected_metric_shift: str = ""  # human-readable expectation
    children: list[CounterfactualBranch] = field(default_factory=list)


__all__ = ["BranchPoint", "CounterfactualBranch"]
