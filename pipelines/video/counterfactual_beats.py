"""What-If Timeline — counterfactual shot branches for parallel narrative exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence

__all__ = [
    "CounterfactualBranch",
    "CounterfactualPlan",
    "parse_counterfactuals",
    "build_counterfactual_plan",
]


@dataclass(slots=True)
class CounterfactualBranch:
    id: str
    parent_shot_id: str
    branch_label: str
    alt_prompt: str
    alt_duration_sec: float = 0.0
    probability: float = 0.5


@dataclass(slots=True)
class CounterfactualPlan:
    branches: List[CounterfactualBranch] = field(default_factory=list)
    merge_strategy: str = "none"  # none | montage | choose_one


def parse_counterfactuals(raw: Any) -> tuple[List[CounterfactualBranch], str]:
    branches: List[CounterfactualBranch] = []
    strategy = "none"
    if isinstance(raw, Mapping):
        strategy = str(raw.get("merge_strategy") or raw.get("strategy") or "none")
        items = raw.get("branches") or raw.get("what_if") or []
    elif isinstance(raw, list):
        items = raw
    else:
        return [], strategy

    for i, row in enumerate(items):
        if not isinstance(row, Mapping):
            continue
        branches.append(
            CounterfactualBranch(
                id=str(row.get("id") or f"cf_{i}"),
                parent_shot_id=str(row.get("after_shot") or row.get("parent_shot") or row.get("shot") or ""),
                branch_label=str(row.get("label") or row.get("branch") or "alternate"),
                alt_prompt=str(row.get("prompt") or row.get("alt_prompt") or ""),
                alt_duration_sec=float(row.get("duration_sec") or row.get("duration") or 0.0),
                probability=float(row.get("probability") or row.get("weight") or 0.5),
            )
        )
    return branches, strategy


def build_counterfactual_plan(
    shots: Sequence[Any],
    branches: Sequence[CounterfactualBranch],
    *,
    merge_strategy: str = "none",
) -> CounterfactualPlan:
    if not branches:
        return CounterfactualPlan(merge_strategy=merge_strategy)
    shot_ids = {str(getattr(s, "id", "")) for s in shots}
    valid = [b for b in branches if not b.parent_shot_id or b.parent_shot_id in shot_ids]
    return CounterfactualPlan(branches=list(valid), merge_strategy=merge_strategy)
