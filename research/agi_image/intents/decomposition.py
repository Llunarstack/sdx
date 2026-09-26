from __future__ import annotations

from .goal_spec import ConstraintKind, GoalSpec


def decompose_goal_stub(goal: GoalSpec) -> list[tuple[str, str]]:
    """
    Placeholder splitter: narrative → labelled sub-sentences.

    Swap for LM / structured parser later; stable API for planners.
    """
    raw = (goal.narrative or "").strip()
    if not raw:
        return []
    clauses = [c.strip() for c in raw.replace(";", ",").split(",") if c.strip()]
    out: list[tuple[str, str]] = []
    for i, c in enumerate(clauses):
        ck = ConstraintKind.soft if goal.ambiguity.open_style else ConstraintKind.hard
        _ = ck  # future: clause-level constraint typing
        out.append((f"clause_{i}", c))
    return out


__all__ = ["decompose_goal_stub"]
