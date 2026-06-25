"""Spend GPU budget where it matters — skip expensive guidance when cheap heuristics suffice."""

from .compute_budget import ComputeBudget, ComputeBudgetPlanner, GuidanceTier

__all__ = ["ComputeBudget", "ComputeBudgetPlanner", "GuidanceTier"]
