"""User intent layering: constraints, ambiguity, hierarchical goals."""

from .decomposition import decompose_goal_stub
from .goal_spec import ConstraintKind, GoalSpec, IntentUncertainty

__all__ = ["ConstraintKind", "GoalSpec", "IntentUncertainty", "decompose_goal_stub"]
