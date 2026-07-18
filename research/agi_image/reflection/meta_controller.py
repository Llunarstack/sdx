from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ControlDecisionKind(str, Enum):
    CONTINUE = "continue"
    SWITCH_VERIFIER = "switch_verifier"
    EXPAND_SAMPLES = "expand_samples"
    DEGRADE_GOAL = "degrade_goal"
    ABORT = "abort"


@dataclass(slots=True)
class MetaControllerState:
    """Rolling advice from a supervisory loop (human or scripted)."""

    decisions: list[ControlDecisionKind] = field(default_factory=list)
    rationales: list[str] = field(default_factory=list)

    def propose(self, kind: ControlDecisionKind, rationale: str) -> None:
        self.decisions.append(kind)
        self.rationales.append(rationale)


__all__ = ["ControlDecisionKind", "MetaControllerState"]
