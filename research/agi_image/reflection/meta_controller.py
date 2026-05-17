from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ControlDecisionKind(str, Enum):
    CONTINUE = "continue"
    SWITCH_VERIFIER = "switch_verifier"
    EXPAND_SAMPLES = "expand_samples"
    DEGRADE_GOAL = "degrade_goal"
    ABORT = "abort"


@dataclass(slots=True)
class MetaControllerState:
    """Rolling advice from a supervisory loop (human or scripted)."""

    decisions: List[ControlDecisionKind] = field(default_factory=list)
    rationales: List[str] = field(default_factory=list)

    def propose(self, kind: ControlDecisionKind, rationale: str) -> None:
        self.decisions.append(kind)
        self.rationales.append(rationale)


__all__ = ["ControlDecisionKind", "MetaControllerState"]
