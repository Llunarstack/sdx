from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from ..planning.generation_plan import GenerationPlan


@dataclass(slots=True)
class PlanProposal:
    planner_id: str
    rationale: str
    plan: GenerationPlan


@dataclass(slots=True)
class PlanAccept:
    reviewer_id: str
    proposal: PlanProposal
    accepted_step_ids: list[str]
    amendments: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationVerdict:
    """Verifier → planner feedback."""

    source: Literal["vit", "ocr", "human", "heuristic"]
    iteration: int
    acceptance: bool
    summary: str = ""
    suggestion: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    artefacts: Dict[str, Any] = field(default_factory=dict)


__all__ = ["PlanAccept", "PlanProposal", "VerificationVerdict"]
