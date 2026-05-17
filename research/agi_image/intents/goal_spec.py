from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal


class ConstraintKind(str, Enum):
    """Hard vs soft commitments to the eventual render."""

    hard = "hard"
    soft = "soft"


@dataclass(slots=True)
class IntentUncertainty:
    """Where the user's brief is intentionally open-ended."""

    open_style: bool = False
    open_palette: bool = False
    open_camera: bool = False
    open_negative_space: bool = False
    notes: str = ""


@dataclass(slots=True)
class GoalSpec:
    """
    Canonical intent object: what must hold vs what may vary.

    Downstream planners map this into ``research.agi_image.planning.GenerationPlan`` steps
    or into ``sample.py`` flag hints (via ``integrations.sample_hints``).
    """

    title: str
    narrative: str
    constraints: List[tuple[ConstraintKind, str]] = field(default_factory=list)
    must_depict: List[str] = field(default_factory=list)
    must_avoid: List[str] = field(default_factory=list)
    reference_roles: Dict[str, str] = field(default_factory=dict)
    ambiguity: IntentUncertainty = field(default_factory=IntentUncertainty)
    style_tokens: List[str] = field(default_factory=list)
    modality_hints: Dict[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot for logs / agent traces."""
        return {
            "title": self.title,
            "narrative": self.narrative,
            "constraints": [[k.value, v] for k, v in self.constraints],
            "must_depict": list(self.must_depict),
            "must_avoid": list(self.must_avoid),
            "reference_roles": dict(self.reference_roles),
            "ambiguity": asdict(self.ambiguity),
            "style_tokens": list(self.style_tokens),
            "modality_hints": dict(self.modality_hints),
        }


LiteralAspect = Literal["subject", "style", "layout", "text", "relation", "temporal"]

__all__ = ["ConstraintKind", "GoalSpec", "IntentUncertainty", "LiteralAspect"]
