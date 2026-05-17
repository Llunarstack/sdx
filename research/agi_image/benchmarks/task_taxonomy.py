from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

TASK_TAXONOMY_VERSION = "2026-05-sdx-agi-scaffold-1"


class ImageCognitiveCapability(str, Enum):
    """
    Coarse capability axes for probing models + planners.

    Not a leaderboard spec — a checklist to design curricula and regressions.
    """

    lexical_grounding = "lexical_grounding"
    counting_and_sets = "counting_and_sets"
    spatial_relations = "spatial_relations"
    perspective_consistency = "perspective_consistency"
    occlusion_reasoning = "occlusion_reasoning"
    procedural_diagram = "procedural_diagram"
    typographic_design = "typographic_design"
    ui_affordances = "ui_affordances"
    causal_interaction_preview = "causal_interaction_preview"
    style_transfer_under_constraints = "style_transfer_under_constraints"
    personalization_from_reference_stack = "personalization_from_reference_stack"
    iterative_refinement_budgeted = "iterative_refinement_budgeted"


@dataclass(frozen=True, slots=True)
class TaskStub:
    """Placeholder row for benchmark registry entries."""

    id: str
    capability: ImageCognitiveCapability
    seed_prompt: str
    verifier_hint: str


def enumerate_capabilities() -> tuple[str, ...]:
    return tuple(sorted(c.value for c in ImageCognitiveCapability))


__all__ = ["ImageCognitiveCapability", "TaskStub", "TASK_TAXONOMY_VERSION", "enumerate_capabilities"]
