"""
AGI-oriented **image generation** research layout (interfaces + data shapes, no training loop).

Themes: layered intent → world/context → multi-step plans → verification → repair → alignment.
Concrete diffusion wiring stays in ``sample.py`` / ``train.py``; this package holds *contracts* and
ideas to grow capability without collapsing everything into monolithic prompts.

Subpackages remain importable **without torch** for docs tools and planners.
"""

from __future__ import annotations

from .benchmarks.task_taxonomy import TASK_TAXONOMY_VERSION, ImageCognitiveCapability
from .evaluation.capability_rubric import CapabilityRubric
from .intents.goal_spec import GoalSpec
from .planning.generation_plan import GenerationPlan, GenerationStepKind
from .schemas.agent_messages import PlanAccept, PlanProposal, VerificationVerdict

__all__ = [
    "CapabilityRubric",
    "GenerationPlan",
    "GenerationStepKind",
    "GoalSpec",
    "ImageCognitiveCapability",
    "PlanAccept",
    "PlanProposal",
    "TASK_TAXONOMY_VERSION",
    "VerificationVerdict",
]
