from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class GenerationStepKind(str, Enum):
    """Composable operations an agent orchestrator might schedule."""

    refine_prompt = "refine_prompt"
    latent_draft = "latent_draft"
    diffusion_sample = "diffusion_sample"
    inpaint_region = "inpaint_region"
    verify_visual = "verify_visual"
    verify_text_ocr = "verify_text_ocr"
    critique_revise = "critique_revise"
    semantic_segment = "semantic_segment"
    knowledge_retrieval = "knowledge_retrieval"
    human_review = "human_review"


@dataclass(slots=True)
class GenerationStep:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    kind: GenerationStepKind = GenerationStepKind.diffusion_sample
    description: str = ""
    deps: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationPlan:
    """DAG-shaped plan (deps on steps); executor walks topsort."""

    goal_id: str
    steps: List[GenerationStep] = field(default_factory=list)

    def add(self, step: GenerationStep) -> GenerationStep:
        self.steps.append(step)
        return step

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "steps": [
                {
                    "id": s.id,
                    "kind": s.kind.value,
                    "description": s.description,
                    "deps": list(s.deps),
                    "kwargs": dict(s.kwargs),
                }
                for s in self.steps
            ],
        }


@dataclass(slots=True)
class StopConditions:
    """Budget + quality guards for iterative systems."""

    max_outer_loops: int = 4
    min_clip_alignment: Optional[float] = None
    min_edge_coherence: Optional[float] = None
    patience_no_gain: int = 2


__all__ = ["GenerationPlan", "GenerationStep", "GenerationStepKind", "StopConditions"]
