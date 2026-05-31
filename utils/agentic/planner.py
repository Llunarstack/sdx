"""
**Planner** — Understand / Think / Plan (UTPC) → ``GenerationPlan`` DAG.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from config.defaults.agentic_stack import AgenticStackDefaults
from research.agi_image.planning.generation_plan import (
    GenerationPlan,
    GenerationStep,
    GenerationStepKind,
    StopConditions,
)


def plan_from_prompt(
    prompt: str,
    *,
    use_rag: bool = True,
    expand: bool = True,
    verify: bool = True,
    self_correct: bool = True,
    goal_id: Optional[str] = None,
) -> GenerationPlan:
    """Build a linear UTPC plan for a single-image request."""
    gid = goal_id or uuid.uuid4().hex[:12]
    plan = GenerationPlan(goal_id=gid)
    prev: List[str] = []

    if use_rag:
        s = GenerationStep(
            kind=GenerationStepKind.knowledge_retrieval,
            description="Retrieve local facts (TF-IDF RAG)",
            kwargs={"tool": "rag_retrieve"},
        )
        plan.add(s)
        prev = [s.id]

    if expand:
        s = GenerationStep(
            kind=GenerationStepKind.refine_prompt,
            description="Expand prompt with quality hints",
            deps=list(prev),
            kwargs={"tool": "expand_prompt"},
        )
        plan.add(s)
        prev = [s.id]

    gen = GenerationStep(
        kind=GenerationStepKind.diffusion_sample,
        description="Generate N candidates + composite pick",
        deps=list(prev),
        kwargs={"tool": "generate", "num": 4},
    )
    plan.add(gen)
    prev = [gen.id]

    if verify:
        v = GenerationStep(
            kind=GenerationStepKind.verify_visual,
            description="Score composite / CLIP / sharpness",
            deps=list(prev),
            kwargs={"tool": "verify"},
        )
        plan.add(v)
        prev = [v.id]

    if self_correct:
        plan.add(
            GenerationStep(
                kind=GenerationStepKind.critique_revise,
                description="CLIP-gated refine if verify fails",
                deps=list(prev),
                kwargs={"tool": "self_correct"},
            )
        )

    return plan


def plan_visual_brain(
    prompt: str,
    *,
    has_references: bool = False,
    web_search: bool = True,
    expected_text: str = "",
    use_rag: bool = True,
    expand: bool = True,
    goal_id: Optional[str] = None,
) -> GenerationPlan:
    """
    Understand → dissect → brief → generate → verify → OCR/edit plan.

    Keeps user prompt in scene brief; never drops original intent.
    """
    gid = goal_id or uuid.uuid4().hex[:12]
    plan = GenerationPlan(goal_id=gid)
    prev: List[str] = []

    if use_rag:
        s = GenerationStep(
            kind=GenerationStepKind.knowledge_retrieval,
            description="Local TF-IDF facts",
            kwargs={"tool": "rag_retrieve"},
        )
        plan.add(s)
        prev = [s.id]

    if web_search:
        s = GenerationStep(
            kind=GenerationStepKind.knowledge_retrieval,
            description="Web image search (DuckDuckGo / Wikimedia)",
            deps=list(prev),
            kwargs={"tool": "web_search"},
        )
        plan.add(s)
        prev = [s.id]

    if has_references or web_search:
        s = GenerationStep(
            kind=GenerationStepKind.semantic_segment,
            description="OCR + VLM understand references",
            deps=list(prev),
            kwargs={"tool": "understand_refs"},
        )
        plan.add(s)
        prev = [s.id]

        s = GenerationStep(
            kind=GenerationStepKind.semantic_segment,
            description="Dissect parts for ControlNet / inpaint init",
            deps=list(prev),
            kwargs={"tool": "dissect_refs"},
        )
        plan.add(s)
        prev = [s.id]

        s = GenerationStep(
            kind=GenerationStepKind.refine_prompt,
            description="Synthesize scene brief from user + references",
            deps=list(prev),
            kwargs={"tool": "build_scene_brief"},
        )
        plan.add(s)
        prev = [s.id]
    elif expand:
        s = GenerationStep(
            kind=GenerationStepKind.refine_prompt,
            description="Expand prompt",
            deps=list(prev),
            kwargs={"tool": "expand_prompt"},
        )
        plan.add(s)
        prev = [s.id]

    gen = GenerationStep(
        kind=GenerationStepKind.diffusion_sample,
        description="Generate with understanding-informed args",
        deps=list(prev),
        kwargs={"tool": "generate"},
    )
    plan.add(gen)
    prev = [gen.id]

    v = GenerationStep(
        kind=GenerationStepKind.verify_visual,
        description="Composite / CLIP / sharpness",
        deps=list(prev),
        kwargs={"tool": "verify"},
    )
    plan.add(v)
    prev = [v.id]

    if expected_text:
        plan.add(
            GenerationStep(
                kind=GenerationStepKind.verify_text_ocr,
                description="OCR match vs expected text",
                deps=list(prev),
                kwargs={"tool": "verify_ocr"},
            )
        )
        prev = [plan.steps[-1].id]

    plan.add(
        GenerationStep(
            kind=GenerationStepKind.inpaint_region,
            description="Inpaint / OCR-fix if verify fails",
            deps=list(prev),
            kwargs={"tool": "inpaint_edit"},
        )
    )
    plan.add(
        GenerationStep(
            kind=GenerationStepKind.critique_revise,
            description="Self-correct pass",
            deps=[plan.steps[-1].id],
            kwargs={"tool": "self_correct"},
        )
    )
    return plan


def build_default_plan(
    prompt: str,
    defaults: Optional[AgenticStackDefaults] = None,
    *,
    use_rag: bool = True,
) -> GenerationPlan:
    d = defaults or AgenticStackDefaults()
    return plan_from_prompt(
        prompt,
        use_rag=use_rag,
        expand=d.expand_prompt,
        verify=True,
        self_correct=d.self_correct,
    )


def default_stop_conditions(defaults: Optional[AgenticStackDefaults] = None) -> StopConditions:
    d = defaults or AgenticStackDefaults()
    return StopConditions(
        max_outer_loops=d.max_reflect_loops,
        min_clip_alignment=d.min_clip_accept,
        patience_no_gain=2,
    )


__all__ = ["build_default_plan", "default_stop_conditions", "plan_from_prompt", "plan_visual_brain"]
