"""Prompt stack orchestrator."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from .context import PromptContext, PromptResult, StackMode

StageFn = Callable[[PromptContext], None]

_DEFAULT_INFERENCE_STAGES: List[StageFn] = []
_PREVIEW_STAGES: List[StageFn] = []
_TRAINING_STAGES: List[StageFn] = []


def _load_stages() -> None:
    global _DEFAULT_INFERENCE_STAGES, _PREVIEW_STAGES, _TRAINING_STAGES
    if _DEFAULT_INFERENCE_STAGES:
        return
    from .stages.content import stage_clauses, stage_content_controls, stage_intelligence
    from .stages.finalize import stage_neg_filter, stage_post_enrich, stage_prompt_breakdown
    from .stages.guidance import stage_guidance
    from .stages.negative import stage_negative_bootstrap
    from .stages.special import stage_special_helpers
    from .stages.style_genome import stage_style_genome

    _TRAINING_STAGES = [stage_guidance]
    _PREVIEW_STAGES = [
        stage_intelligence,
        stage_style_genome,
        stage_special_helpers,
        stage_guidance,
        stage_negative_bootstrap,
        stage_content_controls,
        stage_clauses,
        stage_neg_filter,
    ]
    _DEFAULT_INFERENCE_STAGES = [
        stage_intelligence,
        stage_style_genome,
        stage_special_helpers,
        stage_guidance,
        stage_negative_bootstrap,
        stage_content_controls,
        stage_clauses,
        stage_post_enrich,
        stage_prompt_breakdown,
        stage_neg_filter,
    ]


def run_prompt_stack(ctx: PromptContext, stages: Optional[Sequence[StageFn]] = None) -> PromptResult:
    """
    Run the prompt stack on *ctx* (mutates positive/negative in place).

    Returns :class:`PromptResult` with trace and analysis metadata.
    """
    _load_stages()
    if stages is None:
        if ctx.mode == StackMode.TRAINING:
            stages = _TRAINING_STAGES
        elif ctx.mode == StackMode.PREVIEW:
            stages = _PREVIEW_STAGES
        else:
            stages = _DEFAULT_INFERENCE_STAGES

    for stage in stages:
        stage(ctx)

    analysis = ctx.metadata.get("analysis")
    analysis_dict = None
    if analysis is not None and hasattr(analysis, "__dataclass_fields__"):
        from dataclasses import asdict

        analysis_dict = asdict(analysis)

    return PromptResult(
        positive=ctx.positive,
        negative=ctx.negative,
        t5_positive_hint=ctx.t5_positive_hint,
        trace=list(ctx.trace),
        analysis=analysis_dict,
        resolved_controls=ctx.metadata.get("resolved_controls"),
    )
