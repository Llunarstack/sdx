"""Content-control stage + optional intelligence clauses."""

from __future__ import annotations

from ..clauses import apply_clauses
from ..context import PromptContext
from ..controls import apply_resolved_controls, resolve_content_controls


def stage_intelligence(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None or not getattr(args, "prompt_stack_intelligence", True):
        return
    from ..intelligence import analyze_prompt, apply_intelligence

    analysis = analyze_prompt(ctx.positive)
    ctx.metadata["analysis"] = analysis
    ctx.positive, extra_controls = apply_intelligence(
        ctx.positive,
        analysis,
        auto_quality=bool(getattr(args, "prompt_stack_auto_quality", True)),
        auto_controls=bool(getattr(args, "auto_content_fix", True)),
    )
    if extra_controls:
        ctx.metadata["intelligence_controls"] = extra_controls
    ctx.trace.append(f"intelligence:{analysis.complexity}")


def stage_content_controls(ctx: PromptContext) -> None:
    args = ctx.args
    auto = bool(getattr(args, "auto_content_fix", True)) if args is not None else True
    state = resolve_content_controls(args, ctx.positive, auto_infer=auto)

    intel = ctx.metadata.get("intelligence_controls") or {}
    if intel:
        from ..controls import merge_content_control_overrides

        state = merge_content_control_overrides(state, intel)

    try:
        pos, neg = apply_resolved_controls(ctx.positive, ctx.negative, state)
        ctx.positive, ctx.negative = pos, neg
        ctx.metadata["resolved_controls"] = {
            k: v for k, v in state.to_apply_kwargs().items() if k.endswith("_mode") or k.endswith("_pack")
        }
        ctx.trace.append("content_controls")
    except Exception as exc:
        ctx.metadata["content_controls_error"] = str(exc)
        ctx.trace.append("content_controls:failed")


def stage_clauses(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None:
        return
    raw = str(getattr(args, "prompt_clauses", "") or "").strip()
    if not raw:
        return
    names = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    ctx.positive, ctx.negative = apply_clauses(ctx.positive, ctx.negative, names)
    ctx.trace.append(f"clauses:{len(names)}")
