"""Prompt enrichment stages: intelligence, style genome, special helpers, content controls and clauses."""

from __future__ import annotations

from typing import Any

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


def _get_active_genome(args: Any):
    if args is None:
        return None
    genome = getattr(args, "_active_style_genome", None)
    if genome is not None:
        return genome
    path = str(getattr(args, "style_genome_file", "") or "").strip()
    if not path:
        return None
    try:
        from utils.prompt.style_explore import resolve_style_genome_for_args

        return resolve_style_genome_for_args(args, str(getattr(args, "prompt", "") or ""))
    except Exception:
        return None


def stage_style_genome(ctx: PromptContext) -> None:
    """Merge genome axes into positive/negative; set style head on args."""
    args = ctx.args
    if args is not None and not getattr(args, "style_genome_enabled", True):
        return
    if args is not None and not (
        getattr(args, "_active_style_genome", None)
        or str(getattr(args, "style_genome_file", "") or "").strip()
        or int(getattr(args, "invent_styles", 0) or 0) > 0
    ):
        return

    genome = _get_active_genome(args)
    if genome is None:
        return

    chaos_level = float(getattr(args, "style_chaos_level", 0.0) or 0.0) if args is not None else 0.0
    if chaos_level > 0.01:
        try:
            from utils.prompt.style_genome_chaos import apply_chaos_level

            genome = apply_chaos_level(genome, chaos_level)
        except Exception:
            pass

    ctx.positive = genome.compile_positive(ctx.positive)
    ctx.negative = genome.compile_negative(ctx.negative)
    ctx.metadata["style_genome"] = genome.to_dict()
    ctx.trace.append(f"style_genome:{genome.id}")
    if chaos_level > 0.35:
        ctx.trace.append(f"style_genome:chaos_{chaos_level:.0%}")

    if args is not None:
        if not (getattr(args, "style", None) or "").strip():
            args.style = genome.style_head_string()
        if hasattr(args, "prompt"):
            args.prompt = ctx.positive


def stage_special_helpers(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None:
        return
    mode = str(getattr(args, "prompt_special_helpers", "auto") or "auto").strip().lower()
    if mode in ("off", "none", "false", "0", "skip"):
        return
    try:
        from utils.prompt.special_prompt_helpers import apply_special_helpers

        pos, neg = apply_special_helpers(ctx.positive, ctx.negative, category=mode)
        if pos != ctx.positive or neg != ctx.negative:
            ctx.positive, ctx.negative = pos, neg
            ctx.trace.append(f"special:{mode}")
    except Exception as exc:
        ctx.metadata["special_helpers_error"] = str(exc)
        ctx.trace.append("special:failed")


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
