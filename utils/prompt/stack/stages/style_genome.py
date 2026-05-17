"""Apply an invented StyleGenome to the prompt stack."""

from __future__ import annotations

from typing import Any

from ..context import PromptContext


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
