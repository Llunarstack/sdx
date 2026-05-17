"""Post-control enrichments, prompt breakdown, and neg/pos conflict filter."""

from __future__ import annotations

import sys

from ..context import PromptContext, StackMode
from ..tokens import append_csv


def stage_post_enrich(ctx: PromptContext) -> None:
    """Character/scene negatives and scale-distortion guard."""
    art = ctx.artifacts
    negative = ctx.negative

    if ctx.apply_scale_distortion_negative:
        scale_neg = (
            "distorted proportions, warped anatomy, stretched limbs, "
            "elongated limbs, tiny head, giant hands, bad perspective scale"
        )
        negative = append_csv(negative, scale_neg)
        ctx.trace.append("post:scale_distortion")

    if art.character_negative:
        negative = append_csv(negative, art.character_negative)
        ctx.trace.append("post:character_neg")
    if art.scene_negative:
        negative = append_csv(negative, art.scene_negative)
        ctx.trace.append("post:scene_neg")

    ctx.negative = negative


def stage_prompt_breakdown(ctx: PromptContext) -> None:
    if ctx.mode == StackMode.PREVIEW:
        return
    args = ctx.args
    if args is None:
        return
    pbrk = str(getattr(args, "prompt_breakdown", "off") or "off").lower()
    if pbrk == "off" or not ctx.positive.strip():
        return
    if getattr(args, "_layout_compiled", None) is not None:
        return
    try:
        from utils.prompt.prompt_breakdown import apply_prompt_breakdown, warrant_prompt_breakdown
        from utils.prompt.prompt_layout import PRESET_SECTION_ORDER

        if pbrk == "on" or (pbrk == "auto" and warrant_prompt_breakdown(ctx.positive)):
            fmt = str(getattr(args, "prompt_breakdown_format", "ordered") or "ordered").lower()
            if fmt not in ("ordered", "labeled"):
                fmt = "ordered"
            order = str(getattr(args, "prompt_breakdown_order", "subject_first") or "subject_first").lower()
            if order not in PRESET_SECTION_ORDER:
                order = "subject_first"
            flat, t5s = apply_prompt_breakdown(
                ctx.positive,
                order=order,  # type: ignore[arg-type]
                output_format=fmt,  # type: ignore[arg-type]
            )
            if flat:
                ctx.positive = flat
                args.prompt = flat
                if t5s != flat:
                    ctx.t5_positive_hint = t5s
                ctx.trace.append(f"breakdown:{pbrk}")
                print(
                    f"Prompt breakdown ({pbrk}, order={order}, format={fmt}).",
                    file=sys.stderr,
                )
    except Exception as exc:
        print(f"Prompt breakdown skipped: {exc}", file=sys.stderr)


def stage_neg_filter(ctx: PromptContext) -> None:
    args = ctx.args
    if args is not None and getattr(args, "no_neg_filter", False):
        ctx.trace.append("neg_filter:skipped")
        return
    from utils.prompt.fast_paths import filter_negative_by_positive

    raw = ctx.negative
    filtered = filter_negative_by_positive(ctx.positive, raw)
    if not filtered.strip():
        filtered = " "
    if filtered != raw:
        ctx.trace.append("neg_filter:applied")
        if ctx.mode != StackMode.PREVIEW:
            print(
                f'Negative prompt filtered (conflict resolution): "{raw[:60]}{"..." if len(raw) > 60 else ""}" '
                f'-> "{filtered[:60]}{"..." if len(filtered) > 60 else ""}"',
                file=sys.stderr,
            )
    else:
        ctx.trace.append("neg_filter:none")
    ctx.negative = filtered
