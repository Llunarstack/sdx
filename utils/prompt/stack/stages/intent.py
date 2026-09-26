"""SFW / NSFW / vague / ambiguous / very-long intent helpers."""

from __future__ import annotations

from ..context import PromptContext


def stage_intent_helpers(ctx: PromptContext) -> None:
    args = ctx.args
    if args is not None and bool(getattr(args, "no_intent_helpers", False)):
        return
    mode = "auto"
    if args is not None:
        mode = str(getattr(args, "prompt_intent", "auto") or "auto").strip().lower()
    if mode in ("off", "none", "false", "0", "skip"):
        return
    try:
        from utils.prompt.intent_helpers import apply_intent_helpers

        uncensored = True
        if args is not None:
            uncensored = bool(getattr(args, "uncensored_mode", True))
        pos, neg, intent = apply_intent_helpers(
            ctx.positive,
            ctx.negative,
            mode=mode,
            uncensored=uncensored,
        )
        if pos != ctx.positive or neg != ctx.negative:
            ctx.positive, ctx.negative = pos, neg
        ctx.metadata["prompt_intent"] = intent.primary
        ctx.trace.append(f"intent:{intent.primary}")
        if args is not None:
            args._prompt_intent = intent.primary
            if intent.is_very_long:
                args._prompt_intent_long = True
    except Exception as exc:
        ctx.metadata["intent_helpers_error"] = str(exc)
        ctx.trace.append("intent:failed")
