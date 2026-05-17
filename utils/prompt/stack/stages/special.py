"""Optional special-prompt helpers (surreal, horror, narrative, technical, …)."""

from __future__ import annotations

from ..context import PromptContext


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
