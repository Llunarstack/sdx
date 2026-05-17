"""Default negative assembly and staged negative fragments."""

from __future__ import annotations

from typing import Any

from ..context import PromptContext, StackMode
from ..tokens import append_csv


def stage_negative_bootstrap(ctx: PromptContext) -> None:
    """Build base negative from config, staged attrs, and guidance negs."""
    args = ctx.args
    positive = ctx.positive
    art = ctx.artifacts

    try:
        from config.defaults.prompt_domains import (
            ANTI_AI_LOOK_NEGATIVE,
            ANTI_AI_LOOK_NEGATIVE_STRONG,
            DEFAULT_NEGATIVE_PROMPT,
            TEXT_IN_IMAGE_NEGATIVE,
            TEXT_IN_IMAGE_PHRASES,
        )
    except ImportError:
        DEFAULT_NEGATIVE_PROMPT = " "
        TEXT_IN_IMAGE_NEGATIVE = "garbled text, watermark, low quality, blurry"
        TEXT_IN_IMAGE_PHRASES = ("sign that says", "text that says", "lettering")
        ANTI_AI_LOOK_NEGATIVE = "oversaturated, plastic skin, synthetic, CGI"
        ANTI_AI_LOOK_NEGATIVE_STRONG = ANTI_AI_LOOK_NEGATIVE

    user_neg = ""
    if args is not None:
        user_neg = (getattr(args, "negative_prompt", None) or "").strip()

    if user_neg:
        negative = user_neg
        ctx.trace.append("negative:user")
    elif args is not None and getattr(args, "text_in_image", False):
        negative = TEXT_IN_IMAGE_NEGATIVE
        ctx.trace.append("negative:text_in_image_flag")
    else:
        p_lower = positive.lower()
        if any(phrase in p_lower for phrase in TEXT_IN_IMAGE_PHRASES):
            negative = TEXT_IN_IMAGE_NEGATIVE
            ctx.trace.append("negative:text_in_image_heuristic")
        else:
            negative = DEFAULT_NEGATIVE_PROMPT
            ctx.trace.append("negative:default")

    for frag, label in (
        (art.layout_negative, "layout"),
        (art.multi_instance_negative, "multi_instance"),
        (art.detailed_scene_negative, "detailed_scene"),
        (art.visual_design_negative, "visual_design"),
    ):
        if frag:
            negative = append_csv(negative, frag)
            ctx.trace.append(f"negative:staged_{label}")

    if args is not None:
        negative = _append_flag_negatives(negative, args, ANTI_AI_LOOK_NEGATIVE, ANTI_AI_LOOK_NEGATIVE_STRONG)

    # Guidance negatives (computed in guidance stage)
    for part in ctx.metadata.get("guidance_neg_parts") or []:
        if part:
            negative = append_csv(negative, part)

    if ctx.mode != StackMode.TRAINING:
        if art.photo_negative:
            negative = append_csv(negative, art.photo_negative)

        if args is not None and getattr(args, "less_ai", False):
            if str(getattr(args, "anti_ai_pack", "none") or "none") == "none":
                args.anti_ai_pack = "lite"
            if str(getattr(args, "human_media_mode", "none") or "none") == "none":
                args.human_media_mode = "photographic"

        if args is not None and len(getattr(args, "lora", []) or []) > 1:
            try:
                from config.defaults.prompt_domains import LORA_STACK_NEGATIVE

                negative = append_csv(negative, LORA_STACK_NEGATIVE)
                ctx.trace.append("negative:lora_stack")
            except ImportError:
                pass

    ctx.negative = negative


def _append_flag_negatives(negative: str, args: Any, anti_ai: str, anti_ai_strong: str) -> str:
    if getattr(args, "naturalize", False):
        neg = anti_ai_strong if getattr(args, "naturalize_deep", False) else anti_ai
        negative = append_csv(negative, neg)
    if getattr(args, "anti_bleed", False):
        try:
            from config.defaults.prompt_domains import CONCEPT_BLEEDING_NEGATIVE

            negative = append_csv(negative, CONCEPT_BLEEDING_NEGATIVE)
        except ImportError:
            pass
    if getattr(args, "diversity", False):
        try:
            from config.defaults.prompt_domains import FLUX_FACE_DIVERSITY_NEGATIVE

            negative = append_csv(negative, FLUX_FACE_DIVERSITY_NEGATIVE)
        except ImportError:
            pass
    if getattr(args, "anti_artifacts", False):
        try:
            from config.defaults.prompt_domains import ARTIFACT_NEGATIVES

            negative = append_csv(negative, ARTIFACT_NEGATIVES)
        except ImportError:
            pass
    if getattr(args, "strong_watermark", False):
        try:
            from config.defaults.prompt_domains import WATERMARK_NEGATIVE_STRONG

            negative = append_csv(negative, WATERMARK_NEGATIVE_STRONG)
        except ImportError:
            pass
    return negative
