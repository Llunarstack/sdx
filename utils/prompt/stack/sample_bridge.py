"""Bridge: sample.py post-checkpoint prompt assembly via PromptStack."""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Tuple

from .context import PromptArtifacts, PromptContext, StackMode
from .runner import run_prompt_stack


def _fallback_negative(args: Any, positive: str) -> str:
    """Match ``stage_negative_bootstrap`` when the full stack fails."""
    user = (getattr(args, "negative_prompt", None) or "").strip() if args is not None else ""
    if user:
        return user
    try:
        from config.defaults.prompt_domains import (
            DEFAULT_NEGATIVE_PROMPT,
            TEXT_IN_IMAGE_NEGATIVE,
            TEXT_IN_IMAGE_PHRASES,
        )
    except ImportError:
        return " "
    if args is not None and getattr(args, "text_in_image", False):
        return TEXT_IN_IMAGE_NEGATIVE
    p_lower = (positive or "").lower()
    if any(phrase in p_lower for phrase in TEXT_IN_IMAGE_PHRASES):
        return TEXT_IN_IMAGE_NEGATIVE
    return DEFAULT_NEGATIVE_PROMPT


def apply_sample_prompt_stack(
    args: Any,
    prompt_to_encode: str,
    *,
    character_negative_additions: str = "",
    scene_negative_additions: str = "",
    apply_scale_distortion: bool = False,
) -> Tuple[str, str]:
    """
    Run the unified prompt stack for ``sample.py`` (guidance → negative → controls → finalize).

    Replaces the former inline block (~300 lines). Mutates *args* where legacy code did
    (``args.prompt``, ``human_media_mode``, photo post fields, etc.).
    """
    artifacts = PromptArtifacts(
        layout_negative=str(getattr(args, "_prompt_layout_negative", "") or "").strip(),
        multi_instance_negative=str(getattr(args, "_multi_instance_negative", "") or "").strip(),
        detailed_scene_negative=str(getattr(args, "_detailed_scene_negative", "") or "").strip(),
        visual_design_negative=str(getattr(args, "_visual_design_negative", "") or "").strip(),
        character_negative=(character_negative_additions or "").strip(),
        scene_negative=(scene_negative_additions or "").strip(),
    )

    if args is not None and (
        int(getattr(args, "invent_styles", 0) or 0) > 0 or str(getattr(args, "style_genome_file", "") or "").strip()
    ):
        try:
            from utils.prompt.style_explore import resolve_style_genome_for_args

            resolve_style_genome_for_args(args, prompt_to_encode)
        except Exception as exc:
            print(f"Warning: style genome resolve failed: {exc}", file=sys.stderr)

    ctx = PromptContext(
        positive=prompt_to_encode,
        negative="",
        mode=StackMode.INFERENCE,
        args=args,
        artifacts=artifacts,
        apply_scale_distortion_negative=apply_scale_distortion,
    )

    try:
        result = run_prompt_stack(ctx)
    except Exception as exc:
        print(f"Warning: prompt stack failed: {exc}", file=sys.stderr)
        if os.environ.get("SDX_DEBUG", "").strip():
            traceback.print_exc()
        return prompt_to_encode, _fallback_negative(args, prompt_to_encode)

    setattr(args, "_encode_t5_positive_hint", result.t5_positive_hint or "")
    if result.analysis:
        setattr(args, "_prompt_stack_analysis", result.analysis)
    if result.resolved_controls:
        setattr(args, "_prompt_stack_controls", result.resolved_controls)
    if os.environ.get("SDX_PROMPT_STACK_TRACE", "").strip():
        print("Prompt stack trace:", " -> ".join(result.trace), file=sys.stderr)

    return result.positive, result.negative
