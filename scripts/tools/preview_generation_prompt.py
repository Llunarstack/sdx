#!/usr/bin/env python3
"""
Preview final positive/negative prompts — delegates to PromptStack v2.

Legacy entrypoint; prefer ``preview_prompt_stack`` for trace/JSON/clauses.

Usage:
  python -m scripts.tools preview_generation_prompt --prompt "1girl, red dress" --quality-pack top
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from types import SimpleNamespace

    from utils.prompt.stack import PromptContext, StackMode, run_prompt_stack

    from config import DEFAULT_NEGATIVE_PROMPT

    p = argparse.ArgumentParser(description="Preview prompts after PromptStack (no model load).")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative-prompt", type=str, default="", help="Empty => DEFAULT_NEGATIVE_PROMPT from config")
    p.add_argument("--quality-pack", type=str, default="none")
    p.add_argument(
        "--adherence-pack",
        type=str,
        default="none",
        help="none|standard|strict — prompt literalism (auto-content-fix can infer from long prompts)",
    )
    p.add_argument("--one-shot-boost", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anti-ai-pack", type=str, default="none", choices=["none", "lite", "strong"])
    p.add_argument("--human-media", dest="human_media_mode", type=str, default="none")
    p.add_argument("--lora-scaffold", type=str, default="none")
    p.add_argument("--lora-count", type=int, default=0, help="If >=2, append LORA_STACK_NEGATIVE like sample.py")
    p.add_argument("--less-ai", action="store_true")
    p.add_argument("--auto-content-fix", dest="auto_content_fix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pose-mode", type=str, default="none")
    p.add_argument("--view-angle", type=str, default="none")
    p.add_argument("--style-mode", type=str, default="none")
    p.add_argument("--composition-mode", type=str, default="none")
    p.add_argument(
        "--artist-composition",
        type=str,
        default="none",
        choices=["none", "lite", "standard", "perspective", "classical", "full"],
    )
    p.add_argument("--people-layout", type=str, default="none")
    p.add_argument("--hand-mode", type=str, default="none")
    p.add_argument("--lighting-mode", type=str, default="none")
    p.add_argument("--clothing-mode", type=str, default="none")
    p.add_argument("--style-lock", action="store_true")
    p.add_argument("--anti-style-bleed", action="store_true")
    p.add_argument("--anti-duplicate-subjects", action="store_true")
    p.add_argument("--anti-perspective-drift", action="store_true")
    p.add_argument("--cleanup-conflicting-tags", action="store_true")
    p.add_argument("--text-in-image", action="store_true")
    args = p.parse_args()

    lora_list = [""] * max(0, int(args.lora_count)) if args.lora_count >= 2 else []
    ns = SimpleNamespace(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        shortcomings_mitigation="auto" if args.less_ai else "none",
        shortcomings_2d=False,
        art_guidance_mode="auto" if args.less_ai else "none",
        anatomy_guidance="none",
        style_guidance_mode="auto" if args.less_ai else "none",
        style_guidance_artists=True,
        no_art_guidance_photography=False,
        auto_photo_realism=False,
        photo_realism_pack="none",
        photo_color_grade="none",
        photo_lighting_technique="none",
        photo_filter="none",
        photo_grain_style="none",
        photo_realism_strength=1.0,
        realism_autopilot=False,
        quality_pack=args.quality_pack,
        adherence_pack=args.adherence_pack,
        auto_content_fix=args.auto_content_fix,
        prompt_clauses="",
        prompt_stack_intelligence=True,
        prompt_stack_auto_quality=True,
        one_shot_boost=args.one_shot_boost,
        less_ai=args.less_ai,
        lora=lora_list,
        text_in_image=args.text_in_image,
        no_neg_filter=False,
        scene_domain="none",
        view_angle=args.view_angle,
        pose_mode=args.pose_mode,
        style_mode=args.style_mode,
        clothing_mode=args.clothing_mode,
        composition_mode=args.composition_mode,
        artist_composition=args.artist_composition,
        people_layout=args.people_layout,
        hand_mode=args.hand_mode,
        lighting_mode=args.lighting_mode,
        anti_ai_pack=args.anti_ai_pack if args.anti_ai_pack != "none" else ("lite" if args.less_ai else "none"),
        human_media_mode=args.human_media_mode
        if args.human_media_mode != "none"
        else ("photographic" if args.less_ai else "none"),
        lora_scaffold=args.lora_scaffold,
        lora_scaffold_auto=False,
        style_lock=args.style_lock,
        anti_style_bleed=args.anti_style_bleed,
        anti_duplicate_subjects=args.anti_duplicate_subjects,
        anti_perspective_drift=args.anti_perspective_drift,
        cleanup_conflicting_tags=args.cleanup_conflicting_tags,
        subject_sex="none",
        background_mode="none",
        relationship_mode="none",
        object_layout="none",
        pose_naturalness="none",
        typography_mode="none",
        skin_detail_mode="none",
        body_proportion="none",
        interaction_intensity="none",
    )

    ctx = PromptContext(
        positive=args.prompt,
        negative=args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        mode=StackMode.PREVIEW,
        args=ns,
    )
    result = run_prompt_stack(ctx)

    print("=== POSITIVE ===")
    print(result.positive)
    print("\n=== NEGATIVE ===")
    print(result.negative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
