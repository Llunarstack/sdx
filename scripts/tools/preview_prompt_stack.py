#!/usr/bin/env python3
"""
Preview the full SDX prompt stack (guidance, controls, clauses, neg filter) without a GPU.

Usage:
  python -m scripts.tools preview_prompt_stack --prompt "1girl, city street, golden hour"
  SDX_PROMPT_STACK_TRACE=1 python -m scripts.tools preview_prompt_stack --prompt "..." --prompt-clauses hands.stable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from utils.prompt.stack import PromptContext, StackMode, run_prompt_stack

    from config import DEFAULT_NEGATIVE_PROMPT

    p = argparse.ArgumentParser(description="Preview SDX PromptStack output (pos/neg + trace).")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--quality-pack", default="none")
    p.add_argument("--adherence-pack", default="none")
    p.add_argument("--auto-content-fix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prompt-clauses", default="", help="Comma-separated clause names (see utils.prompt.stack.clauses)")
    p.add_argument("--no-prompt-stack-intelligence", action="store_true")
    p.add_argument("--prompt-special-helpers", default="auto")
    p.add_argument("--less-ai", action="store_true")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = p.parse_args()

    ns = SimpleNamespace(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        shortcomings_mitigation="auto" if args.less_ai else "none",
        shortcomings_2d=False,
        art_guidance_mode="auto" if args.less_ai else "none",
        anatomy_guidance="none",
        style_guidance_mode="auto" if args.less_ai else "none",
        style_guidance_artists=True,
        no_art_guidance_photography=False,
        auto_photo_realism=True,
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
        prompt_clauses=args.prompt_clauses,
        prompt_stack_intelligence=not args.no_prompt_stack_intelligence,
        prompt_special_helpers=args.prompt_special_helpers,
        prompt_stack_auto_quality=True,
        one_shot_boost=True,
        less_ai=args.less_ai,
        lora=[],
        text_in_image=False,
        no_neg_filter=False,
        scene_domain="none",
        view_angle="none",
        pose_mode="none",
        style_mode="none",
        clothing_mode="none",
        composition_mode="none",
        artist_composition="none",
        people_layout="none",
        hand_mode="none",
        lighting_mode="none",
        anti_ai_pack="lite" if args.less_ai else "none",
        human_media_mode="photographic" if args.less_ai else "none",
        lora_scaffold="none",
        lora_scaffold_auto=False,
    )

    ctx = PromptContext(positive=args.prompt, mode=StackMode.PREVIEW, args=ns)
    result = run_prompt_stack(ctx)

    if args.json:
        print(
            json.dumps(
                {
                    "positive": result.positive,
                    "negative": result.negative,
                    "trace": result.trace,
                    "analysis": result.analysis,
                    "resolved_controls": result.resolved_controls,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print("=== POSITIVE ===")
    print(result.positive)
    print("\n=== NEGATIVE ===")
    print(result.negative)
    if result.analysis:
        print("\n=== ANALYSIS ===")
        print(f"complexity: {result.analysis.get('complexity')}  domains: {result.analysis.get('domains')}")
    print("\n=== TRACE ===")
    print(" -> ".join(result.trace))
    if os.environ.get("SDX_PROMPT_STACK_TRACE", "").strip() and result.resolved_controls:
        print("\n=== CONTROLS ===")
        for k, v in sorted(result.resolved_controls.items()):
            if v and v != "none":
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
