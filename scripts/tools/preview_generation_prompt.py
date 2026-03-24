#!/usr/bin/env python3
"""
Preview final positive/negative prompts after content_controls + optional neg filter — no GPU, no checkpoint.

Does not replicate every sample.py step (character sheets, scene blueprints, hard-style prefix,
naturalize prefix, boost-quality, emphasis weights). Use for tuning --safety-mode, --quality-pack,
--anti-ai-pack, --human-media, Civitai packs, etc.

Usage:
  python scripts/tools/preview_generation_prompt.py --prompt "1girl, red dress" --safety-mode nsfw
  SDX_DEBUG=1 python scripts/tools/preview_generation_prompt.py --prompt "..." --less-ai
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from config import DEFAULT_NEGATIVE_PROMPT
    from utils.prompt.content_controls import apply_content_controls, infer_content_controls_from_prompt
    from utils.prompt.neg_filter import filter_negative_by_positive

    p = argparse.ArgumentParser(description="Preview prompts after content_controls (no model load).")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative-prompt", type=str, default="", help="Empty => DEFAULT_NEGATIVE_PROMPT from config")
    p.add_argument("--no-neg-filter", action="store_true", help="Skip pos/neg token conflict filter")
    p.add_argument("--safety-mode", type=str, default="none", choices=["none", "sfw", "nsfw"])
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
    p.add_argument("--people-layout", type=str, default="none")
    p.add_argument("--hand-mode", type=str, default="none")
    p.add_argument("--lighting-mode", type=str, default="none")
    p.add_argument("--nsfw-pack", type=str, default="none")
    p.add_argument("--sex-position", type=str, default="none")
    p.add_argument("--clothing-mode", type=str, default="none")
    p.add_argument("--nsfw-civitai-pack", type=str, default="none")
    p.add_argument("--civitai-trigger-bank", type=str, default="none")
    p.add_argument("--civitai-model-bank-csv", type=str, default="")
    p.add_argument("--civitai-frequency-txt", type=str, default="")
    p.add_argument("--style-lock", action="store_true")
    p.add_argument("--anti-style-bleed", action="store_true")
    p.add_argument("--anti-duplicate-subjects", action="store_true")
    p.add_argument("--anti-perspective-drift", action="store_true")
    p.add_argument("--cleanup-conflicting-tags", action="store_true")
    p.add_argument("--text-in-image", action="store_true")
    args = p.parse_args()

    human_media_mode = str(args.human_media_mode or "none")
    anti_ai_pack = str(args.anti_ai_pack or "none")
    lora_scaffold_ef = str(args.lora_scaffold or "none")

    if args.less_ai:
        if anti_ai_pack == "none":
            anti_ai_pack = "lite"
        if human_media_mode == "none":
            human_media_mode = "photographic"

    pos = (args.prompt or "").strip()
    neg = (args.negative_prompt or "").strip() or DEFAULT_NEGATIVE_PROMPT

    if int(args.lora_count) >= 2:
        try:
            from config.prompt_domains import LORA_STACK_NEGATIVE

            neg = f"{neg}, {LORA_STACK_NEGATIVE}".strip()
        except ImportError:
            pass

    scene_domain = "none"
    view_angle = str(args.view_angle or "none")
    pose_mode = str(args.pose_mode or "none")
    style_mode = str(args.style_mode or "none")
    safety_mode = str(args.safety_mode or "none")
    clothing_mode = str(args.clothing_mode or "none")
    composition_mode = str(args.composition_mode or "none")
    people_layout = str(args.people_layout or "none")
    hand_mode = str(args.hand_mode or "none")
    lighting_mode = str(args.lighting_mode or "none")
    nsfw_pack = str(args.nsfw_pack or "none")
    sex_position = str(args.sex_position or "none")
    adherence_pack = str(getattr(args, "adherence_pack", "none") or "none")

    if args.auto_content_fix:
        inferred = infer_content_controls_from_prompt(pos)
        if scene_domain == "none":
            scene_domain = inferred.get("scene_domain", scene_domain)
        if view_angle == "none":
            view_angle = inferred.get("view_angle", view_angle)
        if pose_mode == "none":
            pose_mode = inferred.get("pose_mode", pose_mode)
        if style_mode == "none":
            style_mode = inferred.get("style_mode", style_mode)
        if safety_mode == "none" and inferred.get("safety_mode"):
            safety_mode = inferred["safety_mode"]
        if composition_mode == "none" and inferred.get("composition_mode"):
            composition_mode = inferred["composition_mode"]
        if people_layout == "none" and inferred.get("people_layout"):
            people_layout = inferred["people_layout"]
        if hand_mode == "none" and inferred.get("hand_mode"):
            hand_mode = inferred["hand_mode"]
        if lighting_mode == "none" and inferred.get("lighting_mode"):
            lighting_mode = inferred["lighting_mode"]
        if clothing_mode == "none" and inferred.get("clothing_mode"):
            clothing_mode = inferred["clothing_mode"]
        if nsfw_pack == "none" and inferred.get("nsfw_pack"):
            nsfw_pack = inferred["nsfw_pack"]
        if sex_position == "none" and inferred.get("sex_position"):
            sex_position = inferred["sex_position"]
        if human_media_mode == "none" and inferred.get("human_media_mode"):
            human_media_mode = inferred["human_media_mode"]
        if adherence_pack == "none" and inferred.get("adherence_pack"):
            adherence_pack = str(inferred["adherence_pack"])

    try:
        pos_o, neg_o = apply_content_controls(
            pos,
            neg,
            safety_mode=safety_mode,
            pose_mode=pose_mode,
            view_angle=view_angle,
            subject_sex="none",
            scene_domain=scene_domain,
            clothing_mode=clothing_mode,
            background_mode="none",
            people_layout=people_layout,
            relationship_mode="none",
            object_layout="none",
            hand_mode=hand_mode,
            pose_naturalness="none",
            typography_mode="none",
            quality_pack=str(args.quality_pack or "none"),
            lighting_mode=lighting_mode,
            skin_detail_mode="none",
            nsfw_pack=nsfw_pack,
            sex_position=sex_position,
            penetration_detail="none",
            body_proportion="none",
            interaction_intensity="none",
            advanced_pose="none",
            object_interaction="none",
            environment_type="none",
            sfw_mood="none",
            sfw_pose="none",
            sfw_clothing="none",
            sfw_environment="none",
            sfw_expression="none",
            style_mode=style_mode,
            style_lock=bool(args.style_lock),
            anti_style_bleed=bool(args.anti_style_bleed),
            composition_mode=composition_mode,
            anti_duplicate_subjects=bool(args.anti_duplicate_subjects),
            anti_perspective_drift=bool(args.anti_perspective_drift),
            cleanup_conflicting_tags=bool(args.cleanup_conflicting_tags),
            allow_text_in_image=bool(args.text_in_image),
            nsfw_civitai_pack=str(args.nsfw_civitai_pack or "none"),
            civitai_trigger_bank=str(args.civitai_trigger_bank or "none"),
            civitai_model_bank_csv=(str(args.civitai_model_bank_csv or "").strip() or None),
            civitai_frequency_txt=(str(args.civitai_frequency_txt or "").strip() or None),
            one_shot_boost=bool(args.one_shot_boost),
            anti_ai_pack=anti_ai_pack,
            human_media_mode=human_media_mode,
            lora_scaffold=lora_scaffold_ef,
            adherence_pack=adherence_pack,
        )
    except Exception as e:
        print(f"apply_content_controls failed: {e}", file=sys.stderr)
        if os.environ.get("SDX_DEBUG", "").strip():
            raise
        return 1

    if not args.no_neg_filter:
        neg_f = filter_negative_by_positive(pos_o, neg_o)
    else:
        neg_f = neg_o

    print("=== effective positive ===")
    print(pos_o)
    print("\n=== effective negative ===")
    print(neg_f if neg_f.strip() else "(empty/space)")
    print(f"\n(positive length chars: {len(pos_o)}, negative chars: {len(neg_f)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
