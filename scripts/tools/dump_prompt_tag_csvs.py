#!/usr/bin/env python3
"""
Regenerate ``data/prompt_tags/*.csv`` from the in-module Python definitions in
``utils.prompt.content_controls``, plus ``12_artist_composition.csv`` (round-trip from
tables already loaded from disk — keep that file as the source of truth for artist
composition tags).

Run from repo root::

    python -m scripts.tools dump_prompt_tag_csvs

Afterwards, trim ``content_controls.py`` tag literals if you migrate to CSV-only loads.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.prompt import content_controls as cc  # noqa: E402

Row = Tuple[str, str, str]
Attr = Union[List[str], Dict[str, List[str]]]


def _rows_from_list(pack: str, items: Sequence[str]) -> List[Row]:
    return [(pack, "_", t) for t in items]


def _rows_from_dict(pack: str, d: Dict[str, List[str]]) -> List[Row]:
    rows: List[Row] = []
    for mode, tags in d.items():
        for t in tags:
            rows.append((pack, mode, t))
    return rows


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pack", "mode", "tag"])
        w.writerows(rows)


def main() -> None:
    out_dir = REPO / "data" / "prompt_tags"

    # --- 01_scores_quality_adherence.csv ---
    r1: List[Row] = []
    r1 += _rows_from_list("best_model_tags", cc._BEST_MODEL_TAGS)
    r1 += _rows_from_list("one_shot_positive", cc._ONE_SHOT_POSITIVE)
    r1 += _rows_from_list("one_shot_negative", cc._ONE_SHOT_NEGATIVE)
    for mode, tags in cc._QUALITY_PACK_POSITIVE.items():
        if mode in ("none", "top", "one_shot"):
            continue
        for t in tags:
            r1.append(("quality_pack_positive", mode, t))
    r1 += _rows_from_list("quality_pack_negative_top", cc._QUALITY_PACK_NEGATIVE["top"])
    r1 += _rows_from_list("quality_pack_negative_one_shot", cc._QUALITY_PACK_NEGATIVE["one_shot"])
    for mode, tags in cc._QUALITY_PACK_NEGATIVE.items():
        if mode in ("none", "top", "one_shot", "micro_detail"):
            continue
        for t in tags:
            r1.append(("quality_pack_negative", mode, t))
    for t in cc._QUALITY_PACK_NEGATIVE["micro_detail"]:
        r1.append(("quality_pack_negative", "micro_detail", t))
    r1 += _rows_from_list("adherence_standard_positive", cc._ADHERENCE_STANDARD_POSITIVE)
    r1 += _rows_from_list("adherence_strict_extra_positive", cc._ADHERENCE_STRICT_EXTRA_POSITIVE)
    r1 += _rows_from_list("adherence_standard_negative", cc._ADHERENCE_STANDARD_NEGATIVE)
    r1 += _rows_from_list("adherence_strict_extra_negative", cc._ADHERENCE_STRICT_EXTRA_NEGATIVE)
    r1 += _rows_from_list("anti_ai_lite_positive", cc._ANTI_AI_LITE_POSITIVE)
    r1 += _rows_from_list("anti_ai_strong_extra_positive", cc._ANTI_AI_STRONG_POSITIVE[len(cc._ANTI_AI_LITE_POSITIVE) :])
    r1 += _rows_from_list("anti_ai_lite_negative", cc._ANTI_AI_LITE_NEGATIVE)
    r1 += _rows_from_list("anti_ai_strong_extra_negative", cc._ANTI_AI_STRONG_NEGATIVE[len(cc._ANTI_AI_LITE_NEGATIVE) :])
    write_csv(out_dir / "01_scores_quality_adherence.csv", r1)

    # --- 02_sfw.csv ---
    r2: List[Row] = []
    pre_best = cc._SFW_POSITIVE[: cc._SFW_POSITIVE.index(cc._BEST_MODEL_TAGS[0])]
    post_best = cc._SFW_POSITIVE[cc._SFW_POSITIVE.index(cc._BEST_MODEL_TAGS[-1]) + 1 :]
    for t in pre_best:
        r2.append(("sfw_positive", "_", t))
    r2.append(("sfw_positive", "_", "__REF__:best_model_tags"))
    for t in post_best:
        r2.append(("sfw_positive", "_", t))
    r2 += _rows_from_list("sfw_negative", cc._SFW_NEGATIVE)
    r2 += _rows_from_dict("sfw_mood_positive", cc._SFW_MOOD_POSITIVE)
    r2 += _rows_from_dict("sfw_pose_positive", cc._SFW_POSE_POSITIVE)
    r2 += _rows_from_dict("sfw_clothing_positive", cc._SFW_CLOTHING_POSITIVE)
    r2 += _rows_from_dict("sfw_environment_positive", cc._SFW_ENVIRONMENT_POSITIVE)
    r2 += _rows_from_dict("sfw_expression_positive", cc._SFW_EXPRESSION_POSITIVE)
    write_csv(out_dir / "02_sfw.csv", r2)

    # --- 03_nsfw_core.csv ---
    r3: List[Row] = []
    pre = cc._NSFW_POSITIVE[: cc._NSFW_POSITIVE.index(cc._BEST_MODEL_TAGS[0])]
    post = cc._NSFW_POSITIVE[cc._NSFW_POSITIVE.index(cc._BEST_MODEL_TAGS[-1]) + 1 :]
    for t in pre:
        r3.append(("nsfw_positive", "_", t))
    r3.append(("nsfw_positive", "_", "__REF__:best_model_tags"))
    for t in post:
        r3.append(("nsfw_positive", "_", t))
    r3 += _rows_from_list("nsfw_negative", cc._NSFW_NEGATIVE)
    r3 += _rows_from_dict("nsfw_pack_positive", cc._NSFW_PACK_POSITIVE)
    r3 += _rows_from_dict("nsfw_pack_negative", cc._NSFW_PACK_NEGATIVE)
    r3 += _rows_from_dict("nsfw_civitai_positive", cc._NSFW_CIVITAI_POSITIVE)
    r3 += _rows_from_dict("nsfw_civitai_negative", cc._NSFW_CIVITAI_NEGATIVE)
    # Remove hits/hits_lite positive rows (dynamic from CIVITAI_HOT_TAGS)
    r3 = [row for row in r3 if not (row[0] == "nsfw_civitai_positive" and row[1] in ("hits", "hits_lite"))]
    r3 += _rows_from_list("civitai_snippet_tags", cc._CIVITAI_SNIPPET_TAGS)
    write_csv(out_dir / "03_nsfw_core.csv", r3)

    # --- 04_scene_people_objects.csv ---
    r4: List[Row] = []
    r4 += _rows_from_dict("domain_positive", cc._DOMAIN_POSITIVE)
    r4 += _rows_from_dict("domain_negative", cc._DOMAIN_NEGATIVE)
    r4 += _rows_from_dict("background_positive", cc._BACKGROUND_POSITIVE)
    r4 += _rows_from_dict("background_negative", cc._BACKGROUND_NEGATIVE)
    r4 += _rows_from_dict("people_layout_positive", cc._PEOPLE_LAYOUT_POSITIVE)
    r4 += _rows_from_dict("people_layout_negative", cc._PEOPLE_LAYOUT_NEGATIVE)
    r4 += _rows_from_dict("relationship_positive", cc._RELATIONSHIP_POSITIVE)
    r4 += _rows_from_dict("relationship_negative", cc._RELATIONSHIP_NEGATIVE)
    r4 += _rows_from_dict("object_layout_positive", cc._OBJECT_LAYOUT_POSITIVE)
    r4 += _rows_from_dict("object_layout_negative", cc._OBJECT_LAYOUT_NEGATIVE)
    r4 += _rows_from_dict("composition_positive", cc._COMPOSITION_POSITIVE)
    r4 += _rows_from_dict("composition_negative", cc._COMPOSITION_NEGATIVE)
    r4 += _rows_from_list("perspective_stability_positive", cc._PERSPECTIVE_STABILITY_POSITIVE)
    r4 += _rows_from_list("perspective_stability_negative", cc._PERSPECTIVE_STABILITY_NEGATIVE)
    r4 += _rows_from_list("duplicate_subject_negative", cc._DUPLICATE_SUBJECT_NEGATIVE)
    write_csv(out_dir / "04_scene_people_objects.csv", r4)

    # --- 05_pose_camera_hands.csv ---
    r5: List[Row] = []
    r5 += _rows_from_dict("pose_positive", cc._POSE_POSITIVE)
    r5 += _rows_from_list("pose_negative_shared", cc._POSE_NEGATIVE)
    r5 += _rows_from_dict("angle_positive", cc._ANGLE_POSITIVE)
    r5 += _rows_from_list("angle_negative_shared", cc._ANGLE_NEGATIVE)
    r5 += _rows_from_dict("sex_view_positive", cc._SEX_VIEW_POSITIVE)
    r5 += _rows_from_dict("hand_positive", cc._HAND_POSITIVE)
    r5 += _rows_from_dict("hand_negative", cc._HAND_NEGATIVE)
    r5 += _rows_from_dict("pose_naturalness_positive", cc._POSE_NATURALNESS_POSITIVE)
    r5 += _rows_from_dict("pose_naturalness_negative", cc._POSE_NATURALNESS_NEGATIVE)
    r5 += _rows_from_dict("typography_positive", cc._TYPOGRAPHY_POSITIVE)
    r5 += _rows_from_dict("typography_negative", cc._TYPOGRAPHY_NEGATIVE)
    write_csv(out_dir / "05_pose_camera_hands.csv", r5)

    # --- 06_clothing_lighting_skin.csv ---
    r6: List[Row] = []
    r6 += _rows_from_dict("clothing_positive", cc._CLOTHING_POSITIVE)
    r6 += _rows_from_dict("clothing_negative", cc._CLOTHING_NEGATIVE)
    r6 += _rows_from_dict("lighting_mode_positive", cc._LIGHTING_MODE_POSITIVE)
    r6 += _rows_from_dict("lighting_mode_negative", cc._LIGHTING_MODE_NEGATIVE)
    r6 += _rows_from_dict("skin_detail_positive", cc._SKIN_DETAIL_POSITIVE)
    r6 += _rows_from_dict("skin_detail_negative", cc._SKIN_DETAIL_NEGATIVE)
    write_csv(out_dir / "06_clothing_lighting_skin.csv", r6)

    # --- 07_nsfw_detail_poses_env.csv ---
    r7: List[Row] = []
    r7 += _rows_from_dict("sex_position_positive", cc._SEX_POSITION_POSITIVE)
    r7 += _rows_from_dict("penetration_positive", cc._PENETRATION_POSITIVE)
    r7 += _rows_from_dict("body_proportion_positive", cc._BODY_PROPORTION_POSITIVE)
    r7 += _rows_from_dict("interaction_intensity_positive", cc._INTERACTION_INTENSITY)
    r7 += _rows_from_dict("advanced_pose_positive", cc._ADVANCED_POSE_POSITIVE)
    r7 += _rows_from_dict("objects_interaction_positive", cc._OBJECTS_INTERACTION_POSITIVE)
    r7 += _rows_from_dict("environment_detail_positive", cc._ENVIRONMENT_DETAIL_POSITIVE)
    write_csv(out_dir / "07_nsfw_detail_poses_env.csv", r7)

    # --- 08_style_media_lora.csv ---
    r8: List[Row] = []
    r8 += _rows_from_dict("style_positive", cc._STYLE_POSITIVE)
    r8 += _rows_from_dict("style_negative", cc._STYLE_NEGATIVE)
    r8 += _rows_from_list("style_lock_positive", cc._STYLE_LOCK_POSITIVE)
    r8 += _rows_from_list("style_lock_negative", cc._STYLE_LOCK_NEGATIVE)
    r8 += _rows_from_dict("human_media_positive", cc._HUMAN_MEDIA_POSITIVE)
    r8 += _rows_from_dict("human_media_negative", cc._HUMAN_MEDIA_NEGATIVE)
    r8 += _rows_from_dict("lora_scaffold_positive", cc._LORA_SCAFFOLD_POSITIVE)
    r8 += _rows_from_dict("lora_scaffold_negative", cc._LORA_SCAFFOLD_NEGATIVE)
    write_csv(out_dir / "08_style_media_lora.csv", r8)

    # --- 09_misc.csv ---
    r9: List[Row] = []
    r9 += _rows_from_list("unwanted_text_negative", cc._UNWANTED_TEXT_NEGATIVE)
    for a, b in cc._CONFLICTING_TAG_PAIRS:
        r9.append(("conflicting_tag_pairs", "_", f"{a}|||{b}"))
    write_csv(out_dir / "09_misc.csv", r9)

    # --- 12_artist_composition.csv (round-trip loaded tables; edit the CSV, then re-run dump to normalize) ---
    from utils.prompt import content_control_tags as cct

    r12: List[Row] = []
    pos_art = {k: v for k, v in cct._ARTIST_COMPOSITION_POSITIVE.items() if k != "none"}
    neg_art = {k: v for k, v in cct._ARTIST_COMPOSITION_NEGATIVE.items() if k != "none"}
    r12 += _rows_from_dict("artist_composition_positive", pos_art)
    r12 += _rows_from_dict("artist_composition_negative", neg_art)
    write_csv(out_dir / "12_artist_composition.csv", r12)

    print(f"Wrote CSVs under {out_dir}")


if __name__ == "__main__":
    main()
