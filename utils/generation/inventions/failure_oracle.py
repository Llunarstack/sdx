"""FAILURE ORACLE — Map likely T2I failure modes → repair skills.

Diagnoses prompt risk *before* generate and optional post-hoc cues, then
emits a repair plan (countgate, bindlock, inpaint regions, friction, etc.).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FAILURE_ATLAS",
    "FailureOracleReport",
    "diagnose_failures",
    "repair_plan_from_failures",
    "save_failure_atlas_jsonl",
]


@dataclass
class FailureOracleReport:
    risks: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_failures(prompt: str, *, negative: str = "") -> FailureOracleReport:
    p = str(prompt or "").lower()
    scores: dict[str, float] = {}

    # Counting risk
    if re.search(r"\b([2-9]|two|three|four|five|six)\s+\w+", p) or re.search(r"\b[2-9](girl|boy|cat)", p):
        scores["count"] = 0.85
    # Binding / spatial
    if re.search(r"\b(left of|right of|above|below|next to)\b", p) or re.search(
        r"\b(red|blue|green)\s+\w+\b.+\b(red|blue|green)\s+\w+", p
    ):
        scores["binding"] = 0.8
    # Negation
    if re.search(r"\b(no|without|not|n't)\s+\w+", p):
        scores["negation"] = 0.9
    # Hands / anatomy
    if re.search(r"\b(hand|hands|fingers|holding|grasp|pose)\b", p):
        scores["anatomy"] = 0.7
    # Text / glyph
    if '"' in prompt or "'" in prompt or re.search(r"\b(text|sign|logo|poster|typography)\b", p):
        scores["glyph"] = 0.75
    # Multi-character bleed
    if re.search(r"\b(2girl|3girl|two girls|multiple|both wearing)\b", p):
        scores["multi_char"] = 0.8
    # Plastic / beauty
    if re.search(r"\b(portrait|skin|beauty|face|close-?up)\b", p):
        scores["plastic"] = 0.55
    # Long prompt truncation
    if len(p) > 400:
        scores["long_prompt"] = 0.65

    risks = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v >= 0.5]
    return FailureOracleReport(risks=risks, scores=scores, notes=[f"{len(risks)} elevated risks"])


def repair_plan_from_failures(report: FailureOracleReport) -> FailureOracleReport:
    repairs: list[dict[str, Any]] = []
    for risk in report.risks:
        if risk == "count":
            repairs.append({"module": "countgate", "action": "apply_countgate_prompts"})
        elif risk == "binding":
            repairs.append({"module": "bindlock", "action": "apply_bindlock_to_prompts"})
        elif risk == "negation":
            repairs.append({"module": "negatron", "action": "apply_negatron"})
        elif risk == "anatomy":
            repairs.append({"module": "edit_skills", "action": "fix_region=hands,face", "post": True})
        elif risk == "glyph":
            repairs.append({"module": "ideogram", "action": "plan_ideogram_layout"})
        elif risk == "multi_char":
            repairs.append({"module": "multi_char_cast", "action": "compile_cast_scene"})
        elif risk == "plastic":
            repairs.append({"module": "friction", "action": "apply_friction_to_prompts"})
        elif risk == "long_prompt":
            repairs.append({"module": "prompt_breakdown", "action": "enable prompt_breakdown=auto"})
    report.repairs = repairs
    return report


# (mode id, name, eval prompts that tend to trigger it)
FAILURE_ATLAS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("count_under", "Wrong count (too few)", ("exactly four red apples on a table", "3girls standing in a row")),
    ("count_over", "Wrong count (too many)", ("exactly two cats, no more", "a single bicycle")),
    ("bind_color", "Attribute binding", ("red cube left of blue sphere", "green backpack on a brown dog")),
    ("spatial_left_right", "Left/right", ("the lamp is left of the chair", "person on the right side of frame")),
    ("negation", "Negation ignored", ("a cat with no hat", "portrait without watermark or text")),
    ("hands", "Hand anatomy", ("close-up of hands holding a coffee cup", "counting on five fingers")),
    ("face_far", "Distant face smear", ("crowd scene with recognizable faces", "two people talking across a room")),
    ("glyph", "Text rendering", ('poster that says "OPEN"', "street sign reading MAIN ST")),
    ("multi_bleed", "Character bleed", ("2girls different outfits, not twins", "boy in red, girl in blue, no swap")),
    ("occlusion", "Occlusion", ("person behind chain-link fence", "cat partially behind a vase")),
    ("reflection", "Reflection", ("person reflected in wet street", "mirror selfie with correct chirality")),
    ("long_prompt", "Long prompt tail", ("masterpiece, " + ", ".join(f"detail{i}" for i in range(40)),)),
    ("plastic", "Plastic skin", ("natural skin texture portrait, pores visible",)),
    ("extra_limb", "Extra limbs", ("yoga pose, anatomically correct", "dancer mid-leap")),
    ("object_fusion", "Merged objects", ("two distinct wine glasses side by side",)),
    ("style_mud", "Style mud", ("in the style of ukiyo-e, sharp linework",)),
    ("lighting_physics", "Lighting", ("single hard side light, long cast shadow",)),
    ("material_metal", "Materials", ("brushed aluminum cylinder, sharp reflections",)),
    ("transparency", "Glass/transparency", ("clear glass cup with water and straw",)),
    ("text_small", "Small text", ("business card with tiny crisp letters",)),
    ("rare_count", "Rare counts", ("seven candles on a cake", "eight matching chairs")),
    ("negation_color", "Negation color", ("a car that is not red", "flowers without yellow")),
    ("relation_stack", "Stacked relations", ("cup on book on table under lamp",)),
    ("action_multi", "Multi-action", ("A waving while B sits reading",)),
    ("crop_fullbody", "Full body crop", ("full body standing person, head to toes in frame",)),
    ("background_amnesia", "BG consistency", ("detailed kitchen background, sharp cabinets",)),
    ("cfg_burn", "CFG burn", ("soft natural colors, no oversat neon skin",)),
)


def save_failure_atlas_jsonl(path: str | Path) -> Path:
    """One ``{"mode", "name", "prompt"}`` row per atlas prompt (an eval pack for failure-mode regressions)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for mode, name, prompts in FAILURE_ATLAS:
            for prompt in prompts:
                f.write(json.dumps({"mode": mode, "name": name, "prompt": prompt}) + "\n")
    return p
