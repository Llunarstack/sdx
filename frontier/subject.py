"""
Subject-aware frontier synthesis — bodies, creatures, mature content, mediums, realism.

Compose into one plan for ``sample.py`` or prompt stack stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from config.defaults.art_mediums import merge_csv_unique

from .anatomy import BodyPlan, BodyPlanner
from .creatures import CreatureTaxonomy
from .mature import MatureGuidance, MaturePlan
from .medium import BrushPlanner, extended_guidance_fragments
from .realism import PhotorealPlan, PhotorealStack


@dataclass
class SubjectFrontierPlan:
    prompt: str
    body: BodyPlan | None = None
    creature_positive: str = ""
    creature_negative: str = ""
    mature: MaturePlan | None = None
    brush_positive: str = ""
    brush_negative: str = ""
    medium_positive: str = ""
    medium_negative: str = ""
    photoreal: PhotorealPlan | None = None
    augmented_prompt: str = ""
    merged_positive: str = ""
    merged_negative: str = ""


def analyze_subject(
    prompt: str,
    *,
    include_photography: bool = True,
    medium_mode: str = "auto",
) -> SubjectFrontierPlan:
    """Run body, creature, mature, medium, brush, and photoreal analyzers."""
    body_planner = BodyPlanner()
    body = body_planner.plan(prompt)

    creature_pos, creature_neg = CreatureTaxonomy().prompt_fragments(prompt)

    mature = MatureGuidance().plan(prompt)

    med_pos, med_neg = extended_guidance_fragments(
        prompt,
        include_photography=include_photography,
        anatomy_mode="none",  # body planner owns anatomy
        base_mode=medium_mode,
    )
    brush_pos, brush_neg = BrushPlanner().fragments(prompt)
    photo = PhotorealStack().plan(prompt)

    pos_parts: List[str] = []
    neg_parts: List[str] = []

    if body and body.positive:
        pos_parts.append(body.positive)
    if body and body.negative:
        neg_parts.append(body.negative)
    if creature_pos:
        pos_parts.append(creature_pos)
    if creature_neg:
        neg_parts.append(creature_neg)
    if mature and mature.positive:
        pos_parts.append(mature.positive)
    if mature and mature.negative:
        neg_parts.append(mature.negative)
    if med_pos:
        pos_parts.append(med_pos)
    if med_neg:
        neg_parts.append(med_neg)
    if brush_pos:
        pos_parts.append(brush_pos)
    if brush_neg:
        neg_parts.append(brush_neg)
    if photo.positive:
        pos_parts.append(photo.positive)
    if photo.negative:
        neg_parts.append(photo.negative)

    merged_pos = merge_csv_unique(*pos_parts)
    merged_neg = merge_csv_unique(*neg_parts)

    augmented = prompt
    if merged_pos and prompt:
        augmented = f"{prompt}, {merged_pos}" if len(merged_pos) < 400 else prompt

    return SubjectFrontierPlan(
        prompt=prompt,
        body=body,
        creature_positive=creature_pos,
        creature_negative=creature_neg,
        mature=mature if mature and mature.content_class.value != "none" else None,
        brush_positive=brush_pos,
        brush_negative=brush_neg,
        medium_positive=med_pos,
        medium_negative=med_neg,
        photoreal=photo if photo.tier.value != "none" else None,
        augmented_prompt=augmented,
        merged_positive=merged_pos,
        merged_negative=merged_neg,
    )


def subject_sample_kwargs(plan: SubjectFrontierPlan, *, base_negative: str = "") -> Dict[str, Any]:
    neg = merge_csv_unique(base_negative, plan.merged_negative)
    cfg_mult = 1.0
    if plan.body:
        cfg_mult = plan.body.cfg_bias
    if plan.body and plan.body.risk.hand_focus:
        cfg_mult = max(cfg_mult, 1.1)

    return {
        "prompt": plan.augmented_prompt or plan.prompt,
        "negative_prompt": neg,
        "cfg_scale_multiplier": cfg_mult,
        "subject_body_mode": plan.body.mode.value if plan.body else "abstract",
        "subject_anatomy_risk": plan.body.risk.score if plan.body else 0.0,
        "subject_mature_class": plan.mature.content_class.value if plan.mature else "none",
        "subject_realism_tier": plan.photoreal.tier.value if plan.photoreal else "none",
        "subject_brush_style": None,
        "subject_merged_positive": plan.merged_positive,
        "subject_merged_negative": plan.merged_negative,
    }


__all__ = ["SubjectFrontierPlan", "analyze_subject", "subject_sample_kwargs"]
