"""
Perfect frontier — compose deep + subject + scene quality + safety into one plan.

This is the "turn everything on" orchestrator for maximum generation quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from config.defaults.art_mediums import merge_csv_unique

from .atmosphere import AtmospherePlanner
from .composition import CompositionPlanner
from .era import EraPlanner
from .harmony import PalettePlanner
from .lighting import LightingPlanner
from .materials import MaterialPlanner
from .motion import MotionPlanner
from .optics import LensCharacterPlanner
from .safety import ContentPolicy, PolicyDecision, PolicyTier, SafetyReport
from .subject import SubjectFrontierPlan, analyze_subject, subject_sample_kwargs
from .synthesis import DeepFrontierPlan, analyze_deep, deep_sample_kwargs
from .typography import TypographyPlanner


@dataclass
class PerfectFrontierPlan:
    prompt: str
    deep: DeepFrontierPlan
    subject: SubjectFrontierPlan
    safety: SafetyReport
    composition_pos: str = ""
    composition_neg: str = ""
    lighting_pos: str = ""
    lighting_neg: str = ""
    atmosphere_pos: str = ""
    atmosphere_neg: str = ""
    materials_pos: str = ""
    materials_neg: str = ""
    harmony_pos: str = ""
    harmony_neg: str = ""
    motion_pos: str = ""
    motion_neg: str = ""
    era_pos: str = ""
    era_neg: str = ""
    typography_pos: str = ""
    typography_neg: str = ""
    optics_pos: str = ""
    optics_neg: str = ""
    final_prompt: str = ""
    final_negative: str = ""
    refused: bool = False
    refuse_reasons: List[str] = field(default_factory=list)


def analyze_perfect(
    prompt: str,
    *,
    num_steps: int = 28,
    serendipity_dial: float = 0.25,
    layout_regions: int = 0,
    auto_resolve_contradictions: bool = False,
    medium_mode: str = "auto",
    safety_tier: PolicyTier = PolicyTier.MODERATE,
) -> PerfectFrontierPlan:
    policy = ContentPolicy(safety_tier)
    safety = policy.evaluate(prompt)
    if safety.decision == PolicyDecision.REFUSE:
        return PerfectFrontierPlan(
            prompt=prompt,
            deep=analyze_deep(prompt, num_steps=num_steps, serendipity_dial=0.0, layout_regions=layout_regions),
            subject=analyze_subject(prompt, medium_mode=medium_mode),
            safety=safety,
            refused=True,
            refuse_reasons=list(safety.reasons),
        )

    working = prompt
    steer_neg = ""
    if safety.decision == PolicyDecision.STEER:
        working, steer_neg = policy.apply_steering(prompt, safety)

    deep = analyze_deep(
        working,
        num_steps=num_steps,
        serendipity_dial=serendipity_dial,
        layout_regions=layout_regions,
        auto_resolve_contradictions=auto_resolve_contradictions,
    )
    subject = analyze_subject(working, medium_mode=medium_mode)

    comp_p, comp_n = CompositionPlanner().fragments(working)
    light_p, light_n = LightingPlanner().fragments(working)
    atm_p, atm_n = AtmospherePlanner().fragments(working)
    mat_p, mat_n = MaterialPlanner().fragments(working)
    har_p, har_n = PalettePlanner().fragments(working)
    mot_p, mot_n = MotionPlanner().fragments(working)
    era_p, era_n = EraPlanner().fragments(working)
    typ_p, typ_n = TypographyPlanner().fragments(working)
    opt_p, opt_n = LensCharacterPlanner().fragments(working)

    scene_pos = merge_csv_unique(comp_p, light_p, atm_p, mat_p, har_p, mot_p, era_p, typ_p, opt_p)
    scene_neg = merge_csv_unique(comp_n, light_n, atm_n, mat_n, har_n, mot_n, era_n, typ_n, opt_n)

    base_prompt = deep.deep_augmented_prompt or working
    final_prompt = merge_csv_unique(base_prompt, subject.merged_positive, scene_pos)
    if len(final_prompt) > 900:
        final_prompt = merge_csv_unique(base_prompt, scene_pos[:400])

    final_neg = merge_csv_unique(
        subject.merged_negative,
        scene_neg,
        steer_neg,
        safety.steer_negative,
    )

    return PerfectFrontierPlan(
        prompt=prompt,
        deep=deep,
        subject=subject,
        safety=safety,
        composition_pos=comp_p,
        composition_neg=comp_n,
        lighting_pos=light_p,
        lighting_neg=light_n,
        atmosphere_pos=atm_p,
        atmosphere_neg=atm_n,
        materials_pos=mat_p,
        materials_neg=mat_n,
        harmony_pos=har_p,
        harmony_neg=har_n,
        motion_pos=mot_p,
        motion_neg=mot_n,
        era_pos=era_p,
        era_neg=era_n,
        typography_pos=typ_p,
        typography_neg=typ_n,
        optics_pos=opt_p,
        optics_neg=opt_n,
        final_prompt=final_prompt or working,
        final_negative=final_neg,
    )


def perfect_sample_kwargs(plan: PerfectFrontierPlan, *, base_negative: str = "") -> Dict[str, Any]:
    if plan.refused:
        return {
            "refused": True,
            "refuse_reasons": plan.refuse_reasons,
            "prompt": plan.prompt,
        }

    deep_kw = deep_sample_kwargs(plan.deep, base_negative=base_negative)
    sub_kw = subject_sample_kwargs(plan.subject, base_negative=plan.final_negative or base_negative)

    neg = merge_csv_unique(base_negative, plan.final_negative, deep_kw.get("negative_prompt", ""))

    cfg = float(deep_kw.get("cfg_scale_multiplier") or 1.0)
    cfg *= float(sub_kw.get("cfg_scale_multiplier") or 1.0)
    typ = TypographyPlanner().plan(plan.prompt)
    if typ.cfg_boost > 1.0:
        cfg *= typ.cfg_boost

    out: Dict[str, Any] = {
        **deep_kw,
        **{k: v for k, v in sub_kw.items() if k not in ("prompt", "negative_prompt")},
        "prompt": plan.final_prompt,
        "negative_prompt": neg,
        "cfg_scale_multiplier": cfg,
        "perfect_frontier": True,
        "safety_decision": plan.safety.decision.value,
        "safety_tier": plan.safety.tier.value,
    }
    return out


__all__ = ["PerfectFrontierPlan", "analyze_perfect", "perfect_sample_kwargs"]
