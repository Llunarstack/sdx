"""Anatomy inventions 22–34 (inference planners + training hooks)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "HandExpertPlan",
    "plan_hand_expert",
    "FingerCriticPlan",
    "plan_finger_critic",
    "FaceIdExpressionPlan",
    "plan_face_id_expression",
    "eye_symmetry_breaker_addon",
    "teeth_mouth_addon",
    "contact_shadow_addon",
    "anatomy_token_normalize",
    "two_stage_body_detail_plan",
    "pose_skeleton_soft_plan",
    "limb_monotonicity_loss",
]


@dataclass
class HandExpertPlan:
    active: bool = False
    lora_hint: str = "hands_expert"
    positive: str = ""
    negative: str = ""
    fix_regions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_hand_expert(prompt: str) -> HandExpertPlan:
    p = str(prompt or "").lower()
    active = bool(re.search(r"\b(hand|hands|fingers|holding|grasp|typing|peace sign)\b", p))
    if not active:
        return HandExpertPlan()
    return HandExpertPlan(
        active=True,
        positive="detailed correct hands, five fingers, natural knuckles",
        negative="extra fingers, fused fingers, melted hands, poorly drawn hands",
        fix_regions=["hands"],
    )


@dataclass
class FingerCriticPlan:
    """Plan for detector→inpaint loop (detector optional at runtime)."""

    max_iters: int = 3
    target_fingers: int = 5
    region: str = "hands"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_finger_critic() -> FingerCriticPlan:
    return FingerCriticPlan(notes=["run hand detector; if fingers!=5 inpaint hands"])


@dataclass
class FaceIdExpressionPlan:
    identity_ref: str = ""
    expression: str = ""
    positive: str = ""
    argv_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_face_id_expression(identity_ref: str = "", expression: str = "neutral") -> FaceIdExpressionPlan:
    pos = f"locked facial identity, expression: {expression}, identity preserved"
    hints = []
    if identity_ref:
        hints.extend(["--reference-image", identity_ref, "--reference-style-mode", "instantstyle"])
    return FaceIdExpressionPlan(identity_ref=identity_ref, expression=expression, positive=pos, argv_hints=hints)


def eye_symmetry_breaker_addon() -> tuple[str, str]:
    return (
        "subtle natural eye asymmetry, lived-in gaze, uneven catchlights",
        "perfectly symmetric doll eyes, identical mirrored pupils",
    )


def teeth_mouth_addon() -> tuple[str, str]:
    return (
        "natural teeth, correct gum line, coherent mouth interior",
        "broken teeth mesh, shark mouth, melted lips, extra teeth",
    )


def contact_shadow_addon() -> tuple[str, str]:
    return (
        "feet planted with soft contact shadows on ground",
        "floating feet, missing contact shadow, hovering subject",
    )


_ANATOMY_CANON = {
    "bad_hands": "bad hands",
    "extra_digit": "extra fingers",
    "mutated_hands": "mutated hands",
    "poorly_drawn_hands": "poorly drawn hands",
    "extra_limbs": "extra limbs",
    "missing_arms": "missing arms",
    "missing_legs": "missing legs",
    "disconnected_limbs": "disconnected limbs",
}


def anatomy_token_normalize(tags: str | list[str]) -> list[str]:
    raw = tags if isinstance(tags, list) else re.split(r"[,\s]+", str(tags))
    out = []
    for t in raw:
        k = t.strip().lower().replace(" ", "_")
        if not k:
            continue
        out.append(_ANATOMY_CANON.get(k, t.strip()))
    return list(dict.fromkeys(out))


def two_stage_body_detail_plan(*, steps_body: int = 12, steps_detail: int = 16) -> dict[str, Any]:
    return {
        "stage1": {"focus": "body_pose_silhouette", "steps": steps_body, "cfg_mult": 1.1},
        "stage2": {
            "focus": "face_hands_texture",
            "steps": steps_detail,
            "init_strength": 0.35,
            "fix_regions": ["face", "hands"],
        },
    }


def pose_skeleton_soft_plan(prompt: str) -> dict[str, Any]:
    human = bool(re.search(r"\b(1girl|1boy|person|pose|standing|sitting|dance)\b", prompt or "", re.I))
    return {
        "enable_openpose_soft": human,
        "control_weight": 0.35 if human else 0.0,
        "notes": "weak pose hint when humanish prompt",
    }


def limb_monotonicity_loss(pred_lengths: list[float], parent_lengths: list[float]) -> float:
    """
    Training hook (#25): forearm should not exceed upper-arm etc.
    pred_lengths[i] child, parent_lengths[i] parent; hinge penalty.
    """
    loss = 0.0
    for c, p in zip(pred_lengths, parent_lengths):
        # child should be <= parent * 1.05
        overflow = max(0.0, float(c) - 1.05 * float(p))
        loss += overflow * overflow
    return float(loss)
