"""
Motion intelligence — prompt → physics-aware motion plan (Wan 'motion intelligence').

Closed models still fail cloth/fluids/multi-body. We emit an explicit plan:
primary action, contacts, secondary motion, forbidden artifacts, and which
superiority gates to tighten — so generation isn't one vague prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["MotionPlan", "compile_motion_intelligence", "plan_to_prompt_fragments"]


@dataclass(slots=True)
class MotionPlan:
    primary_action: str = "idle"
    contacts: list[str] = field(default_factory=list)
    secondary: list[str] = field(default_factory=list)
    camera: str = ""
    physics_hints: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    gate_boosts: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


_ACTION_VERBS = (
    ("run", "running"),
    ("walk", "walking"),
    ("jump", "jumping"),
    ("fall", "falling"),
    ("dance", "dancing"),
    ("fight", "fighting"),
    ("drive", "driving"),
    ("fly", "flying"),
    ("swim", "swimming"),
    ("pour", "pouring liquid"),
    ("explode", "explosion"),
    ("turntable", "product turntable orbit"),
)


def compile_motion_intelligence(prompt: str, *, style: str = "") -> MotionPlan:
    text = f"{prompt} {style}".lower()
    plan = MotionPlan()

    for key, action in _ACTION_VERBS:
        if key in text:
            plan.primary_action = action
            break

    # Contacts
    if any(k in text for k in ("foot", "walk", "run", "stand", "ground", "floor")):
        plan.contacts.append("feet_to_ground")
        plan.gate_boosts["min_contact"] = 0.45
        plan.physics_hints.append("weight transfer through feet")
    if any(k in text for k in ("hand", "grab", "hold", "touch", "shake hands")):
        plan.contacts.append("hand_contact")
        plan.gate_boosts["min_extremity"] = 0.4
    if any(k in text for k in ("sit", "chair", "lean")):
        plan.contacts.append("body_to_furniture")

    # Secondary
    if any(k in text for k in ("hair", "cloth", "dress", "coat", "cape")):
        plan.secondary.append("cloth_follow_through")
        plan.physics_hints.append("cloth lag behind body")
        plan.forbid.append("cloth clipping through limbs")
    if any(k in text for k in ("water", "rain", "splash", "pour", "liquid")):
        plan.secondary.append("fluid_secondary")
        plan.physics_hints.append("gravity on droplets")
        plan.forbid.append("water floating upward")
        plan.gate_boosts["min_physics"] = 0.55
    if any(k in text for k in ("bag", "phone", "gun", "sword", "cup")):
        plan.secondary.append("prop_attachment")
        plan.gate_boosts["min_secondary"] = 0.4

    # Multi-body
    if any(k in text for k in ("crowd", "two people", "three ", "group", "together")):
        plan.physics_hints.append("independent body trajectories")
        plan.forbid.append("merged silhouettes")
        plan.gate_boosts["min_count"] = 0.45
        plan.gate_boosts["min_occlusion"] = 0.35

    # Camera from prompt
    for cam in ("orbit", "dolly", "handheld", "crane", "static", "hitchcock"):
        if cam in text:
            plan.camera = cam
            break

    plan.forbid.extend(
        [
            "teleporting limbs",
            "anti-gravity falls",
            "sliding on ice feet",
            "identity morph mid-action",
        ]
    )
    plan.notes.append(f"action={plan.primary_action}")
    return plan


def plan_to_prompt_fragments(plan: MotionPlan) -> tuple[str, str]:
    pos = [
        plan.primary_action,
        *plan.physics_hints,
        *plan.secondary,
        *plan.contacts,
    ]
    if plan.camera:
        pos.append(f"camera:{plan.camera}")
    neg = list(plan.forbid)
    return ", ".join(p for p in pos if p), ", ".join(neg)


def apply_gate_boosts(opts: Any, plan: MotionPlan) -> Any:
    """Raise ProcessOptions min_* thresholds from the motion plan."""
    from dataclasses import replace

    kw = {}
    for k, v in plan.gate_boosts.items():
        cur = float(getattr(opts, k, 0.0) or 0.0)
        kw[k] = max(cur, float(v))
    # Enable related repairs
    if "feet_to_ground" in plan.contacts:
        kw["contact_ground"] = True
    if "prop_attachment" in plan.secondary:
        kw["secondary_track"] = True
    if "fluid_secondary" in plan.secondary or plan.gate_boosts.get("min_physics"):
        kw["physics_gate"] = True
    if not kw:
        return opts
    return replace(opts, **kw)
