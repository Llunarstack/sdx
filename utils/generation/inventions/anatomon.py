"""ANATOMON — Anatomical monotonicity hints at inference (prompt + region plan).

Novel angle: encode *joint chain constraints* as language + fix-region priors
("forearm shorter than upper arm", "five fingers", etc.) before a full mesh model.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["AnatomonPlan", "plan_anatomon", "apply_anatomon_prompts"]

_HUMANISH = re.compile(
    r"\b(1girl|1boy|girl|boy|woman|man|person|hands?|fingers?|pose|standing|sitting|"
    r"holding|arm|legs?|body|portrait|selfie)\b",
    re.IGNORECASE,
)

_POS = (
    "anatomically correct limbs",
    "five fingers per hand",
    "correct joint articulation",
    "plausible bone lengths",
    "shoulders connected to arms",
    "no extra limbs",
)
_NEG = (
    "extra fingers",
    "fused fingers",
    "broken wrists",
    "inverted elbows",
    "disconnected limbs",
    "melted hands",
    "extra arms",
    "bad anatomy",
)


@dataclass
class AnatomonPlan:
    active: bool = False
    positive_addon: str = ""
    negative_addon: str = ""
    fix_regions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_anatomon(prompt: str) -> AnatomonPlan:
    active = bool(_HUMANISH.search(prompt or ""))
    if not active:
        return AnatomonPlan(notes=["anatomon inactive"])
    return AnatomonPlan(
        active=True,
        positive_addon=", ".join(_POS),
        negative_addon=", ".join(_NEG),
        fix_regions=["hands", "face"],
        notes=["anatomon priors on"],
    )


def apply_anatomon_prompts(positive: str, negative: str = "") -> tuple[str, str, AnatomonPlan]:
    plan = plan_anatomon(positive)
    if not plan.active:
        return positive, negative, plan
    pos = f"{str(positive).rstrip(',')}, {plan.positive_addon}"
    neg = f"{str(negative or '').rstrip(',')}, {plan.negative_addon}".strip(", ")
    return pos, neg, plan
