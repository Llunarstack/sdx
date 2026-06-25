"""Anticipation Borrow — steals shot time before action verbs for Disney-style wind-up frames."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "WindupSpec",
    "AnticipationPlan",
    "parse_anticipation_config",
    "plan_anticipation_windups",
]

_ACTION_VERBS = (
    "jumps",
    "jump",
    "punches",
    "punch",
    "throws",
    "throw",
    "kicks",
    "kick",
    "runs",
    "run",
    "leaps",
    "leap",
    "swings",
    "swing",
    "slams",
    "slam",
    "explodes",
    "explode",
    "falls",
    "fall",
    "dives",
    "dive",
    "attacks",
    "attack",
    "strikes",
    "strike",
    "launches",
    "launch",
)


@dataclass(slots=True)
class WindupSpec:
    shot_id: str
    verb: str
    borrow_sec: float
    windup_prompt: str
    keyframe_anchor: str  # pre_action | action_peak


@dataclass(slots=True)
class AnticipationPlan:
    enabled: bool
    windups: List[WindupSpec] = field(default_factory=list)
    total_borrowed_sec: float = 0.0


def parse_anticipation_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, bool):
        return {"enabled": raw}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "borrow_ratio": float(raw.get("borrow_ratio") or raw.get("ratio") or 0.18),
            "max_borrow_sec": float(raw.get("max_borrow_sec") or 0.45),
            "min_shot_sec": float(raw.get("min_shot_sec") or 1.0),
        }
    return {"enabled": False}


def _first_action_verb(prompt: str) -> str:
    p = (prompt or "").lower()
    for v in _ACTION_VERBS:
        if re.search(rf"\b{re.escape(v)}\b", p):
            return v
    return ""


_WINDUP_FRAGMENTS: Dict[str, str] = {
    "jump": "anticipation crouch, coiled legs, wind-up before jump",
    "leap": "coiled stance, gathering energy before leap",
    "punch": "shoulder wind-up, fist pulled back, anticipation before punch",
    "kick": "planted foot, hip coil before kick",
    "throw": "arm drawn back, weight shift before throw",
    "run": "lean forward, preparatory step before sprint",
    "swing": "weapon drawn back, breath before swing",
    "slam": "raised arms, breath in before slam",
    "explode": "tension building, subtle rumble before explosion",
    "fall": "loss of balance, tipping anticipation before fall",
    "dive": "knees bend, arms ready before dive",
    "attack": "combat stance wind-up before attack",
    "strike": "coiled strike preparation",
    "launch": "charged posture before launch",
}


def plan_anticipation_windups(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> AnticipationPlan:
    if not config.get("enabled"):
        return AnticipationPlan(enabled=False)
    ratio = float(config.get("borrow_ratio") or 0.18)
    max_borrow = float(config.get("max_borrow_sec") or 0.45)
    min_shot = float(config.get("min_shot_sec") or 1.0)

    windups: List[WindupSpec] = []
    total = 0.0
    for sh in shots:
        prompt = str(getattr(sh, "prompt", "") or "")
        verb = _first_action_verb(prompt)
        if not verb:
            continue
        dur = float(getattr(sh, "duration_sec", 0) or 0)
        if dur < min_shot:
            continue
        borrow = min(max_borrow, dur * ratio)
        frag = _WINDUP_FRAGMENTS.get(verb, f"anticipation wind-up before {verb}")
        windups.append(
            WindupSpec(
                shot_id=str(getattr(sh, "id", "")),
                verb=verb,
                borrow_sec=round(borrow, 3),
                windup_prompt=frag,
                keyframe_anchor="pre_action",
            )
        )
        total += borrow
    return AnticipationPlan(enabled=True, windups=windups, total_borrowed_sec=round(total, 3))
