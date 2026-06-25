"""Mise-en-scène Grammar — lead room, headroom, and balance rules as compositional prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "CompositionSpec",
    "parse_mise_config",
    "compose_shot_framing",
]

_SHOT_GRAMMAR: Dict[str, Dict[str, str]] = {
    "close_up": {"headroom": "tight", "lead_room": "minimal", "balance": "face dominant center-third"},
    "close-up": {"headroom": "tight", "lead_room": "minimal", "balance": "face dominant center-third"},
    "medium": {"headroom": "standard", "lead_room": "moderate", "balance": "subject on vertical third"},
    "wide": {"headroom": "generous sky", "lead_room": "wide negative space", "balance": "subject small in frame"},
    "establishing": {
        "headroom": "architectural scale",
        "lead_room": "deep staging",
        "balance": "environment tells story",
    },
    "over_shoulder": {
        "headroom": "standard",
        "lead_room": "space toward listener",
        "balance": "foreground shoulder frames subject",
    },
    "pov": {"headroom": "immersive", "lead_room": "center-weighted", "balance": "hands or weapon lower third optional"},
    "low_angle": {"headroom": "compressed", "lead_room": "upward power", "balance": "subject towers from below"},
    "high_angle": {"headroom": "vulnerable", "lead_room": "downward diminish", "balance": "subject isolated below"},
}


@dataclass(slots=True)
class CompositionSpec:
    shot_id: str
    grammar_key: str
    prompt_suffix: str
    negative_suffix: str


def parse_mise_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": True}
    if isinstance(raw, bool):
        return {"enabled": raw}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "style": str(raw.get("style") or "classical"),
            "force_rule_of_thirds": bool(raw.get("rule_of_thirds", True)),
        }
    return {"enabled": True}


def _grammar_key(shot: Any) -> str:
    st = str(getattr(shot, "shot_type", "") or "").lower().replace("-", "_")
    cam = str(getattr(shot, "camera", "") or "").lower()
    prompt = str(getattr(shot, "prompt", "") or "").lower()
    if st in _SHOT_GRAMMAR:
        return st
    if "ots" in cam or "over the shoulder" in prompt:
        return "over_shoulder"
    if "pov" in cam:
        return "pov"
    if "low angle" in prompt or "low-angle" in prompt:
        return "low_angle"
    if "high angle" in prompt or "bird" in prompt:
        return "high_angle"
    if "establishing" in prompt or "wide" in prompt:
        return "establishing"
    if "close" in prompt:
        return "close_up"
    return "medium"


def compose_shot_framing(shot: Any, *, config: Mapping[str, Any]) -> Optional[CompositionSpec]:
    if not config.get("enabled", True):
        return None
    key = _grammar_key(shot)
    gram = _SHOT_GRAMMAR.get(key, _SHOT_GRAMMAR["medium"])
    style = str(config.get("style") or "classical")
    thirds = ", rule of thirds composition" if config.get("force_rule_of_thirds") else ""
    pos = f"{style} framing, headroom {gram['headroom']}, lead room {gram['lead_room']}, {gram['balance']}{thirds}"
    neg = "centered bullseye framing, clipped head, cramped composition"
    return CompositionSpec(
        shot_id=str(getattr(shot, "id", "")),
        grammar_key=key,
        prompt_suffix=pos,
        negative_suffix=neg,
    )


def compose_all_shots(shots: Sequence[Any], config: Mapping[str, Any]) -> List[CompositionSpec]:
    out: List[CompositionSpec] = []
    for sh in shots:
        spec = compose_shot_framing(sh, config=config)
        if spec:
            out.append(spec)
    return out
