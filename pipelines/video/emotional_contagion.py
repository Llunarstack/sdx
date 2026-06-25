"""Emotional Contagion — background/crowd inherits hero emotional state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["ContagionPlan", "parse_contagion_config", "plan_emotional_contagion"]

_EMOTION_CROWD: Dict[str, str] = {
    "fear": "background extras anxious, wide eyes, protective postures, scattered retreat",
    "joy": "background crowd smiling, relaxed shoulders, celebratory micro-gestures",
    "anger": "background onlookers tense, clenched jaws, confrontational stances",
    "grief": "background figures subdued, downcast eyes, slow movement",
    "awe": "background crowd frozen, upward gazes, open mouths",
    "panic": "background chaos, running figures, flailing arms at edges",
    "calm": "background extras neutral, unhurried movement, soft body language",
}


@dataclass(slots=True)
class ContagionPlan:
    shot_id: str
    source_emotion: str
    source_entity: str
    prompt_suffix: str


def parse_contagion_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "source": str(raw.get("source") or raw.get("hero") or ""),
            "radius": str(raw.get("radius") or "crowd"),
        }
    return {"enabled": bool(raw)}


def _shot_emotion(shot: Any) -> str:
    em = str(getattr(shot, "emotion", "") or "").lower()
    if em:
        return em
    p = str(getattr(shot, "prompt", "")).lower()
    for key in _EMOTION_CROWD:
        if key in p:
            return key
    return ""


def plan_emotional_contagion(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> List[ContagionPlan]:
    if not config.get("enabled"):
        return []
    default_source = str(config.get("source") or "")
    plans: List[ContagionPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        em = _shot_emotion(sh)
        if not em:
            continue
        chars = list(getattr(sh, "characters", []) or [])
        source = default_source or (chars[0] if chars else "subject")
        frag = _EMOTION_CROWD.get(em, f"background reflects {em} mood")
        plans.append(ContagionPlan(shot_id=sid, source_emotion=em, source_entity=source, prompt_suffix=frag))
    return plans
