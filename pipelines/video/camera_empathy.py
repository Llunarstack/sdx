"""Camera Empathy — the camera has emotional attachment; it retreats, lingers, or trembles with subjects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["EmpathyMove", "parse_empathy_config", "plan_camera_empathy"]


@dataclass(slots=True)
class EmpathyMove:
    emotion: str
    camera_prompt: str
    negative: str
    edit_overrides: Dict[str, Any]


_EMPATHY_MAP = {
    "intimacy": EmpathyMove(
        "intimacy",
        "slow tender dolly inward, shallow depth, camera breathes with subject",
        "aggressive whip pans",
        {"camera_stabilize": True, "velocity_ease": True},
    ),
    "fear": EmpathyMove(
        "fear",
        "hesitant handheld retreat, slight delay following subject, negative space growing",
        "confident steady push in",
        {"camera_stabilize": False},
    ),
    "awe": EmpathyMove(
        "awe",
        "slow crane rise, camera stillness then subtle push, vast reveal",
        "chaotic shaky cam",
        {"camera_stabilize": True, "camera_stabilize_strength": 0.9},
    ),
    "rage": EmpathyMove(
        "rage",
        "aggressive handheld push, micro jitters synced to subject",
        "static detached framing",
        {"camera_stabilize": False, "velocity_ease": False},
    ),
    "loneliness": EmpathyMove(
        "loneliness",
        "slow pull back leaving subject small in frame, lingering hold",
        "tight crowded framing",
        {"velocity_ease": True, "velocity_ease_mode": "ease_out"},
    ),
    "suspense": EmpathyMove(
        "suspense",
        "almost imperceptible push in, camera holds breath",
        "fast cutting energy",
        {"keyframe_interval": 5},
    ),
}


def parse_empathy_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {"enabled": bool(raw.get("enabled", True)), "default": str(raw.get("default") or "")}
    return {"enabled": bool(raw)}


def _infer_emotion(shot: Any, tension: float = 0.5) -> str:
    em = str(getattr(shot, "emotion", "") or "").lower()
    if em in _EMPATHY_MAP:
        return em
    p = str(getattr(shot, "prompt", "")).lower()
    for key in _EMPATHY_MAP:
        if key in p:
            return key
    if tension >= 0.75:
        return "suspense"
    if tension <= 0.25:
        return "loneliness"
    return "intimacy"


def plan_camera_empathy(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
    tension_by_shot: Mapping[str, float] | None = None,
) -> List[tuple[str, EmpathyMove]]:
    if not config.get("enabled"):
        return []
    tension_by_shot = tension_by_shot or {}
    out: List[tuple[str, EmpathyMove]] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        em = _infer_emotion(sh, tension_by_shot.get(sid, 0.5))
        move = _EMPATHY_MAP.get(em) or _EMPATHY_MAP.get(str(config.get("default") or "intimacy"))
        if move:
            out.append((sid, move))
    return out
