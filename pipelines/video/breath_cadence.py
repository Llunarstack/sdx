"""Breath Cadence — micro motion amplitude tied to tension (characters breathe with the scene)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "BreathCadence",
    "plan_breath_cadence",
    "parse_breath_config",
]


@dataclass(slots=True)
class BreathCadence:
    shot_id: str
    bpm: float
    amplitude: float
    prompt_suffix: str


def parse_breath_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "base_bpm": float(raw.get("base_bpm") or 14.0),
        }
    return {"enabled": bool(raw)}


def plan_breath_cadence(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
    tension_by_shot: Optional[Mapping[str, float]] = None,
) -> List[BreathCadence]:
    if not config.get("enabled"):
        return []
    base_bpm = float(config.get("base_bpm") or 14.0)
    tension_by_shot = tension_by_shot or {}
    out: List[BreathCadence] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        t = float(tension_by_shot.get(sid, 0.4))
        bpm = base_bpm + t * 18.0
        amp = 0.15 + t * 0.55
        if t < 0.35:
            frag = "subtle idle breathing, calm micro motion in chest and shoulders"
        elif t < 0.7:
            frag = "visible breathing rhythm, anxious micro sway"
        else:
            frag = "heavy rapid breathing, pronounced chest heave, trembling hands"
        out.append(BreathCadence(shot_id=sid, bpm=round(bpm, 1), amplitude=round(amp, 3), prompt_suffix=frag))
    return out
