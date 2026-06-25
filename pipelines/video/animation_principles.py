"""Disney 12 principles + style presets as tunable sliders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

__all__ = [
    "AnimationPrinciples",
    "principles_from_dict",
    "principles_prompt",
    "preset_principles",
]


@dataclass(slots=True)
class AnimationPrinciples:
    squash_stretch: float = 0.5
    anticipation: float = 0.5
    staging: float = 0.5
    straight_ahead_pose: float = 0.5
    follow_through: float = 0.5
    slow_in_slow_out: float = 0.5
    arc: float = 0.5
    secondary_action: float = 0.5
    timing: float = 0.5
    exaggeration: float = 0.5
    appeal: float = 0.5
    solid_drawing: float = 0.5


_PRESETS: Dict[str, AnimationPrinciples] = {
    "pixar": AnimationPrinciples(
        squash_stretch=0.75, anticipation=0.7, follow_through=0.85, appeal=0.9, exaggeration=0.6
    ),
    "ghibli": AnimationPrinciples(
        squash_stretch=0.35, anticipation=0.5, follow_through=0.6, timing=0.4, appeal=0.85, exaggeration=0.3
    ),
    "anime_tv": AnimationPrinciples(squash_stretch=0.45, anticipation=0.55, exaggeration=0.7, timing=0.65, appeal=0.75),
    "looney": AnimationPrinciples(squash_stretch=0.95, anticipation=0.9, exaggeration=0.95, timing=0.85, appeal=0.8),
    "spider_verse": AnimationPrinciples(
        squash_stretch=0.6, exaggeration=0.8, timing=0.75, secondary_action=0.7, appeal=0.85
    ),
    "stop_motion": AnimationPrinciples(
        squash_stretch=0.25, timing=0.55, follow_through=0.35, exaggeration=0.2, solid_drawing=0.7
    ),
    "realistic": AnimationPrinciples(squash_stretch=0.15, exaggeration=0.1, appeal=0.5, follow_through=0.4),
}


def preset_principles(name: str) -> AnimationPrinciples:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _PRESETS.get(key, AnimationPrinciples())


def principles_from_dict(raw: Mapping[str, Any] | None, *, preset: str = "") -> AnimationPrinciples:
    base = preset_principles(preset) if preset else AnimationPrinciples()
    if not raw:
        return base
    fields = {f.name for f in AnimationPrinciples.__dataclass_fields__.values()}
    vals = {k: float(raw[k]) for k in raw if k in fields}
    return AnimationPrinciples(**{**base.__dict__, **vals})


def principles_prompt(p: AnimationPrinciples) -> str:
    parts: list[str] = []
    if p.squash_stretch > 0.6:
        parts.append("strong squash and stretch")
    elif p.squash_stretch < 0.3:
        parts.append("rigid form, minimal squash")
    if p.anticipation > 0.6:
        parts.append("clear anticipation poses")
    if p.follow_through > 0.65:
        parts.append("follow through and overlap on hair, cloth, limbs")
    if p.exaggeration > 0.7:
        parts.append("exaggerated poses and silhouettes")
    elif p.exaggeration < 0.25:
        parts.append("subtle restrained motion")
    if p.timing > 0.65:
        parts.append("snappy timing")
    elif p.timing < 0.35:
        parts.append("slow deliberate timing")
    if p.appeal > 0.75:
        parts.append("appealing readable silhouettes")
    if p.secondary_action > 0.6:
        parts.append("secondary motion on props and hair")
    return ", ".join(parts) if parts else "balanced animation timing"
