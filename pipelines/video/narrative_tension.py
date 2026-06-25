"""Narrative Tension Thermostat — emotion curve drives camera, motion, and grade per shot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "TensionPoint",
    "TensionCurve",
    "ShotTensionProfile",
    "parse_tension_curve",
    "sample_tension_for_shots",
    "tension_shot_overrides",
]


@dataclass(slots=True)
class TensionPoint:
    at: float  # 0..1 normalized timeline
    value: float  # 0..1 tension


@dataclass(slots=True)
class TensionCurve:
    points: List[TensionPoint] = field(default_factory=list)
    genre_bias: str = ""

    def sample(self, u: float) -> float:
        if not self.points:
            return 0.5
        pts = sorted(self.points, key=lambda p: p.at)
        u = max(0.0, min(1.0, u))
        if u <= pts[0].at:
            return pts[0].value
        if u >= pts[-1].at:
            return pts[-1].value
        for a, b in zip(pts, pts[1:]):
            if a.at <= u <= b.at:
                span = max(1e-6, b.at - a.at)
                t = (u - a.at) / span
                return a.value + t * (b.value - a.value)
        return pts[-1].value


@dataclass(slots=True)
class ShotTensionProfile:
    shot_id: str
    shot_index: int
    tension: float
    prompt_suffix: str
    negative_suffix: str
    edit_overrides: Dict[str, Any] = field(default_factory=dict)


_GENRE_CURVES: Dict[str, List[tuple[float, float]]] = {
    "horror": [(0.0, 0.25), (0.5, 0.55), (0.85, 0.95), (1.0, 0.4)],
    "action": [(0.0, 0.5), (0.3, 0.85), (0.7, 0.95), (1.0, 0.35)],
    "romance": [(0.0, 0.2), (0.5, 0.45), (0.8, 0.7), (1.0, 0.5)],
    "comedy": [(0.0, 0.15), (0.6, 0.35), (0.9, 0.8), (1.0, 0.2)],
    "documentary": [(0.0, 0.3), (0.5, 0.4), (1.0, 0.35)],
}


def parse_tension_curve(raw: Any, *, genre: str = "") -> Optional[TensionCurve]:
    if raw is None:
        genre_key = (genre or "").lower()
        if genre_key in _GENRE_CURVES:
            return TensionCurve(
                points=[TensionPoint(at=a, value=v) for a, v in _GENRE_CURVES[genre_key]],
                genre_bias=genre_key,
            )
        return None
    if isinstance(raw, list):
        pts: List[TensionPoint] = []
        for i, v in enumerate(raw):
            if isinstance(v, (int, float)):
                at = i / max(1, len(raw) - 1)
                pts.append(TensionPoint(at=at, value=float(max(0.0, min(1.0, v)))))
            elif isinstance(v, Mapping):
                pts.append(
                    TensionPoint(
                        at=float(v.get("at") or v.get("t") or 0.0),
                        value=float(max(0.0, min(1.0, float(v.get("value") or v.get("tension") or 0.5)))),
                    )
                )
        return TensionCurve(points=pts, genre_bias=genre) if pts else None
    if isinstance(raw, Mapping):
        if raw.get("auto_genre") and genre:
            return parse_tension_curve(None, genre=genre)
        pts2 = (
            [
                TensionPoint(at=float(a), value=float(b))
                for a, b in (raw.get("points") or raw.get("curve") or {}).items()
            ]
            if isinstance(raw.get("points") or raw.get("curve"), Mapping)
            else []
        )
        if not pts2 and isinstance(raw.get("values"), list):
            return parse_tension_curve(raw["values"], genre=genre)
        return TensionCurve(points=pts2, genre_bias=str(raw.get("genre") or genre)) if pts2 else None
    return None


def sample_tension_for_shots(
    curve: TensionCurve,
    shots: Sequence[Any],
    *,
    total_duration: float,
) -> List[ShotTensionProfile]:
    if not shots:
        return []
    dur_sum = sum(float(getattr(s, "duration_sec", 0) or 0) for s in shots)
    total = dur_sum if dur_sum > 0 else total_duration
    elapsed = 0.0
    out: List[ShotTensionProfile] = []
    for i, sh in enumerate(shots):
        d = float(getattr(sh, "duration_sec", 0) or (total / max(1, len(shots))))
        mid = (elapsed + d * 0.5) / max(1e-6, total)
        t = curve.sample(mid)
        elapsed += d
        prompt, neg, edits = tension_shot_overrides(t)
        out.append(
            ShotTensionProfile(
                shot_id=str(getattr(sh, "id", f"shot_{i}")),
                shot_index=i,
                tension=round(t, 3),
                prompt_suffix=prompt,
                negative_suffix=neg,
                edit_overrides=edits,
            )
        )
    return out


def tension_shot_overrides(t: float) -> tuple[str, str, Dict[str, Any]]:
    """High tension → handheld, fast cuts feel, contrast; low → still, soft."""
    t = max(0.0, min(1.0, t))
    if t >= 0.75:
        return (
            "high stakes intensity, urgent energy, sharp contrast",
            "flat calm, sleepy pacing",
            {
                "velocity_ease": False,
                "camera_stabilize": False,
                "keyframe_interval": max(3, int(6 - t * 3)),
                "motion_beat_keyframes": True,
            },
        )
    if t >= 0.45:
        return (
            "building suspense, controlled momentum",
            "chaotic shaky blur",
            {"velocity_ease": True, "velocity_ease_mode": "ease_in"},
        )
    return (
        "quiet atmosphere, gentle stillness, observational pacing",
        "hyperactive shake, strobe chaos",
        {"camera_stabilize": True, "camera_stabilize_strength": 0.8, "keyframe_interval": 8},
    )
