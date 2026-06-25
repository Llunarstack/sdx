"""Chromatic Story Arc — color palette evolves with emotion independent of grade LUT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["ChromaticBeat", "parse_chromatic_arc", "sample_chromatic_for_shots"]

_PALETTES: Dict[str, str] = {
    "hope": "warm gold and soft green accents",
    "dread": "sickly yellow-green and deep violet shadows",
    "rage": "saturated crimson highlights, charcoal blacks",
    "grief": "desaturated blue-grey, muted skin tones",
    "wonder": "prismatic highlights, cyan-magenta split",
    "neutral": "balanced naturalistic palette",
    "nostalgia": "faded amber, lifted blacks, gentle sepia",
    "betrayal": "cold steel blue against warm false comfort",
}


@dataclass(slots=True)
class ChromaticBeat:
    shot_id: str
    palette_key: str
    prompt_suffix: str
    post_grade: str


def parse_chromatic_arc(raw: Any) -> List[tuple[float, str]]:
    """Return list of (normalized_time, palette_key)."""
    pts: List[tuple[float, str]] = []
    if isinstance(raw, list):
        for i, v in enumerate(raw):
            if isinstance(v, str):
                pts.append((i / max(1, len(raw) - 1), v))
            elif isinstance(v, Mapping):
                pts.append((float(v.get("at") or 0), str(v.get("palette") or v.get("color") or "neutral")))
    elif isinstance(raw, Mapping):
        for k, v in raw.items():
            try:
                pts.append((float(k), str(v)))
            except ValueError:
                pts.append((0.0, str(v)))
    return sorted(pts, key=lambda x: x[0])


def _sample_palette(pts: List[tuple[float, str]], u: float) -> str:
    if not pts:
        return "neutral"
    u = max(0.0, min(1.0, u))
    if u <= pts[0][0]:
        return pts[0][1]
    if u >= pts[-1][0]:
        return pts[-1][1]
    for a, b in zip(pts, pts[1:]):
        if a[0] <= u <= b[0]:
            return a[1] if (u - a[0]) < (b[0] - u) else b[1]
    return pts[-1][1]


def sample_chromatic_for_shots(
    arc: List[tuple[float, str]],
    shots: Sequence[Any],
    *,
    total_duration: float,
) -> List[ChromaticBeat]:
    if not arc:
        return []
    dur_sum = sum(float(getattr(s, "duration_sec", 0) or 0) for s in shots)
    total = dur_sum if dur_sum > 0 else total_duration
    elapsed = 0.0
    out: List[ChromaticBeat] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        d = float(getattr(sh, "duration_sec", 0) or total / max(1, len(shots)))
        mid = (elapsed + d * 0.5) / max(1e-6, total)
        key = _sample_palette(arc, mid)
        frag = _PALETTES.get(key, f"{key} color palette")
        out.append(
            ChromaticBeat(
                shot_id=sid,
                palette_key=key,
                prompt_suffix=f"color script: {frag}",
                post_grade=key if key in ("grief", "nostalgia", "dread") else "",
            )
        )
        elapsed += d
    return out
