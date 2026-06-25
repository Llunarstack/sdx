"""Weather Inertia — climate state persists; storms don't vanish between cuts without reason."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["WeatherState", "WeatherIssue", "WeatherInertiaReport", "parse_weather_config", "track_weather_inertia"]

_WEATHER_LEVELS = ("clear", "overcast", "rain", "storm", "snow", "fog", "sandstorm")
_ORDER = {w: i for i, w in enumerate(_WEATHER_LEVELS)}


@dataclass(slots=True)
class WeatherState:
    condition: str
    intensity: float  # 0..1


@dataclass(slots=True)
class WeatherIssue:
    level: str
    code: str
    message: str
    shot_id: str


@dataclass(slots=True)
class WeatherInertiaReport:
    timeline: List[Dict[str, Any]]
    issues: List[WeatherIssue]
    prompt_injections: Dict[str, str]


def parse_weather_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "max_jump": int(raw.get("max_jump") or 2),
            "initial": str(raw.get("initial") or raw.get("start") or ""),
        }
    return {"enabled": bool(raw)}


def _infer_weather(prompt: str) -> WeatherState:
    p = (prompt or "").lower()
    if any(x in p for x in ("blizzard", "snow", "snowing")):
        return WeatherState("snow", 0.8)
    if any(x in p for x in ("thunder", "storm", "lightning")):
        return WeatherState("storm", 0.9)
    if any(x in p for x in ("rain", "downpour", "drizzle")):
        return WeatherState("rain", 0.7)
    if any(x in p for x in ("fog", "mist", "haze")):
        return WeatherState("fog", 0.6)
    if any(x in p for x in ("sandstorm", "dust storm")):
        return WeatherState("sandstorm", 0.85)
    if any(x in p for x in ("overcast", "cloudy")):
        return WeatherState("overcast", 0.4)
    if any(x in p for x in ("clear sky", "sunny", "bright day")):
        return WeatherState("clear", 0.2)
    return WeatherState("", 0.0)


def _shot_weather(shot: Any) -> WeatherState:
    w = getattr(shot, "weather_spec", None) or getattr(shot, "weather", None)
    if isinstance(w, Mapping):
        return WeatherState(str(w.get("condition") or w.get("type") or ""), float(w.get("intensity") or 0.5))
    if isinstance(w, str) and w:
        return WeatherState(w.lower(), 0.6)
    return _infer_weather(str(getattr(shot, "prompt", "")))


_WEATHER_FRAG: Dict[str, str] = {
    "rain": "continued rain, wet atmosphere",
    "storm": "ongoing storm, wind and rain persistence",
    "snow": "continuing snowfall, cold air",
    "fog": "persistent fog layer",
    "sandstorm": "sand still in air, reduced visibility",
    "overcast": "overcast sky continuity",
}


def track_weather_inertia(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> WeatherInertiaReport:
    if not config.get("enabled"):
        return WeatherInertiaReport(timeline=[], issues=[], prompt_injections={})
    max_jump = int(config.get("max_jump") or 2)
    init = str(config.get("initial") or "")
    prev = WeatherState(init, 0.5) if init else WeatherState("", 0.0)
    timeline: List[Dict[str, Any]] = []
    issues: List[WeatherIssue] = []
    injections: Dict[str, str] = {}

    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        cur = _shot_weather(sh)
        if not cur.condition and prev.condition:
            injections[sid] = _WEATHER_FRAG.get(prev.condition, f"continued {prev.condition}")
            cur = prev
        elif cur.condition and prev.condition:
            jump = abs(_ORDER.get(cur.condition, 0) - _ORDER.get(prev.condition, 0))
            if jump > max_jump and "interior" not in str(getattr(sh, "prompt", "")).lower():
                issues.append(
                    WeatherIssue(
                        level="warn",
                        code="weather_discontinuity",
                        message=f"Weather {prev.condition}→{cur.condition} without transition",
                        shot_id=sid,
                    )
                )
        if cur.condition:
            prev = cur
        timeline.append({"shot_id": sid, "condition": cur.condition, "intensity": cur.intensity})
    return WeatherInertiaReport(timeline=timeline, issues=issues, prompt_injections=injections)
