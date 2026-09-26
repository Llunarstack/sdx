"""Offscreen Space Map — what happens just outside frame affects sound, light, and reactions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OffscreenEvent", "OffscreenPlan", "parse_offscreen_map", "plan_offscreen_space"]

_OFFSCREEN_PRESETS: dict[str, dict[str, str]] = {
    "crowd_left": {
        "audio": "muffled crowd noise from offscreen left",
        "light": "occasional flash from offscreen left",
        "reaction": "characters glance offscreen left",
    },
    "monster_right": {
        "audio": "deep growl from offscreen right",
        "light": "shadow sweep from offscreen right",
        "reaction": "characters recoil looking offscreen right",
    },
    "traffic_behind": {
        "audio": "distant traffic hum behind camera",
        "light": "headlight streaks in background bokeh",
        "reaction": "none",
    },
    "storm_above": {
        "audio": "thunder rumble above frame",
        "light": "lightning flicker from above",
        "reaction": "characters flinch upward",
    },
    "river_below": {
        "audio": "rushing water below frame edge",
        "light": "cool blue bounce from below",
        "reaction": "balance cautious near edge",
    },
}


@dataclass(slots=True)
class OffscreenEvent:
    zone: str
    audio: str = ""
    light: str = ""
    reaction: str = ""


@dataclass(slots=True)
class OffscreenPlan:
    shot_id: str
    events: list[OffscreenEvent] = field(default_factory=list)
    prompt_suffix: str = ""


def parse_offscreen_map(raw: Any) -> dict[str, list[OffscreenEvent]]:
    out: dict[str, list[OffscreenEvent]] = {}
    if not isinstance(raw, Mapping):
        return out
    for shot_id, spec in raw.items():
        events: list[OffscreenEvent] = []
        zones = spec if isinstance(spec, list) else (spec.get("zones") or spec.get("offscreen") or [])
        if isinstance(zones, str):
            zones = [zones]
        for z in zones:
            zname = str(z)
            preset = _OFFSCREEN_PRESETS.get(zname, {})
            if isinstance(z, Mapping):
                events.append(
                    OffscreenEvent(
                        zone=str(z.get("zone") or zname),
                        audio=str(z.get("audio") or ""),
                        light=str(z.get("light") or ""),
                        reaction=str(z.get("reaction") or ""),
                    )
                )
            else:
                events.append(
                    OffscreenEvent(
                        zone=zname,
                        audio=preset.get("audio", ""),
                        light=preset.get("light", ""),
                        reaction=preset.get("reaction", ""),
                    )
                )
        if events:
            out[str(shot_id)] = events
    return out


def plan_offscreen_space(
    shots: Sequence[Any],
    offscreen_map: Mapping[str, list[OffscreenEvent]],
) -> list[OffscreenPlan]:
    plans: list[OffscreenPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        inline = getattr(sh, "offscreen", None)
        events = list(offscreen_map.get(sid, []))
        if isinstance(inline, list):
            for z in inline:
                events.append(OffscreenEvent(zone=str(z)))
        elif isinstance(inline, str):
            events.append(OffscreenEvent(zone=inline))
        if not events:
            continue
        frags: list[str] = []
        for ev in events:
            if ev.audio:
                frags.append(ev.audio)
            if ev.light:
                frags.append(ev.light)
            if ev.reaction and ev.reaction != "none":
                frags.append(ev.reaction)
        plans.append(OffscreenPlan(shot_id=sid, events=events, prompt_suffix=", ".join(frags)))
    return plans
