"""Silence Map — deliberate audio/visual silence beats for horror, comedy, and tension."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["SilenceBeat", "parse_silence_map", "plan_silence_beats"]


@dataclass(slots=True)
class SilenceBeat:
    shot_id: str
    duration_sec: float
    visual_mode: str
    prompt_suffix: str
    audio_note: str


def parse_silence_map(raw: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for sid, spec in raw.items():
            if isinstance(spec, (int, float)):
                out[str(sid)] = {"duration_sec": float(spec)}
            elif isinstance(spec, Mapping):
                out[str(sid)] = dict(spec)
    return out


def plan_silence_beats(
    shots: Sequence[Any],
    silence_map: Mapping[str, dict[str, Any]],
) -> list[SilenceBeat]:
    beats: list[SilenceBeat] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        spec = silence_map.get(sid, {})
        if getattr(sh, "silence", False):
            spec = {**spec, "duration_sec": spec.get("duration_sec") or 0.8}
        if not spec:
            continue
        dur = float(spec.get("duration_sec") or spec.get("duration") or 0.5)
        mode = str(spec.get("visual") or spec.get("mode") or "hold")
        if mode == "horror":
            frag = "held frozen frame, dread silence, micro-movement only"
        elif mode == "comedy":
            frag = "awkward pause beat, characters frozen mid-action"
        else:
            frag = "deliberate stillness, held breath moment"
        beats.append(
            SilenceBeat(
                shot_id=sid,
                duration_sec=dur,
                visual_mode=mode,
                prompt_suffix=frag,
                audio_note=str(spec.get("audio") or "drop all sound"),
            )
        )
    return beats
