"""Stinger Frames — last frames of a shot get impact/sakuga emphasis (smear, hold, flash)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["StingerSpec", "parse_stinger_config", "plan_stinger_frames"]

_ACTION_END = ("hits", "lands", "explodes", "strikes", "slams", "punch", "kick", "impact", "crash")


@dataclass(slots=True)
class StingerSpec:
    shot_id: str
    frame_count: int
    style: str
    prompt_suffix: str
    edit_overrides: Dict[str, Any]


def parse_stinger_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "frames": int(raw.get("frames") or raw.get("frame_count") or 3),
            "style": str(raw.get("style") or "impact"),
        }
    return {"enabled": bool(raw)}


def _needs_stinger(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(v in p for v in _ACTION_END)


def plan_stinger_frames(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> List[StingerSpec]:
    if not config.get("enabled"):
        return []
    n = max(1, min(8, int(config.get("frames") or 3)))
    style = str(config.get("style") or "impact")
    specs: List[StingerSpec] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        prompt = str(getattr(sh, "prompt", ""))
        force = bool(getattr(sh, "stinger", False))
        if not force and not _needs_stinger(prompt):
            continue
        if style == "anime":
            frag = f"anime sakuga stinger, smear frames on last {n} frames, impact hold"
        elif style == "horror":
            frag = f"horror stinger, sudden micro-zoom last {n} frames, breath hold"
        else:
            frag = f"cinematic impact stinger, motion accent last {n} frames"
        specs.append(
            StingerSpec(
                shot_id=sid,
                frame_count=n,
                style=style,
                prompt_suffix=frag,
                edit_overrides={"motion_beat_keyframes": True, "keyframe_interval": max(2, 6 - n)},
            )
        )
    return specs
