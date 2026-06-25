"""Kling-style storyboard cuts with camera verbs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

__all__ = [
    "CAMERA_VERBS",
    "StoryboardCut",
    "camera_prompt_fragment",
    "parse_storyboard",
]


CAMERA_VERBS: Dict[str, str] = {
    "static": "locked tripod, static frame",
    "pan_left": "smooth pan left",
    "pan_right": "smooth pan right",
    "tilt_up": "tilt up",
    "tilt_down": "tilt down",
    "push_in": "slow dolly push in",
    "pull_back": "dolly pull back reveal",
    "tracking": "tracking shot following subject",
    "orbit": "orbiting camera around subject",
    "crane": "crane shot rising",
    "handheld": "handheld documentary camera",
    "whip_pan": "fast whip pan",
    "zoom_in": "slow zoom in",
    "zoom_out": "zoom out",
    "ots": "over the shoulder shot",
    "pov": "point of view shot",
    "establishing": "wide establishing shot",
    "close_up": "close-up shot",
}


@dataclass(slots=True)
class StoryboardCut:
    id: str = ""
    prompt: str = ""
    duration_sec: float = 0.0
    camera: str = ""
    shot_type: str = ""
    transition: str = "cut"
    characters: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    start_image: str = ""
    end_image: str = ""
    flf2v: bool = False
    motion_brush: Dict[str, Any] = field(default_factory=dict)
    elements: List[str] = field(default_factory=list)
    bindings: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def camera_prompt_fragment(camera: str) -> str:
    key = (camera or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in CAMERA_VERBS:
        return CAMERA_VERBS[key]
    if key:
        return f"{camera} camera movement"
    return ""


def _infer_shot_type(camera: str, prompt: str) -> str:
    c = (camera or "").lower()
    p = (prompt or "").lower()
    if "establishing" in c or "wide" in p:
        return "establishing"
    if "close" in c or "close_up" in c:
        return "close_up"
    if "ots" in c or "over the shoulder" in p:
        return "over_shoulder"
    if "pov" in c:
        return "pov"
    return "medium"


def parse_storyboard(raw: Any) -> List[StoryboardCut]:
    if not raw:
        return []
    cuts_raw: Any = raw
    if isinstance(raw, Mapping):
        cuts_raw = raw.get("cuts") or raw.get("shots") or []
    if not isinstance(cuts_raw, list):
        return []
    cuts: List[StoryboardCut] = []
    for i, row in enumerate(cuts_raw):
        if isinstance(row, str):
            cuts.append(StoryboardCut(id=f"cut_{i}", prompt=row))
            continue
        if not isinstance(row, Mapping):
            continue
        cuts.append(
            StoryboardCut(
                id=str(row.get("id") or f"cut_{i}"),
                prompt=str(row.get("prompt") or row.get("description") or ""),
                duration_sec=float(row.get("duration_sec") or row.get("duration") or 0.0),
                camera=str(row.get("camera") or row.get("camera_move") or ""),
                shot_type=str(row.get("shot_type") or row.get("type") or ""),
                transition=str(row.get("transition") or "cut").lower(),
                characters=_str_list(row.get("characters") or row.get("cast")),
                objects=_str_list(row.get("objects") or row.get("props")),
                effects=_str_list(row.get("effects")),
                start_image=str(row.get("start_image") or row.get("start") or ""),
                end_image=str(row.get("end_image") or row.get("end") or ""),
                flf2v=bool(
                    row.get("flf2v") or row.get("first_last") or (row.get("start_image") and row.get("end_image"))
                ),
                motion_brush=dict(row.get("motion_brush") or {}),
                elements=_str_list(row.get("elements") or row.get("bind_elements")),
                bindings=dict(row.get("bindings") or {}),
            )
        )
    return cuts


def _str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[,;]", v) if x.strip()]
    return [str(x) for x in v if str(x).strip()]
