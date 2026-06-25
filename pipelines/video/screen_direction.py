"""Screen Direction Lock — movement vector must stay consistent across cuts (vector 180° rule)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "ScreenDirectionIssue",
    "ScreenDirectionReport",
    "parse_screen_direction_config",
    "track_screen_direction",
]

_DIR_ALIASES = {
    "l": "left",
    "r": "right",
    "left": "left",
    "right": "right",
    "frame_left": "left",
    "frame_right": "right",
}

_MOVE_LEFT = (r"\bmoves?\s+left\b", r"\bwalks?\s+left\b", r"\benters?\s+from\s+right\b", r"\bexits?\s+left\b")
_MOVE_RIGHT = (r"\bmoves?\s+right\b", r"\bwalks?\s+right\b", r"\benters?\s+from\s+left\b", r"\bexits?\s+right\b")


@dataclass(slots=True)
class ScreenDirectionIssue:
    level: str
    code: str
    message: str
    shot_id: str
    related_shot_id: str = ""


@dataclass(slots=True)
class ScreenDirectionReport:
    directions: Dict[str, str]
    issues: List[ScreenDirectionIssue]
    prompt_patches: Dict[str, str]


def parse_screen_direction_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {"enabled": bool(raw.get("enabled", True)), "strict": bool(raw.get("strict", False))}
    return {"enabled": bool(raw)}


def _norm(d: str) -> str:
    return _DIR_ALIASES.get((d or "").strip().lower().replace(" ", "_"), "")


def _infer_direction(prompt: str) -> str:
    p = prompt or ""
    for pat in _MOVE_LEFT:
        if re.search(pat, p, re.I):
            return "left"
    for pat in _MOVE_RIGHT:
        if re.search(pat, p, re.I):
            return "right"
    return ""


def _shot_direction(shot: Any) -> str:
    explicit = _norm(str(getattr(shot, "screen_direction", "") or ""))
    if explicit:
        return explicit
    return _infer_direction(str(getattr(shot, "prompt", "")))


def track_screen_direction(shots: Sequence[Any], *, config: Mapping[str, Any]) -> ScreenDirectionReport:
    if not config.get("enabled"):
        return ScreenDirectionReport(directions={}, issues=[], prompt_patches={})
    strict = bool(config.get("strict"))
    level = "error" if strict else "warn"
    directions: Dict[str, str] = {}
    issues: List[ScreenDirectionIssue] = []
    patches: Dict[str, str] = {}

    prev_dir = ""
    prev_id = ""
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        d = _shot_direction(sh)
        if d:
            directions[sid] = d
        if prev_dir and d and prev_dir != d:
            if str(getattr(sh, "transition", "cut") or "cut") in ("cut", ""):
                issues.append(
                    ScreenDirectionIssue(
                        level=level,
                        code="screen_direction_flip",
                        message=f"Movement {prev_dir}→{d} across cut breaks screen direction continuity",
                        shot_id=prev_id,
                        related_shot_id=sid,
                    )
                )
                patches[sid] = f"continue movement toward screen {prev_dir}, match prior trajectory"
        if d:
            prev_dir = d
            prev_id = sid
    return ScreenDirectionReport(directions=directions, issues=issues, prompt_patches=patches)
