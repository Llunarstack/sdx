"""Continuity validators — eyeline, props, light motivation, silhouette readability."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "ContinuityIssue",
    "ContinuityReport",
    "ValidatorConfig",
    "parse_validator_config",
    "run_continuity_validation",
    "format_continuity_report",
]

_GAZE_ALIASES: Dict[str, str] = {
    "left": "frame_left",
    "right": "frame_right",
    "camera": "camera",
    "down": "down",
    "up": "up",
    "off_left": "off_screen_left",
    "off_right": "off_screen_right",
    "frame_left": "frame_left",
    "frame_right": "frame_right",
    "off_screen_left": "off_screen_left",
    "off_screen_right": "off_screen_right",
}

_LIGHT_SOURCES = (
    "window",
    "sun",
    "moon",
    "moonlight",
    "candle",
    "fire",
    "torch",
    "neon",
    "lamp",
    "streetlight",
    "headlight",
    "screen glow",
    "phone screen",
    "lightning",
    "spotlight",
)

_UNMOTIVATED_LIGHT = (
    "studio lighting",
    "flat lighting",
    "even lighting",
    "three point",
    "beauty dish",
    "ring light",
)

_SILHOUETTE_POSITIVE = (
    "silhouette",
    "backlit",
    "rim light",
    "against sky",
    "profile",
    "outline",
    "shadow figure",
)

_CLUTTER_WORDS = (
    "busy background",
    "crowded",
    "cluttered",
    "detailed background",
    "many objects",
)


@dataclass(slots=True)
class ContinuityIssue:
    level: str  # error | warn | info
    code: str
    message: str
    shot_id: str = ""
    related_shot_id: str = ""


@dataclass(slots=True)
class ValidatorConfig:
    eyeline: bool = True
    props: bool = True
    light_motivation: bool = True
    silhouette: bool = True
    strict: bool = False


@dataclass(slots=True)
class ContinuityReport:
    ok: bool
    issues: List[ContinuityIssue] = field(default_factory=list)

    def errors(self) -> List[ContinuityIssue]:
        return [i for i in self.issues if i.level == "error"]

    def warnings(self) -> List[ContinuityIssue]:
        return [i for i in self.issues if i.level == "warn"]


def parse_validator_config(raw: Mapping[str, Any] | None) -> ValidatorConfig:
    if not isinstance(raw, Mapping):
        return ValidatorConfig()
    v = raw.get("validators")
    if not isinstance(v, Mapping):
        v = raw
    return ValidatorConfig(
        eyeline=bool(v.get("eyeline", True)),
        props=bool(v.get("props", True)),
        light_motivation=bool(v.get("light_motivation", v.get("lighting", True))),
        silhouette=bool(v.get("silhouette", v.get("silhouette_readability", True))),
        strict=bool(v.get("strict", raw.get("strict", False))),
    )


def _norm_gaze(g: str) -> str:
    key = (g or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _GAZE_ALIASES.get(key, key)


def _infer_gaze_from_prompt(prompt: str) -> str:
    p = (prompt or "").lower()
    if re.search(r"\blooks?\s+(left|to the left)\b", p):
        return "frame_left"
    if re.search(r"\blooks?\s+(right|to the right)\b", p):
        return "frame_right"
    if "over the shoulder" in p or " ots " in f" {p} ":
        return "frame_right"
    if "pov" in p or "point of view" in p:
        return "camera"
    if re.search(r"\b(gaze|eyes?)\s+(down|downward)\b", p):
        return "down"
    if re.search(r"\b(gaze|eyes?)\s+(up|upward)\b", p):
        return "up"
    return ""


def _shot_gaze(shot: Any) -> str:
    g = str(getattr(shot, "gaze", "") or "")
    if g:
        return _norm_gaze(g)
    bindings = getattr(shot, "bindings", None) or {}
    if isinstance(bindings, Mapping):
        for val in bindings.values():
            if isinstance(val, Mapping) and val.get("gaze"):
                return _norm_gaze(str(val["gaze"]))
    return _infer_gaze_from_prompt(str(getattr(shot, "prompt", "")))


def _is_dialogue_pair(g1: str, g2: str) -> bool:
    """Reverse-shot eyeline: subjects should look toward each other across cuts."""
    pair = {g1, g2}
    return pair == {"frame_left", "frame_right"} or pair == {"off_screen_left", "off_screen_right"}


def validate_eyeline(shots: Sequence[Any], *, strict: bool = False) -> List[ContinuityIssue]:
    issues: List[ContinuityIssue] = []
    level = "error" if strict else "warn"
    for i in range(len(shots) - 1):
        a, b = shots[i], shots[i + 1]
        if str(getattr(a, "transition", "cut") or "cut") not in ("cut", ""):
            continue
        ga = _shot_gaze(a)
        gb = _shot_gaze(b)
        if not ga or not gb:
            continue
        ca = list(getattr(a, "characters", []) or [])
        cb = list(getattr(b, "characters", []) or [])
        same_chars = bool(ca and cb and set(ca) & set(cb))
        if same_chars and ga == gb and ga in ("frame_left", "frame_right"):
            issues.append(
                ContinuityIssue(
                    level=level,
                    code="eyeline_180_rule",
                    message=f"Cut {getattr(a, 'id', i)} → {getattr(b, 'id', i + 1)}: same gaze "
                    f"({ga}) on shared character — reverse shot may break 180° rule",
                    shot_id=str(getattr(a, "id", "")),
                    related_shot_id=str(getattr(b, "id", "")),
                )
            )
        if ca and cb and not (set(ca) & set(cb)) and not _is_dialogue_pair(ga, gb):
            if ga in ("frame_left", "frame_right") and gb in ("frame_left", "frame_right") and ga == gb:
                issues.append(
                    ContinuityIssue(
                        level=level,
                        code="eyeline_mismatch",
                        message=f"Dialogue cut {getattr(a, 'id', i)} → {getattr(b, 'id', i + 1)}: "
                        f"both gaze {ga} — subjects may not appear to look at each other",
                        shot_id=str(getattr(a, "id", "")),
                        related_shot_id=str(getattr(b, "id", "")),
                    )
                )
        cam_b = str(getattr(b, "camera", "") or "").lower()
        if "ots" in cam_b and gb == "frame_left" and ga not in ("frame_right", "camera", ""):
            issues.append(
                ContinuityIssue(
                    level="info",
                    code="eyeline_ots_hint",
                    message=f"Shot {getattr(b, 'id', i + 1)}: OTS usually needs subject gaze frame_right",
                    shot_id=str(getattr(b, "id", "")),
                )
            )
    return issues


def _props_state(shot: Any) -> Dict[str, str]:
    raw = getattr(shot, "props_state", None) or {}
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def validate_prop_continuity(
    shots: Sequence[Any],
    ledger: Mapping[str, Any],
    *,
    strict: bool = False,
) -> List[ContinuityIssue]:
    issues: List[ContinuityIssue] = []
    level = "error" if strict else "warn"
    tracked: Dict[str, str] = {}
    for pid, spec in ledger.items():
        if isinstance(spec, Mapping):
            tracked[str(pid)] = str(spec.get("initial") or spec.get("state") or "")
        elif isinstance(spec, str):
            tracked[str(pid)] = spec

    last_state: Dict[str, str] = dict(tracked)
    irreversible = ("broken", "destroyed", "empty", "gone", "shattered", "consumed")

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        state = _props_state(sh)
        shot_objects = set(getattr(sh, "objects", []) or [])
        for pid, st in state.items():
            if pid not in shot_objects and pid not in ledger:
                issues.append(
                    ContinuityIssue(
                        level="info",
                        code="prop_unlisted",
                        message=f"Shot {sid}: props_state.{pid} set but prop not in shot.objects",
                        shot_id=sid,
                    )
                )
            prev = last_state.get(pid)
            if prev and prev != st:
                if any(x in prev.lower() for x in irreversible) and st.lower() in ("pristine", "full", "new", "intact"):
                    issues.append(
                        ContinuityIssue(
                            level=level,
                            code="prop_state_reset",
                            message=f"Shot {sid}: {pid} jumped {prev!r} → {st!r} without narrative reset",
                            shot_id=sid,
                        )
                    )
            last_state[pid] = st
    return issues


def _shot_lighting(shot: Any) -> Dict[str, Any]:
    raw = getattr(shot, "lighting", None) or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def validate_light_motivation(shots: Sequence[Any], *, strict: bool = False) -> List[ContinuityIssue]:
    issues: List[ContinuityIssue] = []
    level = "error" if strict else "warn"
    last_sun = ""

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        prompt = str(getattr(sh, "prompt", "") or "").lower()
        lit = _shot_lighting(sh)
        motivated = lit.get("motivated")
        key = str(lit.get("key") or lit.get("source") or "").lower()

        has_source = bool(key) or any(s in prompt for s in _LIGHT_SOURCES)
        unmotivated = any(u in prompt for u in _UNMOTIVATED_LIGHT)

        if motivated is False or (unmotivated and not has_source):
            issues.append(
                ContinuityIssue(
                    level=level,
                    code="light_unmotivated",
                    message=f"Shot {sid}: lighting may lack in-scene motivation (add lighting.key or source in prompt)",
                    shot_id=sid,
                )
            )

        sun_dir = str(lit.get("sun_direction") or lit.get("sun") or "")
        if not sun_dir:
            if "sun from left" in prompt or "sunlight from left" in prompt:
                sun_dir = "left"
            elif "sun from right" in prompt or "sunlight from right" in prompt:
                sun_dir = "right"
        if sun_dir and last_sun and sun_dir != last_sun:
            issues.append(
                ContinuityIssue(
                    level="warn",
                    code="light_sun_flip",
                    message=f"Shot {sid}: sun direction {last_sun} → {sun_dir} without time/location change",
                    shot_id=sid,
                )
            )
        if sun_dir:
            last_sun = sun_dir
    return issues


def validate_silhouette_readability(shots: Sequence[Any]) -> List[ContinuityIssue]:
    issues: List[ContinuityIssue] = []
    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", f"shot_{i}"))
        prompt = str(getattr(sh, "prompt", "") or "").lower()
        shot_type = str(getattr(sh, "shot_type", "") or "").lower()
        if not prompt:
            continue
        has_positive = any(w in prompt for w in _SILHOUETTE_POSITIVE)
        has_clutter = any(w in prompt for w in _CLUTTER_WORDS)
        action_heavy = any(w in prompt for w in ("fight", "chase", "battle", "runs", "jumps", "explosion"))
        if action_heavy and not has_positive and has_clutter:
            issues.append(
                ContinuityIssue(
                    level="warn",
                    code="silhouette_cluttered",
                    message=f"Shot {sid}: action beat with cluttered background — silhouette may be unreadable",
                    shot_id=sid,
                )
            )
        if shot_type in ("wide", "establishing") and "wide" in prompt and not has_positive:
            if prompt.count(",") >= 4:
                issues.append(
                    ContinuityIssue(
                        level="info",
                        code="silhouette_busy_wide",
                        message=f"Shot {sid}: busy wide shot — consider stronger subject/background separation",
                        shot_id=sid,
                    )
                )
    return issues


def run_continuity_validation(
    shots: Sequence[Any],
    *,
    continuity: Optional[Mapping[str, Any]] = None,
    config: Optional[ValidatorConfig] = None,
) -> ContinuityReport:
    cfg = config or parse_validator_config(continuity or {})
    ledger = {}
    if isinstance(continuity, Mapping):
        ledger = continuity.get("props_ledger") or continuity.get("props") or {}
        if not isinstance(ledger, Mapping):
            ledger = {}

    issues: List[ContinuityIssue] = []
    if cfg.eyeline and shots:
        issues.extend(validate_eyeline(shots, strict=cfg.strict))
    if cfg.props and shots:
        issues.extend(validate_prop_continuity(shots, ledger, strict=cfg.strict))
    if cfg.light_motivation and shots:
        issues.extend(validate_light_motivation(shots, strict=cfg.strict))
    if cfg.silhouette and shots:
        issues.extend(validate_silhouette_readability(shots))

    ok = not any(i.level == "error" for i in issues)
    return ContinuityReport(ok=ok, issues=issues)


def format_continuity_report(report: ContinuityReport) -> str:
    lines = [f"Continuity: {'OK' if report.ok else 'ISSUES'}"]
    for i in report.issues:
        rel = f" (↔ {i.related_shot_id})" if i.related_shot_id else ""
        lines.append(f"  [{i.level.upper()}] {i.code} [{i.shot_id}]{rel}: {i.message}")
    return "\n".join(lines)
