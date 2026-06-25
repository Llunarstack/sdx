"""Causal Ripple Engine — story events automatically trigger camera, FX, and prop reactions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Set

__all__ = [
    "CausalRule",
    "CausalRipple",
    "parse_causal_rules",
    "detect_triggers_in_prompt",
    "apply_causal_ripples",
]

# Built-in physics-adjacent narrative causality (not pixel physics — story physics)
_BUILTIN_RULES: Dict[str, List[str]] = {
    "explosion": ["camera_shake", "dust_cloud", "characters_flinch", "audio_rumble"],
    "gunshot": ["camera_jolt", "muzzle_flash", "echo_reverb"],
    "thunder": ["lightning_flash", "rain_intensify", "characters_glance_up"],
    "door_slam": ["camera_micro_shake", "dust_motes", "hanging_lamp_sway"],
    "car_crash": ["debris_spray", "camera_whiplash", "glass_shatter"],
    "magic_cast": ["light_pulse", "wind_gust", "particles_swirl"],
    "scream": ["birds_scatter", "silence_after", "characters_turn"],
    "rain_start": ["wet_surfaces", "fabric_darken", "footstep_splash"],
    "wind_gust": ["hair_motion", "cloth_flutter", "leaves_blow"],
    "footstep_close": ["floor_creak", "camera_subtle_push"],
}


@dataclass(slots=True)
class CausalRule:
    trigger: str
    effects: List[str]
    camera: str = ""
    props: Dict[str, str] = field(default_factory=dict)
    delay_sec: float = 0.0


@dataclass(slots=True)
class CausalRipple:
    shot_id: str
    trigger: str
    injected_effects: List[str]
    prompt_suffix: str
    camera_hint: str
    props_patch: Dict[str, str]


def parse_causal_rules(raw: Any) -> List[CausalRule]:
    rules: List[CausalRule] = []
    if isinstance(raw, Mapping):
        for trig, spec in raw.items():
            if isinstance(spec, list):
                rules.append(CausalRule(trigger=str(trig), effects=[str(x) for x in spec]))
            elif isinstance(spec, Mapping):
                rules.append(
                    CausalRule(
                        trigger=str(trig),
                        effects=[str(x) for x in (spec.get("then") or spec.get("effects") or [])],
                        camera=str(spec.get("camera") or ""),
                        props={str(k): str(v) for k, v in (spec.get("props") or {}).items()},
                        delay_sec=float(spec.get("delay_sec") or 0.0),
                    )
                )
    elif isinstance(raw, list):
        for row in raw:
            if not isinstance(row, Mapping):
                continue
            trig = str(row.get("when") or row.get("trigger") or "")
            effects = row.get("then") or row.get("effects") or []
            rules.append(
                CausalRule(
                    trigger=trig,
                    effects=[str(x) for x in effects] if isinstance(effects, list) else [str(effects)],
                    camera=str(row.get("camera") or ""),
                    props={str(k): str(v) for k, v in (row.get("props") or {}).items()},
                    delay_sec=float(row.get("delay_sec") or 0.0),
                )
            )
    return rules


def _trigger_patterns(trigger: str) -> List[re.Pattern[str]]:
    t = trigger.lower().replace("_", " ")
    return [re.compile(rf"\b{re.escape(t)}\b", re.I), re.compile(rf"\b{re.escape(trigger)}\b", re.I)]


def detect_triggers_in_prompt(prompt: str, rules: Sequence[CausalRule]) -> List[str]:
    found: List[str] = []
    p = prompt or ""
    for rule in rules:
        for pat in _trigger_patterns(rule.trigger):
            if pat.search(p):
                found.append(rule.trigger)
                break
    # Also scan built-ins when prompt matches common verbs
    for trig in _BUILTIN_RULES:
        if trig in found:
            continue
        for pat in _trigger_patterns(trig):
            if pat.search(p):
                found.append(trig)
                break
    return found


_EFFECT_PROMPTS: Dict[str, str] = {
    "camera_shake": "handheld camera shake from impact",
    "camera_jolt": "sharp camera jolt",
    "camera_whiplash": "violent camera whiplash",
    "camera_micro_shake": "subtle camera vibration",
    "dust_cloud": "billowing dust cloud",
    "dust_motes": "dust motes in air",
    "characters_flinch": "characters flinch from blast",
    "muzzle_flash": "brief muzzle flash",
    "lightning_flash": "lightning illuminates scene",
    "rain_intensify": "rain intensifies",
    "wet_surfaces": "surfaces glistening wet",
    "hair_motion": "hair blown by wind",
    "cloth_flutter": "clothes flutter in gust",
    "particles_swirl": "swirling magical particles",
    "birds_scatter": "birds scatter from noise",
    "debris_spray": "debris spraying outward",
    "glass_shatter": "shattering glass fragments",
    "light_pulse": "pulsing magical light",
    "wind_gust": "strong wind gust through scene",
}


def apply_causal_ripples(
    shots: Sequence[Any],
    rules: Sequence[CausalRule],
    *,
    use_builtins: bool = True,
) -> List[CausalRipple]:
    merged_rules = list(rules)
    if use_builtins:
        existing = {r.trigger for r in merged_rules}
        for trig, effects in _BUILTIN_RULES.items():
            if trig not in existing:
                merged_rules.append(CausalRule(trigger=trig, effects=effects))

    ripples: List[CausalRipple] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        prompt = str(getattr(sh, "prompt", "") or "")
        triggers = detect_triggers_in_prompt(prompt, merged_rules)
        if not triggers:
            continue
        effects: Set[str] = set()
        camera_hints: List[str] = []
        props_patch: Dict[str, str] = {}
        for trig in triggers:
            for rule in merged_rules:
                if rule.trigger != trig:
                    continue
                effects.update(rule.effects)
                if rule.camera:
                    camera_hints.append(rule.camera)
                props_patch.update(rule.props)
            if use_builtins:
                effects.update(_BUILTIN_RULES.get(trig, []))

        prompt_bits = [_EFFECT_PROMPTS.get(e, e.replace("_", " ")) for e in sorted(effects)]
        ripples.append(
            CausalRipple(
                shot_id=sid,
                trigger=",".join(triggers),
                injected_effects=sorted(effects),
                prompt_suffix=", ".join(prompt_bits),
                camera_hint=", ".join(camera_hints),
                props_patch=props_patch,
            )
        )
    return ripples
