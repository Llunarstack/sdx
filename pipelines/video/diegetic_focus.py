"""Narrative Spotlight — diegetic focus budget: hero detail vs impressionistic periphery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "FocusBudget",
    "ShotFocusPlan",
    "parse_focus_config",
    "plan_diegetic_focus",
]


@dataclass(slots=True)
class FocusBudget:
    hero_entities: List[str]
    periphery_mode: str  # soft | bokeh | abstract | pixel_melt
    periphery_strength: float
    background_entities: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ShotFocusPlan:
    shot_id: str
    in_focus: List[str]
    out_of_focus: List[str]
    prompt_suffix: str
    edit_overrides: Dict[str, Any] = field(default_factory=dict)


_PERIPHERY_PROMPTS: Dict[str, str] = {
    "soft": "shallow depth of field, soft background blur",
    "bokeh": "creamy bokeh, subject razor sharp",
    "abstract": "peripheral elements impressionistic, center narrative sharp",
    "pixel_melt": "dreamlike soft periphery, sharp story focus",
    "fog_of_narrative": "detail only where story looks, edges dissolve into atmosphere",
}


def parse_focus_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "default_mode": str(raw.get("periphery_mode") or raw.get("mode") or "fog_of_narrative"),
            "strength": float(raw.get("periphery_strength") or raw.get("strength") or 0.65),
            "heroes": list(raw.get("heroes") or raw.get("focus") or []),
            "background": list(raw.get("background") or []),
        }
    return {"enabled": bool(raw)}


def _entities_in_shot(shot: Any) -> List[str]:
    chars = list(getattr(shot, "characters", []) or [])
    objs = list(getattr(shot, "objects", []) or [])
    return [str(x) for x in chars + objs]


def plan_diegetic_focus(
    shots: Sequence[Any],
    config: Mapping[str, Any],
    cast: Mapping[str, Any],
) -> List[ShotFocusPlan]:
    if not config.get("enabled"):
        return []
    mode = str(config.get("default_mode") or "fog_of_narrative")
    strength = float(config.get("strength") or 0.65)
    global_heroes = [str(x) for x in (config.get("heroes") or [])]
    global_bg = [str(x) for x in (config.get("background") or [])]

    plans: List[ShotFocusPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        ents = _entities_in_shot(sh)
        heroes = [e for e in ents if e in global_heroes] or ([ents[0]] if ents else global_heroes[:1])
        bg = [e for e in ents if e in global_bg] or [e for e in ents if e not in heroes]
        periphery = _PERIPHERY_PROMPTS.get(mode, _PERIPHERY_PROMPTS["fog_of_narrative"])
        hero_desc = []
        for h in heroes:
            ent = cast.get(h)
            if ent is not None:
                hero_desc.append(getattr(ent, "description", "") or h)
            else:
                hero_desc.append(h)
        focus_frag = f"narrative spotlight on {', '.join(hero_desc)}" if hero_desc else "narrative spotlight on subject"
        if bg:
            focus_frag += f", background {', '.join(bg)} rendered {mode}"
        plans.append(
            ShotFocusPlan(
                shot_id=sid,
                in_focus=heroes,
                out_of_focus=bg,
                prompt_suffix=f"{focus_frag}, {periphery}",
                edit_overrides={
                    "depth_interpolate": strength > 0.4,
                    "identity_lock": bool(heroes),
                    "identity_lock_strength": min(0.95, 0.75 + strength * 0.2),
                },
            )
        )
    return plans
