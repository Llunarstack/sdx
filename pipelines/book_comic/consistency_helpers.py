"""
Prompt helpers for cross-page consistency: recurring characters, props, vehicles,
settings, palette/lighting, and lettering (hard cases for diffusion).

These are **soft** text cues—pair with inpaint anchoring (face/edge/bubbles) and
``--character-sheet`` / ``--character-prompt-extra`` where your stack supports them.

Used by ``generate_book.py`` (``--consistency-*`` flags, optional JSON spec).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

# ---------------------------------------------------------------------------
# Negative add-ons (append to your base negative; keep lexicon negatives too)
# ---------------------------------------------------------------------------

CONSISTENCY_NEGATIVE_LIGHT = (
    "different face in every panel, random hair color change, outfit changes mid-page, "
    "morphing vehicle silhouette, prop redesign each panel, duplicate clones of same character"
)

CONSISTENCY_NEGATIVE_STRONG = (
    CONSISTENCY_NEGATIVE_LIGHT
    + ", melted props, floating disconnected objects, illegible blob text, mirrored lettering errors"
)

CHARACTER_SILHOUETTE_HINT = (
    "consistent character silhouette and proportions across panels, recognizable same person"
)

TEXT_LEGIBILITY_POSITIVE = (
    "hand-lettered comic dialogue, crisp balloon tails, readable printed letters, "
    "high contrast ink inside speech balloons"
)


def character_appearance_clause(
    *,
    label: str = "",
    hair: str = "",
    eyes: str = "",
    skin: str = "",
    build: str = "",
    face: str = "",
    signature_items: str = "",
    age_vibe: str = "",
    extra: str = "",
) -> str:
    """Stable visual tokens for one recurring cast member."""
    bits: List[str] = []
    if (label or "").strip():
        bits.append(f"recurring character {label.strip()}")
    else:
        bits.append("same recurring protagonist")
    for part in (age_vibe, build, face, skin, hair, eyes, signature_items, extra):
        s = (part or "").strip()
        if s:
            bits.append(s)
    bits.append(CHARACTER_SILHOUETTE_HINT)
    return merge_prompt_fragments(*bits)


def costume_lock_clause(outfit: str) -> str:
    s = (outfit or "").strip()
    if not s:
        return ""
    return f"unchanging outfit and accessories throughout: {s}"


def object_prop_clause(description: str, *, codename: str = "") -> str:
    """Single important prop or object that should read the same across pages."""
    d = (description or "").strip()
    if not d:
        return ""
    tag = (codename or "").strip()
    if tag:
        return f"same important prop ({tag}): {d}, identical object design across panels"
    return f"same recurring prop: {d}, identical object details across shots"


def vehicle_clause(description: str, *, interior: bool = False) -> str:
    """Recurring vehicle—exterior (and optional interior) stability."""
    d = (description or "").strip()
    if not d:
        return ""
    tail = (
        ", same interior: seats, dashboard, steering wheel details"
        if interior
        else ", same exterior details and proportions"
    )
    return f"same recurring vehicle: {d}{tail}"


def setting_continuity_clause(
    location: str,
    *,
    time_of_day: str = "",
    weather: str = "",
    era: str = "",
    architecture_style: str = "",
) -> str:
    """Location and atmosphere that should stay coherent across the run."""
    loc = (location or "").strip()
    if not loc:
        return ""
    bits = [f"continuous setting: {loc}"]
    for part in (time_of_day, weather, era, architecture_style):
        s = (part or "").strip()
        if s:
            bits.append(s)
    bits.append("consistent background geography and props in environment")
    return merge_prompt_fragments(*bits)


def palette_lighting_clause(*, palette: str = "", lighting: str = "") -> str:
    """Color script + light direction (reduces drift when inpaint chain is weak)."""
    bits: List[str] = []
    p = (palette or "").strip()
    l = (lighting or "").strip()
    if p:
        bits.append(f"locked color palette: {p}")
    if l:
        bits.append(f"consistent lighting: {l}")
    if not bits:
        return ""
    bits.append("same color grading across panels")
    return merge_prompt_fragments(*bits)


def lettering_hard_clause(
    *,
    language: str = "",
    all_caps_dialogue: bool = False,
    emphasize_quotes: bool = True,
) -> str:
    """Extra positive cues when dialogue and SFX are critical (pair with OCR / expected text)."""
    bits = [TEXT_LEGIBILITY_POSITIVE]
    lang = (language or "").strip()
    if lang:
        bits.append(f"dialogue language {lang}")
    if all_caps_dialogue:
        bits.append("clear all-caps dialogue lettering where appropriate")
    if emphasize_quotes:
        bits.append("quoted dialogue text must match lettering exactly")
    return merge_prompt_fragments(*bits)


def creature_or_mascot_clause(description: str, *, species_lock: str = "") -> str:
    """Pet, robot sidekick, or non-human recurring companion."""
    d = (description or "").strip()
    if not d:
        return ""
    sp = (species_lock or "").strip()
    head = f"same recurring creature companion: {d}"
    if sp:
        head = f"{head}, species locked as {sp}"
    return f"{head}, consistent anatomy and markings across appearances"


def merge_consistency_positive(*fragments: str) -> str:
    """Join non-empty consistency fragments (comma-separated)."""
    return merge_prompt_fragments(*fragments)


def consistency_negative_addon(level: str) -> str:
    lv = (level or "none").lower().strip()
    if lv == "light":
        return CONSISTENCY_NEGATIVE_LIGHT
    if lv == "strong":
        return CONSISTENCY_NEGATIVE_STRONG
    return ""


# ---------------------------------------------------------------------------
# JSON + CLI merge (spec consumed by generate_book)
# ---------------------------------------------------------------------------

_CHAR_KEYS = frozenset(
    {"label", "hair", "eyes", "skin", "build", "face", "signature_items", "age_vibe", "extra"}
)


def load_consistency_json(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("consistency JSON root must be an object")
    return data


def _character_from_mapping(d: Mapping[str, Any]) -> str:
    kw = {k: str(v).strip() for k, v in d.items() if k in _CHAR_KEYS and str(v).strip()}
    return character_appearance_clause(**kw)  # type: ignore[arg-type]


def _props_from_spec_value(props: Any) -> List[str]:
    out: List[str] = []
    if isinstance(props, list):
        for p in props:
            if isinstance(p, str) and p.strip():
                out.append(object_prop_clause(p.strip()))
            elif isinstance(p, dict):
                desc = str(p.get("description", "")).strip()
                if desc:
                    out.append(object_prop_clause(desc, codename=str(p.get("label", "") or "")))
    elif isinstance(props, str) and props.strip():
        for seg in props.split(";"):
            s = seg.strip()
            if s:
                out.append(object_prop_clause(s))
    return out


def positive_block_from_mapping(spec: Mapping[str, Any]) -> str:
    """Build one positive consistency block from a dict (JSON file or merged CLI)."""
    fragments: List[str] = []

    ch = spec.get("character")
    if isinstance(ch, dict):
        fragments.append(_character_from_mapping(ch))
    elif isinstance(ch, str) and ch.strip():
        fragments.append(f"consistent protagonist appearance: {ch.strip()}")

    costume = spec.get("costume")
    if isinstance(costume, str) and costume.strip():
        fragments.append(costume_lock_clause(costume))

    fragments.extend(_props_from_spec_value(spec.get("props") if "props" in spec else spec.get("objects")))

    veh = spec.get("vehicle")
    if isinstance(veh, dict):
        desc = str(veh.get("description", "")).strip()
        if desc:
            fragments.append(vehicle_clause(desc, interior=bool(veh.get("interior"))))
    elif isinstance(veh, str) and veh.strip():
        fragments.append(vehicle_clause(veh))

    st = spec.get("setting")
    if isinstance(st, dict):
        loc = str(st.get("location", "")).strip()
        if loc:
            fragments.append(
                setting_continuity_clause(
                    loc,
                    time_of_day=str(st.get("time_of_day", "") or ""),
                    weather=str(st.get("weather", "") or ""),
                    era=str(st.get("era", "") or ""),
                    architecture_style=str(st.get("architecture_style", "") or ""),
                )
            )
    elif isinstance(st, str) and st.strip():
        fragments.append(setting_continuity_clause(st))

    mascot = spec.get("creature") or spec.get("mascot") or spec.get("pet")
    if isinstance(mascot, dict):
        desc = str(mascot.get("description", "")).strip()
        if desc:
            fragments.append(
                creature_or_mascot_clause(desc, species_lock=str(mascot.get("species", "") or ""))
            )
    elif isinstance(mascot, str) and mascot.strip():
        fragments.append(creature_or_mascot_clause(mascot))

    pl = spec.get("palette_lighting")
    if isinstance(pl, dict):
        frag = palette_lighting_clause(
            palette=str(pl.get("palette", "") or ""),
            lighting=str(pl.get("lighting", "") or ""),
        )
        if frag:
            fragments.append(frag)

    let = spec.get("lettering")
    if isinstance(let, dict):
        fragments.append(
            lettering_hard_clause(
                language=str(let.get("language", "") or ""),
                all_caps_dialogue=bool(let.get("all_caps_dialogue", False)),
                emphasize_quotes=bool(let.get("emphasize_quotes", True)),
            )
        )
        ex = str(let.get("extra", "")).strip()
        if ex:
            fragments.append(ex)
    elif spec.get("lettering_hard") is True:
        fragments.append(lettering_hard_clause())

    ve = spec.get("visual_extra") or spec.get("extras")
    if isinstance(ve, str) and ve.strip():
        fragments.append(ve.strip())

    return merge_prompt_fragments(*fragments)


def negative_level_from_spec(spec: Mapping[str, Any], cli_level: Optional[str]) -> str:
    """Resolve negative tier: explicit CLI beats spec when CLI is not None."""
    if cli_level is not None:
        return (cli_level or "none").lower().strip()
    nl = spec.get("consistency_negative") or spec.get("negative_level")
    if isinstance(nl, str) and nl.strip():
        return nl.strip().lower()
    return "none"


def overlay_cli_on_spec(spec: MutableMapping[str, Any], args: Any) -> None:
    """Mutate *spec* with non-empty generate_book CLI consistency fields."""

    def _s(name: str) -> str:
        return str(getattr(args, name, "") or "").strip()

    if _s("consistency_character"):
        spec["character"] = _s("consistency_character")
    if _s("consistency_costume"):
        spec["costume"] = _s("consistency_costume")
    props = _s("consistency_props")
    if props:
        spec["props"] = [p.strip() for p in props.split(";") if p.strip()]
    if _s("consistency_vehicle"):
        spec["vehicle"] = _s("consistency_vehicle")
    if _s("consistency_setting"):
        spec["setting"] = _s("consistency_setting")
    if _s("consistency_creature"):
        spec["creature"] = _s("consistency_creature")
    pal = _s("consistency_palette")
    lig = _s("consistency_lighting")
    if pal or lig:
        existing = spec.get("palette_lighting")
        ep = str(existing.get("palette", "")).strip() if isinstance(existing, dict) else ""
        el = str(existing.get("lighting", "")).strip() if isinstance(existing, dict) else ""
        spec["palette_lighting"] = {
            "palette": pal or ep,
            "lighting": lig or el,
        }
    vx = _s("consistency_visual_extra")
    if vx:
        spec["visual_extra"] = merge_prompt_fragments(str(spec.get("visual_extra", "")), vx)
    if getattr(args, "consistency_lettering_hard", False):
        spec["lettering_hard"] = True
