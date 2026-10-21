"""
Prompt helpers for cross-page consistency: recurring characters, props, vehicles,
settings, palette/lighting, and lettering (hard cases for diffusion).

These are **soft** text cues—pair with inpaint anchoring (face/edge/bubbles) and
``--character-sheet`` / ``--character-prompt-extra`` where your stack supports them.

Used by ``generate_book.py`` (``--consistency-*`` flags, optional JSON spec).

Optional JSON key ``text_continuity`` (object): locked dialogue phrases, object labels, strict script
flags — see ``book_text_continuity.text_continuity_clause``.

Optional key ``challenging_content`` (object): ``pack`` / ``tags`` / ``extra`` for hard-case prompts;
see ``book_challenging_content.challenging_content_from_mapping``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from pipelines.book_comic.book_challenging_content import challenging_content_from_mapping
from pipelines.book_comic.book_text_continuity import text_continuity_clause
from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

# Negative add-ons are appended to the base negative (lexicon negatives stay).
CONSISTENCY_NEGATIVE_LIGHT = (
    "different face in every panel, random hair color change, outfit changes mid-page, "
    "morphing vehicle silhouette, prop redesign each panel, duplicate clones of same character"
)

CONSISTENCY_NEGATIVE_STRONG = (
    CONSISTENCY_NEGATIVE_LIGHT
    + ", melted props, floating disconnected objects, illegible blob text, mirrored lettering errors"
)

CHARACTER_SILHOUETTE_HINT = "consistent character silhouette and proportions across panels, recognizable same person"

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
    label = (label or "").strip()
    head = f"recurring character {label}" if label else "same recurring protagonist"
    return merge_prompt_fragments(
        head, age_vibe, build, face, skin, hair, eyes, signature_items, extra, CHARACTER_SILHOUETTE_HINT
    )


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
    return merge_prompt_fragments(
        f"continuous setting: {loc}",
        time_of_day,
        weather,
        era,
        architecture_style,
        "consistent background geography and props in environment",
    )


def palette_lighting_clause(*, palette: str = "", lighting: str = "") -> str:
    """Color script + light direction (reduces drift when inpaint chain is weak)."""
    palette = (palette or "").strip()
    lighting = (lighting or "").strip()
    if not (palette or lighting):
        return ""
    return merge_prompt_fragments(
        f"locked color palette: {palette}" if palette else "",
        f"consistent lighting: {lighting}" if lighting else "",
        "same color grading across panels",
    )


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


def consistency_negative_addon(level: str) -> str:
    """Negative add-on for ``light`` / ``strong``; "" otherwise."""
    lv = (level or "").lower().strip()
    return {"light": CONSISTENCY_NEGATIVE_LIGHT, "strong": CONSISTENCY_NEGATIVE_STRONG}.get(lv, "")


_CHAR_KEYS = frozenset({"label", "hair", "eyes", "skin", "build", "face", "signature_items", "age_vibe", "extra"})


def load_consistency_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("consistency JSON root must be an object")
    return data


def _field(d: Mapping[str, Any], key: str) -> str:
    """``d[key]`` as stripped text; missing / null / empty -> "" (never the literal text None)."""
    return str(d.get(key) or "").strip()


def _character_from_mapping(d: Mapping[str, Any]) -> str:
    return character_appearance_clause(**{k: _field(d, k) for k in _CHAR_KEYS if k in d})


def _props_from_spec_value(props: Any) -> list[str]:
    """Prop clauses from a ``;``-separated string or a list of strings / ``{description, label}`` dicts."""
    if isinstance(props, str):
        props = props.split(";")
    if not isinstance(props, list):
        return []
    out: list[str] = []
    for p in props:
        if isinstance(p, str):
            out.append(object_prop_clause(p))
        elif isinstance(p, dict):
            out.append(object_prop_clause(_field(p, "description"), codename=_field(p, "label")))
    return out


def positive_block_from_mapping(spec: Mapping[str, Any]) -> str:
    """Build one positive consistency block from a dict (JSON file or merged CLI)."""
    fragments: list[str] = []

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
        desc = _field(veh, "description")
        if desc:
            fragments.append(vehicle_clause(desc, interior=bool(veh.get("interior"))))
    elif isinstance(veh, str) and veh.strip():
        fragments.append(vehicle_clause(veh))

    st = spec.get("setting")
    if isinstance(st, dict):
        loc = _field(st, "location")
        if loc:
            fragments.append(
                setting_continuity_clause(
                    loc,
                    time_of_day=_field(st, "time_of_day"),
                    weather=_field(st, "weather"),
                    era=_field(st, "era"),
                    architecture_style=_field(st, "architecture_style"),
                )
            )
    elif isinstance(st, str) and st.strip():
        fragments.append(setting_continuity_clause(st))

    mascot = spec.get("creature") or spec.get("mascot") or spec.get("pet")
    if isinstance(mascot, dict):
        desc = _field(mascot, "description")
        if desc:
            fragments.append(creature_or_mascot_clause(desc, species_lock=_field(mascot, "species")))
    elif isinstance(mascot, str) and mascot.strip():
        fragments.append(creature_or_mascot_clause(mascot))

    pl = spec.get("palette_lighting")
    if isinstance(pl, dict):
        fragments.append(palette_lighting_clause(palette=_field(pl, "palette"), lighting=_field(pl, "lighting")))

    let = spec.get("lettering")
    if isinstance(let, dict):
        fragments.append(
            lettering_hard_clause(
                language=_field(let, "language"),
                all_caps_dialogue=bool(let.get("all_caps_dialogue", False)),
                emphasize_quotes=bool(let.get("emphasize_quotes", True)),
            )
        )
        fragments.append(_field(let, "extra"))
    elif spec.get("lettering_hard") is True:
        fragments.append(lettering_hard_clause())

    ve = spec.get("visual_extra") or spec.get("extras")
    if isinstance(ve, str) and ve.strip():
        fragments.append(ve.strip())

    tc = spec.get("text_continuity")
    if isinstance(tc, dict):
        fragments.append(text_continuity_clause(tc))

    challenge = spec.get("challenging_content")
    if isinstance(challenge, dict):
        fragments.append(challenging_content_from_mapping(challenge))

    return merge_prompt_fragments(*fragments)


def negative_level_from_spec(spec: Mapping[str, Any], cli_level: str | None) -> str:
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

    for key in ("character", "costume", "vehicle", "setting", "creature"):
        value = _s(f"consistency_{key}")
        if value:
            spec[key] = value
    props = _s("consistency_props")
    if props:
        spec["props"] = [p.strip() for p in props.split(";") if p.strip()]
    pal = _s("consistency_palette")
    lig = _s("consistency_lighting")
    if pal or lig:
        existing = spec.get("palette_lighting")
        if not isinstance(existing, dict):
            existing = {}
        spec["palette_lighting"] = {
            "palette": pal or _field(existing, "palette"),
            "lighting": lig or _field(existing, "lighting"),
        }
    vx = _s("consistency_visual_extra")
    if vx:
        spec["visual_extra"] = merge_prompt_fragments(str(spec.get("visual_extra", "")), vx)
    if getattr(args, "consistency_lettering_hard", False):
        spec["lettering_hard"] = True
