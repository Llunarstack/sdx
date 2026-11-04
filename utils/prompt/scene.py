"""Scene and subject prompting: creature / physics add-ons, busy scenes, multi-instance and multi-subject hints, scene blueprints and shape scaffolds, facets, auto original characters."""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import IO, Any, Literal

from config.prompts import (
    CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_COMMON_NEGATIVE_ADDON,
    CREATURE_NSFW_CONTEXT_POSITIVE,
    CREATURE_NSFW_NEGATIVE_ADDON,
    CREATURE_SFW_CONTEXT_POSITIVE,
    CREATURE_SFW_NEGATIVE_ADDON,
    HUMANOID_MONSTER_PROMPT_KEYWORDS,
    PHYSICS_COMMON_NEGATIVE_ADDON,
    PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN,
    dedupe_tags,
    merge_csv_unique,
)
from config.styles import ANATOMY_NEG_LITE, ANATOMY_NEG_STRONG, ANATOMY_POS_LITE, ANATOMY_POS_STRONG

from utils.prompt.text import str_list
from utils.runtime import json_loads
from utils.safety.minor_guard import is_sexual

CreatureRating = Literal["sfw", "nsfw", "auto"]

_ANTHRO_KW = ("anthro", "anthropomorphic", "furry", "fursona", "kemono", "muzzle", "digitigrade", "plantigrade", "paw")
_ROBOT_KW = ("robot", "android", "mech", "mecha", "cyborg", "gundam", "automaton", "synthetic humanoid")
_BEAST_KW = tuple(
    k
    for k in (
        "dragon",
        "griffin",
        "gryphon",
        "wyvern",
        "kaiju",
        "beast",
        "creature",
        "alien",
        "xenomorph",
        "mermaid",
        "merfolk",
        "golem",
    )
    if k not in frozenset(HUMANOID_MONSTER_PROMPT_KEYWORDS)
)

_PHYSICS_RULES: tuple[tuple[tuple[str, ...], str, int], ...] = (
    (
        (
            "water",
            "liquid",
            "fluid",
            "splash",
            "pour",
            "wave",
            "rain",
            "ocean",
            "underwater",
            "droplet",
            "meniscus",
            "steam",
            "mist",
            "fog",
        ),
        "fluids",
        0,
    ),
    (
        (
            "glass",
            "transparent",
            "translucent",
            "refraction",
            "frosted",
            "acrylic",
            "caustic",
            "see-through",
            "crystalline",
        ),
        "transparency",
        0,
    ),
    (("fabric", "cloth", "drap", "silk", "velvet", "hair", "ponytail", "cape", "dress"), "soft_bodies", 0),
    (
        ("stack", "balance", "leaning", "resting on", "sitting on", "standing on", "contact shadow"),
        "physics_materials",
        2,
    ),
    (("smoke", "fire", "flame", "explosion", "volumetric", "god ray"), "physics_materials", 0),
)


def suggest_creature_prompt_addons(prompt: str, *, rating: CreatureRating = "auto") -> tuple[str, str]:
    """Add-ons for anthro, robot/mech, humanoid-monster and beast prompts.

    ``rating="auto"`` switches to the mature packs only for sexual prompts (``minor_guard.is_sexual``).
    """
    lower = prompt.lower()
    mature = rating == "nsfw" or (rating == "auto" and is_sexual(lower))
    if rating == "sfw":
        domain_map = CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN
        ctx_pos, extra_neg = CREATURE_SFW_CONTEXT_POSITIVE, CREATURE_SFW_NEGATIVE_ADDON
    elif mature:
        domain_map = CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN
        ctx_pos, extra_neg = CREATURE_NSFW_CONTEXT_POSITIVE, CREATURE_NSFW_NEGATIVE_ADDON
    else:
        domain_map = CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN
        ctx_pos, extra_neg = "", ""

    pos = [
        domain_map[domain][0]
        for keywords, domain in (
            (_ANTHRO_KW, "anthro_furry"),
            (_ROBOT_KW, "robots_mechs"),
            (HUMANOID_MONSTER_PROMPT_KEYWORDS, "humanoid_monsters"),
            (_BEAST_KW, "creatures"),
        )
        if any(k in lower for k in keywords)
    ]
    if not pos:
        return "", ""
    neg = ", ".join(x for x in (CREATURE_COMMON_NEGATIVE_ADDON, extra_neg) if x)
    return ", ".join(([ctx_pos] if ctx_pos else []) + pos), neg


def suggest_physics_prompt_addons(prompt: str) -> tuple[str, str]:
    """Add-ons for fluids, transparency, soft bodies, rigid contact and volumetrics."""
    lower = prompt.lower()
    pos = [
        PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN[domain][idx]
        for keywords, domain, idx in _PHYSICS_RULES
        if any(k in lower for k in keywords)
    ]
    if not pos:
        return "", ""
    return ", ".join(pos), PHYSICS_COMMON_NEGATIVE_ADDON


DetailedSceneMode = Literal["off", "auto", "on"]
StrengthMode = Literal["lite", "strong"]

# Crowd / multi-count / ensemble cues.
_GROUP_HINTS = (
    "group",
    "crowd",
    "team",
    "family",
    "soldiers",
    "knights",
    "ensemble",
    "cast of",
    "several ",
    "multiple ",
    "many ",
    "each ",
    "gathered",
    "standing together",
)

# ``five cats``, ``3 people``, etc.
_COUNT_PHRASE = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|pair|pairs)\b|\b\d{1,2}\s+(people|person|men|women|characters|children|kids|figures|objects|items|books|posters|soldiers|horses|cats|dogs|robots)\b",
    re.IGNORECASE,
)

_PEOPLE_HINT = re.compile(
    r"\b(person|people|man|woman|men|women|boy|girl|child|children|kid|kids|character|characters|portrait|crowd|knight|warrior|soldier|dancer|couple|family)\b",
    re.IGNORECASE,
)


def detailed_scene_warrants_boost(prompt: str) -> bool:
    """Conservative: busy or explicitly multi-entity prompts."""
    p = (prompt or "").strip()
    if not p:
        return False
    low = p.lower()
    if any(h in low for h in _GROUP_HINTS):
        return True
    if _COUNT_PHRASE.search(p):
        return True
    # Long tag-style prompts with several distinct clauses (avoid short boilerplate tags).
    segments = [s.strip() for s in p.split(",") if len(s.strip()) >= 16]
    if len(segments) >= 6:
        return True
    if " and " in low and len(p) > 100:
        return True
    return False


def extract_key_segments(prompt: str, *, max_segments: int = 12) -> list[str]:
    """Rough noun-phrases from comma-separated prompt chunks (for logging / UI)."""
    raw = [s.strip() for s in (prompt or "").replace("\n", ",").split(",") if s.strip()]
    # Drop very short noise tokens
    out = [s for s in raw if len(s) >= 4 and s.lower() not in {"best quality", "masterpiece", "detailed", "8k"}]
    return out[:max_segments]


def apply_detailed_scene_boost(
    prompt: str,
    mode: DetailedSceneMode,
    *,
    strength: StrengthMode = "lite",
) -> tuple[str, str]:
    """
    Return ``(augmented_prompt, negative_fragment)``.

    Pulls in:
      - entity separation / anti-identity-swap language
      - pose, weight, contact (ground plane)
      - anatomy packs when people are likely
      - physics / creature heuristics via existing research helpers
    """
    prompt = (prompt or "").strip()
    if mode == "off" or not prompt:
        return prompt, ""

    if mode == "auto" and not detailed_scene_warrants_boost(prompt):
        return prompt, ""

    sep_pos = (
        "spatially separated readable subjects, clear negative space or depth between figures and props, "
        "consistent scale relationships, foreground midground background readable"
    )
    sep_neg = (
        "merged subjects, melted silhouettes, conjoined torsos, props fused between characters, "
        "ambiguous which limb belongs to whom, scale doubling errors"
    )

    noun_pos = (
        "each principal noun in the description keeps its own palette, material, and silhouette; "
        "no attribute bleed between adjacent entities"
    )
    noun_neg = (
        "wrong outfit on wrong character, swapped hair or faces between people, duplicated same person "
        "with different labels, object A drawn with B's texture"
    )

    pose_lite_pos = (
        "plausible stance, weight supported by feet or seat, coherent center of mass, "
        "joints flex in natural ranges, feet meet ground plane or visible support"
    )
    pose_lite_neg = (
        "broken spine twist, floating feet, impossible lean without support, dislocated hips, "
        "limbs emerging from wrong torso, gravity-defying group pose"
    )

    pose_strong_pos = (
        pose_lite_pos + ", contact shadows under shoes and props, consistent cast shadow direction for the whole scene"
    )
    pose_strong_neg = pose_lite_neg + ", contradictory shadow directions, hovering furniture"

    if strength == "strong":
        pose_pos, pose_neg = pose_strong_pos, pose_strong_neg
    else:
        pose_pos, pose_neg = pose_lite_pos, pose_lite_neg

    pos_parts: list[str] = [sep_pos, noun_pos, pose_pos]
    neg_parts: list[str] = [sep_neg, noun_neg, pose_neg]

    if _PEOPLE_HINT.search(prompt):
        if strength == "strong":
            pos_parts.append(ANATOMY_POS_STRONG)
            neg_parts.append(ANATOMY_NEG_STRONG)
        else:
            pos_parts.append(ANATOMY_POS_LITE)
            neg_parts.append(ANATOMY_NEG_LITE)

    try:
        cpos, cneg = suggest_creature_prompt_addons(prompt, rating="auto")
        if cpos:
            pos_parts.append(cpos)
        if cneg:
            neg_parts.append(cneg)
    except Exception:
        pass

    try:
        ppos, pneg = suggest_physics_prompt_addons(prompt)
        if ppos:
            pos_parts.append(ppos)
        if pneg:
            neg_parts.append(pneg)
    except Exception:
        pass

    pos_merged = merge_csv_unique(*pos_parts)
    neg_merged = merge_csv_unique(*neg_parts)

    # Dedupe against prompt body for positives
    low = prompt.lower()
    pos_tokens = [t.strip() for t in pos_merged.split(",") if t.strip() and t.strip().lower() not in low]
    pos_final = ", ".join(pos_tokens[:24])  # cap length
    if pos_final:
        out_prompt = f"{prompt}, {pos_final}".strip().strip(",")
    else:
        out_prompt = prompt

    return out_prompt, neg_merged


@dataclass(frozen=True, slots=True)
class MultiInstanceAugment:
    """Returned by :func:`apply_multi_instance_preset`."""

    prompt_suffix: str
    negative_suffix: str
    min_candidates: int
    suggested_expected_count: int | None
    workflow_note: str


_PRESETS: dict[str, MultiInstanceAugment] = {
    "distinct_objects": MultiInstanceAugment(
        prompt_suffix=(
            "multiple separate framed or clearly spaced items, each with its own distinct "
            "graphic design, palette, and subject; readable separation between instances; "
            "no symmetry-driven duplication; frontal or mild perspective on each piece"
        ),
        negative_suffix=(
            "cloned identical posters, tiled repetition, mirrored duplicate artwork, "
            "merged faces across frames, copy-paste motifs, kaleidoscope symmetry, "
            "single repeating texture, melted together designs"
        ),
        min_candidates=8,
        suggested_expected_count=None,
        workflow_note=(
            "Use --pick-best combo_count + --multi-instance-count; --grid to compare; "
            "dissect/inpaint for per-poster control."
        ),
    ),
    "stacked_media": MultiInstanceAugment(
        prompt_suffix=(
            "orderly stack of volumes; each visible cover and spine belongs to a clearly "
            "different book with its own color blocking and title zone; believable "
            "foreshortening; page edges aligned; no smeared single rectangle of noise"
        ),
        negative_suffix=(
            "single illegible collage, fused covers, duplicated spine gibberish, melted "
            "stack, one blurry book repeated, mirrored spines, text soup"
        ),
        min_candidates=6,
        suggested_expected_count=None,
        workflow_note=("Name titles/colors per book; combo_count + text constraints; inpaint spine rows."),
    ),
    "turnaround_sheet": MultiInstanceAugment(
        prompt_suffix=(
            "character turnaround model sheet layout: labeled orthographic rows or columns "
            "for front view, three-quarter view, side view, back view; identical outfit "
            "and proportions in every panel; neutral studio lighting; flat neutral background"
        ),
        negative_suffix=(
            "different outfit per view, hairstyle changing between panels, mismatched shoes, "
            "asymmetric props swapping sides, mirrored clone artifacts, cropped missing views"
        ),
        min_candidates=8,
        suggested_expected_count=4,
        workflow_note=("Prefer --reference-image + strength; otherwise render panels separately and composite."),
    ),
    "panel_strip": MultiInstanceAugment(
        prompt_suffix=(
            "horizontal storyboard strip of clearly separated sequential panels "
            "(visible gutters or bezels); each panel shows a visually distinct beat; "
            "consistent line weight and lettering style across panels"
        ),
        negative_suffix=(
            "panels melting together, duplicated frame content, same screenshot repeated, ambiguous panel boundaries"
        ),
        min_candidates=8,
        suggested_expected_count=None,
        workflow_note=("Set --multi-instance-count = panel count; --composition-brief auto for legibility."),
    ),
    "group_portrait": MultiInstanceAugment(
        prompt_suffix=(
            "group shot with individually readable faces spaced apart; "
            "each person has distinct clothing, hairstyle, and body silhouette; "
            "no fused faces or shared jaws; proportional heads"
        ),
        negative_suffix=(
            "conjoined faces, fused heads, mirrored twins error, "
            "duplicate same person multiplied, anatomically merged limbs, "
            "extra limbs shared between subjects"
        ),
        min_candidates=8,
        suggested_expected_count=None,
        workflow_note=("Set --multi-instance-count = people; widen --width; combo_count aligns with head count."),
    ),
}


def describe_limitation_short() -> str:
    """One-line rationale for UX / stderr."""
    return (
        "Single-pass diffusion cannot 'remember' prior regions; multi-instance presets add "
        "structure, negatives, and candidate pressure — not true scene memory."
    )


def apply_multi_instance_preset(
    prompt: str,
    preset: str,
    *,
    user_expected_count: int = 0,
) -> tuple[str, str, int, int | None, str]:
    """Returns ``( augmented_prompt, negative_fragment, min_candidates, expected_count_hint, workflow_note )``."""
    prompt = (prompt or "").strip()
    key = (preset or "none").lower().strip()
    if key in ("none", "") or not prompt:
        return prompt, "", 1, None, ""

    aug = _PRESETS.get(key)
    if aug is None:
        return prompt, "", 1, None, ""

    suf = aug.prompt_suffix
    fp = suf[:80].lower()
    if fp not in (prompt or "").lower():
        out_prompt = f"{prompt}, {suf}".strip().strip(",")
    else:
        out_prompt = prompt

    ec: int | None = user_expected_count if user_expected_count > 0 else aug.suggested_expected_count
    neg = aug.negative_suffix

    return out_prompt, neg, aug.min_candidates, ec, aug.workflow_note


def multi_instance_auto_settings(preset_key: str) -> dict[str, Any]:
    """
    Companion defaults for ``--multi-instance-auto`` in ``sample.py``.

    Keys: ``expected_count_target`` (``people`` | ``objects``), ``pick_best``, ``composition_brief_boost``.
    """
    pk = (preset_key or "").lower().strip()
    people_presets = {"turnaround_sheet", "group_portrait"}
    return {
        "expected_count_target": ("people" if pk in people_presets else "objects"),
        "pick_best": "combo_count",
        "composition_brief_boost": True,
    }


def print_multi_instance_hints(preset_key: str, *, stream: IO[str] | None = None) -> None:
    """stderr checklist for tougher multi-instance generations."""
    out = stream or sys.stderr
    aug = _PRESETS.get((preset_key or "").lower().strip())
    aug_line = aug.workflow_note if aug else ""
    print(
        "\nMulti-instance checklist:\n"
        "  • Reference lock: --reference-image + strength for repeated identity.\n"
        "  • Regions: --dissect-refs + --auto-init-from-dissection → composite init/mask.\n"
        "  • Search: high --num, --pick-best combo_count + --multi-instance-count / --expected-count.\n"
        "  • Per-entity pose/physics: --detailed-scene-boost auto (or on) + --detailed-scene-strength.\n"
        "  • Chain: refine pass, img2img on crops, or external compositing.\n"
        + (f"  • Preset: {aug_line}\n" if aug_line else ""),
        file=out,
    )


# Extra negatives when multiple distinct identities/outfits must not bleed together.
MULTI_SUBJECT_NEGATIVE_EXTRAS: tuple[str, ...] = (
    "same outfit on both",
    "outfit swap",
    "clothes mixed between characters",
    "wrong outfit on wrong person",
    "identical twins when not intended",
    "cloned face on both",
    "merged bodies",
    "conjoined",
    "shared limbs between people",
    "pose copied onto wrong figure",
    "mirror duplicate pose error",
)

# Positives that encourage separability (used with composition_mode ``multi_character``).
MULTI_SUBJECT_POSITIVE_EXTRAS: tuple[str, ...] = (
    "each person has distinct clothing",
    "each figure has own pose",
    "readable silhouettes",
    "spatial separation between subjects",
)


def default_subject_labels(n: int) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return ["subject"]
    return [f"character {i + 1}" for i in range(n)]


def merge_character_sheet_positives(
    blocks: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> str:
    """
    Join multiple character-sheet positive strings with **labeled** segments so the
    text encoder sees separate subjects (not one blended description).
    """
    parts: list[str] = []
    raw = [b.strip() for b in blocks if b and str(b).strip()]
    if not raw:
        return ""
    if len(raw) == 1:
        return raw[0]
    lab = list(labels) if labels is not None else default_subject_labels(len(raw))
    while len(lab) < len(raw):
        lab.append(f"character {len(lab) + 1}")
    for i, chunk in enumerate(raw):
        label = lab[i].strip() if i < len(lab) else f"character {i + 1}"
        parts.append(f"({label}: {chunk})")
    return ", ".join(parts)


def multi_sheet_extra_negatives_csv() -> str:
    """Comma-separated negatives focused on outfit/face/pose confusion (multi-sheet scenes)."""
    return ", ".join(MULTI_SUBJECT_NEGATIVE_EXTRAS)


def _actor_tokens(actor: dict[str, Any], index: int) -> tuple[list[str], list[str]]:
    pos: list[str] = []
    neg: list[str] = []
    label = str(actor.get("id", f"actor_{index + 1}")).strip() or f"actor_{index + 1}"
    role = str(actor.get("role", "")).strip()
    anchor = str(actor.get("spatial_anchor", "") or actor.get("screen_position", "")).strip()
    if anchor:
        pos.append(f"{label} position: {anchor}")

    if role:
        pos.append(f"{label} role: {role}")
    else:
        pos.append(f"{label} subject")

    for k in (
        "identity",
        "appearance",
        "body",
        "wardrobe",
        "pose",
        "expression",
        "props",
        "style",
        "locks",
    ):
        pos.extend(str_list(actor.get(k)))

    neg.extend(str_list(actor.get("avoid")))
    return pos, neg


def _relation_tokens(rel: dict[str, Any]) -> tuple[list[str], list[str]]:
    pos: list[str] = []
    neg: list[str] = []
    a = str(rel.get("a", "")).strip()
    b = str(rel.get("b", "")).strip()
    kind = str(rel.get("kind", "")).strip()
    detail = str(rel.get("detail", "")).strip()
    if a and b and kind:
        base = f"{a} {kind} {b}"
        if detail:
            base = f"{base}, {detail}"
        pos.append(base)
    neg.extend(str_list(rel.get("avoid")))
    return pos, neg


def _compile_blueprint_dict(data: dict[str, Any], strength: float = 1.0) -> tuple[str, str]:
    pos: list[str] = []
    neg: list[str] = []

    # Global scene controls
    for key in (
        "subject",
        "scene",
        "composition",
        "camera",
        "lighting",
        "style",
        "color_script",
        "background",
        "objects",
        "constraints",
        "quality",
    ):
        pos.extend(str_list(data.get(key)))

    neg.extend(str_list(data.get("avoid")))

    # Actors
    for i, actor in enumerate(data.get("actors", []) or []):
        if isinstance(actor, dict):
            p, n = _actor_tokens(actor, i)
            pos.extend(p)
            neg.extend(n)

    # Relations / spatial constraints
    for rel in data.get("relations", []) or []:
        if isinstance(rel, dict):
            p, n = _relation_tokens(rel)
            pos.extend(p)
            neg.extend(n)

    # Optional scale directives
    for scale in data.get("scale_directives", []) or []:
        if isinstance(scale, str) and scale.strip():
            pos.append(scale.strip())

    # Optional anti-artifact directives
    if data.get("anti_artifacts", False):
        neg.extend(
            [
                "warped anatomy",
                "merged limbs",
                "floating objects",
                "inconsistent perspective",
                "duplicate face",
                "extra limbs",
            ]
        )

    # Strength controls how many high-signal scene controls we duplicate with emphasis.
    s = max(0.5, min(float(strength), 2.0))
    if s > 1.0:
        for tag in str_list(data.get("constraints"))[:6]:
            pos.append(f"({tag})")

    pos = dedupe_tags(pos)
    neg = dedupe_tags(neg)
    return ", ".join(pos), ", ".join(neg)


def load_scene_blueprint(path: str, strength: float = 1.0) -> tuple[str, str]:
    """
    Load scene blueprint JSON and return `(positive_additions, negative_additions)`.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"scene-blueprint not found: {p}")
    data = json_loads(p.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, dict):
        raise ValueError("scene-blueprint must be a JSON object")
    return _compile_blueprint_dict(data, strength=strength)


def compile_scene_blueprint_dict(data: dict[str, Any], strength: float = 1.0) -> tuple[str, str]:
    """
    Compile an in-memory scene blueprint dict into `(positive_additions, negative_additions)`.
    """
    if not isinstance(data, dict):
        raise ValueError("scene blueprint must be a dict")
    return _compile_blueprint_dict(data, strength=strength)


_ACTOR_HINTS = (
    "man",
    "woman",
    "girl",
    "boy",
    "person",
    "character",
    "warrior",
    "knight",
    "mage",
    "robot",
    "dragon",
    "cat",
    "dog",
    "car",
)

_CAMERA_HINTS = (
    "close-up",
    "close up",
    "wide shot",
    "full body",
    "portrait",
    "overhead",
    "low angle",
    "high angle",
    "cinematic",
)

_LIGHT_HINTS = ("rim light", "soft light", "hard light", "dramatic lighting", "sunset", "neon", "moonlight")

_RELATION_PATTERNS = (
    (re.compile(r"\bleft of\b", flags=re.IGNORECASE), "left_of"),
    (re.compile(r"\bright of\b", flags=re.IGNORECASE), "right_of"),
    (re.compile(r"\bin front of\b", flags=re.IGNORECASE), "in_front_of"),
    (re.compile(r"\bbehind\b", flags=re.IGNORECASE), "behind"),
    (re.compile(r"\bon top of\b", flags=re.IGNORECASE), "on_top_of"),
    (re.compile(r"\bholding\b", flags=re.IGNORECASE), "holding"),
    (re.compile(r"\bnear\b", flags=re.IGNORECASE), "near"),
)


def _extract_phrases(prompt: str) -> list[str]:
    parts = [p.strip() for p in str(prompt).split(",")]
    return [p for p in parts if p]


def _extract_actors(prompt: str, max_actors: int = 4) -> list[dict[str, Any]]:
    phrases = _extract_phrases(prompt)
    actors: list[dict[str, Any]] = []
    for ph in phrases:
        low = ph.lower()
        if any(h in low for h in _ACTOR_HINTS):
            actors.append(
                {
                    "id": f"actor_{len(actors) + 1}",
                    "role": ph,
                    "appearance": [ph],
                    "locks": ["consistent proportions", "clean silhouette"],
                }
            )
        if len(actors) >= max(1, int(max_actors)):
            break
    if not actors:
        actors.append(
            {
                "id": "actor_1",
                "role": "main subject",
                "appearance": ["clear primary silhouette"],
                "locks": ["stable anatomy"],
            }
        )
    return actors


def _extract_relations(prompt: str) -> list[dict[str, Any]]:
    low = str(prompt).lower()
    rels: list[dict[str, Any]] = []
    for rx, tag in _RELATION_PATTERNS:
        if rx.search(low):
            rels.append({"a": "actor_1", "kind": tag, "b": "actor_2" if "actor_2" in low else "subject_2"})
    return rels


def infer_shape_blueprint(prompt: str, max_actors: int = 4) -> dict[str, Any]:
    low = str(prompt).lower()
    phrases = _extract_phrases(prompt)
    camera = [p for p in phrases if any(h in p.lower() for h in _CAMERA_HINTS)]
    lighting = [p for p in phrases if any(h in p.lower() for h in _LIGHT_HINTS)]
    constraints = [
        "clear foreground/midground/background separation",
        "anatomically plausible proportions",
        "coherent perspective and horizon",
        "no merged limbs, no duplicate body parts",
    ]
    if "text" in low or "title" in low or "logo" in low:
        constraints.append("legible text with correct spelling")
    return {
        "subject": phrases[:2] or [prompt],
        "composition": ["strong silhouette readability", "shape-first composition"],
        "camera": camera or ["balanced camera framing"],
        "lighting": lighting or ["coherent single-key lighting"],
        "style": ["high-fidelity, intentional brushwork/texture"],
        "constraints": constraints,
        "actors": _extract_actors(prompt, max_actors=max_actors),
        "relations": _extract_relations(prompt),
        "anti_artifacts": True,
    }


def compile_shape_scaffold(prompt: str, strength: float = 1.0, max_actors: int = 4) -> tuple[str, str, dict[str, Any]]:
    blueprint = infer_shape_blueprint(prompt, max_actors=max_actors)
    pos, neg = compile_scene_blueprint_dict(blueprint, strength=strength)
    return pos, neg, blueprint


_STYLE_RE = re.compile(
    r"(?:^|,)\s*((?:by|art by|drawn by|painted by|in the style of|style of)\s+[^,]+|"
    r"[^,]*\s+style)\s*(?=,|$)",
    re.IGNORECASE,
)
_ATTIRE_RE = re.compile(
    r"\b("
    r"lingerie|bikini|swimsuit|dress|skirt|shirt|blouse|jacket|coat|hoodie|"
    r"armor|uniform|kimono|hanfu|qipao|suit|tie|stockings|thighhighs|"
    r"gloves|boots|heels|jewelry|necklace|choker|collar|leotard|bodysuit|"
    r"nude|naked|topless|bottomless|clothing|outfit|wear|attire"
    r")\b",
    re.IGNORECASE,
)
_POSE_RE = re.compile(
    r"\b("
    r"standing|sitting|kneeling|lying|reclining|walking|running|jumping|"
    r"looking at viewer|from behind|from side|profile|cowboy shot|full body|"
    r"upper body|portrait|close-up|dynamic pose|contrapposto|arms up|"
    r"crossed arms|hand on hip|leaning|squatting|all fours|spread"
    r")\b",
    re.IGNORECASE,
)
_PLACE_RE = re.compile(
    r"\b("
    r"indoors|outdoors|bedroom|bathroom|kitchen|office|classroom|street|"
    r"park|forest|beach|ocean|city|cyberpunk|neon|cafe|bar|club|studio|"
    r"white background|black background|simple background|detailed background|"
    r"room|interior|exterior|sky|sunset|night|day"
    r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PromptFacets:
    raw: str
    subject: list[str] = field(default_factory=list)
    attire: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)
    pose: list[str] = field(default_factory=list)
    place: list[str] = field(default_factory=list)
    nsfw: bool = False
    artist_hint: str = ""
    missing: list[str] = field(default_factory=list)
    search_queries: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_tags(prompt: str) -> list[str]:
    text = str(prompt or "").strip().strip("[]")
    if not text:
        return []
    if "," in text:
        return [t.strip() for t in text.split(",") if t.strip()]
    return [t.strip() for t in re.split(r"\s{2,}|\n+", text) if t.strip()]


def _is_nsfw(text: str) -> bool:
    return bool(
        re.search(
            r"\b(nsfw|nude|naked|lingerie|explicit|uncensored|xxx|porn|hentai)\b",
            text,
            re.IGNORECASE,
        )
    )


def decompose_prompt_facets(prompt: str) -> PromptFacets:
    """Heuristic facet split tuned for danbooru-style and short prose prompts."""
    raw = str(prompt or "").strip()
    facets = PromptFacets(raw=raw, nsfw=_is_nsfw(raw))
    tags = _split_tags(raw)

    for m in _STYLE_RE.finditer(raw):
        phrase = m.group(1).strip(" ,")
        if phrase and phrase.lower() not in {s.lower() for s in facets.style}:
            facets.style.append(phrase)

    artist = ""
    try:
        from config.styles import extract_style_from_text

        artist = str(extract_style_from_text(raw) or "").strip()
    except Exception:
        artist = ""
    if artist:
        facets.artist_hint = artist
        if artist.lower() not in {s.lower() for s in facets.style}:
            facets.style.append(f"{artist} style")

    for tag in tags:
        low = tag.lower()
        if any(s.lower() == low for s in facets.style):
            continue
        if "style" in low or low.startswith("by ") or "art by" in low:
            facets.style.append(tag)
            continue
        if _ATTIRE_RE.search(tag):
            facets.attire.append(tag)
            continue
        if _POSE_RE.search(tag):
            facets.pose.append(tag)
            continue
        if _PLACE_RE.search(tag):
            facets.place.append(tag)
            continue
        facets.subject.append(tag)

    cleaned: list[str] = []
    for t in facets.subject:
        if _ATTIRE_RE.search(t) or _POSE_RE.search(t) or _PLACE_RE.search(t):
            continue
        if "style" in t.lower():
            continue
        cleaned.append(t)
    facets.subject = cleaned or ([tags[0]] if tags else [])

    if not facets.pose:
        facets.missing.append("pose")
    if not facets.place:
        facets.missing.append("place")
    if not facets.style and not facets.artist_hint:
        facets.missing.append("style")

    subj = ", ".join(facets.subject[:6]) or "character"
    if facets.subject:
        facets.search_queries["subject"] = subj
    if facets.attire:
        attire = ", ".join(facets.attire[:4])
        facets.search_queries["attire"] = f"{subj}, {attire}"
        facets.search_queries["attire_only"] = attire
    if facets.style or facets.artist_hint:
        styl = facets.artist_hint or ", ".join(facets.style[:3])
        facets.search_queries["style"] = f"{styl} art style reference"
        facets.search_queries["artist"] = styl
    return facets


_OC_TRIGGER_RE = re.compile(
    r"\boriginal character\b|\bmy oc\b|\bnew oc\b|\bcreate (?:an|a) character\b|"
    r"\bcharacter design\b|\bdesign (?:an|a) oc\b|\binvent (?:an|a) character\b|\boc\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AutoOriginalCharacterProfile:
    name: str
    archetype: str
    visual_traits: str
    wardrobe: str
    silhouette: str
    color_motifs: str
    expression_sheet: str
    negative_block: str

    def to_prompt_block(self) -> str:
        bits = [
            f"original character {self.name}, consistent identity across frames",
            f"archetype {self.archetype.replace('_', ' ')}",
            f"signature traits: {self.visual_traits}",
            f"consistent wardrobe: {self.wardrobe}",
            f"silhouette lock: {self.silhouette}",
            f"color motif: {self.color_motifs}",
            f"expression sheet anchors: {self.expression_sheet}",
            "same face structure and hairstyle in every frame",
        ]
        return ", ".join([b for b in bits if b.strip()])


_FIRST_NAMES = (
    "Astra",
    "Nia",
    "Kairo",
    "Sora",
    "Riven",
    "Lyra",
    "Vega",
    "Kael",
    "Mira",
    "Noel",
    "Tali",
    "Rune",
)
_LAST_NAMES = (
    "Vale",
    "Ashen",
    "Kestrel",
    "Morrow",
    "Locke",
    "Voss",
    "Drake",
    "Rowan",
    "Quill",
    "Sterling",
)

_ARCHETYPES = (
    "shonen_lead",
    "cool_rival",
    "mentor",
    "antihero",
    "magical_girl",
    "noir_detective",
    "space_pilot",
)

_ARCHETYPE_KEYWORDS: dict[str, str] = {
    "noir_detective": "detective noir trench mystery crime",
    "space_pilot": "space sci-fi pilot spaceship mecha cyberpunk",
    "magical_girl": "magical idol sparkle transformation wand",
    "mentor": "mentor teacher master sage veteran",
    "antihero": "antihero rogue assassin dark grim",
    "cool_rival": "rival elite duelist prodigy",
    "shonen_lead": "hero adventure battle protagonist",
}

_STYLE_ARCHETYPE_KEYWORDS: dict[str, str] = {
    "space_pilot": "anime 3d game sci-fi cyberpunk mecha pbr unreal",
    "noir_detective": "noir ink crosshatch detective",
    "magical_girl": "idol shoujo magical pastel vtuber",
    "cool_rival": "seinen rival edgy dark",
    "shonen_lead": "shonen battle hero action",
}

_TRAITS = (
    "short asymmetric hair",
    "silver streaked bangs",
    "sharp amber eyes",
    "soft heterochromia",
    "cheek scar",
    "ear cuff and utility comms",
    "freckles across nose bridge",
    "distinctive brow shape",
    "signature glove design",
    "small beauty mark under eye",
)

_WARDROBE_BY_ARCHETYPE: dict[str, tuple[str, ...]] = {
    "space_pilot": ("flight jacket with utility harness", "reinforced boots and tactical gloves"),
    "noir_detective": ("long trench coat and waistcoat", "fedora-inspired headwear with leather holster"),
    "magical_girl": ("iconic transformation outfit with ribbon motifs", "knee-high boots and emblem accessory"),
    "mentor": ("layered coat with ceremonial accents", "minimalist robe and signature pendant"),
    "antihero": ("high-collar asymmetrical coat", "combat boots with utility straps"),
    "cool_rival": ("tailored jacket with geometric trim", "sleek gloves and metallic accessories"),
    "shonen_lead": ("sporty jacket with bold emblem", "battle-ready belt and fingerless gloves"),
}

_SILHOUETTE_BY_ARCHETYPE: dict[str, str] = {
    "space_pilot": "broad upper silhouette with tapered lower body",
    "noir_detective": "long vertical silhouette with coat-tail rhythm",
    "magical_girl": "clear hourglass silhouette with iconic accessory shapes",
    "mentor": "stable triangular silhouette with grounded stance",
    "antihero": "angular asymmetrical silhouette with sharp shoulder cues",
    "cool_rival": "tall narrow silhouette with precise edge language",
    "shonen_lead": "dynamic V-shape torso with athletic stance",
}

_COLOR_MOTIFS = (
    "teal and orange accents on dark neutral base",
    "crimson and charcoal with silver highlights",
    "cobalt and white with neon accent details",
    "lavender and gold with soft pastel support",
    "forest green and bronze with warm shadows",
    "black and ivory with one signature accent color",
)

_EXPRESSIONS_BY_ARCHETYPE: dict[str, str] = {
    "space_pilot": "confident smirk, focused glare, determined command shout",
    "noir_detective": "skeptical glance, dry half-smile, intense confrontation stare",
    "magical_girl": "gentle smile, determined shout, radiant heroic expression",
    "mentor": "calm smile, stern warning, reflective concern",
    "antihero": "cold stare, restrained smirk, explosive battle shout",
    "cool_rival": "composed stare, contempt smirk, competitive grin",
    "shonen_lead": "energetic grin, battle shout, resilient determined face",
}

_DEFAULT_NEGATIVE = (
    "identity drift, inconsistent face structure, hairstyle changing between shots, "
    "outfit inconsistency, color motif drift, random extra accessories"
)

_STYLE_TRAIT_BOOSTS: dict[str, tuple[str, ...]] = {
    "anime 3d game": ("clean anime face planes", "stylized topology-friendly hair silhouette"),
    "ink": ("ink-friendly shape breakup", "strong contour readability under monochrome"),
    "oil": ("painterly facial plane transitions", "brush-character-friendly hair masses"),
}


def prompt_requests_original_character(prompt: str) -> bool:
    p = str(prompt or "").strip()
    if not p:
        return False
    return bool(_OC_TRIGGER_RE.search(p))


def _seeded_rng(prompt: str, seed: int = 0) -> Random:
    h = hashlib.sha256(f"{prompt}|{int(seed)}".encode("utf-8", errors="ignore")).hexdigest()
    return Random(int(h[:16], 16))


def _pick_archetype(prompt: str, style_context: str, rng: Random) -> str:
    p = f"{str(prompt or '').lower()} {str(style_context or '').lower()}"
    for archetype, kw_blob in _ARCHETYPE_KEYWORDS.items():
        if any(k in p for k in kw_blob.split()):
            return archetype
    for archetype, kw_blob in _STYLE_ARCHETYPE_KEYWORDS.items():
        if any(k in p for k in kw_blob.split()):
            return archetype
    return _ARCHETYPES[rng.randrange(len(_ARCHETYPES))]


def infer_auto_original_character(
    prompt: str,
    *,
    seed: int = 0,
    style_context: str = "",
) -> AutoOriginalCharacterProfile | None:
    """
    Return a synthesized OC profile only when prompt asks for OC/character design.
    """
    p = str(prompt or "").strip()
    if not prompt_requests_original_character(p):
        return None
    rng = _seeded_rng(p, seed=seed)
    archetype = _pick_archetype(p, style_context, rng)
    first = _FIRST_NAMES[rng.randrange(len(_FIRST_NAMES))]
    last = _LAST_NAMES[rng.randrange(len(_LAST_NAMES))]
    picked_traits = list(rng.sample(_TRAITS, k=3))
    sc = str(style_context or "").lower()
    for key, extra in _STYLE_TRAIT_BOOSTS.items():
        if key in sc:
            picked_traits.append(extra[rng.randrange(len(extra))])
            break
    traits = ", ".join(picked_traits)
    wardrobe = ", ".join(rng.sample(_WARDROBE_BY_ARCHETYPE.get(archetype, _WARDROBE_BY_ARCHETYPE["shonen_lead"]), k=2))
    silhouette = _SILHOUETTE_BY_ARCHETYPE.get(archetype, _SILHOUETTE_BY_ARCHETYPE["shonen_lead"])
    color_motifs = _COLOR_MOTIFS[rng.randrange(len(_COLOR_MOTIFS))]
    expression_sheet = _EXPRESSIONS_BY_ARCHETYPE.get(archetype, _EXPRESSIONS_BY_ARCHETYPE["shonen_lead"])
    return AutoOriginalCharacterProfile(
        name=f"{first} {last}",
        archetype=archetype,
        visual_traits=traits,
        wardrobe=wardrobe,
        silhouette=silhouette,
        color_motifs=color_motifs,
        expression_sheet=expression_sheet,
        negative_block=_DEFAULT_NEGATIVE,
    )
