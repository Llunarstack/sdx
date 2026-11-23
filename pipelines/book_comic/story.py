"""Book prompt planning: text continuity, challenging content packs, style fusion/authenticity,
cross-page consistency clauses, visual memory and prompt intelligence.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments


def _first_text(spec: Mapping[str, Any], *keys: str) -> str:
    """First truthy value among *keys*, stringified and stripped."""
    for key in keys:
        value = spec.get(key)
        if value:
            return str(value).strip()
    return ""


def lettering_visual_memory_fragment(lettering: Mapping[str, Any]) -> str:
    """Build a fragment from visual-memory ``lettering`` object."""
    bits: list[str] = []
    for template, keys in (
        ("consistent speech balloon style: {}", ("balloon_style",)),
        ("hand-lettered typography mood: {}", ("typography_mood", "font_vibe")),
        ("integrated sfx lettering: {}", ("sfx_style",)),
        ("dialogue orthography lock: {}", ("orthography", "language_lock")),
        ("{}", ("extra",)),
    ):
        value = _first_text(lettering, *keys)
        if value:
            bits.append(template.format(value))
    if lettering.get("match_quoted_script"):
        bits.append("printed dialogue must match quoted script text exactly in balloons")
    return merge_prompt_fragments(*bits)


def text_continuity_clause(spec: Mapping[str, Any]) -> str:
    """
    Strong continuity for recurring on-page text, object labels, and chapter leitmotifs.

    Expected keys (all optional):

    - ``locked_phrases`` / ``must_include``: list[str] or semicolon str
    - ``object_labels``: list of ``{"id": "sword", "label": "EXCALIBUR"}`` for prop text
    - ``chapter_motto``: recurring title fragment
    - ``narration_caption_style``: e.g. "italic rectangular captions top-left"
    - ``strict_script``: if true, insist lettering matches provided wording
    """
    bits: list[str] = []
    if spec.get("strict_script"):
        bits.append("strict script fidelity: every visible word in balloons and captions matches the writer script")

    raw_p = spec.get("locked_phrases") or spec.get("must_include") or spec.get("must_include_phrases")
    phrases: list[str] = []
    if isinstance(raw_p, list):
        phrases = [str(x).strip() for x in raw_p if str(x).strip()]
    elif isinstance(raw_p, str) and raw_p.strip():
        phrases = [p.strip() for p in raw_p.split(";") if p.strip()]
    if phrases:
        blob = "; ".join(phrases[:12])
        tail = " (and same recurring wording elsewhere)" if len(phrases) > 12 else ""
        bits.append(f"recurring exact text on page when shown: {blob}{tail}")

    motto = _first_text(spec, "chapter_motto")
    if motto:
        bits.append(f"chapter leitmotif text: {motto}")

    cap = _first_text(spec, "narration_caption_style")
    if cap:
        bits.append(f"caption box discipline: {cap}")

    objs = spec.get("object_labels")
    if isinstance(objs, list):
        for o in objs:
            if not isinstance(o, dict):
                continue
            oid = _first_text(o, "id")
            lab = _first_text(o, "label")
            if oid and lab:
                bits.append(f"object {oid} always reads as printed text {lab!r} when labeled")
            elif lab:
                bits.append(f"recurring printed label: {lab}")

    ex = _first_text(spec, "extra")
    if ex:
        bits.append(ex)
    return merge_prompt_fragments(*bits)


# Keyed by visual_memory ``challenge_tags`` / consistency JSON tag names.
CHALLENGE_TAG_FRAGMENTS: dict[str, str] = {
    "mature_narrative_fidelity": (
        "mature-rated scenes follow the writer brief without arbitrary modesty drift, "
        "wardrobe or blocking changes that contradict the stated page intent"
    ),
    "surreal_weird": (
        "surreal or dream-logic staging reads intentional: coherent internal rules, "
        "stable focal hierarchy, readable silhouettes despite odd proportions"
    ),
    "non_human_morph": (
        "non-human or hybrid anatomy stays internally consistent across panels: "
        "same limb count, markings, material reads, and silhouette logic"
    ),
    "extreme_perspective": (
        "aggressive foreshortening and wide-angle framing with correct overlap and scale cues, "
        "ground plane reads clearly, figures stay anchored in space"
    ),
    "crowd_scale": (
        "dense crowd: many distinct readable figures, depth layering, no face cloning soup, "
        "clear staging of foreground vs background actors"
    ),
    "hands_heavy": (
        "hands-heavy composition: deliberate finger poses, correct thumb side, "
        "clean overlaps, props contacted with believable grip contact"
    ),
    "reflections_glass": (
        "mirrors, glass, and water reflections: consistent viewpoint, "
        "plausible flipped detail, no impossible duplicate geometry"
    ),
    "horror_mood": (
        "horror-adjacent illustration mood: readable dread through lighting and staging, "
        "no gratuitous gore emphasis unless the page prompt explicitly requests it"
    ),
    "technical_materials": (
        "difficult materials rendered coherently: chrome, wet skin, thin fabric folds, "
        "subsurface hints where appropriate, stable contact shadows"
    ),
}


def _tags(*keys: str) -> str:
    return merge_prompt_fragments(*(CHALLENGE_TAG_FRAGMENTS[k] for k in keys))


_PACK_POSITIVE: dict[str, str] = {
    "mature_coherence": _tags("mature_narrative_fidelity"),
    "surreal_weird": _tags("surreal_weird", "non_human_morph"),
    "technical_hard": _tags("extreme_perspective", "reflections_glass", "technical_materials"),
    "horror_mood": _tags("horror_mood"),
    "crowd_hands": _tags("crowd_scale", "hands_heavy"),
    "max": _tags(
        "mature_narrative_fidelity",
        "surreal_weird",
        "non_human_morph",
        "extreme_perspective",
        "crowd_scale",
        "hands_heavy",
        "reflections_glass",
        "technical_materials",
        "horror_mood",
    ),
}

CHALLENGE_PACK_CHOICES = frozenset({"none", *_PACK_POSITIVE})

CHALLENGE_NEGATIVE_ADDON = (
    "random unexplained censor bars, lens censorship blur, arbitrary black voids over anatomy, "
    "sanitized rewrite of the written scene, sudden modesty costume insertion"
)


def _pack_name(pack: str | None) -> str:
    return str(pack or "none").strip().lower()


def _tag_list(raw: Any) -> list[str]:
    """Accept a comma-separated string or a list of tag names."""
    if isinstance(raw, str):
        return raw.split(",")
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return []


def merge_challenge_tags(tags: Sequence[str]) -> str:
    """Map challenge tag names to lexicon fragments; unknown tags are skipped."""
    out: list[str] = []
    for raw in tags or []:
        key = str(raw or "").strip().lower().replace("-", "_")
        frag = CHALLENGE_TAG_FRAGMENTS.get(key)
        if frag:
            out.append(frag)
    return merge_prompt_fragments(*out)


def challenge_pack_positive(pack: str) -> str:
    """Resolve a named challenge pack to a positive prompt block ("" for none/unknown)."""
    return _PACK_POSITIVE.get(_pack_name(pack), "")


def challenge_pack_negative(pack: str) -> str:
    """Negative fragment for any real pack; "" for none/unknown."""
    return CHALLENGE_NEGATIVE_ADDON if _pack_name(pack) in _PACK_POSITIVE else ""


def challenging_content_from_mapping(block: Mapping[str, Any]) -> str:
    """
    JSON block shape::

        {"pack": "surreal_weird", "tags": ["hands_heavy"], "extra": "freeform"}

    *pack* uses the same names as ``--book-challenge-pack``. *tags* add CHALLENGE_TAG_FRAGMENTS keys.
    """
    if not block:
        return ""
    extra = str(block.get("extra", "") or "").strip()
    tags = _tag_list(block.get("tags") or block.get("challenge_tags"))
    return merge_prompt_fragments(challenge_pack_positive(block.get("pack")), merge_challenge_tags(tags), extra)


def visual_memory_challenge_clause(root: Mapping[str, Any]) -> str:
    """
    Read optional visual-memory keys:

    - ``book_challenge_pack``: same as CLI pack name
    - ``challenge_tags`` / ``challenge_tag``: list or comma string of tag keys
    - ``challenging_content``: object passed to :func:`challenging_content_from_mapping`
    - ``weird_character_notes`` / ``unusual_character_notes``: freeform string

    The mature-fidelity fragment is always included (once).
    """
    bits: list[str] = []
    ch = root.get("challenging_content")
    if isinstance(ch, dict):
        bits.append(challenging_content_from_mapping(ch))
    bits.append(challenge_pack_positive(root.get("book_challenge_pack")))
    bits.append(merge_challenge_tags(_tag_list(root.get("challenge_tags") or root.get("challenge_tag"))))

    mature = CHALLENGE_TAG_FRAGMENTS["mature_narrative_fidelity"]
    if mature.lower() not in merge_prompt_fragments(*bits).lower():
        bits.append(mature)

    for key in ("weird_character_notes", "unusual_character_notes"):
        w = str(root.get(key, "") or "").strip()
        if w:
            bits.append(
                f"unusual recurring character design locked across pages: {w}, "
                "same distinctive traits in every appearance unless the page prompt overrides"
            )
            break

    return merge_prompt_fragments(*bits)


# Short idiom anchors (avoid importing visual_memory to prevent cycles)
_IDIOM: dict[str, str] = {
    "manga": (
        "manga-native pacing: expressive eyes, screentone-friendly values, speed-line grammar, "
        "right-to-left flow where appropriate"
    ),
    "webtoon": "vertical-scroll webtoon readability: large shapes, soft painterly blends, thumb-scroll staging",
    "graphic_novel": "graphic-novel wash and ink fusion: painterly passages with disciplined sequential gutters",
    "comic_us": (
        "American comic construction: bold spot blacks, zip texture discipline, left-to-right gutters, "
        "heroic figure drawing"
    ),
    "illustration": "illustration-forward finish: plate clarity, editorial focal hierarchy",
    "manhwa": "full-color manhwa finish: clean color holds, soft gradients, web-first readability",
    "storyboard": "production storyboard linework, readable staging, thumbnail energy not final render",
}

_HARMONIZE = (
    "single unified art direction: one production pipeline blending both influences, "
    "no split-screen style clash, consistent line family across panels"
)

# Preset name -> (primary_key, secondary_key, extra_tail)
_FUSION_KEYS: dict[str, tuple[str, str, str]] = {
    "manga_comic": (
        "manga",
        "comic_us",
        "hybrid manga-comic book: Japanese figure expression with western ink holds and balloon grammar",
    ),
    "webtoon_manga": (
        "webtoon",
        "manga",
        "webtoon vertical color story with manga black-outline accents on key characters",
    ),
    "graphic_comic": (
        "graphic_novel",
        "comic_us",
        "painterly graphic novel page with comic-book spot black discipline in focal figures",
    ),
    "manhwa_western": (
        "manhwa",
        "comic_us",
        "full-color manhwa rendering with western panel rhythm and sfx lettering habits",
    ),
    "illustration_manga": (
        "illustration",
        "manga",
        "illustrated book plate polish with manga silhouette clarity and expressive faces",
    ),
}


_BOOK_TYPE_PRIMARY: dict[str, str] = {
    "manga": "manga",
    "comic": "comic_us",
    "novel_cover": "illustration",
    "storyboard": "storyboard",
}


def _key(name: str | None) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


def freeform_style_fusion(primary: str, secondary: str) -> str:
    """Fuse two canonical style keys (see visual-memory ``book_style`` names); "" if they match."""
    a = _key(primary) or "manga"
    b = _key(secondary)
    if not b or b == a:
        return ""
    pa = _IDIOM.get(a, _IDIOM["manga"])
    pb = _IDIOM.get(b, "")
    if not pb:
        return merge_prompt_fragments(
            pa,
            f"secondary influence blended throughout: {b.replace('_', ' ')}",
            _HARMONIZE,
        )
    return merge_prompt_fragments(
        f"primary idiom ({a}): {pa}",
        f"secondary idiom ({b}): {pb}",
        _HARMONIZE,
    )


def fusion_fragment_from_preset(preset: str) -> str:
    """Return merged prompt for a named hybrid, or empty if unknown/none."""
    row = _FUSION_KEYS.get((preset or "").strip().lower().replace("-", "_"))
    if not row:
        return ""
    primary, secondary, tail = row
    return merge_prompt_fragments(freeform_style_fusion(primary, secondary), tail)


def fusion_from_cli(*, preset: str, secondary: str, primary_book_style: str = "manga") -> str:
    """
    CLI resolver: *preset* wins when set; else *secondary* + *primary_book_style* from pipeline
    (e.g. ``--book-type`` / memory ``book_style``).
    """
    return fusion_fragment_from_preset(preset) or freeform_style_fusion(primary_book_style, secondary)


def primary_style_from_book_type(book_type: str) -> str:
    """Map ``generate_book`` ``--book-type`` to a fusion primary idiom."""
    return _BOOK_TYPE_PRIMARY.get((book_type or "").strip().lower(), "manga")


# Aliases -> canonical medium keys of _AUTH_POSITIVE; anything else falls back to "manga".
_MEDIUM_ALIASES: dict[str, str] = {
    "manhwa": "webtoon",
    "graphicnovel": "graphic_novel",
    "comic": "comic_us",
    "western": "comic_us",
    "us_comic": "comic_us",
    "kids": "children",
    "picture_book": "children",
    "novel_cover": "illustration",
}

_LEVEL_TIERS: dict[str, str] = {
    "lite": "lite",
    "standard": "standard",
    "balanced": "standard",
    "medium": "standard",
    "strong": "strong",
    "max": "strong",
    "maximum": "strong",
    "heavy": "strong",
}

# Medium-independent human-craft lines appended per tier.
_TIER_CRAFT_LINES: dict[str, tuple[str, ...]] = {
    "lite": (),
    "standard": ("single consistent artist hand across the page, intentional staging not stock collage",),
    "strong": (
        "convincing human art direction: one author's judgment calls on detail density and rest areas",
        "avoid template panel rhythm and duplicated background geometry",
    ),
}

# Positives: medium → tier → fragment
_AUTH_POSITIVE: dict[str, dict[str, str]] = {
    "manga": {
        "lite": (
            "convincing hand-inked manga line, natural taper on strokes, readable screentone dots "
            "without muddy gray soup"
        ),
        "standard": (
            "professional manga craft: intentional line weight hierarchy, controlled screentone "
            "gradation, clear black fills, believable hand correction marks, not vector-perfect curves"
        ),
        "strong": (
            "editorial manga finish with human pacing: varied sodegari and hatching rhythm, "
            "non-uniform tone buildup, organic small ink bleeds at stress points, faces feel drawn "
            "by one artist not cloned"
        ),
    },
    "webtoon": {
        "lite": "vertical-scroll friendly shapes, restrained digital smoothing, readable silhouette at phone scale",
        "standard": (
            "webtoon-native painting: soft gradients with hand-placed breaks, intentional rim-light "
            "variation, avoid plastic skin sheen, costume folds drawn not airbrushed flat"
        ),
        "strong": (
            "longform webtoon consistency: human color picking jitter, non-repeating fabric wrinkle logic, "
            "background props with slight perspective wobble, believable production art not render pass"
        ),
    },
    "graphic_novel": {
        "lite": "painterly sequential page, mixed edge softness, readable values in gutters",
        "standard": (
            "graphic novel reproduction quality: believable wash edges, ink dry-brush texture, "
            "page-sized value planning, characters grounded with cast shadow logic"
        ),
        "strong": (
            "award-trades craft: human brush economy, non-symmetrical detail distribution, "
            "lettering-adjacent art integration, no glossy 3d shader look in painted areas"
        ),
    },
    "comic_us": {
        "lite": "western comic inks, confident holds, clear spot blacks",
        "standard": (
            "American comic production: Kirby-class readable silhouettes, disciplined zip patterns, "
            "brush spot variance, lettering-friendly negative space, not sterile traced outlines"
        ),
        "strong": (
            "print comic authenticity: slight misregistration charm in flat colors, hand-inked feathering, "
            "dynamic figure drawing with weight shifts, avoid same-face hero clones across panels"
        ),
    },
    "illustration": {
        "lite": "illustration plate clarity, natural material read, single-image focus",
        "standard": (
            "trad-digital illustration hybrid: believable paper or canvas behavior, purposeful edge chaos, "
            "non-perfect symmetry in foliage and fabric"
        ),
        "strong": (
            "published book-plate quality: editorial composition, human color harmony quirks, "
            "avoid AI halo outlines and overcooked micro-detail noise"
        ),
    },
    "children": {
        "lite": "warm picture-book charm, simple readable shapes, gentle texture",
        "standard": (
            "children's book illustration warmth: hand-painted feel, slight shape asymmetry, "
            "inviting brushy edges, not glossy 3d toy render"
        ),
        "strong": (
            "classic picture-book authenticity: consistent character hand across spreads, "
            "organic color blocking, print-friendly texture without banding"
        ),
    },
    "storyboard": {
        "lite": "production storyboard clarity, fast readable poses, light construction honesty",
        "standard": (
            "working storyboard: gestural line confidence, camera intent obvious, loose but intentional "
            "perspective, not final-render polish"
        ),
        "strong": (
            "studio board craft: varied line energy, annotation-friendly framing, human thumbnail "
            "imperfections, avoid hyper-detailed AI board that reads like a still frame"
        ),
    },
}

_AUTH_NEGATIVE: dict[str, dict[str, str]] = {
    "manga": {
        "lite": "uniform vector manga lines, muddy screentone smear, same face every panel",
        "standard": (
            "AI manga soup, repetitive hatching pattern, waxy skin on ink pages, "
            "symmetry-cloned eyes, dead digital gradients in black areas"
        ),
        "strong": (
            "synthetic cel shading on ink work, tiling texture noise, rubber anatomy, "
            "overclean fills, duplicated speech balloon tails, stock screentone overlay"
        ),
    },
    "webtoon": {
        "lite": "plastic webtoon skin, flat airbrush only, duplicated gradient ramps",
        "standard": (
            "generic manhwa filter look, same hair shine on every character, "
            "background asset paste repetition, HDR bloom on flat comic color"
        ),
        "strong": (
            "render-engine lighting on 2d comic, subsurface abuse, copy-paste eyes, waxy lips, procedural fabric noise"
        ),
    },
    "graphic_novel": {
        "lite": "over-smoothed painterly, digital mud in shadows",
        "standard": "AI painterly soup, uniform brush stamps, posterized value steps, uncanny hands",
        "strong": (
            "CGI sheen in painted comics, texture tiling, symmetrical faces across cast, stock photo integration seams"
        ),
    },
    "comic_us": {
        "lite": "thin uniform outlines, flat color with no ink personality",
        "standard": "vector comic look, duplicated zip tones, same nose on every character",
        "strong": (
            "Marvel Legends plastic rendering in ink comic, dead-eyed symmetry, "
            "over-detailed micro-lines without macro design"
        ),
    },
    "illustration": {
        "lite": "generic stock illustration, oversaturated candy colors",
        "standard": "AI illustration texture tiling, halo outlines, waxy skin, cloned motifs",
        "strong": "midjourney sheen, over-sharpened detail, symmetrical faces, dead composition centering",
    },
    "children": {
        "lite": "3d render picture book, overly glossy, creepy perfect smiles",
        "standard": "AI children's art sameness, neon gradients, uncanny doll faces",
        "strong": "Pixar-still pasted as illustration, plastic toys, duplicated character poses",
    },
    "storyboard": {
        "lite": "over-rendered boards, cinematic lens spam without staging",
        "standard": "AI storyboard stills, hyperdetail in thumbnails, wrong focal hierarchy",
        "strong": "final-frame beauty passes pretending to be boards, duplicated camera angles",
    },
}


def peek_visual_memory_book_style(path: Path) -> str:
    """Read only ``book_style`` from a visual-memory JSON ("" when unreadable)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers JSONDecodeError and UnicodeDecodeError
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("book_style", "") or "").strip().lower()


def normalize_medium(name: str) -> str:
    """Canonical medium key for *name* (aliases resolved; unknown and ``auto`` -> ``manga``)."""
    k = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    k = _MEDIUM_ALIASES.get(k, k)
    return k if k in _AUTH_POSITIVE else "manga"


def resolve_effective_medium(
    *,
    medium: str,
    book_type: str = "",
    lexicon_style: str = "",
    visual_memory_book_style: str = "",
) -> str:
    """Explicit *medium* wins, then visual-memory ``book_style``, then lexicon style, then book type."""
    m = (medium or "auto").lower().strip()
    if m != "auto":
        return normalize_medium(m)
    vm = (visual_memory_book_style or "").strip().lower()
    if vm:
        return normalize_medium(vm)
    ls = (lexicon_style or "").lower().strip()
    if "webtoon" in ls or "manhwa" in ls:
        return "webtoon"
    if "graphic" in ls or ls in ("seinen", "editorial"):
        return "graphic_novel"
    if ls in ("western_comic", "us_comic", "comic"):
        return "comic_us"
    return primary_style_from_book_type(book_type)


def resolve_authenticity_bundle(
    *,
    level: str,
    medium: str = "auto",
    book_type: str = "",
    lexicon_style: str = "",
    visual_memory_book_style: str = "",
) -> dict[str, str]:
    """
    Return ``{"positive": str, "negative": str, "effective_medium": str}``.
    *level*: ``none``/``off`` | ``lite`` | ``standard`` (default for unknown) | ``strong``.
    """
    lv = (level or "none").lower().strip()
    if lv in ("", "none", "off"):
        return {"positive": "", "negative": "", "effective_medium": ""}

    tier = _LEVEL_TIERS.get(lv, "standard")
    em = resolve_effective_medium(
        medium=medium,
        book_type=book_type,
        lexicon_style=lexicon_style,
        visual_memory_book_style=visual_memory_book_style,
    )
    return {
        "positive": merge_prompt_fragments(_AUTH_POSITIVE[em][tier], *_TIER_CRAFT_LINES[tier]),
        "negative": _AUTH_NEGATIVE[em][tier],
        "effective_medium": em,
    }


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


MEMORY_VERSION = 1

BOOK_STYLE_HINTS: dict[str, str] = {
    "manga": (
        "Japanese manga readability: clear silhouette reads, screentone-friendly values, "
        "consistent line weight hierarchy face vs costume"
    ),
    "webtoon": (
        "vertical-scroll webtoon: bold readable shapes, generous negative space, "
        "soft gradients and clean separation between vertical tiers"
    ),
    "graphic_novel": (
        "graphic novel sequential art: painterly cohesion, cinematic framing, stable inking across panels"
    ),
    "comic_us": (
        "American comic style: confident figure drawing, bold holds, dynamic foreshortening, "
        "consistent costume construction"
    ),
    "illustration": (
        "illustrated book plates: single-image clarity, rich materials, stable character read "
        "when repeated across chapters"
    ),
    "other": "",
}

ENTITY_KIND_LABEL = {
    "character": "recurring character",
    "prop": "recurring prop/object",
    "object": "recurring prop/object",
    "vehicle": "recurring vehicle",
    "setting": "recurring setting element",
}

_STRUCTURE_LABELS = {
    "proportions": "body proportions",
    "head_to_body_ratio": "head-to-body ratio",
    "height_class": "relative height",
    "default_viewing_angle": "preferred camera / viewing angle",
    "scale_notes": "scale vs other cast",
    "silhouette": "silhouette read",
    "pose_language": "typical pose language",
}

_GLOBAL_KEYS = ("global_camera_language", "global_rendering", "global_color_script", "global_continuity")


def _text(value: Any) -> str:
    """Stripped string form of a JSON value; ``None`` becomes ``""`` rather than ``"None"``."""
    return "" if value is None else str(value).strip()


def _labelled(label: str, value: Any) -> str:
    s = _text(value)
    return f"{label}: {s}" if s else ""


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _normalize_entities(root: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = root.get("entities")
    if isinstance(raw, dict):
        return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in raw.items()}
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            eid = _text(item.get("id"))
            if eid:
                out[eid] = {k: v for k, v in item.items() if k != "id"}
        return out
    return {}


def _structure_clause(structure: Mapping[str, Any]) -> str:
    parts = [_labelled(label, structure.get(key)) for key, label in _STRUCTURE_LABELS.items()]
    parts.append(_text(structure.get("extra")))
    parts = [p for p in parts if p]
    return "structured design lock: " + "; ".join(parts) if parts else ""


def _entity_prompt(eid: str, data: Mapping[str, Any]) -> str:
    kind = _text(data.get("kind") or "character").lower()
    head = ENTITY_KIND_LABEL.get(kind, ENTITY_KIND_LABEL["character"])
    name = _text(data.get("display_name") or eid)
    structure = data.get("structure")
    return merge_prompt_fragments(
        f'{head} "{name}" (id {eid})',
        _labelled("established look", data.get("canonical_look")),
        _structure_clause(structure) if isinstance(structure, dict) else "",
        _labelled("line/rendering", data.get("line_and_rendering")),
        _labelled("locked wardrobe", data.get("costume_lock")),
        _labelled("signature items", data.get("signature_props")),
        _text(data.get("notes")),
    )


def _page_window(ov: Mapping[str, Any]) -> tuple[int, int] | None:
    """Inclusive ``(lo, hi)`` page range of an override; ``None`` when ``from_page`` is unparseable."""
    try:
        lo = int(ov.get("from_page", 0))
    except (TypeError, ValueError):
        return None
    try:
        hi = int(ov.get("to_page", lo))
    except (TypeError, ValueError):
        hi = lo
    return (lo, hi) if lo <= hi else (hi, lo)


def _merge_entity_for_page(base_entity: Mapping[str, Any], page_index: int) -> dict[str, Any]:
    """Apply every override covering *page_index*, earliest ``from_page`` first so later arcs win."""
    ent = copy.deepcopy(dict(base_entity))
    overrides = ent.get("page_overrides")
    if not isinstance(overrides, list):
        return ent
    active: list[tuple[int, dict[str, Any]]] = []
    for ov in overrides:
        window = _page_window(ov) if isinstance(ov, dict) else None
        if window is not None and window[0] <= page_index <= window[1]:
            active.append((window[0], ov))
    active.sort(key=lambda t: t[0])
    for _, ov in active:
        patch = ov.get("patch")
        if isinstance(patch, dict):
            ent = _deep_merge(ent, patch)
    return ent


@dataclass(slots=True)
class BookVisualMemory:
    """Loaded visual-memory document with helpers for prompts and updates."""

    root: dict[str, Any]
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> BookVisualMemory:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("visual memory JSON root must be an object")
        ver = int(raw.get("version", MEMORY_VERSION))
        if ver != MEMORY_VERSION:
            raise ValueError(f"unsupported visual memory version {ver!r} (expected {MEMORY_VERSION})")
        ent = _normalize_entities(raw)
        root = dict(raw)
        root["entities"] = ent
        return cls(root=root, entities=ent)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def entity_ids(self) -> list[str]:
        return sorted(self.entities)

    def effective_entity(self, entity_id: str, page_index: int) -> dict[str, Any]:
        if entity_id not in self.entities:
            raise KeyError(f"unknown entity {entity_id!r}")
        return _merge_entity_for_page(self.entities[entity_id], page_index)

    def apply_entity_page_patch(
        self,
        entity_id: str,
        *,
        from_page: int,
        to_page: int | None = None,
        patch: Mapping[str, Any] | None = None,
    ) -> None:
        """Remember a design change from *from_page* (to *to_page*, inclusive) onward.

        *patch* is deep-merged into the entity for prompts on pages in range; an unknown
        *entity_id* is created as a character.
        """
        ent = self.entities.setdefault(entity_id, {"kind": "character", "display_name": entity_id})
        overrides = ent.get("page_overrides")
        if not isinstance(overrides, list):
            overrides = ent["page_overrides"] = []
        overrides.append(
            {
                "from_page": int(from_page),
                "to_page": int(from_page if to_page is None else to_page),
                "patch": dict(patch or {}),
            }
        )
        self.root["entities"] = self.entities

    def add_page_patch(
        self,
        *,
        from_page: int,
        to_page: int | None = None,
        extra_prompt: str = "",
    ) -> None:
        s = (extra_prompt or "").strip()
        if not s:
            return
        patches = self.root.get("page_patches")
        if not isinstance(patches, list):
            patches = self.root["page_patches"] = []
        patches.append(
            {"from_page": int(from_page), "to_page": int(from_page if to_page is None else to_page), "extra_prompt": s}
        )

    def _global_fragment(self) -> str:
        style = _text(self.root.get("book_style") or "manga").lower()
        hint = BOOK_STYLE_HINTS.get(style, "")
        return merge_prompt_fragments(
            f"book visual language ({style}): {hint}" if hint else "",
            *(_text(self.root.get(key)) for key in _GLOBAL_KEYS),
        )

    def _lettering_fragment(self) -> str:
        letter = self.root.get("lettering")
        return lettering_visual_memory_fragment(letter) if isinstance(letter, dict) else ""

    def _style_mix_fragment(self) -> str:
        sm = self.root.get("style_mix")
        if not isinstance(sm, dict):
            return ""
        return fusion_from_cli(
            preset=_text(sm.get("preset")),
            secondary=_text(sm.get("secondary")),
            primary_book_style=str(self.root.get("book_style") or "manga"),
        )

    def _page_patch_fragments(self, page_index: int) -> str:
        patches = self.root.get("page_patches")
        if not isinstance(patches, list):
            return ""
        frags: list[str] = []
        for p in patches:
            window = _page_window(p) if isinstance(p, dict) else None
            if window is not None and window[0] <= page_index <= window[1]:
                frags.append(_text(p.get("extra_prompt")))
        return merge_prompt_fragments(*frags)

    def _entities_fragment(self, page_index: int | None) -> str:
        """Entity prompts for *page_index*; ``None`` means the cover (canonical design, no overrides)."""
        parts: list[str] = []
        for eid in sorted(self.entities):
            base = self.entities[eid]
            if page_index is None:
                merged = {k: v for k, v in base.items() if k != "page_overrides"}
            else:
                merged = _merge_entity_for_page(base, page_index)
            parts.append(_entity_prompt(eid, merged))
        return merge_prompt_fragments(*parts)

    def _compose(self, entities: str, *tail: str) -> str:
        return merge_prompt_fragments(
            self._global_fragment(),
            entities,
            visual_memory_challenge_clause(self.root),
            self._lettering_fragment(),
            self._style_mix_fragment(),
            _text(self.root.get("user_style_anchor")),
            *tail,
        )

    def prompt_fragment_for_cover(self) -> str:
        cov = self.root.get("cover")
        cover_bits = (
            (_text(cov.get("extra_prompt")), _text(cov.get("entity_treatment"))) if isinstance(cov, dict) else ()
        )
        return self._compose(self._entities_fragment(None), *cover_bits)

    def prompt_fragment_for_page(self, page_index: int) -> str:
        """0-based page index; merges globals, entities (with overrides), and run page patches."""
        return self._compose(
            self._entities_fragment(page_index),
            self._page_patch_fragments(page_index),
            _text(self.root.get("extra_prompt_all_pages")),
        )


def load_visual_memory(path: str | Path) -> BookVisualMemory:
    """Load from JSON path (see module docstring)."""
    return BookVisualMemory.load(path)


def approximate_token_estimate(text: str, *, chars_per_token: float = 3.2) -> int:
    """Rough upper-ish token count (English-ish prose; OK for budgeting, not exact)."""
    s = (text or "").strip()
    if not s:
        return 0
    return max(1, int(len(s) / max(0.5, float(chars_per_token))))


def composed_prompt_length_report(
    *,
    user_prompt: str,
    narration_prefix: str = "",
    consistency_block: str = "",
    panel_hint: str = "",
    rolling_context: str = "",
    visual_memory_fragment: str = "",
    oc_block: str = "",
) -> dict[str, int]:
    """Return per-block character counts and total for a composed book page prompt."""
    parts = {
        "narration_prefix": len((narration_prefix or "").strip()),
        "consistency_block": len((consistency_block or "").strip()),
        "panel_hint": len((panel_hint or "").strip()),
        "rolling_context": len((rolling_context or "").strip()),
        "visual_memory": len((visual_memory_fragment or "").strip()),
        "oc_block": len((oc_block or "").strip()),
        "user_prompt": len((user_prompt or "").strip()),
    }
    parts["total_chars"] = sum(parts.values())
    blocks = (
        narration_prefix,
        consistency_block,
        panel_hint,
        rolling_context,
        visual_memory_fragment,
        oc_block,
        user_prompt,
    )
    # Join only non-empty blocks so empty slots do not inflate the estimate with ", " separators.
    parts["approx_tokens"] = approximate_token_estimate(", ".join(b.strip() for b in blocks if b and b.strip()))
    return parts


@dataclass(slots=True)
class CastMentionResult:
    """Which cast labels were found in a page prompt (case-insensitive substring)."""

    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def soft_reminder_fragment(self) -> str:
        """Short positive fragment nudging unnamed cast into the page line (empty if none missing)."""
        if not self.missing:
            return ""
        names = ", ".join(self.missing[:6])
        tail = " (and others)" if len(self.missing) > 6 else ""
        return f"this page should visibly include: {names}{tail}"


def find_cast_mentions(page_prompt: str, cast_names: Sequence[str]) -> CastMentionResult:
    """Heuristic: a cast name is "mentioned" if it appears as a case-insensitive substring."""
    text = (page_prompt or "").lower()
    found: set[str] = set()
    missing: set[str] = set()
    for raw in cast_names:
        name = str(raw).strip()
        if name:
            (found if name.lower() in text else missing).add(name)
    return CastMentionResult(found=sorted(found), missing=sorted(missing))
