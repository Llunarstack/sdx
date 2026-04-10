"""
**Sequential-art authenticity** — extra positive/negative prompt layers so book/comic output
reads as **human-produced** (craft cadence, print/ink honesty) rather than generic AI polish.

Complements ``prompt_lexicon`` humanize bundles; use together with ``--humanize-pack balanced``
and craft flags. Controlled by ``generate_book.py`` ``--book-authenticity`` / ``--book-authenticity-medium``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

# Aliases → canonical medium keys used below
_MEDIUM_ALIASES: Dict[str, str] = {
    "auto": "auto",
    "manga": "manga",
    "manhwa": "webtoon",
    "webtoon": "webtoon",
    "graphic_novel": "graphic_novel",
    "graphicnovel": "graphic_novel",
    "comic": "comic_us",
    "comic_us": "comic_us",
    "western": "comic_us",
    "us_comic": "comic_us",
    "illustration": "illustration",
    "children": "children",
    "kids": "children",
    "picture_book": "children",
    "storyboard": "storyboard",
    "novel_cover": "illustration",
    "other": "manga",
}

# Positives: medium → tier → fragment
_AUTH_POSITIVE: Dict[str, Dict[str, str]] = {
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
        "lite": (
            "vertical-scroll friendly shapes, restrained digital smoothing, readable silhouette at phone scale"
        ),
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

_AUTH_NEGATIVE: Dict[str, Dict[str, str]] = {
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
            "render-engine lighting on 2d comic, subsurface abuse, copy-paste eyes, "
            "waxy lips, procedural fabric noise"
        ),
    },
    "graphic_novel": {
        "lite": "over-smoothed painterly, digital mud in shadows",
        "standard": "AI painterly soup, uniform brush stamps, posterized value steps, uncanny hands",
        "strong": (
            "CGI sheen in painted comics, texture tiling, symmetrical faces across cast, "
            "stock photo integration seams"
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
    """Read only ``book_style`` from a visual-memory JSON (fast; no full validation)."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return str(data.get("book_style", "") or "").strip().lower()
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return ""


def normalize_medium(name: str) -> str:
    k = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if k in _MEDIUM_ALIASES:
        v = _MEDIUM_ALIASES[k]
        return v if v != "auto" else "manga"
    if k in _AUTH_POSITIVE:
        return k
    return "manga"


def resolve_effective_medium(
    *,
    medium: str,
    book_type: str = "",
    lexicon_style: str = "",
    visual_memory_book_style: str = "",
) -> str:
    m = (medium or "auto").lower().strip()
    if m != "auto":
        return normalize_medium(m)
    vm = (visual_memory_book_style or "").strip().lower()
    if vm:
        return normalize_medium(vm)
    ls = (lexicon_style or "").lower().strip()
    if "webtoon" in ls or "manhwa" in ls or ls in ("webtoon", "manhwa_color"):
        return "webtoon"
    if "graphic" in ls or ls in ("graphic_novel", "seinen", "editorial"):
        return "graphic_novel"
    if ls in ("western_comic", "us_comic", "comic"):
        return "comic_us"
    bt = (book_type or "").lower().strip()
    if bt == "comic":
        return "comic_us"
    if bt == "novel_cover":
        return "illustration"
    if bt == "storyboard":
        return "storyboard"
    if bt == "manga":
        return "manga"
    return "manga"


def resolve_authenticity_bundle(
    *,
    level: str,
    medium: str = "auto",
    book_type: str = "",
    lexicon_style: str = "",
    visual_memory_book_style: str = "",
) -> Dict[str, str]:
    """
    Return ``{"positive": str, "negative": str, "effective_medium": str}``.
    *level*: ``none`` | ``lite`` | ``standard`` | ``strong``.
    """
    lv = (level or "none").lower().strip()
    if lv in ("", "none", "off"):
        return {"positive": "", "negative": "", "effective_medium": ""}

    if lv in ("lite", "standard", "strong"):
        tier = lv
    elif lv in ("balanced", "medium"):
        tier = "standard"
    elif lv in ("max", "maximum", "heavy"):
        tier = "strong"
    else:
        tier = "standard"
    em = resolve_effective_medium(
        medium=medium,
        book_type=book_type,
        lexicon_style=lexicon_style,
        visual_memory_book_style=visual_memory_book_style,
    )
    pos_map = _AUTH_POSITIVE.get(em, _AUTH_POSITIVE["manga"])
    neg_map = _AUTH_NEGATIVE.get(em, _AUTH_NEGATIVE["manga"])
    pos = pos_map.get(tier, pos_map.get("standard", ""))
    neg = neg_map.get(tier, neg_map.get("standard", ""))
    # Cross-cutting human-craft line (light) always for standard+
    if tier == "standard":
        pos = merge_prompt_fragments(
            pos,
            "single consistent artist hand across the page, intentional staging not stock collage",
        )
    elif tier == "strong":
        pos = merge_prompt_fragments(
            pos,
            "convincing human art direction: one author's judgment calls on detail density and rest areas",
            "avoid template panel rhythm and duplicated background geometry",
        )
    return {"positive": pos.strip(), "negative": neg.strip(), "effective_medium": em}


def authenticity_preset_names() -> Tuple[str, ...]:
    """CLI / docs: valid ``--book-authenticity`` levels."""
    return ("none", "lite", "standard", "strong")
