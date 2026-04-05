"""Style-domain + artist/game reference guidance packs (auto-detect or full)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

__all__ = [
    "StyleGuidanceMode",
    "StyleSpec",
    "STYLE_SPECS",
    "STYLE_IDS",
    "ARTIST_REFERENCE_TAGS",
    "detect_style_ids",
    "merge_csv_unique",
    "style_guidance_fragments",
]

StyleGuidanceMode = Literal["none", "auto", "all"]


@dataclass(frozen=True)
class StyleSpec:
    id: str
    keywords: Tuple[str, ...]
    positive_hints: str
    negative_hints: str


def _word_re(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def _compile_keywords(keywords: Tuple[str, ...]) -> Tuple[re.Pattern, ...]:
    return tuple(_word_re(k) for k in keywords)


def _matches(prompt: str, patterns: Sequence[re.Pattern]) -> bool:
    return any(p.search(prompt) for p in patterns)


def merge_csv_unique(*chunks: str) -> str:
    seen = set()
    out: List[str] = []
    for chunk in chunks:
        if not chunk or not str(chunk).strip():
            continue
        for part in str(chunk).split(","):
            p = part.strip()
            if not p:
                continue
            k = p.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
    return ", ".join(out)


# Representative cross-medium style/artist/game tags for detection and conditioning.
ARTIST_REFERENCE_TAGS: Tuple[str, ...] = (
    # Traditional / modern painters
    "john singer sargent",
    "claude monet",
    "vincent van gogh",
    "rembrandt",
    "caravaggio",
    "edward hopper",
    "gustav klimt",
    "j.c. leyendecker",
    "norman rockwell",
    "alphonse mucha",
    # Concept / fantasy / sci-fi illustrators
    "greg rutkowski",
    "feng zhu",
    "syd mead",
    "moebius",
    "katsuhiro otomo",
    "yoshiaki kawajiri",
    "h.r. giger",
    "frank frazetta",
    "boris vallejo",
    "ian mccaig",
    # Anime / manga / game-adjacent
    "akira toriyama",
    "naoko takeuchi",
    "takehiko inoue",
    "hajime isayama",
    "makoto shinkai",
    "hayao miyazaki",
    "yoji shinkawa",
    "tetsuya nomura",
    "kohei horikoshi",
    "clamp",
    # Western comics / illustration
    "jim lee",
    "alex ross",
    "frank miller",
    "mike mignola",
    "jack kirby",
    "moebius style",
    # Studios / game style anchors
    "studio ghibli",
    "pixar",
    "disney",
    "riot games",
    "blizzard style",
    "fortnite style",
    "valorant style",
    "overwatch style",
    "genshin impact style",
    "zelda wind waker style",
)


STYLE_SPECS: Tuple[StyleSpec, ...] = (
    StyleSpec(
        id="anime_manga",
        keywords=("anime", "manga", "shonen", "shoujo", "seinen", "isekai", "light novel", "visual novel"),
        positive_hints="consistent anime style language, clean silhouette readability, controlled cel/value hierarchy",
        negative_hints="style drift between painterly and anime regions, inconsistent eye/line conventions",
    ),
    StyleSpec(
        id="western_comic_graphic_novel",
        keywords=("comic book", "graphic novel", "western comic", "superhero comic", "inked comic", "panel art"),
        positive_hints="panel-first storytelling composition, controlled ink-weight rhythm, readable action silhouettes",
        negative_hints="muddy panel readability, inconsistent inking cadence, over-rendered clutter",
    ),
    StyleSpec(
        id="editorial_children_book",
        keywords=("editorial illustration", "children's book", "picture book", "storybook", "book illustration"),
        positive_hints="clear narrative beats, shape-language clarity, friendly value grouping, print-friendly readability",
        negative_hints="overcomplex detail noise, weak focal hierarchy, inconsistent age-appropriate stylization",
    ),
    StyleSpec(
        id="concept_fantasy_sci_fi",
        keywords=("concept art", "fantasy art", "sci-fi art", "matte painting", "keyframe", "environment concept"),
        positive_hints="strong concept read, disciplined focal hierarchy, coherent worldbuilding motifs",
        negative_hints="generic design mush, conflicting visual motifs, weak narrative shape language",
    ),
    StyleSpec(
        id="game_stylized_3d",
        keywords=("stylized game art", "fortnite style", "valorant style", "overwatch style", "stylized 3d"),
        positive_hints="stylized form simplification, cohesive shape exaggeration, clear gameplay-readable silhouettes",
        negative_hints="half-real half-toon inconsistency, noisy materials that break gameplay readability",
    ),
    StyleSpec(
        id="game_realism_pbr",
        keywords=("pbr", "physically based", "aaa realism", "real-time render", "unreal engine", "photoreal game art"),
        positive_hints="physically plausible material response, coherent roughness-metalness logic, grounded contact lighting",
        negative_hints="plastic roughness errors, inconsistent BRDF cues, fake specular patches",
    ),
    StyleSpec(
        id="pixel_voxel_lowpoly",
        keywords=("pixel art", "voxel", "low poly", "retro game", "isometric pixel", "blocky style"),
        positive_hints="crisp shape language by medium constraints, consistent scale grammar, deliberate stylized simplification",
        negative_hints="mixed-resolution artifacts, inconsistent simplification level, blurry interpolation",
    ),
    StyleSpec(
        id="film_photo_language",
        keywords=("cinematic photo", "film still", "35mm", "kodak", "street photo", "fashion photo", "documentary"),
        positive_hints="photographic composition discipline, coherent lens and exposure language, believable tonal roll-off",
        negative_hints="overprocessed digital crunch, contradictory camera language, fake HDR halos",
    ),
)

STYLE_IDS: Tuple[str, ...] = tuple(s.id for s in STYLE_SPECS)
_STYLE_PATTERNS: Dict[str, Tuple[re.Pattern, ...]] = {s.id: _compile_keywords(s.keywords) for s in STYLE_SPECS}
_ARTIST_PATTERNS: Tuple[re.Pattern, ...] = tuple(_word_re(a) for a in ARTIST_REFERENCE_TAGS)


def detect_style_ids(prompt: str) -> Tuple[str, ...]:
    if not prompt or not prompt.strip():
        return ()
    out: List[str] = []
    for spec in STYLE_SPECS:
        if _matches(prompt, _STYLE_PATTERNS[spec.id]):
            out.append(spec.id)
    return tuple(out)


def _artist_reference_fragments(prompt: str, enabled: bool) -> Tuple[str, str]:
    if not enabled:
        return "", ""
    if _matches(prompt, _ARTIST_PATTERNS):
        return (
            "style-faithful motif consistency, coherent brush/line grammar across the whole frame",
            "style token drift, mixed incompatible artist signatures in one subject region",
        )
    return "", ""


def style_guidance_fragments(
    prompt: str,
    mode: StyleGuidanceMode,
    *,
    include_artist_refs: bool,
) -> Tuple[str, str]:
    m = (mode or "none").lower()
    if m == "none":
        ap, an = _artist_reference_fragments(prompt, include_artist_refs)
        return ap, an
    if m == "all":
        specs = list(STYLE_SPECS)
    else:
        specs = [s for s in STYLE_SPECS if s.id in detect_style_ids(prompt)]
    pos = merge_csv_unique(*(s.positive_hints for s in specs))
    neg = merge_csv_unique(*(s.negative_hints for s in specs))
    ap, an = _artist_reference_fragments(prompt, include_artist_refs)
    return merge_csv_unique(pos, ap), merge_csv_unique(neg, an)

