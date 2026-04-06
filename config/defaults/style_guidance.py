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


STYLE_ALIAS_TERMS: Dict[str, Tuple[str, ...]] = {
    "shonen": ("battle shonen", "shonen battle", "battle anime"),
    "shojo romance": ("shoujo romance", "shojo romance anime", "romance shoujo"),
    "seinen gritty": ("gritty seinen", "seinen dark", "mature seinen"),
    "mecha anime": ("mecha", "robot anime", "anime mecha"),
    "webtoon": ("vertical webtoon", "scroll comic", "vertical scroll comic"),
    "newspaper comic": ("comic strip", "newspaper strip", "daily strip"),
    "manga horror": ("horror manga", "psychological manga horror"),
    "baroque": ("baroque painting", "caravaggio style", "dramatic chiaroscuro art"),
    "impressionist": ("impressionism", "impressionist painting"),
    "surrealist": ("surrealism", "dreamlike surreal art"),
    "art nouveau": ("art-nouveau", "nouveau poster style"),
    "ukiyo-e": ("ukiyoe", "japanese woodblock print"),
    "octane render": ("octane", "octane-like render"),
    "eevee render": ("eevee", "blender eevee"),
    "archviz": ("architectural render", "arch viz", "archviz render"),
    "product cgi": ("product render", "commercial cgi product"),
    "clay render": ("clay shaded render", "matcap clay render"),
    "toon render": ("toon shaded render", "cel rendered 3d"),
    "mixed media": ("mixed-media", "media collage"),
    "paper cut": ("paper-cut", "papercut style"),
    "risograph": ("riso print", "riso style"),
    "screenprint": ("silk screen print", "silkscreen"),
    "retro pulp cover": ("pulp cover", "vintage pulp illustration"),
    "poster graphic": ("graphic poster", "poster design style"),
    "35mm": ("35 mm film", "35mm film look"),
    "film still": ("bw film look", "black and white film look", "monochrome film look"),
    "cinematic grade": ("teal orange grade", "teal-and-orange", "cinematic color grade"),
    "bleach bypass": ("bleach-bypass", "silver retention look"),
    "kodachrome": ("kodachrome film look", "kodachrome palette"),
    "diorama anime": ("anime diorama style", "miniature anime scene"),
}

_STYLE_ALIAS_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE), canonical)
    for canonical, aliases in STYLE_ALIAS_TERMS.items()
    for alias in aliases
)


def _expand_style_aliases(prompt: str) -> str:
    p = str(prompt or "").strip()
    if not p:
        return p
    found: List[str] = []
    for pat, canonical in _STYLE_ALIAS_PATTERNS:
        if pat.search(p):
            found.append(canonical)
    if not found:
        return p
    extras = ", ".join(sorted(set(found)))
    return f"{p}, {extras}"


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
    # Additional high-demand style anchors
    "yoji shinkawa style",
    "makoto shinkai style",
    "satoshi kon style",
    "clamp style",
    "studio trigger style",
    "arcanes style",
    "arcane style",
    "riot splash art",
    "fromsoftware style",
    "dark souls style",
    "bloodborne style",
    "elden ring style",
    "final fantasy style",
    "persona style",
    "vaporwave style",
    "art nouveau style",
    "ukiyo-e style",
    "baroque painting style",
    "impressionist style",
    "surrealist style",
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
    StyleSpec(
        id="anime_specializations",
        keywords=(
            "anime movie",
            "anime tv",
            "shonen battle",
            "shojo romance",
            "seinen gritty",
            "isekai fantasy",
            "mecha anime",
            "idol anime",
        ),
        positive_hints="substyle-consistent anime grammar, stable character-on-model identity, controlled emotional staging",
        negative_hints="mixed incompatible anime substyles, unstable facial grammar, inconsistent line/shading language",
    ),
    StyleSpec(
        id="comic_webtoon_substyles",
        keywords=(
            "web comic",
            "webtoon",
            "vertical scroll comic",
            "newspaper comic",
            "gag strip",
            "manga horror",
            "noir comic",
            "superhero modern",
        ),
        positive_hints="format-aware panel rhythm, dialogue-first readability, coherent ink/color language per substyle",
        negative_hints="format drift across panels, unreadable beat progression, conflicting inking/color conventions",
    ),
    StyleSpec(
        id="fine_art_movements",
        keywords=(
            "baroque",
            "rococo",
            "impressionist",
            "expressionist",
            "cubist",
            "surrealist",
            "art deco",
            "art nouveau",
            "ukiyo-e",
        ),
        positive_hints="movement-faithful motif discipline, coherent historical shape-language cues, style-consistent mark economy",
        negative_hints="historical-style token soup, mixed era contradictions, decorative overload without focal hierarchy",
    ),
    StyleSpec(
        id="render_pipeline_styles",
        keywords=(
            "octane render",
            "eevee render",
            "cycles render",
            "ray traced render",
            "archviz",
            "product cgi",
            "clay render",
            "toon render",
        ),
        positive_hints="renderer-aware material/lighting coherence, stable shading language, consistent post-process treatment",
        negative_hints="mixed renderer artifacts, contradictory material response, inconsistent light transport cues",
    ),
    StyleSpec(
        id="mixed_media_print_styles",
        keywords=(
            "mixed media",
            "collage",
            "paper cut",
            "risograph",
            "screenprint",
            "retro pulp cover",
            "poster graphic",
        ),
        positive_hints="print-aware composition hierarchy, deliberate medium texture interplay, clear silhouette-led communication",
        negative_hints="texture clutter, weak print readability, accidental medium conflicts reducing style clarity",
    ),
    StyleSpec(
        id="cinema_color_grading",
        keywords=(
            "cinematic grade",
            "teal orange",
            "bleach bypass",
            "kodachrome",
            "fujifilm look",
            "deakins lighting",
        ),
        positive_hints="intentional cinematic color separation, coherent filmic contrast mapping, stable highlight roll-off and shadow color logic",
        negative_hints="random LUT stacking, crushed blacks with clipped highlights, contradictory color grading intent",
    ),
    StyleSpec(
        id="anime_hybrid_rendering",
        keywords=(
            "diorama anime",
            "anime diorama",
            "2.5d anime",
            "anime 3d hybrid",
            "anime cinematic hybrid",
        ),
        positive_hints="cohesive 2d-3d anime hybrid grammar, stable character model fidelity, controlled painterly-composited depth cues",
        negative_hints="disjoint 2d/3d compositing seams, unstable character model in hybrid scenes, mismatched lighting language",
    ),
)

STYLE_IDS: Tuple[str, ...] = tuple(s.id for s in STYLE_SPECS)
_STYLE_PATTERNS: Dict[str, Tuple[re.Pattern, ...]] = {s.id: _compile_keywords(s.keywords) for s in STYLE_SPECS}
_ARTIST_PATTERNS: Tuple[re.Pattern, ...] = tuple(_word_re(a) for a in ARTIST_REFERENCE_TAGS)


def detect_style_ids(prompt: str) -> Tuple[str, ...]:
    if not prompt or not prompt.strip():
        return ()
    prompt = _expand_style_aliases(prompt)
    out: List[str] = []
    for spec in STYLE_SPECS:
        if _matches(prompt, _STYLE_PATTERNS[spec.id]):
            out.append(spec.id)
    return tuple(out)


def _artist_reference_fragments(prompt: str, enabled: bool) -> Tuple[str, str]:
    if not enabled:
        return "", ""
    prompt = _expand_style_aliases(prompt)
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

