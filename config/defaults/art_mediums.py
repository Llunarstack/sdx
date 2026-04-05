"""Artist-first medium guidance packs for sampling and training captions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

__all__ = [
    "GuidanceMode",
    "AnatomyGuidanceMode",
    "MediumSpec",
    "MEDIUM_SPECS",
    "MEDIUM_IDS",
    "detect_medium_ids",
    "merge_csv_unique",
    "guidance_fragments",
]

GuidanceMode = Literal["none", "auto", "all"]
AnatomyGuidanceMode = Literal["none", "lite", "strong"]


@dataclass(frozen=True)
class MediumSpec:
    id: str
    keywords: Tuple[str, ...]
    positive_hints: str
    negative_hints: str
    is_photography: bool = False


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


MEDIUM_SPECS: Tuple[MediumSpec, ...] = (
    # --- Traditional ---
    MediumSpec(
        id="oil_painting",
        keywords=("oil painting", "oil paint", "impasto", "alla prima", "canvas oil"),
        positive_hints="intentional brush strokes, impasto texture, warm-cool color harmony, painterly edge control",
        negative_hints="plastic airbrush blending, textureless paint, muddy over-blended forms",
    ),
    MediumSpec(
        id="watercolor",
        keywords=("watercolor", "water colour", "wet-on-wet", "wash painting"),
        positive_hints="transparent washes, paper tooth, controlled blooms, layered glaze clarity",
        negative_hints="opaque plastic gradients, random noise stain, muddy watercolor bleed",
    ),
    MediumSpec(
        id="gouache_acrylic",
        keywords=("gouache", "acrylic", "acrylic paint", "poster paint"),
        positive_hints="matte opaque color blocks, decisive shape design, clean edge hierarchy",
        negative_hints="waxy CGI sheen, inconsistent opacity, uncontrolled edge noise",
    ),
    MediumSpec(
        id="graphite_charcoal_ink",
        keywords=("graphite", "charcoal", "ink drawing", "pen and ink", "crosshatching", "etching"),
        positive_hints="confident mark-making, controlled hatching direction, varied line weight and pressure",
        negative_hints="uniform line thickness, smudged anatomy, chaotic hatch noise",
    ),
    MediumSpec(
        id="pastel_crayon",
        keywords=("pastel", "oil pastel", "chalk pastel", "crayon"),
        positive_hints="soft pigment grain, directional blending, visible medium texture",
        negative_hints="over-smoothed surface, synthetic gradient plastic look",
    ),
    # --- Digital art ---
    MediumSpec(
        id="digital_painting",
        keywords=("digital painting", "digital art", "photoshop", "procreate", "clip studio", "krita", "paintover"),
        positive_hints="intentional digital brushwork, edge variety, clear focal rendering, coherent layer blending",
        negative_hints="ai soup smear, airbrushed plastic skin, muddy midtones, texture spam",
    ),
    MediumSpec(
        id="concept_matte",
        keywords=("concept art", "matte painting", "photobash", "environment design", "key art", "artstation"),
        positive_hints="strong design read, perspective consistency, unified light and color grade across elements",
        negative_hints="floating cutouts, scale mismatch, conflicting light direction, generic muddy forms",
    ),
    MediumSpec(
        id="vector_flat",
        keywords=("vector", "flat illustration", "app icon", "ui illustration", "infographic", "svg style"),
        positive_hints="clean vector geometry, consistent stroke rules, deliberate flat color silhouettes",
        negative_hints="wobbly anchors, fuzzy edges, accidental photoreal leakage, inconsistent stroke widths",
    ),
    MediumSpec(
        id="pixel_art",
        keywords=("pixel art", "pixelart", "sprite", "spritesheet", "8-bit", "16-bit", "retro game"),
        positive_hints="crisp pixel grid, disciplined limited palette, intentional dithering and cluster readability",
        negative_hints="subpixel blur, muddy gradients, inconsistent pixel scale, accidental anti-alias mush",
    ),
    MediumSpec(
        id="stylized_game_texture",
        keywords=("hand-painted texture", "stylized 3d", "game asset", "texture sheet", "albedo map", "low poly"),
        positive_hints="cohesive hand-painted texture flow, consistent texel density, readable albedo value design",
        negative_hints="random noisy albedo, swimming details, inconsistent brush scale across surfaces",
    ),
    # --- Photography / realism ---
    MediumSpec(
        id="portrait_photo",
        keywords=("portrait photo", "headshot", "studio portrait", "fashion portrait", "beauty photo"),
        positive_hints="natural skin pores, realistic facial asymmetry, coherent key-fill-rim lighting, true lens depth",
        negative_hints="plastic skin retouch, over-symmetry, dead eyes, inconsistent portrait lighting",
        is_photography=True,
    ),
    MediumSpec(
        id="street_documentary_photo",
        keywords=("street photography", "documentary photo", "photojournalism", "candid photo"),
        positive_hints="authentic candid timing, environmental storytelling, grounded perspective and lens behavior",
        negative_hints="staged mannequin poses, overprocessed HDR, unrealistic crowd anatomy",
        is_photography=True,
    ),
    MediumSpec(
        id="product_architecture_photo",
        keywords=("product photo", "product photography", "architecture photo", "interior photo", "real estate photo"),
        positive_hints="clean geometric perspective, physically plausible reflections, controlled material highlights",
        negative_hints="warped verticals, impossible reflections, floating objects, inconsistent shadows",
        is_photography=True,
    ),
    MediumSpec(
        id="macro_wildlife_photo",
        keywords=("macro photo", "wildlife photo", "nature photo", "telephoto", "close-up photo"),
        positive_hints="optical depth realism, sharp subject isolation, natural texture fidelity and detail falloff",
        negative_hints="uniform fake sharpness, texture hallucination, physically implausible depth",
        is_photography=True,
    ),
)

MEDIUM_IDS: Tuple[str, ...] = tuple(s.id for s in MEDIUM_SPECS)
_MEDIUM_PATTERNS: Dict[str, Tuple[re.Pattern, ...]] = {s.id: _compile_keywords(s.keywords) for s in MEDIUM_SPECS}

_PERSON_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    _word_re(k)
    for k in (
        "person",
        "people",
        "portrait",
        "man",
        "woman",
        "girl",
        "boy",
        "face",
        "hands",
        "full body",
        "character",
    )
)

ANATOMY_POS_LITE = "accurate anatomy, natural body proportions, coherent joints and limb attachment, natural hands"
ANATOMY_NEG_LITE = "bad anatomy, broken joints, noodle limbs, fused fingers, extra fingers"
ANATOMY_POS_STRONG = (
    "accurate anatomy and proportions, realistic skeletal-muscular structure, natural hand articulation and finger count, "
    "grounded body mechanics and weight distribution"
)
ANATOMY_NEG_STRONG = (
    "bad anatomy, broken or impossible joints, dislocated shoulders, twisted limbs, noodle limbs, malformed hands, "
    "extra fingers, fused fingers, floating feet"
)


def detect_medium_ids(prompt: str, *, include_photography: bool) -> Tuple[str, ...]:
    if not prompt or not prompt.strip():
        return ()
    out: List[str] = []
    for spec in MEDIUM_SPECS:
        if spec.is_photography and not include_photography:
            continue
        if _matches(prompt, _MEDIUM_PATTERNS[spec.id]):
            out.append(spec.id)
    return tuple(out)


def _anatomy_fragments(prompt: str, mode: AnatomyGuidanceMode) -> Tuple[str, str]:
    m = (mode or "none").lower()
    if m == "none":
        return "", ""
    if m == "strong":
        return ANATOMY_POS_STRONG, ANATOMY_NEG_STRONG
    # lite: only if people are present
    if _matches(prompt, _PERSON_PATTERNS):
        return ANATOMY_POS_LITE, ANATOMY_NEG_LITE
    return "", ""


def guidance_fragments(
    prompt: str,
    mode: GuidanceMode,
    *,
    include_photography: bool,
    anatomy_mode: AnatomyGuidanceMode = "none",
) -> Tuple[str, str]:
    m = (mode or "none").lower()
    if m == "none":
        return _anatomy_fragments(prompt, anatomy_mode)
    if m == "all":
        specs = [s for s in MEDIUM_SPECS if include_photography or not s.is_photography]
    else:
        specs = [s for s in MEDIUM_SPECS if s.id in detect_medium_ids(prompt, include_photography=include_photography)]
    pos = merge_csv_unique(*(s.positive_hints for s in specs))
    neg = merge_csv_unique(*(s.negative_hints for s in specs))
    apos, aneg = _anatomy_fragments(prompt, anatomy_mode)
    return merge_csv_unique(pos, apos), merge_csv_unique(neg, aneg)

