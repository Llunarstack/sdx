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


MEDIUM_ALIAS_TERMS: Dict[str, Tuple[str, ...]] = {
    "digital painting": ("digital paint", "painted digitally", "procreate painting", "photoshop painting"),
    "concept art": ("concept key art", "keyframe concept", "environment concept art"),
    "pixel art": ("pixelart", "retro pixel", "sprite art"),
    "hard surface": ("hard-surface", "mecha hard surface", "mechanical hard-surface"),
    "archviz": ("architectural visualization", "arch viz", "interior visualization render"),
    "toon 3d": ("toon-shaded 3d", "cel-shaded 3d", "cartoon 3d render"),
    "street photography": ("street photo", "street documentary"),
    "sports photography": ("sports action photo", "action sports shot"),
    "wedding photo": ("wedding photography", "bridal photo", "ceremony photo shoot"),
    "food photography": ("food photo", "culinary photography", "editorial food shot"),
    "black and white film": ("bw film photo", "black-and-white film", "monochrome analog photo"),
    "ink wash": ("sumi-e", "sumi e", "brush ink wash"),
    "storyboard sketch": ("animatic storyboard", "pencil storyboard", "film storyboard sketch"),
    "ceramic sculpture": ("ceramic art", "clay sculpture", "glazed ceramic sculpture"),
    "miniature diorama photo": ("miniature photo", "diorama photography", "tabletop diorama shot"),
}

_MEDIUM_ALIAS_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE), canonical)
    for canonical, aliases in MEDIUM_ALIAS_TERMS.items()
    for alias in aliases
)


def _expand_medium_aliases(prompt: str) -> str:
    p = str(prompt or "").strip()
    if not p:
        return p
    found: List[str] = []
    for pat, canonical in _MEDIUM_ALIAS_PATTERNS:
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
    MediumSpec(
        id="tempera_fresco",
        keywords=("tempera", "fresco", "egg tempera", "wall painting"),
        positive_hints="matte mineral pigment feel, controlled historical brush layering, stable value grouping",
        negative_hints="glossy plastic finish, random texture noise, inconsistent surface response",
    ),
    MediumSpec(
        id="ink_wash_calligraphy",
        keywords=("ink wash", "sumi-e", "calligraphy brush", "brush ink"),
        positive_hints="intentional brush-pressure rhythm, controlled dilution transitions, elegant negative-space composition",
        negative_hints="muddy gray soup, chaotic stroke thickness, accidental blot artifacts",
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
    MediumSpec(
        id="anime_digital_render",
        keywords=("anime render", "anime illustration", "cel shading", "toon shading", "anime style"),
        positive_hints="stable anime facial grammar, controlled cel-value breakup, clean contour readability",
        negative_hints="anime-realism style drift, muddy cel boundaries, inconsistent eye/line conventions",
    ),
    MediumSpec(
        id="ui_icon_vector",
        keywords=("ui icon", "icon design", "logo mark", "minimal vector", "flat icon"),
        positive_hints="crisp vector silhouette logic, scalable shape clarity, consistent corner and stroke rules",
        negative_hints="blurry edges, uneven stroke widths, accidental texture bleed into flat design",
    ),
    MediumSpec(
        id="matte_concept_dev",
        keywords=("visual development", "color script", "concept sheet", "keyframe art", "lookdev"),
        positive_hints="strong composition intent, iterative design readability, coherent mood and color continuity",
        negative_hints="generic concept mush, unfocused detail clutter, conflicting scene language",
    ),
    # --- 3D / render-focused ---
    MediumSpec(
        id="hard_surface_3d",
        keywords=("hard surface", "mecha 3d", "mechanical design", "sci-fi machinery", "cad render"),
        positive_hints="clean bevel logic, articulated part readability, coherent material break-up",
        negative_hints="mushy panel seams, random greeble noise, impossible mechanical articulation",
    ),
    MediumSpec(
        id="archviz_3d",
        keywords=("archviz", "architectural visualization", "interior render", "exterior render", "real estate render"),
        positive_hints="vertical perspective discipline, believable global illumination, premium material plausibility",
        negative_hints="warped architecture lines, inconsistent interior lighting, implausible reflective behavior",
    ),
    MediumSpec(
        id="toon_3d",
        keywords=("toon 3d", "cartoon 3d", "stylized 3d render", "anime 3d"),
        positive_hints="shape-first stylization, clean toon-ramp transitions, readable silhouette priority",
        negative_hints="half-toon half-photoreal inconsistencies, noisy specular clutter, unstable contour logic",
    ),
    MediumSpec(
        id="storyboard_sketch",
        keywords=("storyboard sketch", "storyboard", "animatic", "thumbnail storyboard", "shot sketch"),
        positive_hints="cinematic shot readability, clear staging arrows and beat intent, disciplined perspective shorthand",
        negative_hints="ambiguous shot language, cluttered line noise, weak panel-to-panel readability",
    ),
    MediumSpec(
        id="ceramic_sculpture",
        keywords=("ceramic sculpture", "ceramic art", "clay sculpture", "glazed ceramic", "stoneware sculpture"),
        positive_hints="material-true clay surface response, believable handcrafted asymmetry, coherent glaze behavior",
        negative_hints="plastic synthetic surface, impossible glaze reflections, uniform machine-perfect form",
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
    MediumSpec(
        id="sports_action_photo",
        keywords=("sports photo", "action photo", "sports photography", "match photo", "stadium photo"),
        positive_hints="decisive action timing, subject isolation under motion, realistic lens compression behavior",
        negative_hints="frozen mannequin motion, smeared limbs, unrealistic shutter-motion artifacts",
        is_photography=True,
    ),
    MediumSpec(
        id="wedding_event_photo",
        keywords=("wedding photo", "event photo", "ceremony photo", "bridal portrait", "reception photo"),
        positive_hints="natural candid emotion capture, flattering skin tone handling, coherent event lighting balance",
        negative_hints="plastic skin retouch, overblown highlights, inconsistent white-balance across subjects",
        is_photography=True,
    ),
    MediumSpec(
        id="food_product_photo",
        keywords=("food photo", "food photography", "product photo", "catalog photo", "commercial product"),
        positive_hints="material appetizing realism, controlled studio highlights, clean composition for commercial read",
        negative_hints="greasy glare overload, texture mush, physically implausible product reflections",
        is_photography=True,
    ),
    MediumSpec(
        id="film_bw_photo",
        keywords=("black and white film", "monochrome film", "35mm film", "analog photo", "film grain"),
        positive_hints="monochrome tonal discipline, natural film-like grain cadence, cinematic contrast control",
        negative_hints="flat gray tonality, fake digital crunch, inconsistent grain/noise layering",
        is_photography=True,
    ),
    MediumSpec(
        id="miniature_diorama_photo",
        keywords=("miniature diorama photo", "miniature photo", "diorama photography", "tabletop miniature", "scale model photo"),
        positive_hints="convincing scale cues, controlled macro depth behavior, realistic miniature lighting ratios",
        negative_hints="scale-breaking blur behavior, toy-like plastic highlights, inconsistent depth-of-field physics",
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

COLOR_THEORY_CORE_POS = (
    "strong value grouping, intentional warm-cool balance, controlled saturation hierarchy, "
    "clear primary-secondary-accent color roles, hue-shifted lights and shadows for material depth"
)
COLOR_THEORY_CORE_NEG = (
    "random palette drift, uncontrolled saturation clipping, value-muddy focal hierarchy, "
    "flat local color without temperature variation"
)

TRADITIONAL_COLOR_MIX_POS = (
    "pigment-aware color mixing, transparent glaze layering, subtractive color behavior, "
    "edge-softening through wet-in-wet transitions"
)
TRADITIONAL_COLOR_MIX_NEG = (
    "digital plastic gradients in traditional media, impossible pigment behavior, chalky muddy overmixing"
)

DIGITAL_2D_RENDER_POS = (
    "clean gradient control, blend-mode discipline, atmospheric perspective color shift by depth, "
    "rim/bounce light color separation, readable shadow family temperature control"
)
DIGITAL_2D_RENDER_NEG = (
    "banded gradients, uncontrolled overlay-color wash, noisy blend-mode artifacts, over-smoothed value planes"
)

RENDER_3D_PBR_POS = (
    "physically based shading logic, coherent roughness/metalness response, global-illumination bounce color, "
    "subsurface scattering where appropriate, filmic tone mapping with preserved highlight rolloff"
)
RENDER_3D_PBR_NEG = (
    "inconsistent BRDF response, broken roughness-metal balance, plastic fake reflections, "
    "non-physical shadow color contamination, clipped highlights with no rolloff"
)

PHOTO_COLOR_GRADE_POS = (
    "photographic color pipeline consistency, scene-referred exposure logic, highlight/shadow color separation, "
    "natural white-balance control, restrained cinematic grade"
)
PHOTO_COLOR_GRADE_NEG = (
    "overprocessed HDR look, crushed blacks with neon highlights, white-balance instability, color-channel clipping"
)


def _color_render_fragments(spec_ids: Sequence[str]) -> Tuple[str, str]:
    """
    Add medium-aware color theory / shading / rendering guidance.
    """
    ids = set(spec_ids)
    pos_parts: List[str] = [COLOR_THEORY_CORE_POS]
    neg_parts: List[str] = [COLOR_THEORY_CORE_NEG]

    has_3d = bool(ids & {"hard_surface_3d", "archviz_3d", "toon_3d", "stylized_game_texture"})
    has_photo = any(i.endswith("_photo") or "photo" in i for i in ids)
    has_traditional = bool(ids & {"oil_painting", "watercolor", "gouache_acrylic", "pastel_crayon", "tempera_fresco", "ink_wash_calligraphy"})
    has_digital_2d = bool(ids & {"digital_painting", "concept_matte", "vector_flat", "pixel_art", "anime_digital_render", "matte_concept_dev"})

    if has_traditional:
        pos_parts.append(TRADITIONAL_COLOR_MIX_POS)
        neg_parts.append(TRADITIONAL_COLOR_MIX_NEG)
    if has_digital_2d:
        pos_parts.append(DIGITAL_2D_RENDER_POS)
        neg_parts.append(DIGITAL_2D_RENDER_NEG)
    if has_3d:
        pos_parts.append(RENDER_3D_PBR_POS)
        neg_parts.append(RENDER_3D_PBR_NEG)
    if has_photo:
        pos_parts.append(PHOTO_COLOR_GRADE_POS)
        neg_parts.append(PHOTO_COLOR_GRADE_NEG)

    return merge_csv_unique(*pos_parts), merge_csv_unique(*neg_parts)


def detect_medium_ids(prompt: str, *, include_photography: bool) -> Tuple[str, ...]:
    if not prompt or not prompt.strip():
        return ()
    prompt = _expand_medium_aliases(prompt)
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
    spec_ids = tuple(s.id for s in specs)
    pos = merge_csv_unique(*(s.positive_hints for s in specs))
    neg = merge_csv_unique(*(s.negative_hints for s in specs))
    cpos, cneg = _color_render_fragments(spec_ids)
    pos = merge_csv_unique(pos, cpos)
    neg = merge_csv_unique(neg, cneg)
    apos, aneg = _anatomy_fragments(prompt, anatomy_mode)
    return merge_csv_unique(pos, apos), merge_csv_unique(neg, aneg)

