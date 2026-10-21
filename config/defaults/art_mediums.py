"""Artist-first medium guidance packs for sampling and training captions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ._lexicon import alias_patterns, compile_keywords, expand_aliases, matches, merge_csv_unique, word_re

__all__ = [
    "GuidanceMode",
    "AnatomyGuidanceMode",
    "MediumSpec",
    "MEDIUM_SPECS",
    "detect_medium_ids",
    "guidance_fragments",
    "MEDIUM_FAMILIES",
    "MediumProfile",
    "detect_all_mediums",
    "detect_medium",
]

GuidanceMode = Literal["none", "auto", "all"]
AnatomyGuidanceMode = Literal["none", "lite", "strong"]


@dataclass(frozen=True, slots=True)
class MediumSpec:
    id: str
    keywords: tuple[str, ...]
    positive_hints: str
    negative_hints: str
    is_photography: bool = False


MEDIUM_ALIAS_TERMS: dict[str, tuple[str, ...]] = {
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

_MEDIUM_ALIAS_PATTERNS = alias_patterns(MEDIUM_ALIAS_TERMS)


MEDIUM_SPECS: tuple[MediumSpec, ...] = (
    # Traditional
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
    # Digital art
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
    # 3D / render-focused
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
    # Craft, print and art-movement mediums
    MediumSpec(
        id="spray_paint_street",
        keywords=("spray paint", "graffiti", "street art", "aerosol mural", "stencil art"),
        positive_hints="cap pressure variation, overspray halo, layered stencil alignment, concrete wall tooth",
        negative_hints="gaussian smear tags, incoherent drip physics, floating letters",
    ),
    MediumSpec(
        id="encaustic_wax",
        keywords=("encaustic", "hot wax painting", "wax medium"),
        positive_hints="molten wax texture, translucent wax layers, scraped reveal marks",
        negative_hints="plastic resin look, uniform gloss, digital gradient wax",
    ),
    MediumSpec(
        id="linocut_woodblock",
        keywords=("linocut", "woodblock print", "woodcut", "block print", "ukiyo-e print"),
        positive_hints="bold carved negative space, ink transfer pressure, registration charm",
        negative_hints="blurry print, random hatch without block logic",
    ),
    MediumSpec(
        id="stained_glass",
        keywords=("stained glass", "lead came", "glass mosaic window"),
        positive_hints="luminous transmitted color, lead line rhythm, cathedral light scatter",
        negative_hints="flat color without transmission, random polygon soup",
    ),
    MediumSpec(
        id="mosaic_tile",
        keywords=("mosaic", "tile mosaic", "byzantine mosaic", "tesserae"),
        positive_hints="consistent tessera scale, grout rhythm, material chip variation",
        negative_hints="noise texture pretending to be tiles, scale drift",
    ),
    MediumSpec(
        id="colored_pencil",
        keywords=("colored pencil", "polychromos", "prismacolor", "wax pencil"),
        positive_hints="directional pencil stroke, wax bloom, paper grain interaction",
        negative_hints="airbrush skin, digital smooth without tooth",
    ),
    MediumSpec(
        id="marker_copic",
        keywords=("copic marker", "alcohol marker", "marker illustration", "marker render"),
        positive_hints="streak-free blender transitions where intended, chisel tip stroke read, clean cap lines",
        negative_hints="muddy marker bleed everywhere, uncontrolled bleed",
    ),
    MediumSpec(
        id="scratchboard",
        keywords=("scratchboard", "scratch art", "white on black scratch"),
        positive_hints="sharp scratch highlight direction, engraved texture rhythm",
        negative_hints="random white noise scratches",
    ),
    MediumSpec(
        id="batik_fabric",
        keywords=("batik", "wax resist fabric", "tie dye", "shibori"),
        positive_hints="wax resist crackle, dye penetration boundaries, fabric fold dye pooling",
        negative_hints="flat procedural pattern, no fabric fold logic",
    ),
    MediumSpec(
        id="collage_mixed",
        keywords=("collage", "mixed media collage", "paper collage", "photomontage"),
        positive_hints="cut paper edge shadow, scale consistency, glue seam logic",
        negative_hints="floating clipart, scale mismatch chaos",
    ),
    MediumSpec(
        id="impressionist",
        keywords=("impressionist", "impressionism", "plein air", "monet style", "broken color"),
        positive_hints="broken color strokes, optical mixing, atmospheric perspective color shift",
        negative_hints="muddy overblend, uniform brush size everywhere",
    ),
    MediumSpec(
        id="expressionist",
        keywords=("expressionist", "expressionism", "bold gestural paint", "die brücke"),
        positive_hints="emotive stroke direction, exaggerated color temperature, raw mark energy",
        negative_hints="random glitch strokes without emotional axis",
    ),
    MediumSpec(
        id="baroque_oil",
        keywords=("baroque painting", "caravaggio", "chiaroscuro oil", "old master oil"),
        positive_hints="dramatic chiaroscuro, tenebrism, glazing depth, historical varnish warmth",
        negative_hints="flat comic shading, modern HDR on old master",
    ),
    MediumSpec(
        id="art_nouveau",
        keywords=("art nouveau", "mucha", "decorative illustration", "ornate floral border"),
        positive_hints="elegant contour rhythm, decorative flat pattern harmony, hair and fabric flow lines",
        negative_hints="generic fantasy portrait without decorative grammar",
    ),
    MediumSpec(
        id="ukiyo_e",
        keywords=("ukiyo-e", "japanese woodblock", "hokusai", "kuniyoshi"),
        positive_hints="flat plane separation, bold outline, limited palette, wave and cloud motifs",
        negative_hints="western shading on ukiyo-e, gradient mush",
    ),
    MediumSpec(
        id="airbrush_80s",
        keywords=("airbrush illustration", "80s airbrush", "fantasy airbrush"),
        positive_hints="controlled airbrush soft focus, chrome highlight discipline, retro gradient skies",
        negative_hints="modern AI plastic skin, random lens flare",
    ),
    MediumSpec(
        id="risograph",
        keywords=("risograph", "riso print", "duotone print"),
        positive_hints="misregistration charm, limited ink layer overlap, grainy halftone",
        negative_hints="full CMYK smooth print, no layer logic",
    ),
    MediumSpec(
        id="chalk_mural",
        keywords=("chalk mural", "sidewalk chalk", "pastel chalk mural"),
        positive_hints="chalk dust at edges, pavement tooth, temporary mural scale",
        negative_hints="digital chalk filter noise",
    ),
    # Photography / realism
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
        keywords=(
            "miniature diorama photo",
            "miniature photo",
            "diorama photography",
            "tabletop miniature",
            "scale model photo",
        ),
        positive_hints="convincing scale cues, controlled macro depth behavior, realistic miniature lighting ratios",
        negative_hints="scale-breaking blur behavior, toy-like plastic highlights, inconsistent depth-of-field physics",
        is_photography=True,
    ),
)

_MEDIUM_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {s.id: compile_keywords(s.keywords) for s in MEDIUM_SPECS}

_PERSON_PATTERNS: tuple[re.Pattern, ...] = tuple(
    word_re(k)
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


_TRADITIONAL_IDS = frozenset(
    {"oil_painting", "watercolor", "gouache_acrylic", "pastel_crayon", "tempera_fresco", "ink_wash_calligraphy"}
)
_DIGITAL_2D_IDS = frozenset(
    {"digital_painting", "concept_matte", "vector_flat", "pixel_art", "anime_digital_render", "matte_concept_dev"}
)
_RENDER_3D_IDS = frozenset({"hard_surface_3d", "archviz_3d", "toon_3d", "stylized_game_texture"})


def _color_render_fragments(specs: Sequence[MediumSpec]) -> tuple[str, str]:
    """Core color-theory pack plus the color/shading pack of every medium family present."""
    ids = {s.id for s in specs}
    pos_parts = [COLOR_THEORY_CORE_POS]
    neg_parts = [COLOR_THEORY_CORE_NEG]
    for present, pos, neg in (
        (ids & _TRADITIONAL_IDS, TRADITIONAL_COLOR_MIX_POS, TRADITIONAL_COLOR_MIX_NEG),
        (ids & _DIGITAL_2D_IDS, DIGITAL_2D_RENDER_POS, DIGITAL_2D_RENDER_NEG),
        (ids & _RENDER_3D_IDS, RENDER_3D_PBR_POS, RENDER_3D_PBR_NEG),
        (any(s.is_photography for s in specs), PHOTO_COLOR_GRADE_POS, PHOTO_COLOR_GRADE_NEG),
    ):
        if present:
            pos_parts.append(pos)
            neg_parts.append(neg)
    return merge_csv_unique(*pos_parts), merge_csv_unique(*neg_parts)


def _eligible(include_photography: bool) -> list[MediumSpec]:
    return [s for s in MEDIUM_SPECS if include_photography or not s.is_photography]


def _detect_specs(prompt: str, include_photography: bool) -> list[MediumSpec]:
    if not prompt or not prompt.strip():
        return []
    prompt = expand_aliases(prompt, _MEDIUM_ALIAS_PATTERNS)
    return [s for s in _eligible(include_photography) if matches(prompt, _MEDIUM_PATTERNS[s.id])]


def detect_medium_ids(prompt: str, *, include_photography: bool) -> tuple[str, ...]:
    """Ids of the mediums named in ``prompt`` (word-boundary keywords, aliases folded to canonical terms)."""
    return tuple(s.id for s in _detect_specs(prompt, include_photography))


def _anatomy_fragments(prompt: str, mode: AnatomyGuidanceMode) -> tuple[str, str]:
    """Anatomy packs only when the prompt has a person; a landscape gets no hands or joints text."""
    m = (mode or "none").lower()
    if m == "none" or not matches(prompt, _PERSON_PATTERNS):
        return "", ""
    if m == "strong":
        return ANATOMY_POS_STRONG, ANATOMY_NEG_STRONG
    return ANATOMY_POS_LITE, ANATOMY_NEG_LITE


def guidance_fragments(
    prompt: str,
    mode: GuidanceMode,
    *,
    include_photography: bool,
    anatomy_mode: AnatomyGuidanceMode = "none",
) -> tuple[str, str]:
    """``(positive, negative)`` medium hints + color/shading packs + optional anatomy pack.

    ``mode="all"`` uses every eligible medium, any other value except ``"none"`` detects them from ``prompt``.
    """
    m = (mode or "none").lower()
    if m == "none":
        return _anatomy_fragments(prompt, anatomy_mode)
    specs = _eligible(include_photography) if m == "all" else _detect_specs(prompt, include_photography)
    cpos, cneg = _color_render_fragments(specs)
    apos, aneg = _anatomy_fragments(prompt, anatomy_mode)
    pos = merge_csv_unique(*(s.positive_hints for s in specs), cpos, apos)
    neg = merge_csv_unique(*(s.negative_hints for s in specs), cneg, aneg)
    return pos, neg


# Coarse medium families used to route finishing and token-space naturalness.
MEDIUM_FAMILIES = ("generic", "photo", "2d", "traditional", "3d")


@dataclass(slots=True)
class MediumProfile:
    """Detected or user-specified medium for naturalness routing."""

    name: str = "unknown"
    family: str = "generic"


_FAMILY_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (name, family, re.compile(rf"\b(?:{pattern})\b"))
    for name, family, pattern in (
        ("pixel_art", "2d", r"pixel ?art|pixelated|sprite ?sheet|8-bit|16-bit"),
        ("vector", "2d", r"vector(?: art| illustration| graphic)?|flat (?:design|illustration|vector)|svg"),
        ("anime", "2d", r"anime|manga|cel[- ]shad\w*|visual novel|chibi|webtoon|1girl|1boy"),
        ("photograph", "photo", r"photo\w*|dslr|raw photo|film still|polaroid|35mm|bokeh|analog film|kodak \w+"),
        ("watercolor", "traditional", r"watercolou?r|gouache"),
        ("oil_painting", "traditional", r"oil (?:painting|paint|on canvas)|acrylic|impasto"),
        (
            "ink",
            "traditional",
            r"ink (?:drawing|wash)|pen and ink|crosshatch\w*|etching|woodcut|linocut|lithograph|sumi-e",
        ),
        ("drawing", "traditional", r"pastel|charcoal|pencil|graphite|sketch\w*"),
        (
            "3d_render",
            "3d",
            r"3d render|octane|blender|unreal engine|cinema ?4d|v-?ray|cgi|claymation|voxel|low[- ]poly",
        ),
        (
            "illustration",
            "2d",
            r"illustration|cartoon|comic|concept art|digital (?:painting|art)|storybook|matte painting|painterly",
        ),
        ("painting", "traditional", r"painting|painted"),
    )
)


def detect_all_mediums(prompt: str) -> list[MediumProfile]:
    """Every medium the prompt names, most specific rule first (mixed media yields several)."""
    text = (prompt or "").lower()
    return [MediumProfile(name, family) for name, family, rx in _FAMILY_RULES if rx.search(text)]


def detect_medium(prompt: str) -> MediumProfile:
    found = detect_all_mediums(prompt)
    return found[0] if found else MediumProfile()
