"""Taxonomy of common generative-image failure modes with prompt/negative fragments that counter them.

Consumed by the PromptStack guidance stage (``sample.py --shortcomings-mitigation`` and the training
caption path via ``TrainConfig.train_shortcomings_mitigation``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ._lexicon import compile_keywords, matches, merge_csv_unique

__all__ = [
    "MitigationMode",
    "ShortcomingSpec",
    "SHORTCOMING_SPECS",
    "spec_by_id",
    "detect_shortcoming_ids",
    "mitigation_fragments",
]

MitigationMode = Literal["none", "auto", "all"]


@dataclass(frozen=True, slots=True)
class ShortcomingSpec:
    id: str
    keywords: tuple[str, ...]
    positive_hints: str
    negative_hints: str
    is_2d_style: bool = False


# Order matters for stable merge of hints (photoreal + digital + CG, then stylized 2D-specific).
SHORTCOMING_SPECS: tuple[ShortcomingSpec, ...] = (
    ShortcomingSpec(
        id="skin_detail_tangents",
        keywords=(
            "portrait",
            "skin",
            "close-up",
            "closeup",
            "freckles",
            "complexion",
            "face",
            "beauty",
        ),
        positive_hints=(
            "natural skin texture, subtle skin detail, soft subsurface scattering, clear overlap and separation of forms"
        ),
        negative_hints=(
            "plastic skin, wax skin, airbrushed, overly smooth skin, doll-like skin, confusing tangents, merged edges"
        ),
    ),
    ShortcomingSpec(
        id="spatial_support",
        keywords=(
            "sitting",
            "leaning",
            "chair",
            "couch",
            "holding",
            "gripping",
            "standing",
            "feet",
            "ground",
            "floor",
            "table",
            "drape",
            "fabric",
        ),
        positive_hints="believable weight and contact, feet planted, fabric creases and drape, hands gripping objects naturally",
        negative_hints="floating, hovering, no ground contact, clipping, stiff cloth, floating hands",
    ),
    ShortcomingSpec(
        id="lighting_gi",
        keywords=(
            "sunlight",
            "studio",
            "neon",
            "lamp",
            "shadow",
            "night",
            "interior",
            "room",
            "window",
            "cinematic",
            "rim light",
        ),
        positive_hints="single coherent light direction, soft bounce light, subtle color bleed, ambient occlusion in contact areas",
        negative_hints="contradictory shadows, inconsistent lighting, disconnected elements, flat ambient",
    ),
    ShortcomingSpec(
        id="line_edges",
        keywords=("lineart", "line art", "ink", "comic", "manga", "outline", "crosshatch", "etching"),
        positive_hints="varied line weight, tapered strokes, lost and found edges, readable silhouette",
        negative_hints="uniform outline, deep-fried edges, over-defined contour everywhere, mechanical line weight",
    ),
    ShortcomingSpec(
        id="anatomy_deep",
        keywords=(
            "hand",
            "hands",
            "fingers",
            "pose",
            "full body",
            "limbs",
            "elbow",
            "knee",
            "shoulder",
            "muscular",
            "anatomy",
        ),
        positive_hints="coherent skeleton and joints, natural limb attachment, correct hands and fingers",
        negative_hints="noodle limbs, broken joints, twisted anatomy, impossible pose, fused fingers",
    ),
    ShortcomingSpec(
        id="composition_flow",
        keywords=("landscape", "cityscape", "wide shot", "poster", "cover", "scene", "crowd", "panorama"),
        positive_hints="rule of thirds, intentional negative space, clear focal hierarchy, visual rest",
        negative_hints="cluttered composition, horror vacui, everything centered, busy corners",
    ),
    ShortcomingSpec(
        id="materials_texture",
        keywords=("metal", "rust", "leather", "marble", "wood", "fabric", "glass", "stone", "armor", "weathered"),
        positive_hints="material-consistent texture, wear where physics and moisture collect, believable folds and highlights",
        negative_hints="random noise texture, plastic everywhere, inconsistent material reads",
    ),
    ShortcomingSpec(
        id="narrative_wear",
        keywords=("battle", "adventurer", "mud", "dust", "traveler", "ruins", "vintage", "worn", "torn", "stained"),
        positive_hints="story-consistent wear and dirt, directional weathering",
        negative_hints="generic scratches, random grime, incoherent damage",
    ),
    ShortcomingSpec(
        id="perspective_foreshortening",
        keywords=("foreshortening", "fisheye", "wide angle", "from above", "from below", "dramatic angle", "pov"),
        positive_hints="coherent perspective, consistent scale in depth, readable overlapping forms",
        negative_hints="warped limbs, broken perspective, mushy depth",
    ),
    ShortcomingSpec(
        id="facial_nuance",
        keywords=("smile", "expression", "emotive", "laughing", "crying", "portrait", "eyes", "gaze"),
        positive_hints="natural facial asymmetry, subtle micro-expression, lifelike eyes",
        negative_hints="perfect symmetry, mask-like expression, dead eyes, uncanny valley face",
    ),
    ShortcomingSpec(
        id="color_value_discipline",
        keywords=("monochrome", "muted", "film", "noir", "zorn", "limited palette", "desaturated", "earth tones"),
        positive_hints="disciplined palette, coherent value structure, mood-consistent grading",
        negative_hints="rainbow oversaturation, neon clutter, chaotic values, posterized noise colors",
    ),
    ShortcomingSpec(
        id="extra_gaps",
        keywords=("typography", "lettering", "sign", "logo", "pattern", "plaid", "stripes", "splash", "smoke", "water"),
        positive_hints="consistent small repeating details, legible intentional text where required",
        negative_hints="garbled text, inconsistent pattern, stiff fluid, merged distant faces",
    ),
    ShortcomingSpec(
        id="cg_render_geometry",
        keywords=(
            "blender",
            "octane",
            "unreal",
            "3d render",
            "cgi",
            "raytrace",
            "subsurface scattering",
            "keyshot",
            "marmoset",
            "substance painter",
            "cycles render",
        ),
        positive_hints="believable hard-surface contact, consistent material response, readable form",
        negative_hints="melting mesh, fused geometry, swimming textures, impossible topology",
    ),
    # Digital painting / illustration / games / UI (cel-anime lives in the is_2d_style specs below).
    ShortcomingSpec(
        id="digital_painting",
        keywords=(
            "digital painting",
            "digital art",
            "digital illustration",
            "photoshop",
            "procreate",
            "clip studio",
            "paint tool sai",
            "krita",
            "corel painter",
            "wacom",
            "speedpaint",
            "speed painting",
            "paintover",
            "drawing tablet",
        ),
        positive_hints=(
            "intentional brushwork, readable stroke economy, coherent edge control, "
            "layer-aware color mixing, purposeful soft and hard transitions"
        ),
        negative_hints=(
            "plastic airbrush smear, uniform over-smoothing, muddy midtones, AI soup blend, "
            "soulless gradient fill, noise pretending to be texture"
        ),
    ),
    ShortcomingSpec(
        id="concept_matte_digital",
        keywords=(
            "concept art",
            "key art",
            "matte painting",
            "photobash",
            "photo bash",
            "environment design",
            "keyframe",
            "artstation",
            "set extension",
            "digital collage",
        ),
        positive_hints=(
            "clear design read, consistent perspective across photobashed elements, "
            "matched grain and color cast, intentional focal hierarchy"
        ),
        negative_hints=(
            "floating cutouts, scale mismatch between assets, conflicting light directions, "
            "generic grey sculpt mush, template composition"
        ),
    ),
    ShortcomingSpec(
        id="pixel_digital",
        keywords=(
            "pixel art",
            "pixelart",
            "sprite",
            "spritesheet",
            "sprite sheet",
            "8-bit",
            "16-bit",
            "32-bit",
            "retro game",
            "nes style",
            "snes",
            "gameboy",
            "isometric pixel",
        ),
        positive_hints=(
            "crisp pixel grid, disciplined limited palette, clean clusters, "
            "intentional dithering or AA where appropriate"
        ),
        negative_hints=(
            "blurry between pixels, accidental gradient ramps, subpixel smear, "
            "inconsistent pixel scale, soft anti-alias where hard pixels are required"
        ),
    ),
    ShortcomingSpec(
        id="vector_flat_digital",
        keywords=(
            "vector",
            "vector art",
            "flat design",
            "flat illustration",
            "app icon",
            "ui illustration",
            "infographic",
            "logo design",
            "svg style",
            "material design",
        ),
        positive_hints=(
            "clean vector shapes, consistent corner and stroke discipline, deliberate flat fills, "
            "readable icon silhouette"
        ),
        negative_hints=(
            "wobbly anchors, accidental 3D or photoreal bleed, fuzzy anti-aliased mush, "
            "inconsistent stroke width, messy boolean overlaps"
        ),
    ),
    ShortcomingSpec(
        id="stylized_game_digital",
        keywords=(
            "hand-painted texture",
            "hand painted texture",
            "stylized 3d",
            "stylized 3d render",
            "game asset",
            "texture sheet",
            "albedo map",
            "character turnaround",
            "model sheet",
            "low poly",
            "hand-painted",
        ),
        positive_hints=(
            "readable stylized materials, consistent texel density, clear albedo vs lighting read, "
            "cohesive hand-painted surface direction"
        ),
        negative_hints=(
            "muddy albedo, random noise mud, swimming hand-painted detail, inconsistent brush scale across UV shells"
        ),
    ),
    # Stylized 2D specs: only considered when the caller enables the 2D pack.
    ShortcomingSpec(
        id="style_drift_2d",
        keywords=("anime", "manga", "cel shaded", "cel-shaded", "visual novel", "light novel", "gacha"),
        positive_hints="consistent stylization, single coherent 2D style across the image, stable character design cues",
        negative_hints="style drift, mixed realism and toon, inconsistent proportions between regions",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="line_art_2d",
        keywords=("anime", "manga", "cel", "toon", "cartoon", "chibi"),
        positive_hints="clean intentional line art, hair strand clarity, garment contour flow",
        negative_hints="extraneous outlines, inconsistent hair lines, mechanical uniform ink",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="cel_lighting_2d",
        keywords=("cel", "flat color", "anime shading", "toon shading", "manga"),
        positive_hints="hard-edged shadow shapes that follow form, disciplined flat colors, coherent toon light",
        negative_hints="random soft gradients in flat-style areas, contradictory toon shadows",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="anatomy_2d",
        keywords=("chibi", "anime", "manga", "super deformed"),
        positive_hints="consistent stylized proportions, stable eye highlights, readable stylized pose",
        negative_hints="inconsistent eye size, broken stylized limbs, floating hands in action pose",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="composition_2d",
        keywords=("manga", "comic", "cover art", "key visual", "splash art", "pin-up"),
        positive_hints="panel-like clarity, generous negative space where appropriate, strong silhouette",
        negative_hints="overcrowded composition, weak focal read in stylized scene",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="medium_2d",
        keywords=("watercolor", "halftone", "screentone", "ink wash", "marker", "copics"),
        positive_hints="medium-consistent texture and bleed, paper or print logic",
        negative_hints="plastic smoothness, random noise pretending to be medium",
        is_2d_style=True,
    ),
    ShortcomingSpec(
        id="extra_2d",
        keywords=("comic strip", "children book", "picture book", "graphic novel"),
        positive_hints="wholesome stylized clarity, consistent line and fill discipline",
        negative_hints="unwanted photoreal drift, kitsch over-polish, garbled lettering in bubbles",
        is_2d_style=True,
    ),
)

_SHORTCOMING_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    s.id: compile_keywords(s.keywords) for s in SHORTCOMING_SPECS
}


def spec_by_id(sid: str) -> ShortcomingSpec:
    for s in SHORTCOMING_SPECS:
        if s.id == sid:
            return s
    raise KeyError(sid)


def _eligible(include_2d: bool) -> list[ShortcomingSpec]:
    return [s for s in SHORTCOMING_SPECS if include_2d or not s.is_2d_style]


def detect_shortcoming_ids(prompt: str, *, include_2d: bool) -> tuple[str, ...]:
    """Ids whose keywords match ``prompt`` (word-boundary, case-insensitive)."""
    if not prompt or not prompt.strip():
        return ()
    return tuple(s.id for s in _eligible(include_2d) if matches(prompt, _SHORTCOMING_PATTERNS[s.id]))


def mitigation_fragments(
    prompt: str,
    mode: MitigationMode,
    *,
    include_2d_pack: bool,
) -> tuple[str, str]:
    """``(positive_suffix, negative_suffix)`` for ``mode``; both empty for ``"none"`` or no match.

    ``"auto"`` uses only the specs whose keywords appear in ``prompt``; ``"all"`` uses every eligible spec.
    """
    if mode == "all":
        specs = _eligible(include_2d_pack)
    elif mode == "auto":
        specs = [spec_by_id(i) for i in detect_shortcoming_ids(prompt, include_2d=include_2d_pack)]
    else:
        return "", ""
    if not specs:
        return "", ""
    pos = merge_csv_unique(*(s.positive_hints for s in specs))
    neg = merge_csv_unique(*(s.negative_hints for s in specs))
    return pos, neg
