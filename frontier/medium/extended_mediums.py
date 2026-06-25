"""
Extended medium specs not yet in ``config.defaults.art_mediums``.

Merged at runtime by ``extended_guidance_fragments`` — keeps core config stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from config.defaults.art_mediums import guidance_fragments, merge_csv_unique

GuidanceMode = str  # re-use literals from art_mediums at call site


@dataclass(frozen=True, slots=True)
class ExtendedMediumSpec:
    id: str
    keywords: Tuple[str, ...]
    positive_hints: str
    negative_hints: str


def _word(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b", re.I)


EXTENDED_MEDIUM_SPECS: Tuple[ExtendedMediumSpec, ...] = (
    ExtendedMediumSpec(
        "spray_paint_street",
        ("spray paint", "graffiti", "street art", "aerosol mural", "stencil art"),
        "cap pressure variation, overspray halo, layered stencil alignment, concrete wall tooth",
        "gaussian smear tags, incoherent drip physics, floating letters",
    ),
    ExtendedMediumSpec(
        "encaustic_wax",
        ("encaustic", "hot wax painting", "wax medium"),
        "molten wax texture, translucent wax layers, scraped reveal marks",
        "plastic resin look, uniform gloss, digital gradient wax",
    ),
    ExtendedMediumSpec(
        "linocut_woodblock",
        ("linocut", "woodblock print", "woodcut", "block print", "ukiyo-e print"),
        "bold carved negative space, ink transfer pressure, registration charm",
        "blurry print, random hatch without block logic",
    ),
    ExtendedMediumSpec(
        "stained_glass",
        ("stained glass", "lead came", "glass mosaic window"),
        "luminous transmitted color, lead line rhythm, cathedral light scatter",
        "flat color without transmission, random polygon soup",
    ),
    ExtendedMediumSpec(
        "mosaic_tile",
        ("mosaic", "tile mosaic", "byzantine mosaic", "tesserae"),
        "consistent tessera scale, grout rhythm, material chip variation",
        "noise texture pretending to be tiles, scale drift",
    ),
    ExtendedMediumSpec(
        "colored_pencil",
        ("colored pencil", "polychromos", "prismacolor", "wax pencil"),
        "directional pencil stroke, wax bloom, paper grain interaction",
        "airbrush skin, digital smooth without tooth",
    ),
    ExtendedMediumSpec(
        "marker_copic",
        ("copic marker", "alcohol marker", "marker illustration", "marker render"),
        "streak-free blender transitions where intended, chisel tip stroke read, clean cap lines",
        "muddy marker bleed everywhere, uncontrolled bleed",
    ),
    ExtendedMediumSpec(
        "scratchboard",
        ("scratchboard", "scratch art", "white on black scratch"),
        "sharp scratch highlight direction, engraved texture rhythm",
        "random white noise scratches",
    ),
    ExtendedMediumSpec(
        "batik_fabric",
        ("batik", "wax resist fabric", "tie dye", "shibori"),
        "wax resist crackle, dye penetration boundaries, fabric fold dye pooling",
        "flat procedural pattern, no fabric fold logic",
    ),
    ExtendedMediumSpec(
        "collage_mixed",
        ("collage", "mixed media collage", "paper collage", "photomontage"),
        "cut paper edge shadow, scale consistency, glue seam logic",
        "floating clipart, scale mismatch chaos",
    ),
    ExtendedMediumSpec(
        "impressionist",
        ("impressionist", "impressionism", "plein air", "monet style", "broken color"),
        "broken color strokes, optical mixing, atmospheric perspective color shift",
        "muddy overblend, uniform brush size everywhere",
    ),
    ExtendedMediumSpec(
        "expressionist",
        ("expressionist", "expressionism", "bold gestural paint", "die brücke"),
        "emotive stroke direction, exaggerated color temperature, raw mark energy",
        "random glitch strokes without emotional axis",
    ),
    ExtendedMediumSpec(
        "baroque_oil",
        ("baroque painting", "caravaggio", "chiaroscuro oil", "old master oil"),
        "dramatic chiaroscuro, tenebrism, glazing depth, historical varnish warmth",
        "flat comic shading, modern HDR on old master",
    ),
    ExtendedMediumSpec(
        "art_nouveau",
        ("art nouveau", "mucha", "decorative illustration", "ornate floral border"),
        "elegant contour rhythm, decorative flat pattern harmony, hair and fabric flow lines",
        "generic fantasy portrait without decorative grammar",
    ),
    ExtendedMediumSpec(
        "ukiyo_e",
        ("ukiyo-e", "japanese woodblock", "hokusai", "kuniyoshi"),
        "flat plane separation, bold outline, limited palette, wave and cloud motifs",
        "western shading on ukiyo-e, gradient mush",
    ),
    ExtendedMediumSpec(
        "airbrush_80s",
        ("airbrush illustration", "80s airbrush", "fantasy airbrush"),
        "controlled airbrush soft focus, chrome highlight discipline, retro gradient skies",
        "modern AI plastic skin, random lens flare",
    ),
    ExtendedMediumSpec(
        "risograph",
        ("risograph", "riso print", "duotone print"),
        "misregistration charm, limited ink layer overlap, grainy halftone",
        "full CMYK smooth print, no layer logic",
    ),
    ExtendedMediumSpec(
        "chalk_mural",
        ("chalk mural", "sidewalk chalk", "pastel chalk mural"),
        "chalk dust at edges, pavement tooth, temporary mural scale",
        "digital chalk filter noise",
    ),
)

_PATTERNS: dict[str, Tuple[re.Pattern, ...]] = {
    s.id: tuple(_word(k) for k in s.keywords) for s in EXTENDED_MEDIUM_SPECS
}


def detect_extended_medium_ids(prompt: str) -> Tuple[str, ...]:
    text = (prompt or "").strip()
    if not text:
        return ()
    out: List[str] = []
    for spec in EXTENDED_MEDIUM_SPECS:
        if any(p.search(text) for p in _PATTERNS[spec.id]):
            out.append(spec.id)
    return tuple(out)


def extended_guidance_fragments(
    prompt: str,
    *,
    include_photography: bool = True,
    anatomy_mode: str = "none",
    base_mode: str = "auto",
) -> Tuple[str, str]:
    """
    Merge core ``art_mediums`` guidance with extended packs + detected extended ids.
    """
    pos, neg = guidance_fragments(
        prompt,
        base_mode,  # type: ignore[arg-type]
        include_photography=include_photography,
        anatomy_mode=anatomy_mode,  # type: ignore[arg-type]
    )
    ext_ids = detect_extended_medium_ids(prompt)
    if base_mode == "all":
        specs = list(EXTENDED_MEDIUM_SPECS)
    else:
        specs = [s for s in EXTENDED_MEDIUM_SPECS if s.id in ext_ids]
    epos = merge_csv_unique(*(s.positive_hints for s in specs))
    eneg = merge_csv_unique(*(s.negative_hints for s in specs))
    return merge_csv_unique(pos, epos), merge_csv_unique(neg, eneg)


__all__ = ["EXTENDED_MEDIUM_SPECS", "ExtendedMediumSpec", "detect_extended_medium_ids", "extended_guidance_fragments"]
