"""
Prompt snippets, negative add-ons, and aspect hints for book/comic/manga workflows.

Grounded in common sequential-art practice (ink, screentone, panels, lettering).
Used by ``generate_book.py`` (``--lexicon-style``, ``--aspect-preset``) and by tools.

This module does **not** call external APIs; it only returns strings and (w, h) tuples.
"""

from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Negative prompts — reduce typical gen-AI failures on comic/manga pages
# ---------------------------------------------------------------------------

LETTERING_NEGATIVE_ADDON = (
    "gibberish text, misspelled words, wrong letters, watermark, artist signature on art, "
    "merged speech bubbles, unreadable tiny font, text outside bubble, subtitles style"
)

ANATOMY_PANEL_NEGATIVE_ADDON = (
    "extra fingers, fused fingers, mangled hands, duplicate faces, asymmetrical eyes, "
    "broken panel borders, random grid overlay"
)

INK_STYLE_NEGATIVE_ADDON = (
    "plastic skin, airbrushed, overly smooth shading, CGI render, 3d model look, "
    "chromatic aberration, heavy jpeg artifacts"
)

ARTIST_LETTERING_STRICT_NEGATIVE = (
    "crossing balloon tails, uncertain speaker attribution, balloon overlap on faces, "
    "tiny unreadable dialogue, caption boxes out of reading order"
)

# Stricter panel / lettering failures (use with --book-accuracy production)
PRODUCTION_TIER_NEGATIVE_ADDON = (
    "cropped dialogue, speech balloon pointing wrong character, overlapping illegible text, "
    "spine misaligned title on cover, barcode on interior page, low dpi moire, muddy screentone"
)

def combined_comic_negative(*, include_lettering: bool = True, include_anatomy: bool = True) -> str:
    parts = [INK_STYLE_NEGATIVE_ADDON]
    if include_lettering:
        parts.append(LETTERING_NEGATIVE_ADDON)
    if include_anatomy:
        parts.append(ANATOMY_PANEL_NEGATIVE_ADDON)
    return ", ".join(parts)

# ---------------------------------------------------------------------------
# Style snippets (append to user prompt or book prefix)
# ---------------------------------------------------------------------------

STYLE_SNIPPETS: Dict[str, str] = {
    "none": "",
    "shonen": "dynamic action lines, speed lines, bold ink weight, expressive eyes, impact frames",
    "shoujo": "delicate linework, soft screentone gradients, sparkles, emotional close-ups, flower motifs",
    "seinen": "realistic proportions, heavy blacks, detailed backgrounds, mature atmosphere, fine hatching",
    "slice_of_life": "everyday setting, calm composition, natural lighting, cozy atmosphere, clear silhouettes",
    "chibi": "super deformed, large head small body, cute proportions, simplified features, comedy timing",
    "webtoon": "vertical composition, mobile reading format, wide establishing shots, scrolling-friendly layout",
    "manhwa_color": "full color, soft gradients, clean lineart, korean webtoon shading, glossy highlights",
    "graphic_novel": "cinematic lighting, painterly ink, cross-hatching, dramatic shadows, graphic novel composition",
    "editorial": "clear hierarchy, readable at small size, professional print margins implied, balanced negative space",
    "light_novel": "light novel cover illustration, ornate title treatment, character pin-up, publisher-ready layout",
    "yonkoma": "four panel strip, gag beat timing, simple backgrounds, punchline panel emphasis",
}

# Artist-oriented production bundles inspired by common comic/manga workflows:
# - clear reading order and eye path
# - controlled shot language (establishing -> medium -> close-up)
# - value hierarchy and texture discipline (screentone/halftone)
ARTIST_CRAFT_PROFILES: Dict[str, str] = {
    "none": "",
    "manga_pro": (
        "clear right-to-left panel flow, dominant-to-secondary focal hierarchy, "
        "establishing shot then medium then close-up rhythm, disciplined screentone values, "
        "clean silhouettes with strategic black fills"
    ),
    "western_comic_pro": (
        "clear left-to-right panel flow, readable gutter transitions, "
        "cinematic shot progression (establishing medium close-up), "
        "strong figure-ground separation, balloon-safe composition"
    ),
    "webtoon_pro": (
        "vertical scroll rhythm, long-to-short beat spacing, mobile-first readability, "
        "staging with strong top-to-bottom eye path, clear dialogue grouping per beat"
    ),
    "children_book": (
        "large readable shapes, simple value grouping, warm storytelling poses, "
        "clean focal hierarchy with generous negative space for text"
    ),
    "cinematic_storyboard": (
        "shot continuity discipline, decisive camera axis, clear staging per beat, "
        "high readability thumbnails, intent-first framing"
    ),
}

SHOT_LANGUAGE_HINTS: Dict[str, str] = {
    "none": "",
    "mixed": "balanced mix of establishing, medium, and close-up shots with clear continuity",
    "cinematic": "film-language shot grammar, motivated camera changes, over-shoulder dialogue coverage",
    "manga_dynamic": "dynamic manga framing, diagonal action staging, impact close-ups with speed emphasis",
    "dialogue_coverage": "dialogue-first shot coverage, over-shoulder and reaction close-ups, readable speaker turns",
}

PACING_PLAN_HINTS: Dict[str, str] = {
    "none": "",
    "decompressed": "decompressed pacing, more panels per action beat, breathing room in gutters",
    "balanced": "balanced pacing with alternating wide setup and tight emotional beats",
    "compressed": "compressed pacing, fewer panels with decisive story beats and efficient transitions",
}

LETTERING_CRAFT_HINTS: Dict[str, str] = {
    "none": "",
    "standard": (
        "speech balloons placed in reading order, tails clearly pointing to speaker, "
        "text kept inside balloons with consistent margin"
    ),
    "strict": (
        "strict lettering discipline, top-to-bottom reading path, non-intersecting balloon tails, "
        "speaker-first balloon placement, avoid covering faces/hands"
    ),
}

VALUE_PLAN_HINTS: Dict[str, str] = {
    "none": "",
    "bw_hierarchy": "black-white value hierarchy, three-step value grouping, focal point highest contrast",
    "color_script": "cohesive color script across page beats, controlled palette shifts, value-first readability",
}

SCREENTONE_PLAN_HINTS: Dict[str, str] = {
    "none": "",
    "clean": "clean screentone application, moire-safe dot scale, controlled gradients on form turns",
    "dramatic": "dramatic screentone contrast, heavy blacks plus selective tone gradients for depth",
}

ORIGINAL_CHARACTER_ARCHETYPES: Dict[str, str] = {
    "none": "",
    "shonen_lead": "energetic protagonist silhouette, readable hero shape language, expressive action-ready posture",
    "cool_rival": "sharp rival silhouette, restrained expression set, angular design accents",
    "mentor": "grounded mentor presence, mature posture language, iconic costume readability",
    "antihero": "edgy antihero contrast, asymmetrical design motifs, controlled intensity in expression",
    "magical_girl": "clean iconic outfit language, transformation-ready silhouette, emotive face readability",
    "noir_detective": "detective silhouette cues, coat/hat shape identity, moody expression control",
    "space_pilot": "functional sci-fi costume logic, helmet/gear identity anchors, practical movement silhouette",
}

ARTIST_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "manga_cinematic": {
        "craft_profile": "manga_pro",
        "shot_language": "manga_dynamic",
        "pacing_plan": "balanced",
        "lettering_craft": "strict",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "dramatic",
    },
    "comic_dialogue": {
        "craft_profile": "western_comic_pro",
        "shot_language": "dialogue_coverage",
        "pacing_plan": "decompressed",
        "lettering_craft": "strict",
        "value_plan": "color_script",
        "screentone_plan": "clean",
    },
    "webtoon_scroll": {
        "craft_profile": "webtoon_pro",
        "shot_language": "mixed",
        "pacing_plan": "decompressed",
        "lettering_craft": "standard",
        "value_plan": "color_script",
        "screentone_plan": "clean",
    },
    "storyboard_fast": {
        "craft_profile": "cinematic_storyboard",
        "shot_language": "cinematic",
        "pacing_plan": "compressed",
        "lettering_craft": "none",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "none",
    },
}

OC_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "heroine_scifi": {
        "archetype": "space_pilot",
        "visual_traits": "short asymmetric hair, sharp brow shape, utility earpiece",
        "wardrobe": "flight jacket, utility belt, reinforced boots",
        "silhouette": "broad upper torso shape with tapered legs",
        "color_motifs": "teal accents on dark neutral base",
        "expression_sheet": "confident smirk, focused glare, determined shout",
    },
    "rival_dark": {
        "archetype": "cool_rival",
        "visual_traits": "narrow eyes, angular fringe, distinct facial mark",
        "wardrobe": "high-collar coat with geometric trim",
        "silhouette": "tall narrow silhouette with sharp shoulder points",
        "color_motifs": "black, crimson, steel gray",
        "expression_sheet": "cold stare, restrained smirk, contempt glance",
    },
    "mentor_classic": {
        "archetype": "mentor",
        "visual_traits": "older face planes, pronounced brow, calm gaze",
        "wardrobe": "layered robe or coat with iconic accessory",
        "silhouette": "stable triangular silhouette",
        "color_motifs": "earth tones with one signature accent color",
        "expression_sheet": "calm smile, stern warning, reflective concern",
    },
}

BOOK_STYLE_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "manga_nsfw_action": {
        "artist_pack": "manga_cinematic",
        "oc_pack": "none",
        "safety_mode": "nsfw",
        "nsfw_pack": "explicit_detail",
        "nsfw_civitai_pack": "action",
        "civitai_trigger_bank": "medium",
    },
    "webtoon_nsfw_romance": {
        "artist_pack": "webtoon_scroll",
        "oc_pack": "none",
        "safety_mode": "nsfw",
        "nsfw_pack": "romantic",
        "nsfw_civitai_pack": "style",
        "civitai_trigger_bank": "light",
    },
    "comic_dialogue_safe": {
        "artist_pack": "comic_dialogue",
        "oc_pack": "none",
        "safety_mode": "sfw",
        "nsfw_pack": "none",
        "nsfw_civitai_pack": "none",
        "civitai_trigger_bank": "none",
    },
    "oc_launch_safe": {
        "artist_pack": "manga_cinematic",
        "oc_pack": "heroine_scifi",
        "safety_mode": "sfw",
        "nsfw_pack": "none",
        "nsfw_civitai_pack": "none",
        "civitai_trigger_bank": "none",
    },
}

HUMANIZE_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "lite": {
        "humanize_profile": "lite",
        "imperfection_level": "lite",
        "materiality_mode": "paper",
        "asymmetry_level": "lite",
        "negative_level": "lite",
    },
    "balanced": {
        "humanize_profile": "balanced",
        "imperfection_level": "balanced",
        "materiality_mode": "print",
        "asymmetry_level": "balanced",
        "negative_level": "balanced",
    },
    "strong": {
        "humanize_profile": "strong",
        "imperfection_level": "strong",
        "materiality_mode": "ink_paper",
        "asymmetry_level": "strong",
        "negative_level": "strong",
    },
    "painterly": {
        "humanize_profile": "painterly",
        "imperfection_level": "balanced",
        "materiality_mode": "canvas",
        "asymmetry_level": "balanced",
        "negative_level": "balanced",
    },
    "filmic": {
        "humanize_profile": "filmic",
        "imperfection_level": "lite",
        "materiality_mode": "film",
        "asymmetry_level": "lite",
        "negative_level": "lite",
    },
}

HUMANIZE_PROFILE_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "subtle hand-drawn irregularities, natural edge variance, avoid sterile symmetry",
    "balanced": "human-made mark-making cadence, varied stroke pressure, intentional imperfection rhythm",
    "strong": "visible handcrafted quirks, non-repeating micro-variation, organic contour wobble",
    "painterly": "human brush economy, purposeful brush breaks, varied paint lay-in and edge softness",
    "filmic": "human-captured photographic feel, mild lens personality, natural scene imperfections",
}

HUMANIZE_IMPERFECTION_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "slight line wobble and tiny spacing variation where appropriate",
    "balanced": "controlled imperfection in line weight, texture breakup, and shape repetition",
    "strong": "pronounced hand-made variance in line rhythm, spacing, and micro-texture",
}

HUMANIZE_MATERIALITY_HINTS: Dict[str, str] = {
    "none": "",
    "paper": "subtle paper tooth interaction, natural ink absorption feel",
    "ink_paper": "ink-on-paper behavior, dry-brush streaks, halftone print texture discipline",
    "canvas": "canvas tooth response, painterly pigment buildup, non-uniform brush drag",
    "print": "print-like halftone behavior, slight registration character, realistic reproduction feel",
    "film": "organic film grain feel, photographic texture depth, non-digital tonal rolloff",
}

HUMANIZE_ASYMMETRY_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "natural facial asymmetry and non-mirrored detail placement",
    "balanced": "human asymmetry in features, posture, and repeated costume details",
    "strong": "clearly non-mirrored human asymmetry across face, pose, and accessories",
}

HUMANIZE_NEGATIVE_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "plastic skin, over-smoothed gradients, sterile perfect symmetry",
    "balanced": "ai soup texture, uniform procedural lines, copy-paste detail repetition, uncanny perfection",
    "strong": "waxy skin, rubbery limbs, overclean vectorized edges, synthetic texture tiling, dead-eyed symmetry",
}

# Reading-order hints for Western vs manga (prompt-only; model follows data)
READING_ORDER_HINT = {
    "manga": "right-to-left reading order cues, manga page layout",
    "comic": "left-to-right comic layout, western gutters",
    "novel_cover": "",
    "storyboard": "sequential storyboard frames, numbered panels implied",
}

# Vertical Japanese lettering (tategaki) — use when your dataset includes JP
TATEGAKI_HINT = (
    "vertical japanese text in speech bubble, tategaki, correct stroke order impression, "
    "legible jp characters"
)

SFX_ONOMATOPOEIA_HINT = (
    "impact sfx typography, hand-drawn sound effects, integrated with art not overlay subtitle"
)

# Optional polish for print / cover work (use with models trained on book art)
PRINT_FINISH_HINT = (
    "print-ready line weight, crisp halftone, no banding, clean margins, professional reproduction"
)

COVER_SPOTLIGHT_HINT = (
    "strong focal point on hero figure, title area reserved, balanced negative space for typography"
)

# Panel / grid hints (sequential art — model follows training; these are soft cues)
PANEL_LAYOUT_HINTS: Dict[str, str] = {
    "none": "",
    "single": "single full-bleed panel, one clear focal composition",
    "two_panel_horizontal": "two horizontal panels stacked, clear gutter between tiers",
    "two_panel_vertical": "two vertical panels side by side, western comic gutters",
    "three_panel_strip": "three panel horizontal strip, equal rhythm, readable flow",
    "four_koma": "four panel vertical strip, yonkoma beat timing, punchline bottom panel",
    "splash": "large splash panel with inset smaller panel, dynamic hierarchy",
    "grid_2x2": "four equal panels in 2x2 grid, consistent line weight across cells",
}

# ---------------------------------------------------------------------------
# Aspect presets (width x height) — suggestions; 0 means “model native” in generate_book
# ---------------------------------------------------------------------------

# Webtoon / scroll: tall canvas; many models trained near 512–768 short side
ASPECT_PRESETS: Dict[str, Tuple[int, int]] = {
    "none": (0, 0),
    "square": (512, 512),
    "print_manga": (768, 1024),  # portrait page-ish
    "webtoon_tall": (512, 1536),  # vertical strip
    "widescreen_panel": (1024, 512),
    "cover_hd": (1024, 1024),
    "double_page_spread": (1536, 1024),
    "print_us_comic": (900, 1400),
}


def style_snippet(name: str) -> str:
    return STYLE_SNIPPETS.get((name or "none").lower().strip(), "")


def reading_order_for_book_type(book_type: str) -> str:
    return READING_ORDER_HINT.get((book_type or "manga").lower().strip(), "")


def merge_prompt_fragments(*parts: str, joiner: str = ", ") -> str:
    """Join non-empty stripped fragments."""
    out = [p.strip() for p in parts if p and str(p).strip()]
    return joiner.join(out)


def enhance_book_prefix(
    base_prefix: str,
    *,
    lexicon_style: str = "none",
    book_type: str = "manga",
    include_tategaki_hint: bool = False,
    include_sfx_hint: bool = False,
    include_print_finish: bool = False,
    include_cover_spotlight: bool = False,
) -> str:
    """
    Merge the existing book-type prefix with optional lexicon style + reading-order hints.
    """
    bits = [base_prefix.strip()]
    sn = style_snippet(lexicon_style)
    if sn:
        bits.append(sn)
    ro = reading_order_for_book_type(book_type)
    if ro:
        bits.append(ro)
    if include_tategaki_hint:
        bits.append(TATEGAKI_HINT)
    if include_sfx_hint:
        bits.append(SFX_ONOMATOPOEIA_HINT)
    if include_print_finish:
        bits.append(PRINT_FINISH_HINT)
    if include_cover_spotlight:
        bits.append(COVER_SPOTLIGHT_HINT)
    return merge_prompt_fragments(*bits)


def suggest_negative_addon(
    *,
    use_lexicon_negative: bool = True,
    user_negative: str = "",
    production_tier: bool = False,
    artist_lettering_strict: bool = False,
) -> str:
    """Combine user negative with lexicon anti-artifact clauses (dedupe loosely by concat)."""
    u = (user_negative or "").strip()
    if not use_lexicon_negative:
        return u
    extra = combined_comic_negative()
    if production_tier:
        extra = f"{extra}, {PRODUCTION_TIER_NEGATIVE_ADDON}"
    if artist_lettering_strict:
        extra = f"{extra}, {ARTIST_LETTERING_STRICT_NEGATIVE}"
    if not u:
        return extra
    return f"{u}, {extra}"


def aspect_dimensions(preset_name: str) -> Tuple[int, int]:
    key = (preset_name or "none").lower().strip()
    return ASPECT_PRESETS.get(key, (0, 0))


def panel_layout_hint(name: str) -> str:
    return PANEL_LAYOUT_HINTS.get((name or "none").lower().strip(), "")


def artist_craft_bundle(
    *,
    craft_profile: str = "none",
    shot_language: str = "none",
    pacing_plan: str = "none",
    lettering_craft: str = "none",
    value_plan: str = "none",
    screentone_plan: str = "none",
) -> str:
    """
    Merge practical artist-facing craft hints for sequential art quality.
    """
    bits = [
        ARTIST_CRAFT_PROFILES.get((craft_profile or "none").lower().strip(), ""),
        SHOT_LANGUAGE_HINTS.get((shot_language or "none").lower().strip(), ""),
        PACING_PLAN_HINTS.get((pacing_plan or "none").lower().strip(), ""),
        LETTERING_CRAFT_HINTS.get((lettering_craft or "none").lower().strip(), ""),
        VALUE_PLAN_HINTS.get((value_plan or "none").lower().strip(), ""),
        SCREENTONE_PLAN_HINTS.get((screentone_plan or "none").lower().strip(), ""),
    ]
    return merge_prompt_fragments(*bits)


def original_character_bundle(
    *,
    name: str = "",
    archetype: str = "none",
    visual_traits: str = "",
    wardrobe: str = "",
    silhouette: str = "",
    color_motifs: str = "",
    expression_sheet: str = "",
) -> str:
    """
    Build an artist-facing original-character (OC) consistency block.
    """
    bits = []
    if str(name).strip():
        bits.append(f"original character {str(name).strip()}, consistent identity across panels")
    arch = ORIGINAL_CHARACTER_ARCHETYPES.get((archetype or "none").lower().strip(), "")
    if arch:
        bits.append(arch)
    if str(visual_traits).strip():
        bits.append(f"signature traits: {str(visual_traits).strip()}")
    if str(wardrobe).strip():
        bits.append(f"consistent wardrobe: {str(wardrobe).strip()}")
    if str(silhouette).strip():
        bits.append(f"silhouette lock: {str(silhouette).strip()}")
    if str(color_motifs).strip():
        bits.append(f"color motif: {str(color_motifs).strip()}")
    if str(expression_sheet).strip():
        bits.append(f"expression sheet anchors: {str(expression_sheet).strip()}")
    bits.append("same face structure and hairstyle in every panel")
    return merge_prompt_fragments(*bits)


def humanize_prompt_bundle(
    *,
    humanize_profile: str = "none",
    imperfection_level: str = "none",
    materiality_mode: str = "none",
    asymmetry_level: str = "none",
) -> str:
    """Build positive prompt cues for a more human-made result."""
    bits = [
        HUMANIZE_PROFILE_HINTS.get((humanize_profile or "none").lower().strip(), ""),
        HUMANIZE_IMPERFECTION_HINTS.get((imperfection_level or "none").lower().strip(), ""),
        HUMANIZE_MATERIALITY_HINTS.get((materiality_mode or "none").lower().strip(), ""),
        HUMANIZE_ASYMMETRY_HINTS.get((asymmetry_level or "none").lower().strip(), ""),
    ]
    return merge_prompt_fragments(*bits)


def humanize_negative_addon(negative_level: str = "none") -> str:
    """Negative prompt addon to suppress common synthetic artifacts."""
    return HUMANIZE_NEGATIVE_HINTS.get((negative_level or "none").lower().strip(), "")


def resolve_artist_controls(
    *,
    artist_pack: str = "none",
    craft_profile: str = "none",
    shot_language: str = "none",
    pacing_plan: str = "none",
    lettering_craft: str = "none",
    value_plan: str = "none",
    screentone_plan: str = "none",
) -> Dict[str, str]:
    """
    Resolve artist controls from a preset pack + explicit CLI overrides.

    Explicit non-``none`` values always win over pack defaults.
    """
    pack = ARTIST_PACK_PRESETS.get((artist_pack or "none").lower().strip(), {})
    out = {
        "craft_profile": pack.get("craft_profile", "none"),
        "shot_language": pack.get("shot_language", "none"),
        "pacing_plan": pack.get("pacing_plan", "none"),
        "lettering_craft": pack.get("lettering_craft", "none"),
        "value_plan": pack.get("value_plan", "none"),
        "screentone_plan": pack.get("screentone_plan", "none"),
    }
    if (craft_profile or "none").lower().strip() != "none":
        out["craft_profile"] = craft_profile
    if (shot_language or "none").lower().strip() != "none":
        out["shot_language"] = shot_language
    if (pacing_plan or "none").lower().strip() != "none":
        out["pacing_plan"] = pacing_plan
    if (lettering_craft or "none").lower().strip() != "none":
        out["lettering_craft"] = lettering_craft
    if (value_plan or "none").lower().strip() != "none":
        out["value_plan"] = value_plan
    if (screentone_plan or "none").lower().strip() != "none":
        out["screentone_plan"] = screentone_plan
    return out


def resolve_oc_controls(
    *,
    oc_pack: str = "none",
    name: str = "",
    archetype: str = "none",
    visual_traits: str = "",
    wardrobe: str = "",
    silhouette: str = "",
    color_motifs: str = "",
    expression_sheet: str = "",
) -> Dict[str, str]:
    """
    Resolve OC controls from a preset pack + explicit CLI overrides.
    """
    pack = OC_PACK_PRESETS.get((oc_pack or "none").lower().strip(), {})
    out = {
        "name": "",
        "archetype": pack.get("archetype", "none"),
        "visual_traits": pack.get("visual_traits", ""),
        "wardrobe": pack.get("wardrobe", ""),
        "silhouette": pack.get("silhouette", ""),
        "color_motifs": pack.get("color_motifs", ""),
        "expression_sheet": pack.get("expression_sheet", ""),
    }
    if str(name).strip():
        out["name"] = str(name).strip()
    if (archetype or "none").lower().strip() != "none":
        out["archetype"] = archetype
    if str(visual_traits).strip():
        out["visual_traits"] = str(visual_traits).strip()
    if str(wardrobe).strip():
        out["wardrobe"] = str(wardrobe).strip()
    if str(silhouette).strip():
        out["silhouette"] = str(silhouette).strip()
    if str(color_motifs).strip():
        out["color_motifs"] = str(color_motifs).strip()
    if str(expression_sheet).strip():
        out["expression_sheet"] = str(expression_sheet).strip()
    return out


def resolve_book_style_controls(
    *,
    book_style_pack: str = "none",
    artist_pack: str = "none",
    oc_pack: str = "none",
    safety_mode: str = "",
    nsfw_pack: str = "",
    nsfw_civitai_pack: str = "",
    civitai_trigger_bank: str = "",
) -> Dict[str, str]:
    """
    Resolve higher-level style controls from one pack + explicit overrides.

    Explicit values win over pack defaults. For `artist_pack` and `oc_pack`,
    `"none"` is treated as unset unless no pack default exists.
    """
    pack = BOOK_STYLE_PACK_PRESETS.get((book_style_pack or "none").lower().strip(), {})
    out = {
        "artist_pack": pack.get("artist_pack", "none"),
        "oc_pack": pack.get("oc_pack", "none"),
        "safety_mode": pack.get("safety_mode", ""),
        "nsfw_pack": pack.get("nsfw_pack", ""),
        "nsfw_civitai_pack": pack.get("nsfw_civitai_pack", ""),
        "civitai_trigger_bank": pack.get("civitai_trigger_bank", ""),
    }
    if (artist_pack or "none").lower().strip() != "none":
        out["artist_pack"] = artist_pack
    if (oc_pack or "none").lower().strip() != "none":
        out["oc_pack"] = oc_pack
    if str(safety_mode).strip():
        out["safety_mode"] = str(safety_mode).strip()
    if str(nsfw_pack).strip():
        out["nsfw_pack"] = str(nsfw_pack).strip()
    if str(nsfw_civitai_pack).strip():
        out["nsfw_civitai_pack"] = str(nsfw_civitai_pack).strip()
    if str(civitai_trigger_bank).strip():
        out["civitai_trigger_bank"] = str(civitai_trigger_bank).strip()
    return out


def resolve_humanize_controls(
    *,
    humanize_pack: str = "none",
    humanize_profile: str = "none",
    imperfection_level: str = "none",
    materiality_mode: str = "none",
    asymmetry_level: str = "none",
    negative_level: str = "none",
) -> Dict[str, str]:
    """
    Resolve humanization controls from one pack + explicit overrides.
    """
    pack = HUMANIZE_PACK_PRESETS.get((humanize_pack or "none").lower().strip(), {})
    out = {
        "humanize_profile": pack.get("humanize_profile", "none"),
        "imperfection_level": pack.get("imperfection_level", "none"),
        "materiality_mode": pack.get("materiality_mode", "none"),
        "asymmetry_level": pack.get("asymmetry_level", "none"),
        "negative_level": pack.get("negative_level", "none"),
    }
    if (humanize_profile or "none").lower().strip() != "none":
        out["humanize_profile"] = humanize_profile
    if (imperfection_level or "none").lower().strip() != "none":
        out["imperfection_level"] = imperfection_level
    if (materiality_mode or "none").lower().strip() != "none":
        out["materiality_mode"] = materiality_mode
    if (asymmetry_level or "none").lower().strip() != "none":
        out["asymmetry_level"] = asymmetry_level
    if (negative_level or "none").lower().strip() != "none":
        out["negative_level"] = negative_level
    return out


def infer_auto_humanize_controls(
    *,
    book_type: str = "manga",
    lexicon_style: str = "none",
    safety_mode: str = "",
) -> Dict[str, str]:
    """
    Infer practical default humanization settings from high-level intent.
    """
    bt = (book_type or "manga").lower().strip()
    ls = (lexicon_style or "none").lower().strip()
    sm = (safety_mode or "").lower().strip()

    if sm == "nsfw":
        return resolve_humanize_controls(humanize_pack="balanced")
    if bt == "storyboard":
        return resolve_humanize_controls(humanize_pack="lite")
    if bt == "novel_cover":
        return resolve_humanize_controls(humanize_pack="painterly")
    if ls in {"graphic_novel", "seinen", "editorial"}:
        return resolve_humanize_controls(humanize_pack="painterly")
    if ls in {"webtoon", "manhwa_color"}:
        return resolve_humanize_controls(humanize_pack="filmic")
    return resolve_humanize_controls(humanize_pack="balanced")
