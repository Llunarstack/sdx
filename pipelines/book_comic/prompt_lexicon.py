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
) -> str:
    """Combine user negative with lexicon anti-artifact clauses (dedupe loosely by concat)."""
    u = (user_negative or "").strip()
    if not use_lexicon_negative:
        return u
    extra = combined_comic_negative()
    if production_tier:
        extra = f"{extra}, {PRODUCTION_TIER_NEGATIVE_ADDON}"
    if not u:
        return extra
    return f"{u}, {extra}"


def aspect_dimensions(preset_name: str) -> Tuple[int, int]:
    key = (preset_name or "none").lower().strip()
    return ASPECT_PRESETS.get(key, (0, 0))


def panel_layout_hint(name: str) -> str:
    return PANEL_LAYOUT_HINTS.get((name or "none").lower().strip(), "")
