"""
**Hybrid book styles** — fuse two sequential-art idioms (e.g. manga × western comic) with a
single coherent prompt layer. Complements ``book_style`` / ``lexicon_style`` and visual-memory
``style_mix``.

Use ``--style-fusion-preset`` on ``generate_book.py`` or set ``style_mix`` in visual-memory JSON.
"""

from __future__ import annotations

from typing import Dict

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

# Short idiom anchors (avoid importing visual_memory to prevent cycles)
_IDIOM: Dict[str, str] = {
    "manga": (
        "manga-native pacing: expressive eyes, screentone-friendly values, speed-line grammar, "
        "right-to-left flow where appropriate"
    ),
    "webtoon": (
        "vertical-scroll webtoon readability: large shapes, soft painterly blends, thumb-scroll staging"
    ),
    "graphic_novel": (
        "graphic-novel wash and ink fusion: painterly passages with disciplined sequential gutters"
    ),
    "comic_us": (
        "American comic construction: bold spot blacks, zip texture discipline, left-to-right gutters, "
        "heroic figure drawing"
    ),
    "illustration": "illustration-forward finish: plate clarity, editorial focal hierarchy",
    "manhwa": "full-color manhwa finish: clean color holds, soft gradients, web-first readability",
    "storyboard": "production storyboard linework, readable staging, thumbnail energy not final render",
}

_HARMONIZE = (
    "single unified art direction: one production pipeline blending both influences, "
    "no split-screen style clash, consistent line family across panels"
)


# Preset name -> (primary_key, secondary_key, extra_tail)
_FUSION_KEYS: Dict[str, tuple[str, str, str]] = {
    "manga_comic": (
        "manga",
        "comic_us",
        "hybrid manga-comic book: Japanese figure expression with western ink holds and balloon grammar",
    ),
    "webtoon_manga": (
        "webtoon",
        "manga",
        "webtoon vertical color story with manga black-outline accents on key characters",
    ),
    "graphic_comic": (
        "graphic_novel",
        "comic_us",
        "painterly graphic novel page with comic-book spot black discipline in focal figures",
    ),
    "manhwa_western": (
        "manhwa",
        "comic_us",
        "full-color manhwa rendering with western panel rhythm and sfx lettering habits",
    ),
    "illustration_manga": (
        "illustration",
        "manga",
        "illustrated book plate polish with manga silhouette clarity and expressive faces",
    ),
}


def freeform_style_fusion(primary: str, secondary: str) -> str:
    """Fuse two canonical style keys (see visual-memory ``book_style`` names)."""
    a = (primary or "manga").strip().lower().replace("-", "_").replace(" ", "_")
    b = (secondary or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not b or b == a:
        return ""
    pa = _IDIOM.get(a, _IDIOM.get("manga"))
    pb = _IDIOM.get(b, "")
    if not pb:
        return merge_prompt_fragments(
            pa,
            f"secondary influence blended throughout: {b.replace('_', ' ')}",
            _HARMONIZE,
        )
    return merge_prompt_fragments(
        f"primary idiom ({a}): {pa}",
        f"secondary idiom ({b}): {pb}",
        _HARMONIZE,
    )


def fusion_fragment_from_preset(preset: str) -> str:
    """Return merged prompt for a named hybrid, or empty if unknown/none."""
    key = (preset or "").strip().lower().replace("-", "_")
    if not key or key in ("none", "off"):
        return ""
    row = _FUSION_KEYS.get(key)
    if not row:
        return ""
    p, s, tail = row
    return merge_prompt_fragments(freeform_style_fusion(p, s), tail)


def fusion_from_cli(*, preset: str, secondary: str, primary_book_style: str = "manga") -> str:
    """
    CLI resolver: *preset* wins when set; else *secondary* + *primary_book_style* from pipeline
    (e.g. ``--book-type`` / memory ``book_style``).
    """
    frag = fusion_fragment_from_preset(preset)
    if frag:
        return frag
    sec = (secondary or "").strip()
    if not sec:
        return ""
    return freeform_style_fusion(primary_book_style, sec)


def list_fusion_presets() -> tuple[str, ...]:
    return tuple(sorted(_FUSION_KEYS.keys()))


def primary_style_from_book_type(book_type: str) -> str:
    """Map ``generate_book`` ``--book-type`` to a fusion primary idiom."""
    m = {
        "manga": "manga",
        "comic": "comic_us",
        "novel_cover": "illustration",
        "storyboard": "storyboard",
    }
    return m.get((book_type or "manga").lower().strip(), "manga")
