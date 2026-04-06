# Style and artist tags from tag-based image boards (PixAI, Danbooru, Gelbooru, etc.).
# Use for strong style conditioning: extract artist/style from captions and use as --style at inference.

__all__ = [
    "ARTIST_STYLE_PATTERNS",
    "ARTIST_STYLE_TAGS",
    "STYLE_PHRASE_PREFIXES",
    "extract_style_from_text",
]

import re
from typing import List, Optional

# Regex and phrases that indicate an artist or style in the caption/prompt.
# Order matters: more specific first. First match is used for style extraction.
ARTIST_STYLE_PATTERNS = [
    # "by artist_name", "art by X", "drawn by X", "illustration by X"
    (re.compile(r"\b(?:art\s+)?by\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\b(?:drawn|painted|illustrated)\s+by\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\bin\s+the\s+style\s+of\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\bstyle\s+of\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\b(?:like|similar to)\s+([^,]+?)(?:\s+style)?(?:,|$)", re.IGNORECASE), 1),
    # Danbooru/Gelbooru: "artist:name" or "artist name"
    (re.compile(r"\bartist:(\S+)", re.IGNORECASE), 1),
    (re.compile(r"\bstyle:(\S+)", re.IGNORECASE), 1),
]

# Prefixes that often start a style phrase (for comma-separated tags).
STYLE_PHRASE_PREFIXES = (
    "by ",
    "art by ",
    "drawn by ",
    "style of ",
    "in the style of ",
    "artist:",
    "style:",
    "art style ",
    "painting style ",
    "illustration style ",
)

# Well-known artist/style tags from tag boards (PixAI, Danbooru, etc.).
# When these appear in a caption, we can use them for style conditioning.
# Add more as needed; model learns from your dataset's actual tags.
ARTIST_STYLE_TAGS: List[str] = [
    # Anime / illustration (Danbooru-style; often use underscore)
    "makoto_shinkai",
    "ghibli",
    "studio_ghibli",
    "miyazaki",
    "hayao_miyazaki",
    "kyoani",
    "shaft",
    "trigger",
    "ufotable",
    "a-1_pictures",
    "kyoto_animation",
    "wit_studio",
    "bones_(studio)",
    "mappa",
    "clamp_(studio)",
    "ufotable_style",
    "trigger_style",
    "digital_art",
    "anime_screencap",
    "official_art",
    "fan_art",
    # Danbooru / Rule34 frequent artist-style tags (2D)
    "sakimichan",
    "wlop",
    "guweiz",
    "redjuice",
    "kantoku",
    "lam_(artist)",
    "ask_(artist)",
    "toi8",
    "huke_(artist)",
    "saitom",
    "krenz_cushart",
    "daito",
    # Booru 3D / game-like style anchors
    "anime_3d",
    "toon_3d",
    "cel_shaded_3d",
    "stylized_3d",
    "game_cg",
    "render",
    "octane_render",
    "eevee_render",
    "blender_(software)",
    "mmd",
    "source_filmmaker",
    "genshin_impact_style",
    "honkai_star_rail_style",
    "zenless_zone_zero_style",
    # Tagboard prompt syntax helpers
    "artist:",
    "style:",
    "masterpiece",
    "best_quality",
    "highres",
    "absurdres",
    "official_style",
    "by_artist",
    # Art styles (PixAI / general)
    "oil_painting",
    "watercolor",
    "digital_painting",
    "concept_art",
    "cel_shading",
    "soft_lighting",
    "dramatic_lighting",
    "cinematic",
    "fantasy_art",
    "character_design",
    "environment_art",
    "pixiv",
    "artstation",
    "behance",
    # Can add more: e.g. specific artist names from your dataset
]


def extract_style_from_text(text: str, known_tags: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract a style or artist string from caption/prompt for style conditioning.
    Returns the first match from ARTIST_STYLE_PATTERNS, or a known tag if present.
    """
    if not (text and text.strip()):
        return None
    text_lower = text.lower()
    tags = known_tags or ARTIST_STYLE_TAGS

    # 1) Try regex patterns
    for pattern, group in ARTIST_STYLE_PATTERNS:
        m = pattern.search(text)  # use original text so captured style keeps case
        if m:
            style = m.group(group).strip()
            if len(style) > 1 and len(style) < 120:
                return style

    # 2) Check for known artist/style tags (e.g. "artist:miyazaki" or "oil painting")
    for tag in tags:
        # Tag might use underscore; caption might use space
        tag_alt = tag.replace("_", " ")
        if tag in text_lower or tag_alt in text_lower:
            return tag.replace("_", " ").strip()

    return None
