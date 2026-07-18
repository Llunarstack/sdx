"""User-friendly prompt composer: ``@artist`` + ``+category`` blocks for anything.

People can write natural bundles instead of memorizing danbooru tag order::

    @wlop +character: 1girl, silver hair, red eyes, school uniform
    +building: art deco skyscraper, night city skyline
    +car: glossy red sports car, low angle
    +scene: rainy tokyo street, neon reflections

``@name`` resolves through :mod:`artist_registry` (every artist in your scraped
dataset). ``+noun`` / ``+adjective`` blocks add category anchor tags plus your
free-text traits, merged into a single danbooru-style caption for T5/CLIP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .artist_registry import ArtistRegistry, get_registry
from .artist_tag import expand_artist_mentions

# ``+category: traits`` or ``+category traits`` — category names are case-insensitive.
_BLOCK_RE = re.compile(
    r"\+\s*([a-z][a-z0-9_-]*)\s*:\s*([^+|]+)|"
    r"\+\s*([a-z][a-z0-9_-]*)\s+([^+|]+)",
    re.IGNORECASE,
)

# Category → booru anchor tags prepended before the user's traits.
_CATEGORY_ANCHORS: dict[str, list[str]] = {
    "character": [],
    "char": [],
    "person": [],
    "building": ["architecture", "building"],
    "architecture": ["architecture"],
    "house": ["house", "building"],
    "skyscraper": ["skyscraper", "cityscape"],
    "city": ["cityscape", "city"],
    "landscape": ["landscape", "scenery"],
    "scenery": ["scenery", "landscape"],
    "vehicle": ["vehicle"],
    "car": ["car", "motor vehicle"],
    "truck": ["truck", "motor vehicle"],
    "motorcycle": ["motorcycle", "motor vehicle"],
    "aircraft": ["aircraft"],
    "ship": ["ship", "watercraft"],
    "train": ["train", "railroad"],
    "object": ["still life", "object focus"],
    "prop": ["object focus"],
    "weapon": ["weapon"],
    "food": ["food"],
    "animal": ["animal"],
    "creature": ["creature"],
    "plant": ["plant", "nature"],
    "flower": ["flower"],
    "tree": ["tree"],
    "furniture": ["furniture"],
    "clothing": ["clothes"],
    "outfit": ["clothes"],
    "armor": ["armor"],
    "scene": [],
    "background": ["background"],
    "interior": ["indoor", "interior"],
    "exterior": ["outdoors"],
    "lighting": [],
    "light": [],
    "weather": [],
    "style": [],
    "mood": [],
    "quality": ["masterpiece", "best quality"],
    "camera": [],
    "pose": [],
    "expression": ["expression"],
    "hair": [],
    "eyes": [],
    "skin": [],
    "noun": [],
    "adj": [],
    "adjective": [],
}


@dataclass
class ComposedPrompt:
    positive: str
    artists: list[str] = field(default_factory=list)
    blocks: dict[str, list[str]] = field(default_factory=dict)
    base_text: str = ""


def _split_traits(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
    return parts


def _merge_unique_csv(*chunks: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        for tok in [t.strip() for t in chunk.split(",") if t.strip()]:
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tok)
    return ", ".join(out)


def parse_blocks(prompt: str) -> tuple[str, dict[str, list[str]]]:
    """Return ``(remainder, {category: [traits...]})``."""
    blocks: dict[str, list[str]] = {}

    def _collect(m: re.Match) -> str:
        cat = (m.group(1) or m.group(3) or "").strip().lower()
        body = (m.group(2) or m.group(4) or "").strip()
        if cat and body:
            blocks.setdefault(cat, []).extend(_split_traits(body))
        return " "

    remainder = _BLOCK_RE.sub(_collect, prompt)
    remainder = re.sub(r"\s+", " ", remainder).strip(" ,|")
    return remainder, blocks


def compose_prompt(
    prompt: str,
    *,
    artist_strength: float = 1.0,
    artist_registry: ArtistRegistry | None = None,
    artist_index: str | None = None,
    prepend_quality: bool = False,
) -> ComposedPrompt:
    """Turn ``@artist`` / ``+category`` syntax into a training-style caption."""
    raw = (prompt or "").strip()
    if not raw:
        return ComposedPrompt(positive="")

    reg = artist_registry or (get_registry(artist_index) if artist_index else get_registry())

    base, blocks = parse_blocks(raw)
    expanded, artists = expand_artist_mentions(base, strength=artist_strength, registry=reg)

    segments: list[str] = []
    if prepend_quality:
        segments.append("masterpiece, best quality")

    if expanded:
        segments.append(expanded)

    for cat, traits in blocks.items():
        anchors = list(_CATEGORY_ANCHORS.get(cat, []))
        trait_str = ", ".join(traits)
        if anchors:
            segments.append(_merge_unique_csv(", ".join(anchors), trait_str))
        else:
            segments.append(trait_str)

    positive = _merge_unique_csv(*segments)
    return ComposedPrompt(positive=positive, artists=artists, blocks=blocks, base_text=base)


def compose_prompt_text(
    prompt: str,
    *,
    artist_strength: float = 1.0,
    artist_index: str | None = None,
) -> str:
    """Convenience: return only the composed positive prompt string."""
    return compose_prompt(prompt, artist_strength=artist_strength, artist_index=artist_index).positive
