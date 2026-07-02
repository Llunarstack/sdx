"""Expand ``@artist`` mentions in a prompt into trained artist-style tags.

Works for **any** artist tag from danbooru, e621, rule34, etc. — not a fixed list.
When ``artist_index.json`` exists (built from your scraped manifests), aliases and
spellings resolve to the exact caption form the model learned.

    "@Kantoku"  "@wlop"  "@artist:hiten_(hitenkei)"  "@'some artist'"
    strength 1.3 -> "(kantoku:1.3)"
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Unicode letters/digits for international artist names (pixiv, etc.).
_ARTIST_BODY = r"[\w\-.()]+"
_ARTIST_EXPLICIT_RE = re.compile(
    rf"@artist:(?:'([^']+)'|\"([^\"]+)\"|({_ARTIST_BODY}))",
    re.UNICODE,
)
_ARTIST_RE = re.compile(
    rf"@(?:'([^']+)'|\"([^\"]+)\"|({_ARTIST_BODY}))",
    re.UNICODE,
)


def normalize_artist_tag(name: str) -> str:
    """Lowercase, underscores->spaces, collapse whitespace (danbooru caption form)."""
    tag = name.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", tag).strip()


def _resolve_name(raw: str, registry) -> str:
    if registry is not None:
        return registry.resolve(raw)
    return normalize_artist_tag(raw)


def expand_artist_mentions(
    prompt: str,
    *,
    strength: float = 1.0,
    registry=None,
) -> Tuple[str, List[str]]:
    """Replace ``@artist`` mentions with trained artist tags.

    Returns ``(expanded_prompt, artists)``. Pass an :class:`ArtistRegistry` (or
    use ``get_registry()``) to resolve aliases from scraped data.
    """
    if not prompt or "@" not in prompt:
        return prompt, []

    if registry is None:
        try:
            from .artist_registry import get_registry

            registry = get_registry()
        except Exception:
            registry = None

    artists: List[str] = []

    def _wrap(tag: str) -> str:
        artists.append(tag)
        if abs(strength - 1.0) > 1e-3:
            return f"({tag}:{strength:.2f})"
        return tag

    def _sub_explicit(m: re.Match) -> str:
        raw = m.group(1) or m.group(2) or m.group(3) or ""
        tag = _resolve_name(raw, registry)
        return _wrap(tag) if tag else m.group(0)

    def _sub(m: re.Match) -> str:
        raw = m.group(1) or m.group(2) or m.group(3) or ""
        if raw.lower() == "artist":
            return m.group(0)
        tag = _resolve_name(raw, registry)
        if not tag:
            return m.group(0)
        return _wrap(tag)

    expanded = _ARTIST_EXPLICIT_RE.sub(_sub_explicit, prompt)
    expanded = _ARTIST_RE.sub(_sub, expanded)
    return expanded, artists
