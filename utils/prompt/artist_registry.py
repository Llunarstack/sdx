"""``@artist`` mentions and the artist name registry built from scraped booru manifests.

Resolves ``@AnyArtist`` (``@Kantoku``, ``@artist:hiten_(hitenkei)``, ``@'some artist'``) to the exact tag
spelling the model saw in training (danbooru / e621 / rule34 artist tags), for every artist in your
downloaded dataset; unknown names still normalize to booru caption form. Strength 1.3 -> ``(kantoku:1.30)``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArtistEntry:
    canonical: str
    aliases: set[str] = field(default_factory=set)
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "aliases": sorted(self.aliases),
            "count": self.count,
        }


def _alias_keys(name: str) -> set[str]:
    """Generate lookup keys for an artist name (case/underscore/space variants)."""
    raw = name.strip()
    if not raw:
        return set()
    norm = normalize_artist_tag(raw)
    keys = {
        raw.lower(),
        norm.lower(),
        norm.lower().replace(" ", "_"),
        norm.lower().replace(" ", ""),
        raw.lower().replace("_", " "),
        raw.lower().replace(" ", "_"),
    }
    return {k for k in keys if k}


class ArtistRegistry:
    """In-memory artist alias map with fuzzy resolution."""

    def __init__(self) -> None:
        self._by_key: dict[str, str] = {}  # alias key -> canonical caption tag
        self._entries: dict[str, ArtistEntry] = {}  # canonical -> entry

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, name: str, *, count: int = 1) -> None:
        canonical = normalize_artist_tag(name)
        if not canonical:
            return
        ent = self._entries.setdefault(canonical, ArtistEntry(canonical=canonical))
        ent.count += int(count)
        for key in _alias_keys(name):
            ent.aliases.add(key)
            # Prefer higher-count artist when aliases collide across artists.
            prev = self._by_key.get(key)
            if prev is None or ent.count >= self._entries[prev].count:
                self._by_key[key] = canonical

    def resolve(self, query: str) -> str:
        """Map a user-typed artist handle to the trained caption tag."""
        q = query.strip()
        if not q:
            return q
        for key in _alias_keys(q):
            if key in self._by_key:
                return self._by_key[key]
        # Unknown artist: still normalize to booru caption form (works after full-site training).
        return normalize_artist_tag(q)

    def known(self, query: str) -> bool:
        return any(k in self._by_key for k in _alias_keys(query))

    def save(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in sorted(self._entries.items())}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ArtistRegistry:
        reg = cls()
        p = Path(path)
        if not p.is_file():
            return reg
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for _key, val in raw.items():
                if isinstance(val, dict):
                    name = val.get("canonical") or _key
                    reg.add(name, count=int(val.get("count") or 0))
                    for alias in val.get("aliases") or []:
                        reg.add(str(alias), count=0)
                elif isinstance(val, (int, float)):
                    reg.add(str(_key), count=int(val))
                else:
                    reg.add(str(_key), count=1)
        return reg


_GLOBAL: ArtistRegistry | None = None


def default_index_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("SDX_ARTIST_INDEX", "").strip()
    if env:
        paths.append(Path(env))
    data = os.environ.get("SDX_DATA", "").strip()
    if data:
        paths.append(Path(data) / "artist_index.json")
        paths.append(Path(data) / "combined" / "artist_index.json")
    # repo-relative fallbacks
    root = Path(__file__).resolve().parents[2]
    paths.append(root / "data" / "artist_index.json")
    return paths


def get_registry(path: str | os.PathLike[str] | None = None) -> ArtistRegistry:
    """Return a cached registry (loads from disk on first call)."""
    global _GLOBAL
    if path is not None:
        return ArtistRegistry.load(path)
    if _GLOBAL is None:
        _GLOBAL = ArtistRegistry()
        for p in default_index_paths():
            if p.is_file():
                _GLOBAL = ArtistRegistry.load(p)
                break
    return _GLOBAL


def build_from_manifests(manifest_paths: Iterable[str | os.PathLike[str]]) -> ArtistRegistry:
    """Scan JSONL manifests for ``artist_tags`` fields and tag captions."""
    reg = ArtistRegistry()
    for mp in manifest_paths:
        p = Path(mp)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            artists = row.get("artist_tags") or []
            if isinstance(artists, str):
                artists = [a.strip() for a in artists.split(",") if a.strip()]
            for a in artists:
                reg.add(str(a), count=1)
    return reg


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
) -> tuple[str, list[str]]:
    """Replace ``@artist`` mentions with trained artist tags.

    Returns ``(expanded_prompt, artists)``. Pass an :class:`ArtistRegistry` (or
    use ``get_registry()``) to resolve aliases from scraped data.
    """
    if not prompt or "@" not in prompt:
        return prompt, []

    if registry is None:
        try:
            registry = get_registry()
        except Exception:
            registry = None

    artists: list[str] = []

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
