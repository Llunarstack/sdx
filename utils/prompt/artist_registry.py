"""Artist name registry built from scraped booru manifests.

Resolves ``@AnyArtist`` mentions to the exact tag spelling the model saw in
training (danbooru / e621 / rule34 artist tags). Works for every artist present
in your downloaded dataset — not a fixed allowlist.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .artist_tag import normalize_artist_tag

_ALIAS_SEP = re.compile(r"[\s_]+")


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
