"""Keyword matching and comma-tag merging shared by the ``config.defaults`` prompt packs (stdlib only)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "alias_patterns",
    "compile_keywords",
    "dedupe_tags",
    "expand_aliases",
    "matches",
    "merge_csv_unique",
    "word_re",
]


def word_re(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def compile_keywords(keywords: Sequence[str]) -> tuple[re.Pattern, ...]:
    return tuple(word_re(k) for k in keywords)


def matches(prompt: str, patterns: Sequence[re.Pattern]) -> bool:
    return any(p.search(prompt) for p in patterns)


def alias_patterns(aliases: Mapping[str, Sequence[str]]) -> tuple[tuple[re.Pattern, str], ...]:
    """``{canonical: (alias, ...)}`` -> ``((alias_regex, canonical), ...)``."""
    return tuple((word_re(alias), canonical) for canonical, group in aliases.items() for alias in group)


def expand_aliases(prompt: str, patterns: Sequence[tuple[re.Pattern, str]]) -> str:
    """Append the canonical term of every alias found, so keyword specs match synonyms."""
    p = str(prompt or "").strip()
    if not p:
        return p
    found = sorted({canonical for pat, canonical in patterns if pat.search(p)})
    return f"{p}, {', '.join(found)}" if found else p


def dedupe_tags(tags: Iterable[str]) -> list[str]:
    """Stripped, non-empty tags without case-insensitive repeats (first casing and order kept)."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        t = str(tag).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def merge_csv_unique(*chunks: str) -> str:
    """Join comma-separated chunks with :func:`dedupe_tags`."""
    return ", ".join(dedupe_tags(part for chunk in chunks for part in str(chunk or "").split(",")))
