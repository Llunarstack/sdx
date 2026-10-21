"""Comma-separated prompt tags: split, join, dedupe-merge and the negative/positive conflict filter."""

from __future__ import annotations

import re
from collections.abc import Sequence

from config.defaults._lexicon import merge_csv_unique

__all__ = [
    "append_csv",
    "append_unique",
    "filter_negative_by_positive",
    "join_tags",
    "merge_csv_unique",
    "split_tags",
    "str_list",
    "token_set",
]


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def str_list(value: object) -> list[str]:
    """Stripped non-empty strings from a JSON field that holds a string or a list of strings."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [x.strip() for x in value if isinstance(x, str) and x.strip()]
    return []


def join_tags(tags: Sequence[str]) -> str:
    return ", ".join(t.strip() for t in tags if t and str(t).strip())


def append_unique(base: str, additions: Sequence[str]) -> str:
    return merge_csv_unique(base or "", join_tags([str(t) for t in additions]))


def append_csv(base: str, fragment: str) -> str:
    """Merge ``fragment`` into ``base`` without repeats; an empty fragment leaves ``base`` as written."""
    frag = (fragment or "").strip().strip(",")
    return merge_csv_unique(base, frag) if frag else (base or "").strip()


def token_set(text: str) -> set[str]:
    """Lowercased words of comma- and space-separated prompt text."""
    return {w.lower() for part in (text or "").split(",") for w in part.split()}


_WORD = re.compile(r"[a-z][a-z0-9'-]*")
_STOPWORDS = frozenset(
    "a an the of on in at to and or with without for from by no not very too over overly is are as into".split()
)
# A multi-word negative tag that names a defect stays even when the positive mentions the same thing:
# "bad hands" or "extra fingers" still belong in the negative of a prompt about hands.
_DEFECT_WORDS = frozenset(
    "bad worst low poor poorly ugly deformed malformed disfigured mutated mutation extra missing fused broken "
    "twisted distorted warped melted melting mangled duplicate duplicated cropped wrong incorrect floating "
    "unnatural gross messy artifacts artifact".split()
)


def filter_negative_by_positive(positive: str, negative: str) -> str:
    """Drop negative tags that name what the positive asks for, so CFG does not push away from it.

    A whole tag goes when one of its content words appears in ``positive``, unless it is a multi-word
    defect description.
    Single words are never cut out of a tag: trimming "plastic skin" to "plastic" or "deformed face"
    to "face" would negate something new.
    """
    wanted = {w for w in _WORD.findall((positive or "").lower()) if w not in _STOPWORDS}
    if not wanted:
        return negative
    kept = []
    for tag in split_tags(negative):
        words = {w for w in _WORD.findall(tag.lower()) if w not in _STOPWORDS}
        if words & wanted and not (len(words) > 1 and words & _DEFECT_WORDS):
            continue
        kept.append(tag)
    return join_tags(kept) or " "
