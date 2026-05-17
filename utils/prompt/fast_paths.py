"""
Fast caption/prompt helpers — prefers ``sdx_native`` (Rust / optimized Python).

Import from here in hot paths (``content_controls``, ``stack``, ``neg_filter``).
"""

from __future__ import annotations

from typing import List, Sequence


def split_tags(text: str) -> List[str]:
    try:
        from sdx_native.caption_csv_fast import split_caption_parts

        return split_caption_parts(text)
    except ImportError:
        return [t.strip() for t in (text or "").split(",") if t.strip()]


def join_tags(tags: Sequence[str]) -> str:
    return ", ".join(t.strip() for t in tags if t and str(t).strip())


def append_unique(base: str, additions: Sequence[str]) -> str:
    frag = join_tags([str(t).strip() for t in additions if str(t).strip()])
    if not frag:
        return (base or "").strip()
    try:
        from sdx_native.prompt_ops_native import maybe_merge_caption_csv

        merged = maybe_merge_caption_csv(base or "", frag)
        if merged is not None:
            return merged
    except ImportError:
        pass
    try:
        from sdx_native.caption_csv_fast import merge_caption_csv

        return merge_caption_csv(base or "", frag)
    except ImportError:
        pass
    existing = split_tags(base)
    seen = {t.lower() for t in existing}
    out = list(existing)
    for token in additions:
        t = str(token).strip()
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return join_tags(out)


def merge_fragments(*parts: str) -> str:
    try:
        from sdx_native.caption_csv_fast import merge_caption_csv

        return merge_caption_csv(*parts)
    except ImportError:
        acc: List[str] = []
        seen: set[str] = set()
        for part in parts:
            for tag in split_tags(part):
                key = tag.lower()
                if key not in seen:
                    seen.add(key)
                    acc.append(tag)
        return join_tags(acc)


def filter_negative_by_positive(positive: str, negative: str) -> str:
    try:
        from sdx_native.prompt_ops_native import maybe_filter_negative_by_positive

        out = maybe_filter_negative_by_positive(positive, negative)
        if out is not None:
            return out
    except ImportError:
        pass
    from utils.prompt.neg_filter import filter_negative_by_positive_python

    return filter_negative_by_positive_python(positive, negative)


def normalize_caption(text: str) -> str:
    try:
        from sdx_native.text_hygiene import normalize_caption_for_training

        return normalize_caption_for_training(text)
    except ImportError:
        return (text or "").strip()


__all__ = [
    "append_unique",
    "filter_negative_by_positive",
    "join_tags",
    "merge_fragments",
    "normalize_caption",
    "split_tags",
]
