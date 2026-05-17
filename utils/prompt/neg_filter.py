"""
Positive / negative prompt conflict resolution for CFG sampling.

Removes from the negative prompt any token that also appears in the positive prompt
so classifier-free guidance does not push away from user-requested concepts.
"""

from __future__ import annotations


def positive_token_set(text: str) -> set:
    """Normalize prompt to a set of tokens (comma/space split, lowercased)."""
    if not (text or text.strip()):
        return set()
    tokens = []
    for part in text.split(","):
        tokens.extend(part.split())
    return {t.strip().lower() for t in tokens if t.strip()}


def filter_negative_by_positive_python(positive: str, negative: str) -> str:
    """
    Remove from the negative prompt any token that also appears in the positive.
    Splits on comma and space; comparison is case-insensitive.
    """
    pos_set = positive_token_set(positive)
    if not pos_set:
        return negative
    kept = []
    for part in negative.split(","):
        words = part.split()
        filtered_words = [w for w in words if w.strip().lower() not in pos_set]
        if filtered_words:
            kept.append(" ".join(filtered_words))
    result = ", ".join(kept).strip()
    return result if result else " "


def filter_negative_by_positive(positive: str, negative: str) -> str:
    """Prefer Rust ``sdx_prompt_ops`` when built; else pure Python."""
    try:
        from sdx_native.prompt_ops_native import maybe_filter_negative_by_positive

        out = maybe_filter_negative_by_positive(positive, negative)
        if out is not None:
            return out
    except ImportError:
        pass
    return filter_negative_by_positive_python(positive, negative)
