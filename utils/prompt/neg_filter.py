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


def filter_negative_by_positive(positive: str, negative: str) -> str:
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
