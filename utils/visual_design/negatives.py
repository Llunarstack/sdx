from __future__ import annotations


def merge_negative_addon(base: str, addon: str) -> str:
    """Append *addon* to *base* negative prompt with comma separation; drops empties."""
    b = (base or "").strip().strip(",")
    a = (addon or "").strip().strip(",")
    if not a:
        return b
    if not b:
        return a
    if a.lower() in b.lower():
        return b
    return f"{b}, {a}"
