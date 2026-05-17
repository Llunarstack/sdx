"""Snapshot dataclass (incl. slots), dict, or ordinary objects to a plain ``dict``."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_plain_dict(obj: Any) -> dict[str, Any]:
    """
    Build a JSON-serialisation-friendly mapping.

    Order: ``dict`` shallow copy → dataclass ``asdict`` → ``__dict__`` copy → ``vars``;
    on total failure returns ``{}`` (manifest / checkpoint writers must not crash).
    """
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    od = getattr(obj, "__dict__", None)
    if isinstance(od, dict):
        return dict(od)
    try:
        return dict(vars(obj))
    except TypeError:
        return {}


__all__ = ["to_plain_dict"]
