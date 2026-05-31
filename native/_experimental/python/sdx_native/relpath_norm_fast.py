"""Fast manifest path normalization (posix keys, dedupe)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


def to_posix_key(path: str) -> str:
    """Forward slashes, strip trailing slashes, no ``os.path.realpath`` (stable for manifests)."""
    p = path.strip().replace("\\", "/")
    while p.endswith("/") and len(p) > 1:
        p = p[:-1]
    return p


def relpath_if_under(path: str, root: str) -> str:
    """If ``path`` is under ``root``, return relative posix path; else ``to_posix_key(path)``."""
    try:
        rp = os.path.relpath(path, root)
        if rp.startswith(".."):
            return to_posix_key(path)
        return to_posix_key(rp)
    except Exception:
        return to_posix_key(path)


def unique_preserve_order(paths: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for p in paths:
        k = to_posix_key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def path_bytes_key(path: Path) -> bytes:
    """Absolute resolved path as utf-8 bytes (for sets); may differ across machines."""
    try:
        return str(path.resolve()).encode("utf-8", errors="surrogateescape")
    except Exception:
        return str(path).encode("utf-8", errors="replace")
