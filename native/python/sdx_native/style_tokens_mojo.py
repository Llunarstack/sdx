"""
Optional Mojo fast path for comma merge/dedupe and style fingerprints.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from sdx_native.native_tools import REPO_ROOT, mojo_cli_path

_MOJO_SRC = REPO_ROOT / "native" / "mojo" / "src" / "sdx_style_tokens.mojo"


def mojo_available() -> bool:
    return mojo_cli_path() is not None and _MOJO_SRC.is_file()


def _run_mojo_cli(extra: List[str], *, timeout: float = 60) -> Optional[subprocess.CompletedProcess[str]]:
    cli = mojo_cli_path()
    if not cli or not _MOJO_SRC.is_file():
        return None
    try:
        return subprocess.run(
            [cli, str(_MOJO_SRC), *extra],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_MOJO_SRC.parent.parent),
        )
    except Exception:
        return None


def maybe_merge_comma_dedupe(text: str) -> Optional[str]:
    if not mojo_available():
        return None
    r = _run_mojo_cli(["merge", text or ""])
    if r is None or r.returncode != 0:
        return None
    return (r.stdout or "").strip()


def maybe_style_fingerprint(text: str) -> Optional[int]:
    if not mojo_available():
        return None
    r = _run_mojo_cli(["fingerprint", text or ""])
    if r is None or r.returncode != 0:
        return None
    raw = (r.stdout or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def merge_comma_dedupe(text: str) -> str:
    out = maybe_merge_comma_dedupe(text)
    if out is not None:
        return out
    seen: set[str] = set()
    parts: list[str] = []
    for part in (text or "").split(","):
        p = part.strip()
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(p)
    return ", ".join(parts)


def style_fingerprint(text: str) -> int:
    from sdx_native.style_ops_native import maybe_fnv1a64

    fp = maybe_fnv1a64(text)
    if fp is not None:
        return fp
    m = maybe_style_fingerprint(text)
    if m is not None:
        return m
    # Python FNV fallback
    h = 0xCBF29CE484222325
    for b in (text or "").encode("utf-8"):
        h ^= b
        h = (h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return int(h)


__all__ = [
    "merge_comma_dedupe",
    "mojo_available",
    "maybe_merge_comma_dedupe",
    "maybe_style_fingerprint",
    "style_fingerprint",
]
