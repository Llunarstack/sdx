"""Ensure repo root and ``native/python`` are on ``sys.path`` for tool scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_PY = _REPO_ROOT / "native" / "python"


def ensure_repo_paths() -> Path:
    for p in (_REPO_ROOT, _NATIVE_PY):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return _REPO_ROOT
