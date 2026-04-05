#!/usr/bin/env python3
"""
Backward-compatible launcher for end-to-end book data prep + training.

Implementation lives in: pipelines/book_comic/scripts/prepare_and_train_book.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MAIN = _ROOT / "pipelines" / "book_comic" / "scripts" / "prepare_and_train_book.py"

if __name__ == "__main__":
    if not _MAIN.is_file():
        raise SystemExit(f"Missing book prepare+train script: {_MAIN}")
    raise SystemExit(subprocess.run([sys.executable, str(_MAIN), *sys.argv[1:]]).returncode)
