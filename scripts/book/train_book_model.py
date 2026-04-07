#!/usr/bin/env python3
"""
Backward-compatible launcher for book/comic/manga training.

Implementation lives in: pipelines/book_comic/scripts/train_book_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.book_comic.scripts.train_book_model import main

if __name__ == "__main__":
    raise SystemExit(main())
