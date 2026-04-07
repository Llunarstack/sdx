#!/usr/bin/env python3
"""
Backward-compatible launcher for end-to-end book data prep + training.

Implementation lives in: pipelines/book_comic/scripts/prepare_and_train_book.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.book_comic.scripts.prepare_and_train_book import main

if __name__ == "__main__":
    raise SystemExit(main())
