#!/usr/bin/env python3
"""Backward-compatible shim — use strip_ai_git_trailers.py instead."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("strip_ai_git_trailers.py")), run_name="__main__")
    sys.exit(0)
