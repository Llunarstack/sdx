#!/usr/bin/env python3
"""Deprecated path — use ``python -m scripts.tools prompt_compose``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "tools" / "prompt" / "prompt_compose.py"
sys.argv = [str(_TARGET), *sys.argv[1:]]
runpy.run_path(str(_TARGET), run_name="__main__")
