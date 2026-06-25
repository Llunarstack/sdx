#!/usr/bin/env python3
"""Shim: python -m scripts.tools video_generate ..."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "pipelines" / "video" / "scripts" / "generate_video.py"
    runpy.run_path(str(target), run_name="__main__")
