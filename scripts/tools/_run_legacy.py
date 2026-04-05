from __future__ import annotations

import runpy
from pathlib import Path


def run_legacy_script(filename: str) -> None:
    """Execute a legacy flat script from scripts/tools/ by filename."""
    here = Path(__file__).resolve().parent
    target = here / filename
    runpy.run_path(str(target), run_name="__main__")

