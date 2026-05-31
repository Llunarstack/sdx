"""Regression: prompt_emphasis must import when torch is absent (smoke_imports path)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prompt_emphasis_import_without_torch():
    code = """
import sys
sys.modules['torch'] = None
from utils.prompt.prompt_emphasis import parse_prompt_emphasis
cleaned, _ = parse_prompt_emphasis('(hi) there')
assert cleaned == 'hi there', cleaned
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
