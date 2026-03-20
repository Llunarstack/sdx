from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.jsonl"
    rows = [
        {"image_path": "a.png", "caption": "cat girl", "negative_caption": "girl"},
        {"image_path": "b.png", "caption": "cat", "negative_caption": "dog"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def run_prompt_lint(args: list[str], jsonl: Path) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / "tools" / "prompt_lint.py"
    cmd = [sys.executable, str(script), str(jsonl), *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def test_prompt_lint_ok_without_fail(sample_jsonl: Path) -> None:
    r = run_prompt_lint([], sample_jsonl)
    assert r.returncode == 0, r.stderr
    assert "pos_neg_overlap_rows: 1" in r.stdout


def test_prompt_lint_fails_on_overlap(sample_jsonl: Path) -> None:
    r = run_prompt_lint(["--fail-on-overlap"], sample_jsonl)
    assert r.returncode == 1
    assert "pos_neg_overlap_rows: 1" in r.stdout

