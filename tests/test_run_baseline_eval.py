from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def baseline_pack(tmp_path: Path) -> Path:
    p = tmp_path / "pack.json"
    p.write_text('[{"id": "a", "prompt": "hello"}]', encoding="utf-8")
    return p


def test_run_baseline_eval_dry_run(capsys: pytest.CaptureFixture[str], baseline_pack: Path) -> None:
    ckpt = _REPO_ROOT / "README.md"
    argv = [
        "run_baseline_eval.py",
        "--ckpt",
        str(ckpt),
        "--pack",
        str(baseline_pack),
        "--out-dir",
        str(baseline_pack.parent / "out"),
    ]
    with mock.patch.object(sys, "argv", argv):
        from examples.run_baseline_eval import main

        assert main() == 0
    out = capsys.readouterr().out
    assert "WOULD RUN:" in out
    assert "sample.py" in out
    assert "hello" in out
