"""Smoke-test core CLIs: ``--help`` must exit 0 without importing heavy GPU stacks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Pytest may already load torch/OpenMP; child processes need this on some Windows/conda stacks.
_SUBPROC_ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "PYTHONIOENCODING": "utf-8",
}


def _run_help(*argv: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, *argv, "--help"]
    return subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env=_SUBPROC_ENV,
    )


def test_train_help():
    r = _run_help("train.py")
    assert r.returncode == 0, r.stderr
    assert "--data-path" in r.stdout or "--manifest-jsonl" in r.stdout


def test_sample_help():
    r = _run_help("sample.py")
    assert r.returncode == 0, r.stderr
    assert "--prompt" in r.stdout
    assert "--ckpt" in r.stdout


def test_demo_help():
    r = _run_help("demo.py")
    assert r.returncode == 0, r.stderr
    assert "--ckpt" in r.stdout


def test_inference_help():
    r = _run_help("inference.py")
    assert r.returncode == 0, r.stderr
    assert "--ckpt" in r.stdout


def test_scripts_tools_help():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.tools", "help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env=_SUBPROC_ENV,
    )
    assert r.returncode == 0, r.stderr
    assert "quick_test" in r.stdout
