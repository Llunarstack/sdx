"""Smoke tests for optional native/ helpers (Rust, pure-Python JSONL, C++ DLL if built)."""

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
        {"image_path": "a.png", "caption": "hello world"},
        {"path": "b.png", "text": "short"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_python_jsonl_stat_runs(sample_jsonl: Path) -> None:
    from sdx_native.jsonl_manifest_pure import jsonl_stat_text

    out = jsonl_stat_text(sample_jsonl)
    assert "rows_ok: 2" in out


def test_rust_jsonl_tools_stats(sample_jsonl: Path) -> None:
    exe = ROOT / "native" / "rust" / "sdx-jsonl-tools" / "target" / "release" / "sdx-jsonl-tools.exe"
    if sys.platform != "win32":
        exe = ROOT / "native" / "rust" / "sdx-jsonl-tools" / "target" / "release" / "sdx-jsonl-tools"
    if not exe.is_file():
        pytest.skip("build Rust first: cd native/rust/sdx-jsonl-tools && cargo build --release")
    r = subprocess.run(
        [str(exe), "stats", str(sample_jsonl)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert "rows_ok: 2" in r.stdout


def test_rust_jsonl_promptlint_runs(sample_jsonl: Path) -> None:
    exe = ROOT / "native" / "rust" / "sdx-jsonl-tools" / "target" / "release" / "sdx-jsonl-tools.exe"
    if sys.platform != "win32":
        exe = ROOT / "native" / "rust" / "sdx-jsonl-tools" / "target" / "release" / "sdx-jsonl-tools"
    if not exe.is_file():
        pytest.skip("build Rust first: cd native/rust/sdx-jsonl-tools && cargo build --release")
    r = subprocess.run(
        [str(exe), "prompt-lint", str(sample_jsonl)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert "promptlint:" in r.stdout
    assert "rows_ok: 2" in r.stdout


def test_cpp_sdx_latent_dll_exists() -> None:
    candidates = [
        ROOT / "native" / "cpp" / "build" / "Debug" / "sdx_latent.dll",
        ROOT / "native" / "cpp" / "build" / "Release" / "sdx_latent.dll",
        ROOT / "native" / "cpp" / "build" / "libsdx_latent.so",
    ]
    if not any(p.is_file() for p in candidates):
        pytest.skip("build C++ first: native/cpp (cmake --build build)")
    dll = next(p for p in candidates if p.is_file())
    assert dll.stat().st_size > 0


def test_python_promptlint_runs(sample_jsonl: Path) -> None:
    from sdx_native.jsonl_manifest_pure import promptlint_text

    out, code = promptlint_text(sample_jsonl)
    assert code == 0
    assert "promptlint:" in out
    assert "rows_ok: 2" in out
