"""Smoke tests for optional native/ helpers (Rust, pure-Python JSONL, C++ DLL if built)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _rust_jsonl_tools_exe() -> Path:
    exe = ROOT / "native" / "rust" / "sdx-jsonl-tools" / "target" / "release" / "sdx-jsonl-tools.exe"
    if sys.platform != "win32":
        exe = ROOT / "native" / "rust" / "sdx-jsonl-tools" / "target" / "release" / "sdx-jsonl-tools"
    return exe


def _streaming_md5_hex(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
    exe = _rust_jsonl_tools_exe()
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
    exe = _rust_jsonl_tools_exe()
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


def test_rust_file_md5_matches_hashlib(tmp_path: Path) -> None:
    """``sdx-jsonl-tools file-md5`` must match ``hashlib.md5`` (streaming, 1 MiB chunks)."""
    exe = _rust_jsonl_tools_exe()
    if not exe.is_file():
        pytest.skip("build Rust first: cd native/rust/sdx-jsonl-tools && cargo build --release")

    from sdx_native.native_tools import maybe_rust_file_md5_hex, run_rust_file_md5

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    want0 = _streaming_md5_hex(empty)
    assert want0 == hashlib.md5(b"").hexdigest()
    r0 = run_rust_file_md5(empty)
    assert r0.returncode == 0, r0.stderr
    assert r0.stdout.strip().lower() == want0
    assert maybe_rust_file_md5_hex(empty) == want0

    small = tmp_path / "small.bin"
    small.write_bytes(b"hello world\n")
    want1 = _streaming_md5_hex(small)
    assert maybe_rust_file_md5_hex(small) == want1

    # Span >1 MiB so Rust/Python streaming paths both see a multi-chunk read.
    big = tmp_path / "big.bin"
    big.write_bytes(b"y" * ((1 << 20) + 17_001))
    want2 = _streaming_md5_hex(big)
    assert maybe_rust_file_md5_hex(big) == want2


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
