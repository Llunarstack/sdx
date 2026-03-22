"""Tests for :mod:`sdx_native.native_tools` (no compiled binaries required)."""

from pathlib import Path

from sdx_native import native_tools as nt


def test_fnv1a64_small_matches_incremental():
    data = b"hello\nworld\n"
    h1 = nt.fnv1a64_bytes(data)
    h2 = _FNV_STREAM(data)
    assert h1 == h2


def _FNV_STREAM(data: bytes) -> int:
    h = nt._FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * nt._FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def test_fnv1a64_file_matches_bytes(tmp_path: Path):
    p = tmp_path / "t.txt"
    raw = b"a\nb\nc\n"
    p.write_bytes(raw)
    h, lines, nbytes = nt.fnv1a64_file(p)
    assert nbytes == len(raw)
    assert lines == 3
    assert h == nt.fnv1a64_bytes(raw)


def test_native_stack_status_keys():
    s = nt.native_stack_status()
    assert "rust_sdx_jsonl_tools" in s
    assert "zig_sdx_pathstat" in s
    assert "repo_root" in s


def test_latent_lib_fallback():
    lib = nt.get_latent_lib()
    n = lib.num_patch_tokens(256, 8, 2)
    assert n == 256
