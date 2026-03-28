"""Tests for :mod:`sdx_native.native_tools` (no compiled binaries required)."""

import hashlib
from pathlib import Path

from sdx_native import native_tools as nt


def test_file_md5_hex_python_streaming(tmp_path: Path):
    p = tmp_path / "t.bin"
    p.write_bytes(b"abc")
    want = hashlib.md5(b"abc").hexdigest()
    assert nt.file_md5_hex(p, prefer_native_md5=False) == want
    assert nt.file_md5_hex(p, prefer_native_md5=True) == want


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
    assert "rust_sdx_noise_schedule" in s
    assert "rust_file_md5_available" in s
    assert s["rust_file_md5_available"] == bool(s.get("rust_sdx_jsonl_tools"))
    assert "zig_sdx_pathstat" in s
    assert "repo_root" in s
    assert "libsdx_inference_timesteps" in s
    assert "libsdx_beta_schedules" in s
    assert "jsonl_manifest_pure" in s
    assert "libsdx_line_stats" in s
    assert "libsdx_cuda_hwc_to_chw" in s
    assert "libsdx_cuda_flow_matching" in s
    assert "libsdx_cuda_nf4" in s
    assert "libsdx_cuda_sdpa_online" in s
    assert "mojo_or_magic_cli" in s


def test_latent_lib_fallback():
    lib = nt.get_latent_lib()
    n = lib.num_patch_tokens(256, 8, 2)
    assert n == 256
