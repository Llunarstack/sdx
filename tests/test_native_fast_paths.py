"""Tests for ``native/python/sdx_native`` fast-path helpers (numpy / scans; optional C DLL)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sdx_native.buffer_scan_fast import scan_file_chunks
from sdx_native.c_buffer_stats_native import count_newlines_py, newline_and_sum, sum_bytes_py
from sdx_native.caption_csv_fast import merge_caption_csv, normalize_caption_csv, token_overlap_ratio
from sdx_native.manifest_line_index import build_line_offset_table, read_line_at
from sdx_native.native_fast_stack_status import fast_numpy_stack_status
from sdx_native.native_tools import fnv1a64_file
from sdx_native.numpy_chw_pack import hwc_u8_to_chw_f32
from sdx_native.numpy_latent_ops import center_crop_hw, latent_mse
from sdx_native.prompt_hash_fast import blake2b_hex, normalized_caption_key
from sdx_native.relpath_norm_fast import to_posix_key, unique_preserve_order
from sdx_native.timestep_grid_fast import linspace_int_timesteps
from sdx_native.uint8_histogram_fast import luminance_histogram_u8


def test_normalize_caption_csv_dedupes():
    s = normalize_caption_csv("foo, bar, FOO,  baz , bar")
    assert s == "foo, bar, baz"


def test_merge_caption_csv():
    assert "a" in merge_caption_csv("a, b", "b, c")


def test_token_overlap():
    assert token_overlap_ratio("a, b", "b, c") > 0


def test_scan_file_chunks_matches_fnv1a64_file(tmp_path: Path):
    p = tmp_path / "t.bin"
    p.write_bytes(b"hello\nworld\n")
    h1, l1, b1 = scan_file_chunks(p)
    h2, l2, b2 = fnv1a64_file(p)
    assert (h1, l1, b1) == (h2, l2, b2)


def test_newline_and_sum_python_fallback():
    buf = b"a\nb\nc"
    n, s = newline_and_sum(buf)
    assert n == count_newlines_py(buf) == 2
    assert s == sum_bytes_py(buf)


def test_hwc_to_chw():
    x = np.zeros((4, 5, 3), dtype=np.uint8)
    x[:, :, 0] = 255
    y = hwc_u8_to_chw_f32(x, scale="n11")
    assert y.shape == (3, 4, 5)
    assert y[0, 0, 0] == pytest.approx(1.0)


def test_center_crop_and_mse():
    a = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    b = center_crop_hw(a, target_h=2, target_w=2)
    assert b.shape == (1, 2, 2)
    assert latent_mse(b, b) == 0.0


def test_luma_histogram():
    im = np.zeros((2, 2, 3), dtype=np.uint8)
    im[:, :, :] = [10, 20, 30]
    h = luminance_histogram_u8(im)
    assert h.shape == (256,)
    assert h.sum() == 4


def test_timestep_linspace():
    t = linspace_int_timesteps(5, 999, 0)
    assert t.shape == (5,)
    assert t[0] >= t[-1]


def test_manifest_line_offsets(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    rows = [{"a": 1}, {"b": 2}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    off = build_line_offset_table(p)
    assert len(off) == 2
    raw0 = read_line_at(p, off[0][0], off[0][1])
    assert json.loads(raw0.decode()) == {"a": 1}


def test_blake2b_and_posix():
    assert len(blake2b_hex("x")) == 32
    assert to_posix_key("a\\b\\c") == "a/b/c"
    assert unique_preserve_order(["a", "a/b", "a"]) == ["a", "a/b"]


def test_normalized_caption_key_stable():
    assert normalized_caption_key("  foo , bar ") == normalized_caption_key("foo, bar,, ")
    assert normalized_caption_key("x") == normalized_caption_key(" x ")


def test_fast_numpy_stack_status():
    st = fast_numpy_stack_status()
    assert st["numpy"] is True
    assert "modules" in st


def test_utils_native_reexports_package_and_tools():
    from utils.native import merge_jsonl_files, native_stack_status, normalize_caption_csv, snr_from_alpha_cumprod_numpy

    assert callable(merge_jsonl_files)
    assert callable(normalize_caption_csv)
    assert callable(snr_from_alpha_cumprod_numpy)
    stack = native_stack_status()
    assert "fast_numpy_helpers" in stack
    assert stack["fast_numpy_helpers"].get("numpy") is True
