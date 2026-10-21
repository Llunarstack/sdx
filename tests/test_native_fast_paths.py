"""Tests for ``sdx_native.native_tools`` pure-Python paths and the lazy ``utils.native`` bridge."""

from __future__ import annotations

import json
from pathlib import Path

from sdx_native.native_tools import (
    SHARED_LIBRARIES,
    fnv1a64_file,
    manifest_fingerprint_line,
    merge_jsonl_files,
    native_stack_status,
)


def test_fnv1a64_file_counts_lines_and_bytes(tmp_path: Path):
    p = tmp_path / "t.bin"
    p.write_bytes(b"hello\nworld\n")
    h, lines, nbytes = fnv1a64_file(p)
    assert (lines, nbytes) == (2, 12)
    assert h == fnv1a64_file(p)[0]
    assert manifest_fingerprint_line(p)


def test_merge_jsonl_files_first_row_wins(tmp_path: Path):
    a, b, out = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "out.jsonl"
    a.write_text(json.dumps({"image_path": "x.png", "v": 1}) + "\n", encoding="utf-8")
    b.write_text(
        json.dumps({"image_path": "x.png", "v": 2}) + "\n" + json.dumps({"path": "y.png"}) + "\n", encoding="utf-8"
    )
    merge_jsonl_files([a, b], out, prefer_go=False)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"image_path": "x.png", "v": 1}, {"path": "y.png"}]


def test_native_stack_status_lists_every_library():
    st = native_stack_status()
    assert all(f"lib{name}" in st for name in SHARED_LIBRARIES)


def test_utils_native_lazy_exports():
    import utils.native as n

    assert callable(n.merge_jsonl_files)
    assert callable(n.maybe_image_stats_native)
    assert "native_stack_status" in dir(n)
