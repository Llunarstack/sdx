"""Optional ``sdx_line_stats`` C++ library (skip if not built)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdx_native.line_stats_native import count_file_bytes_newlines
from sdx_native.native_tools import line_stats_shared_library_path


def test_line_stats_parity_with_python(tmp_path: Path) -> None:
    if line_stats_shared_library_path() is None:
        pytest.skip("sdx_line_stats not built (cmake native/cpp)")
    p = tmp_path / "t.txt"
    raw = b"a\nb\nc\n"
    p.write_bytes(raw)
    py_n = raw.count(b"\n")
    got = count_file_bytes_newlines(p)
    assert got is not None
    b, n = got
    assert b == len(raw)
    assert n == py_n
