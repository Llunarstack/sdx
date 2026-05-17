"""Tests for sdx_native prompt ops (Rust optional) and fast_paths."""

from __future__ import annotations

from utils.prompt.fast_paths import append_unique, filter_negative_by_positive, merge_fragments
from utils.prompt.neg_filter import filter_negative_by_positive_python


def test_filter_negative_python_matches_contract():
    pos = "red dress, sky"
    neg = "red, blurry, dress, artifact"
    out = filter_negative_by_positive_python(pos, neg)
    assert "blurry" in out
    assert "red" not in out.split()
    assert "dress" not in out.split()


def test_filter_negative_unified():
    pos = "1girl, blue eyes"
    neg = "1girl, extra limbs, blue"
    out = filter_negative_by_positive(pos, neg)
    assert "extra" in out or "limbs" in out


def test_append_unique_and_merge():
    assert "c" in append_unique("a, b", ["B", "c"])
    m = merge_fragments("x, y", "Y, z")
    assert m == "x, y, z"


def test_rust_prompt_ops_if_built():
    from sdx_native.prompt_ops_native import get_prompt_ops_lib, maybe_filter_negative_by_positive

    lib = get_prompt_ops_lib()
    if not lib.available:
        return
    out = maybe_filter_negative_by_positive("cat, dog", "cat, blurry")
    assert out is not None
    assert "cat" not in out.lower().split() or out.strip() == " "
