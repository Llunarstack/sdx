"""Tests for ``utils.prompt.tags`` comma-tag helpers."""

from __future__ import annotations

from utils.prompt.tags import append_unique, filter_negative_by_positive, merge_csv_unique


def test_filter_negative_python_matches_contract():
    pos = "red dress, sky"
    neg = "red, blurry, dress, artifact"
    out = filter_negative_by_positive(pos, neg)
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
    m = merge_csv_unique("x, y", "Y, z")
    assert m == "x, y, z"
