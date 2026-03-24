"""Tests for Danbooru general-tag bucketing (underscore-safe matching)."""

from scripts.tools.split_danbooru_general_tags import matches_pattern


def test_matches_pattern_underscore_token():
    assert matches_pattern("school_uniform", "uniform")
    assert matches_pattern("blue_sailor_collar", "sailor")
    assert not matches_pattern("cat", "at")  # '_' + at + '_' not in _cat_
    assert matches_pattern("photorealistic", "photorealistic")
