"""Smoke tests for utils/architecture_map.py (2026 theme → SDX mapping)."""

from utils.architecture.architecture_map import (
    ParityStatus,
    THEMES,
    iter_themes,
    summary_table_md,
    theme_by_id,
    themes_as_dict,
)


def test_themes_non_empty():
    assert len(THEMES) >= 12


def test_theme_by_id():
    t = theme_by_id("hybrid_ar_diffusion")
    assert t is not None
    assert t.status == ParityStatus.PARTIAL
    assert "--num-ar-blocks" in t.cli_flags
    tt = theme_by_id("test_time_inference_scaling")
    assert tt is not None
    assert tt.status == ParityStatus.PARTIAL
    assert "--pick-best" in tt.cli_flags


def test_themes_as_dict_roundtrip_keys():
    rows = themes_as_dict()
    assert all("theme_id" in r and "status" in r for r in rows)


def test_summary_table_md():
    md = summary_table_md()
    assert "Flow matching" in md or "flow" in md.lower()
    assert "|" in md
