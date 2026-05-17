"""Tests for multi-instance scene prompting (posters, stacks, turnarounds)."""

from __future__ import annotations

import io

from utils.prompt.multi_instance_scene import (
    apply_multi_instance_preset,
    describe_limitation_short,
    multi_instance_auto_settings,
    print_multi_instance_hints,
)


def test_none_is_noop():
    p, neg, n, ec, note = apply_multi_instance_preset("a cat", "none")
    assert p == "a cat" and neg == "" and n == 1 and ec is None


def test_distinct_objects_augment():
    p, neg, n, ec, note = apply_multi_instance_preset(
        "wall of concert posters",
        "distinct_objects",
        user_expected_count=0,
    )
    assert "multiple separate framed" in p.lower()
    assert "cloned identical posters" in neg
    assert n >= 8
    assert ec is None
    assert note


def test_user_count_sets_expected_hint():
    _p, _neg, _n, ec, _note = apply_multi_instance_preset(
        "five ads",
        "distinct_objects",
        user_expected_count=5,
    )
    assert ec == 5


def test_turnaround_suggested_count():
    _p, _neg, _n, ec, _note = apply_multi_instance_preset(
        "hero design",
        "turnaround_sheet",
        user_expected_count=0,
    )
    assert ec == 4
    assert "turnaround model sheet" in _p.lower()


def test_limitation_blurb_nonempty():
    assert "memory" in describe_limitation_short().lower()


def test_panel_strip_augment():
    p, neg, n, _ec, _note = apply_multi_instance_preset("my comic", "panel_strip")
    assert "storyboard strip" in p.lower()
    assert "panels melting" in neg
    assert n >= 8


def test_auto_settings_people_vs_objects():
    assert multi_instance_auto_settings("group_portrait")["expected_count_target"] == "people"
    assert multi_instance_auto_settings("distinct_objects")["expected_count_target"] == "objects"
    assert multi_instance_auto_settings("group_portrait")["pick_best"] == "combo_count"


def test_print_hints_smoke():
    buf = io.StringIO()
    print_multi_instance_hints("stacked_media", stream=buf)
    assert "dissect-refs" in buf.getvalue().lower()
