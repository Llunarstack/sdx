"""Tests for pipelines.book_comic.book_helpers."""

from types import SimpleNamespace

from pipelines.book_comic.book_helpers import (
    BookAccuracyPreset,
    append_sample_py_quality_flags,
    expected_text_for_pick,
    preset_for_book_accuracy,
    resolve_book_sample_settings,
)


def test_preset_balanced_has_multi_candidate():
    p = preset_for_book_accuracy("balanced")
    assert p.sample_candidates >= 2
    assert p.pick_best == "combo"


def test_preset_production_stronger_than_maximum():
    prod = preset_for_book_accuracy("production")
    mx = preset_for_book_accuracy("maximum")
    assert prod.sample_candidates >= mx.sample_candidates
    assert prod.post_sharpen >= mx.post_sharpen


def test_resolve_forces_combo_when_multi_candidate():
    args = SimpleNamespace(
        book_accuracy="balanced",
        sample_candidates=3,
        pick_best="auto",
        boost_quality=False,
        no_boost_quality=False,
        subject_first=False,
        no_subject_first=False,
        save_prompt=False,
        post_sharpen=-1.0,
        post_naturalize=False,
        no_post_naturalize=False,
        post_grain=-1.0,
        post_micro_contrast=-1.0,
        prepend_quality_if_short=False,
        no_prepend_quality_if_short=False,
    )
    s = resolve_book_sample_settings(args)
    assert s.sample_candidates == 3
    assert s.pick_best == "combo"


def test_expected_text_for_pick():
    assert expected_text_for_pick(["", "HELLO"]) == "HELLO"
    assert expected_text_for_pick([]) == ""


def test_append_sample_py_quality_flags():
    cmd = ["python", "sample.py", "--prompt", "x", "--out", "o.png"]
    st = BookAccuracyPreset(
        sample_candidates=2,
        pick_best="combo",
        boost_quality=True,
        subject_first=False,
        save_prompt_sidecar=False,
        post_sharpen=0.0,
        post_naturalize=False,
        post_grain=0.0,
        post_micro_contrast=1.0,
        prepend_quality_if_short=False,
    )
    append_sample_py_quality_flags(cmd, st, pick_expected_text="HI")
    assert "--num" in cmd
    assert "2" in cmd
    assert "--pick-best" in cmd
    assert "--boost-quality" in cmd
