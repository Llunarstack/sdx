"""Tests for pipelines.book_comic.book_helpers shortcomings wiring."""

from types import SimpleNamespace

from pipelines.book_comic import book_helpers


def _args(**overrides):
    base = dict(
        book_accuracy="none",
        sample_candidates=0,
        pick_best="auto",
        no_boost_quality=False,
        boost_quality=False,
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
        shortcomings_mitigation="",
        shortcomings_2d=False,
        no_shortcomings_2d=False,
        art_guidance_mode="",
        art_guidance_photography=False,
        no_art_guidance_photography=False,
        anatomy_guidance="",
        style_guidance_mode="",
        style_guidance_artists=False,
        no_style_guidance_artists=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_preset_defaults_for_book_accuracy():
    fast = book_helpers.preset_for_book_accuracy("fast")
    assert fast.shortcomings_mitigation == "none"
    assert fast.shortcomings_2d is False
    balanced = book_helpers.preset_for_book_accuracy("balanced")
    assert balanced.shortcomings_mitigation == "auto"
    assert balanced.shortcomings_2d is True
    assert balanced.art_guidance_mode == "auto"
    assert balanced.anatomy_guidance == "lite"
    assert balanced.style_guidance_mode == "auto"


def test_resolve_overrides_shortcomings():
    settings = book_helpers.resolve_book_sample_settings(
        _args(
            book_accuracy="balanced",
            shortcomings_mitigation="all",
            no_shortcomings_2d=True,
        )
    )
    assert settings.shortcomings_mitigation == "all"
    assert settings.shortcomings_2d is False


def test_append_sample_py_quality_flags_includes_shortcomings():
    settings = book_helpers.resolve_book_sample_settings(_args(book_accuracy="balanced"))
    cmd = ["python", "sample.py"]
    book_helpers.append_sample_py_quality_flags(cmd, settings, pick_expected_text="")
    assert "--shortcomings-mitigation" in cmd
    assert "--shortcomings-2d" in cmd
    assert "--art-guidance-mode" in cmd
    assert "--anatomy-guidance" in cmd
    assert "--style-guidance-mode" in cmd


def test_append_sample_py_quality_flags_combo_count_forwards_count_args():
    settings = book_helpers.resolve_book_sample_settings(_args(book_accuracy="none", sample_candidates=2, pick_best="combo_count"))
    cmd = ["python", "sample.py"]
    book_helpers.append_sample_py_quality_flags(
        cmd,
        settings,
        pick_expected_text="",
        pick_expected_count=5,
        pick_expected_count_target="objects",
        pick_expected_count_object="coin",
    )
    joined = " ".join(cmd)
    assert "--pick-best combo_count" in joined
    assert "--expected-count 5" in joined
    assert "--expected-count-target objects" in joined
    assert "--expected-count-object coin" in joined


def test_build_extra_ocr_flags_includes_shortcomings():
    settings = book_helpers.resolve_book_sample_settings(_args(book_accuracy="production"))
    flags = book_helpers.build_extra_ocr_sample_flags(settings)
    assert "--shortcomings-mitigation" in flags
    assert "all" in flags
    assert "--shortcomings-2d" in flags
    assert "--art-guidance-mode" in flags
    assert "--anatomy-guidance" in flags
    assert "--style-guidance-mode" in flags

