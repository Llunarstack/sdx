"""Tests for utils.prompt.photo_realism helpers."""

from utils.prompt.photo_realism import (
    infer_photo_realism_controls,
    is_photographic_prompt,
    photo_realism_fragments,
    recommend_photo_post_profile,
)


def test_infer_photo_realism_controls_from_prompt():
    out = infer_photo_realism_controls("cinematic still with teal-orange grade and pro mist filter at golden hour")
    assert out.get("photo_realism_pack") == "cinematic"
    assert out.get("photo_color_grade") == "teal_orange"
    assert out.get("photo_filter") == "pro_mist"
    assert out.get("photo_lighting_technique") == "golden_hour"


def test_photo_realism_fragments_builds_positive_and_negative():
    pos, neg = photo_realism_fragments(
        photo_realism_pack="studio_portrait",
        photo_color_grade="kodak_portra",
        photo_lighting_technique="rembrandt",
        photo_filter="clean_digital",
        photo_grain_style="fine_35mm",
        strength=1.0,
    )
    assert "studio portrait realism" in pos
    assert "kodak portra" in pos
    assert "rembrandt portrait lighting technique" in pos
    assert "fine 35mm-like grain structure" in pos
    assert "plastic skin retouch" in neg


def test_is_photographic_prompt_detects_camera_language():
    assert is_photographic_prompt("cinematic still portrait, 35mm lens, golden hour")
    assert not is_photographic_prompt("anime cel shade line art character sheet")


def test_recommend_photo_post_profile_prefers_realism_metric():
    rec = recommend_photo_post_profile(
        photo_realism_pack="film_analog",
        photo_color_grade="cinestill_800t",
        photo_filter="none",
        photo_grain_style="none",
    )
    assert rec["pick_best_metric"] == "combo_realism"
    assert float(rec["photo_post_strength"]) >= 0.7
    assert rec["photo_grain_style"] == "fine_35mm"

