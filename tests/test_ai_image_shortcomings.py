"""Tests for config.defaults.ai_image_shortcomings (prompt mitigation packs)."""

from config.defaults.ai_image_shortcomings import (
    detect_shortcoming_ids,
    merge_csv_unique,
    mitigation_fragments,
    spec_by_id,
)
from data.caption_utils import apply_shortcomings_to_caption_pair


def test_merge_csv_unique_dedupes():
    assert merge_csv_unique("a, b", "b, c") == "a, b, c"


def test_detect_portrait_skin():
    ids = detect_shortcoming_ids("portrait, detailed face, natural skin", include_2d=False)
    assert "skin_detail_tangents" in ids


def test_detect_digital_painting_keywords():
    ids = detect_shortcoming_ids("warrior, digital painting, procreate, detailed", include_2d=False)
    assert "digital_painting" in ids


def test_detect_pixel_art():
    ids = detect_shortcoming_ids("16-bit pixel art sprite, rpg enemy", include_2d=False)
    assert "pixel_digital" in ids


def test_mitigation_auto_digital_art():
    pos, neg = mitigation_fragments("concept art environment, matte painting, artstation", "auto", include_2d_pack=False)
    assert "design" in pos.lower() or "perspective" in pos.lower()
    assert "cutout" in neg.lower() or "mismatch" in neg.lower()


def test_detect_anime_requires_2d_flag():
    ids_no = detect_shortcoming_ids("1girl, anime style, cel shaded", include_2d=False)
    assert "style_drift_2d" not in ids_no
    ids_yes = detect_shortcoming_ids("1girl, anime style, cel shaded", include_2d=True)
    assert "style_drift_2d" in ids_yes


def test_mitigation_auto_non_empty_for_matched():
    pos, neg = mitigation_fragments("woman sitting on a chair in sunlight", "auto", include_2d_pack=False)
    assert pos
    assert neg
    assert "floating" in neg.lower() or "contradictory" in neg.lower()


def test_mitigation_all_includes_many_hints():
    pos, neg = mitigation_fragments("", "all", include_2d_pack=False)
    assert len(pos) > 80
    assert len(neg) > 80


def test_spec_by_id():
    assert spec_by_id("lighting_gi").id == "lighting_gi"


def test_apply_shortcomings_to_caption_pair_none_passthrough():
    c, n = apply_shortcomings_to_caption_pair("hello", "bad", mode="none", include_2d=False)
    assert c == "hello"
    assert n == "bad"


def test_apply_shortcomings_to_caption_pair_auto_merges_negative():
    c, n = apply_shortcomings_to_caption_pair(
        "portrait, woman sitting in sunlight",
        "blurry",
        mode="auto",
        include_2d=False,
    )
    assert "blurry" in n
    assert "floating" in n.lower() or "contradictory" in n.lower()
    assert len(c) > len("portrait, woman sitting in sunlight")
