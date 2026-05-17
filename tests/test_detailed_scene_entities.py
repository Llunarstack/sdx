"""Tests for detailed multi-entity / pose / physics prompt boosting."""

from __future__ import annotations

from utils.prompt.detailed_scene_entities import (
    apply_detailed_scene_boost,
    detailed_scene_warrants_boost,
    extract_key_segments,
)


def test_warrant_group():
    assert detailed_scene_warrants_boost("a crowd of knights in the square")


def test_warrant_count():
    assert detailed_scene_warrants_boost("five people waiting at a bus stop")


def test_no_warrant_short():
    assert not detailed_scene_warrants_boost("a red apple")


def test_apply_on_adds_pose_and_negatives():
    p, neg = apply_detailed_scene_boost(
        "two warriors dueling on a stone bridge over water",
        "on",
        strength="lite",
    )
    assert "water" in p.lower() or "bridge" in p.lower()
    assert "meniscus" in p.lower() or "liquid" in p.lower() or "spatially" in p.lower()
    assert neg
    assert "merged" in neg.lower() or "floating" in neg.lower()


def test_extract_key_segments():
    s = extract_key_segments("masterpiece, a tall knight in armor, glowing sword")
    assert any("knight" in x.lower() for x in s)


def test_off_noop():
    p, neg = apply_detailed_scene_boost("crowd scene", "off")
    assert p == "crowd scene" and neg == ""


def test_auto_skips_when_not_warranted():
    p, neg = apply_detailed_scene_boost("a red apple on a table", "auto")
    assert p == "a red apple on a table" and neg == ""
