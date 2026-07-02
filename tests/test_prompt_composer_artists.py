"""Tests for @artist resolution and +category prompt composer."""

from __future__ import annotations

from utils.prompt.artist_registry import ArtistRegistry
from utils.prompt.artist_tag import expand_artist_mentions, normalize_artist_tag
from utils.prompt.prompt_composer import compose_prompt, parse_blocks


def test_normalize_artist_tag():
    assert normalize_artist_tag("Kantoku") == "kantoku"
    assert normalize_artist_tag("hiten_(hitenkei)") == "hiten (hitenkei)"


def test_registry_resolves_aliases():
    reg = ArtistRegistry()
    reg.add("kantoku", count=100)
    reg.add("wlop", count=50)
    assert reg.resolve("Kantoku") == "kantoku"
    assert reg.resolve("WLOP") == "wlop"
    assert reg.resolve("unknown_artist_xyz") == "unknown artist xyz"


def test_expand_any_artist():
    reg = ArtistRegistry()
    reg.add("some_pixiv_name", count=1)
    out, artists = expand_artist_mentions("@Some_Pixiv_Name", registry=reg)
    assert "some pixiv name" in out
    assert artists == ["some pixiv name"]


def test_parse_blocks():
    base, blocks = parse_blocks(
        "@wlop sunset +character: 1girl, blue hair +building: art deco tower +car: red sports car"
    )
    assert "@" not in base or "wlop" in base.lower()
    assert "character" in blocks
    assert "building" in blocks
    assert "car" in blocks


def test_compose_full():
    cp = compose_prompt(
        "@wlop +character: 1girl, silver hair +vehicle: red sports car, low angle",
        artist_registry=ArtistRegistry(),
    )
    assert "wlop" in cp.positive
    assert "1girl" in cp.positive
    assert "silver hair" in cp.positive
    assert "motor vehicle" in cp.positive or "vehicle" in cp.positive
    assert "red sports car" in cp.positive
