"""Tests for ``pipelines.book_comic.book_challenging_content``."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.book_comic.book_challenging_content import (
    challenge_pack_negative,
    challenge_pack_positive,
    challenging_content_from_mapping,
    merge_challenge_tags,
)
from pipelines.book_comic.consistency_helpers import positive_block_from_mapping
from pipelines.book_comic.visual_memory import load_visual_memory


def test_merge_challenge_tags_unknown_skipped() -> None:
    out = merge_challenge_tags(["crowd_scale", "not_a_real_tag"])
    assert "crowd" in out.lower()
    assert "distinct" in out.lower()


def test_mature_coherence_requires_nsfw_mode() -> None:
    assert challenge_pack_positive("mature_coherence", safety_mode="") == ""
    assert challenge_pack_positive("mature_coherence", safety_mode="sfw") == ""
    frag = challenge_pack_positive("mature_coherence", safety_mode="nsfw")
    assert frag
    assert "mature-rated" in frag.lower()


def test_max_includes_mature_only_when_nsfw() -> None:
    sfw = challenge_pack_positive("max", safety_mode="sfw")
    nsfw = challenge_pack_positive("max", safety_mode="nsfw")
    assert "mature-rated" not in sfw.lower()
    assert "mature-rated" in nsfw.lower()
    assert "surreal" in nsfw.lower() or "dream" in nsfw.lower()


def test_challenge_pack_negative() -> None:
    assert challenge_pack_negative("none") == ""
    assert "censor" in challenge_pack_negative("surreal_weird").lower()


def test_challenging_content_from_mapping_tags() -> None:
    block = {"pack": "none", "tags": ["hands_heavy"], "extra": "custom tail"}
    out = challenging_content_from_mapping(block, safety_mode="nsfw")
    assert "hand" in out.lower()
    assert "custom tail" in out


def test_positive_block_from_mapping_challenging_content() -> None:
    spec = {
        "character": "test hero",
        "challenging_content": {"pack": "horror_mood", "tags": ["reflections_glass"]},
    }
    out = positive_block_from_mapping(spec, safety_mode="nsfw")
    assert "test hero" in out
    assert "horror" in out.lower() or "dread" in out.lower()
    assert "reflection" in out.lower() or "glass" in out.lower()


def test_visual_memory_challenge_tags_and_weird_notes(tmp_path: Path) -> None:
    data = {
        "version": 1,
        "book_style": "manga",
        "challenge_tags": ["surreal_weird"],
        "weird_character_notes": "three-eyed courier with asymmetrical horns",
        "entities": {},
    }
    p = tmp_path / "vm.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    mem = load_visual_memory(p)
    frag = mem.prompt_fragment_for_page(0, safety_mode="sfw")
    assert "surreal" in frag.lower() or "dream" in frag.lower()
    assert "three-eyed" in frag.lower()
