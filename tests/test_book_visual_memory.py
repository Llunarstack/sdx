"""Tests for ``pipelines.book_comic.visual_memory``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pipelines.book_comic.visual_memory import load_visual_memory


def test_load_and_entity_override_prompt(tmp_path: Path) -> None:
    data = {
        "version": 1,
        "book_style": "webtoon",
        "entities": {
            "a": {
                "kind": "character",
                "display_name": "A",
                "canonical_look": "blue hair",
                "structure": {"default_viewing_angle": "eye level"},
                "page_overrides": [
                    {
                        "from_page": 1,
                        "to_page": 1,
                        "patch": {"structure": {"default_viewing_angle": "dramatic low angle"}},
                    }
                ],
            }
        },
        "page_patches": [{"from_page": 0, "to_page": 0, "extra_prompt": "establishing wide shot"}],
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    mem = load_visual_memory(p)

    p0 = mem.prompt_fragment_for_page(0)
    assert "establishing wide shot" in p0
    assert "eye level" in p0

    p1 = mem.prompt_fragment_for_page(1)
    assert "dramatic low angle" in p1
    assert "blue hair" in p1


def test_cover_skips_page_overrides(tmp_path: Path) -> None:
    data = {
        "version": 1,
        "entities": {
            "a": {
                "display_name": "A",
                "canonical_look": "round glasses",
                "structure": {"default_viewing_angle": "bird's eye"},
                "page_overrides": [
                    {"from_page": 0, "to_page": 99, "patch": {"canonical_look": "no glasses"}},
                ],
            }
        },
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    mem = load_visual_memory(p)
    cov = mem.prompt_fragment_for_cover()
    assert "round glasses" in cov
    assert "no glasses" not in cov


def test_entities_list_form(tmp_path: Path) -> None:
    data = {
        "version": 1,
        "entities": [
            {"id": "x", "kind": "prop", "display_name": "Sword", "canonical_look": "curved saber"}
        ],
    }
    p = tmp_path / "l.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    mem = load_visual_memory(p)
    assert "x" in mem.entity_ids()
    assert "saber" in mem.prompt_fragment_for_page(0).lower()


def test_apply_entity_page_patch_and_save(tmp_path: Path) -> None:
    p = tmp_path / "mut.json"
    p.write_text(json.dumps({"version": 1, "entities": {}}), encoding="utf-8")
    mem = load_visual_memory(p)
    mem.apply_entity_page_patch("new_char", from_page=3, to_page=10, patch={"canonical_look": "red cloak"})
    out = tmp_path / "out.json"
    mem.save(out)
    mem2 = load_visual_memory(out)
    assert "new_char" in mem2.entity_ids()
    assert "red cloak" in mem2.prompt_fragment_for_page(5)


def test_effective_entity_keyerror(tmp_path: Path) -> None:
    p = tmp_path / "empty_ent.json"
    p.write_text(json.dumps({"version": 1, "entities": {}}), encoding="utf-8")
    m = load_visual_memory(p)
    with pytest.raises(KeyError):
        m.effective_entity("missing", 0)


def test_visual_memory_lettering_and_style_mix(tmp_path: Path) -> None:
    p = tmp_path / "mix.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "book_style": "manga",
                "lettering": {"balloon_style": "oval", "typography_mood": "hand ink"},
                "style_mix": {"preset": "manga_comic"},
                "user_style_anchor": "pastel wash under ink",
                "entities": {},
            }
        ),
        encoding="utf-8",
    )
    mem = load_visual_memory(p)
    frag = mem.prompt_fragment_for_page(0)
    assert "oval" in frag.lower() or "balloon" in frag.lower()
    assert "hybrid" in frag.lower() or "manga" in frag.lower()
    assert "pastel" in frag.lower()


def test_invalid_version(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": 99, "entities": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_visual_memory(p)
