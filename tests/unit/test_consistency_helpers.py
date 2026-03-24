"""Tests for book/comic consistency prompt helpers."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_positive_block_from_json_like_mapping() -> None:
    from pipelines.book_comic import consistency_helpers as ch

    spec = {
        "character": {"label": "Mira", "hair": "blue bob", "eyes": "gold"},
        "props": ["bronze compass; worn leather strap"],
        "vehicle": "rust orange hatchback, roof rack",
        "setting": {"location": "harbor docks", "time_of_day": "dusk", "weather": "light rain"},
        "lettering_hard": True,
    }
    s = ch.positive_block_from_mapping(spec)
    assert "Mira" in s
    assert "compass" in s.lower()
    assert "hatchback" in s
    assert "harbor" in s
    assert "balloon" in s.lower()


def test_consistency_negative_addon() -> None:
    from pipelines.book_comic import consistency_helpers as ch

    assert ch.consistency_negative_addon("none") == ""
    assert "morphing vehicle" in ch.consistency_negative_addon("light")
    assert "mirrored" in ch.consistency_negative_addon("strong")


def test_load_consistency_json_and_overlay() -> None:
    from pipelines.book_comic import consistency_helpers as ch

    class Args:
        consistency_character = ""
        consistency_costume = "trench coat"
        consistency_props = ""
        consistency_vehicle = ""
        consistency_setting = ""
        consistency_creature = ""
        consistency_palette = ""
        consistency_lighting = ""
        consistency_visual_extra = ""
        consistency_lettering_hard = False

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        p.write_text(
            json.dumps({"character": "old man", "negative_level": "light"}),
            encoding="utf-8",
        )
        spec = dict(ch.load_consistency_json(p))
        ch.overlay_cli_on_spec(spec, Args())
        assert spec.get("costume") == "trench coat"
        assert ch.negative_level_from_spec(spec, None) == "light"


def test_object_prop_clause() -> None:
    from pipelines.book_comic import consistency_helpers as ch

    assert "same recurring prop" in ch.object_prop_clause("silver locket")
    assert "omega" in ch.object_prop_clause("watch", codename="omega")
