"""Tests for pipelines.book_comic.visual_memory_bridge."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.book_comic.visual_memory import load_visual_memory
from pipelines.book_comic.visual_memory_bridge import (
    export_cast_sheet_lines,
    merge_consistency_dicts,
    minimal_consistency_dict_from_visual_memory,
)


def test_export_cast_sheet(tmp_path: Path):
    p = tmp_path / "v.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "entities": {
                    "h": {
                        "kind": "character",
                        "display_name": "Hero",
                        "canonical_look": "red cape",
                        "costume_lock": "armor",
                        "structure": {"proportions": "tall"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    mem = load_visual_memory(p)
    lines = export_cast_sheet_lines(mem)
    assert any("Hero" in ln for ln in lines)
    assert any("red cape" in ln for ln in lines)


def test_minimal_consistency_merge(tmp_path: Path):
    p = tmp_path / "v.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "entities": {"x": {"kind": "character", "display_name": "X", "canonical_look": "blue"}},
            }
        ),
        encoding="utf-8",
    )
    mem = load_visual_memory(p)
    d0 = minimal_consistency_dict_from_visual_memory(mem, page_index=0)
    assert "visual_extra" in d0
    assert d0.get("character")


def test_merge_consistency_dicts_visual_extra():
    m = merge_consistency_dicts({"a": 1, "visual_extra": "one"}, {"visual_extra": "two", "b": 2})
    assert m["a"] == 1 and m["b"] == 2
    assert "one" in m["visual_extra"] and "two" in m["visual_extra"]
