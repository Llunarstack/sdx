"""Tests for corpus retrieval, multi-char cast, edit/ideogram skills."""

from __future__ import annotations

import json
from pathlib import Path

from utils.generation.corpus_retrieve import (
    build_corpus_index_from_jsonl,
    hits_to_moodboard_payload,
    retrieve_corpus_refs,
    save_corpus_index_meta,
)
from utils.generation.edit_skills import plan_edit_skills, plan_ideogram_layout
from utils.generation.multi_char_cast import compile_cast_scene, load_cast_scene


def test_corpus_index_and_retrieve(tmp_path: Path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    jsonl = tmp_path / "train.jsonl"
    rows = [
        {"image": str(img), "text": "1girl, catgirl, black hair, eipril", "source": "danbooru"},
        {"image": str(tmp_path / "missing.png"), "text": "1boy, armor, knight", "source": "gelbooru"},
        {"image": str(img), "tags": ["1girl", "lingerie", "cat_ears"], "source": "danbooru"},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    idx = build_corpus_index_from_jsonl(jsonl)
    assert len(idx) == 3
    hits = retrieve_corpus_refs(idx, "catgirl eipril", top_k=2, source_filter="danbooru")
    assert hits
    assert "cat" in hits[0].caption.lower() or "eipril" in hits[0].caption.lower() or hits[0].tags
    mb = hits_to_moodboard_payload(hits)
    assert "images" in mb
    out = save_corpus_index_meta(idx, tmp_path / "index.jsonl")
    assert out.is_file()


def test_multi_char_cast_anti_blend():
    scene = load_cast_scene(
        {
            "actors": [
                {
                    "id": "A",
                    "wardrobe": ["red hoodie"],
                    "pose": ["arms crossed"],
                    "action": ["standing"],
                    "spatial_anchor": "left",
                },
                {
                    "id": "B",
                    "wardrobe": ["blue dress"],
                    "pose": ["waving"],
                    "action": ["smiling"],
                    "spatial_anchor": "right",
                },
            ],
            "relations": [{"a": "A", "b": "B", "kind": "facing"}],
            "anti_blend": True,
        }
    )
    compiled = compile_cast_scene(scene)
    assert "A wears red hoodie" in compiled.positive
    assert "B wears blue dress" in compiled.positive
    assert "outfit swap" in compiled.negative or "character blending" in compiled.negative
    assert compiled.box_layout is not None
    assert len(compiled.box_layout["regions"]) == 2


def test_ideogram_and_edit_skills(tmp_path: Path):
    ideo = plan_ideogram_layout('cool poster with "HELLO"', texts=["HELLO"], palette_hex=["#ff0000"])
    assert "HELLO" in ideo.glyph_texts
    assert ideo.sample_argv
    img = tmp_path / "out.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    plan = plan_edit_skills(
        init_image=str(img),
        prompt="fix face",
        fix_regions=["face"],
        work_dir=tmp_path / "edit",
        width=256,
        height=256,
    )
    assert plan.skills
    assert "--mask" in plan.sample_argv
