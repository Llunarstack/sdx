"""utils.prompt.prompt_layout — layered prompt compiler."""

import json

import pytest

from utils.prompt.prompt_layout import (
    PRESET_SECTION_ORDER,
    compile_prompt_layout,
    layout_tail_suffix,
    load_prompt_layout_file,
    merge_prompt_with_layout,
    substitute_compiled_layout_in_t5_prompt,
    t5_segment_texts_for_full_prompt,
    t5_segment_texts_from_layout,
    triple_clip_caption,
)


def test_compile_minimal_intent_and_subject():
    c = compile_prompt_layout({"intent": "a hero portrait", "subjects": ["1girl, armor, cape"]})
    assert "hero portrait" in c.positive
    assert "1girl" in c.positive or "armor" in c.positive


def test_subject_first_puts_intent_before_quality():
    c = compile_prompt_layout(
        {
            "preset_order": "subject_first",
            "intent": "ZZZ_intent",
            "quality": "QQQ_quality",
            "subjects": ["SSS_subject"],
        }
    )
    pos = c.positive
    assert pos.index("ZZZ_intent") < pos.index("SSS_subject")
    assert pos.index("SSS_subject") < pos.index("QQQ_quality")


def test_quality_first_orders_quality_early():
    c = compile_prompt_layout(
        {
            "preset_order": "quality_first",
            "quality": "QQQ",
            "intent": "III",
            "subjects": ["SSS"],
        }
    )
    assert c.positive.index("QQQ") < c.positive.index("III")


def test_labeled_subjects_and_negatives():
    c = compile_prompt_layout(
        {
            "subjects": [
                {"label": "A", "tokens": ["red dress"], "negative": ["blue dress"]},
                {"label": "B", "tokens": ["blue suit"]},
            ],
            "negative": ["watermark"],
        }
    )
    assert "(A:" in c.positive and "red dress" in c.positive
    assert "(B:" in c.positive
    assert "blue dress" in c.negative
    assert "watermark" in c.negative


def test_load_file_roundtrip(tmp_path):
    p = tmp_path / "lay.json"
    data = {"intent": "test", "quality": ["a", "b"]}
    p.write_text(json.dumps(data), encoding="utf-8")
    c = load_prompt_layout_file(p)
    assert "test" in c.positive


def test_merge_prompt_with_layout():
    assert merge_prompt_with_layout("L", "U") == "L, U"
    assert merge_prompt_with_layout("", "U") == "U"
    assert merge_prompt_with_layout("L", "", layout_first=False) == "L"


def test_unknown_preset_falls_back():
    c = compile_prompt_layout({"preset_order": "nope", "intent": "x"})
    assert c.preset == "subject_first"
    assert "x" in c.positive


def test_all_presets_have_same_section_sets():
    keys = [frozenset(v) for v in PRESET_SECTION_ORDER.values()]
    assert len(set(keys)) == 1


def test_section_blocks_match_positive_sections():
    c = compile_prompt_layout({"intent": "III", "subjects": ["SSS"], "quality": "QQQ"})
    assert len(c.section_blocks) == 3
    bodies = [b for _, b in c.section_blocks]
    for frag in ("III", "SSS", "QQQ"):
        assert any(frag in b for b in bodies)


def test_to_t5_encoder_string_has_labels():
    c = compile_prompt_layout({"intent": "read a book", "subjects": ["1girl"]})
    s = c.to_t5_encoder_string()
    assert "INTENT:" in s
    assert "SUBJECTS:" in s
    assert "read a book" in s


def test_substitute_replaces_flat_positive_in_full_prompt():
    c = compile_prompt_layout({"intent": "X", "subjects": ["Y"]})
    full = f"prefix, {c.positive}, suffix"
    out = substitute_compiled_layout_in_t5_prompt(full, c)
    assert c.positive not in out
    assert "INTENT:" in out
    assert "suffix" in out
    assert "prefix" in out


def test_t5_segment_texts_from_layout():
    c = compile_prompt_layout({"intent": "a", "environment": "b"})
    segs = t5_segment_texts_from_layout(c)
    assert len(segs) >= 2
    assert any("INTENT" in s for s in segs)


def test_layout_tail_suffix_after_core():
    c = compile_prompt_layout({"intent": "X", "subjects": ["Y"]})
    full = merge_prompt_with_layout(c.positive, "extra tags", layout_first=True)
    assert layout_tail_suffix(full, c) == "extra tags"


def test_triple_clip_caption_includes_labeled_sections_and_tail():
    c = compile_prompt_layout({"intent": "read", "subjects": ["1girl"]})
    full = merge_prompt_with_layout(c.positive, "masterpiece", layout_first=True)
    cap = triple_clip_caption(c, full)
    assert "INTENT:" in cap and "SUBJECTS:" in cap
    assert "masterpiece" in cap


def test_t5_segment_texts_for_full_prompt_adds_other_tail():
    c = compile_prompt_layout({"intent": "a"})
    full = merge_prompt_with_layout(c.positive, "tail bit", layout_first=True)
    segs = t5_segment_texts_for_full_prompt(c, full)
    assert any(s.startswith("OTHER:") and "tail bit" in s for s in segs)
