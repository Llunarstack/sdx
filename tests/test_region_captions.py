"""Regional / layout caption merge for T5 (see docs/REGION_CAPTIONS.md)."""

from data.caption_utils import (
    format_parts_dict,
    format_region_captions_block,
    merge_region_captions_into_caption,
)


def test_format_parts_dict_ordering():
    parts = {
        "background": "forest",
        "subject": "1girl",
        "clothing": "dress",
    }
    s = format_parts_dict(parts)
    assert "subject:" in s and s.index("subject") < s.index("clothing")
    assert "background" in s


def test_format_region_captions_list():
    regions = [
        {"label": "a", "text": "one"},
        "raw: string",
    ]
    assert "a: one" in format_region_captions_block(regions)
    assert "raw: string" in format_region_captions_block(regions)


def test_merge_append_and_prefix():
    base = "masterpiece, 1girl"
    parts = {"subject": "woman in red"}
    m = merge_region_captions_into_caption(base, parts, mode="append")
    assert "[layout]" in m
    assert base in m
    assert "subject:" in m
    p = merge_region_captions_into_caption(base, parts, mode="prefix")
    assert p.startswith("[layout]")


def test_merge_off():
    base = "solo"
    assert merge_region_captions_into_caption(base, {"a": "b"}, mode="off") == "solo"


def test_combined_parts_and_list():
    regions = {
        "parts": {"subject": "cat"},
        "region_captions": [{"label": "bg", "text": "sofa"}],
    }
    b = format_region_captions_block(regions)
    assert "subject:" in b and "bg:" in b
