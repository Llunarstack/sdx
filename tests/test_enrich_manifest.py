"""Tests for enrich_manifest_captions helpers."""

from __future__ import annotations

from setup.enrich_manifest_captions import _merge_research_row, _row_needs_enrich
from utils.caption.prompt_research import PromptResearchResult


def test_row_needs_enrich_prompt_research():
    assert _row_needs_enrich({"tag_sources": ["creative_rag"]}, use_prompt_research=True) is False
    assert _row_needs_enrich({"tag_sources": ["booru"]}, use_prompt_research=True) is True
    assert _row_needs_enrich({"scene_summary": "x"}, use_prompt_research=False) is False


def test_merge_research_row_preserves_booru_identity():
    row = {
        "caption": "1girl, solo, blue_hair",
        "character_tags": ["hatsune_miku"],
        "copyright_tags": ["vocaloid"],
        "artist_tags": ["artist_name"],
        "tag_sources": ["booru"],
    }
    researched = PromptResearchResult(
        diffusion_prompt="solo girl standing on stage, spotlight, detailed illustration",
        sources=["vlm_uncensored", "creative_rag"],
    )
    merged = _merge_research_row(row, researched)
    cap = merged["caption"].lower()
    assert "hatsune miku" in cap
    assert "vocaloid" in cap
    assert "artist name" in cap
    assert "stage" in cap
    assert merged["booru_caption"] == row["caption"]
    assert "creative_rag" in merged["tag_sources"]
