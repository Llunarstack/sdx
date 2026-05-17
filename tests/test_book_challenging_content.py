"""Tests for book challenging-content prompt fragments."""

from __future__ import annotations

from pipelines.book_comic import book_challenging_content
from pipelines.book_comic.consistency_helpers import positive_block_from_mapping
def test_challenge_pack_positive_mature_coherence() -> None:
    frag = book_challenging_content.challenge_pack_positive("mature_coherence")
    assert "mature-rated" in frag


def test_challenge_pack_positive_max_includes_mature() -> None:
    frag = book_challenging_content.challenge_pack_positive("max")
    assert "mature-rated" in frag
    assert "surreal" in frag


def test_challenging_content_from_mapping() -> None:
    block = {"pack": "surreal_weird", "tags": ["hands_heavy"], "extra": "dream logic"}
    out = book_challenging_content.challenging_content_from_mapping(block)
    assert "surreal" in out
    assert "hands-heavy" in out.lower() or "hands" in out.lower()
    assert "dream logic" in out


def test_positive_block_from_mapping_challenging_content() -> None:
    spec = {
        "challenging_content": {"pack": "technical_hard", "tags": ["reflections_glass"]},
    }
    out = positive_block_from_mapping(spec)
    assert "perspective" in out or "reflection" in out


def test_visual_memory_challenge_clause_adds_mature_fidelity() -> None:
    frag = book_challenging_content.visual_memory_challenge_clause({"book_challenge_pack": "none"})
    assert "mature-rated" in frag
