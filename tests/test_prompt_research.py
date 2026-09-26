"""Tests for RAG-based prompt research."""

from __future__ import annotations

import json
from unittest.mock import patch

from utils.caption.prompt_research import (
    PromptResearchResult,
    _seed_from_description,
    research_prompt_for_image,
    retrieve_rag_facts,
)


def test_seed_from_description_uses_first_sentence():
    assert _seed_from_description("A girl in a field. More text.", "") == "A girl in a field"


def test_retrieve_rag_facts_from_corpus(tmp_path):
    corpus = tmp_path / "rag.jsonl"
    rows = [
        {"text": "hatsune miku, vocaloid, 1girl, turquoise hair, stage"},
        {"text": "landscape, mountains, sunset, clouds"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    facts = retrieve_rag_facts("hatsune miku vocaloid stage", corpus, top_k=2)
    assert facts
    assert any("miku" in f.lower() for f in facts)


def test_research_without_models_uses_fallback(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"not-a-real-png")
    corpus = tmp_path / "rag.jsonl"
    corpus.write_text(
        json.dumps({"text": "1girl, solo, blue hair, outdoors"}) + "\n",
        encoding="utf-8",
    )

    fake = PromptResearchResult(
        diffusion_prompt="1girl, solo, blue hair, outdoors, detailed",
        image_description="A girl with blue hair outdoors.",
        retrieved_facts=["1girl, solo, blue hair, outdoors"],
        sources=["vlm_uncensored", "rag:1", "creative_rag"],
    )

    with patch("utils.caption.prompt_research.describe_image_uncensored", return_value=fake.image_description):
        with patch("utils.caption.prompt_research.retrieve_rag_facts", return_value=fake.retrieved_facts):
            with patch("utils.prompt.creative_rag.CreativeRAGEngine", side_effect=RuntimeError("no gpu")):
                result = research_prompt_for_image(
                    img,
                    rag_corpus=corpus,
                    use_creative_rag=True,
                )
    assert "blue hair" in result.diffusion_prompt.lower()
    assert result.retrieved_facts
