"""Tests for utils/superior stack."""

from __future__ import annotations

import numpy as np
from utils.superior.auto_stack import apply_superior_prompt_stack
from utils.superior.composite_ranker import CompositeRanker
from utils.superior.glyph_encoder import ByteHashGlyphEncoder
from utils.superior.inference_pipeline import SuperiorInferenceConfig, build_superior_sample_argv
from utils.superior.retrieval import TfidfFactIndex, build_tfidf_index_from_jsonl
from utils.superior.self_correct import SelfCorrectConfig, SelfCorrectPolicy


def test_tfidf_retrieval_orders_relevant_facts() -> None:
    idx = TfidfFactIndex(
        facts=[
            "neon signs reflect on wet pavement in cyberpunk alleys",
            "pastoral meadow with sheep at sunrise",
            "database indexing with B-trees",
        ]
    )
    hits = idx.query("cyberpunk alley neon rain", top_k=2)
    assert "cyberpunk" in hits[0].lower()


def test_apply_superior_prompt_stack_no_rag() -> None:
    out = apply_superior_prompt_stack("a red fox", append_domain_tips=False)
    assert out == "a red fox"


def test_build_superior_sample_argv() -> None:
    cfg = SuperiorInferenceConfig(num_candidates=4, local_rag_jsonl="facts.jsonl")
    argv = build_superior_sample_argv(ckpt="m.pt", prompt="cat", out="o.png", config=cfg)
    assert "--pick-best" in argv
    assert "superior_composite" in argv
    assert "--local-rag-jsonl" in argv


def test_composite_ranker_picks_sharper_image() -> None:
    blur = np.full((32, 32, 3), 128, dtype=np.uint8)
    sharp = blur.copy()
    sharp[8:24, 8:24] = 255
    sharp[8:24, 8:24:1] = 0
    ranker = CompositeRanker()
    idx, _ = ranker.pick_best_index([blur, sharp], prompt="test", device="cpu")
    assert idx in (0, 1)


def test_glyph_encoder_shape() -> None:
    enc = ByteHashGlyphEncoder(embed_dim=32, max_bytes=16)
    t = enc.encode_utf8(["HELLO"], device=__import__("torch").device("cpu"))
    assert t.shape == (1, 16, 32)


def test_self_correct_policy_threshold() -> None:
    pol = SelfCorrectPolicy(SelfCorrectConfig(align_threshold=0.4))
    assert pol.needs_correction(0.2)
    assert not pol.needs_correction(0.9)


def test_build_index_missing_file() -> None:
    idx = build_tfidf_index_from_jsonl("/nonexistent/path.jsonl")
    assert idx.n_docs == 0
