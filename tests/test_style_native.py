"""Tests for style genome native fast paths (Python fallbacks always run)."""

from __future__ import annotations

import numpy as np

from utils.prompt.style_genome import StyleGenome
from utils.prompt.style_native import (
    genome_style_fingerprint,
    merge_genome_axes_native,
    native_stack_status,
    pick_best_embedding_index,
    text_overlap,
)


def test_native_stack_status_keys():
    st = native_stack_status()
    assert "rust_style_ops" in st
    assert "cuda_style_pick" in st
    assert "go_explore" in st
    assert "mojo_style_tokens" in st


def test_text_overlap_and_fingerprint():
    a = "red dress, blue sky, cinematic"
    b = "red dress, golden hour"
    o = text_overlap(a, b)
    assert 0.0 < o < 1.0
    g = StyleGenome(id="t1", name="Test", palette="red dress cinematic", signature="x")
    fp = genome_style_fingerprint(g)
    assert fp > 0
    merged = merge_genome_axes_native(g)
    assert "red" in merged.lower() or "dress" in merged.lower()


def test_explore_stats_python_fallback(tmp_path):
    import json

    from utils.prompt.style_native import explore_stats_python

    p = tmp_path / "m.jsonl"
    rows = [
        {"caption": "a", "style_genome_id": "g1", "candidate_kind": "base"},
        {"caption": "b", "style_genome_id": "g1", "candidate_kind": "chimera"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = explore_stats_python(p)
    assert "rows: 2" in out
    assert "g1: 2" in out
    assert "chimera: 1" in out


def test_rust_style_ops_when_built():
    from sdx_native.style_ops_native import get_style_ops_lib, maybe_fnv1a64, maybe_token_jaccard

    lib = get_style_ops_lib()
    if not lib.available:
        import pytest

        pytest.skip("sdx_prompt_ops release DLL not built or missing style symbols")
    j = maybe_token_jaccard("a, b", "a, c")
    assert j is not None and j > 0
    assert maybe_fnv1a64("test") is not None


def test_pick_best_embedding_numpy_fallback():
    dim = 8
    query = np.random.randn(dim).astype(np.float32)
    candidates = np.random.randn(5, dim).astype(np.float32)
    candidates[2] = query + 0.01
    idx, score = pick_best_embedding_index(query, candidates)
    assert 0 <= idx < 5
    assert score > 0.5
