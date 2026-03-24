"""Tests for originality token injection (sample + train)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_inject_originality_inserts_after_subject_tags() -> None:
    from utils.prompt.originality_augment import inject_originality_tokens

    rng = np.random.default_rng(0)
    tokens = ["alpha novelty", "beta twist"]
    out = inject_originality_tokens("1girl, red dress, outdoors", 0.5, rng, tokens=tokens)
    assert out.startswith("1girl,")
    assert "alpha novelty" in out or "beta twist" in out
    assert "red dress" in out


def test_inject_originality_empty_or_zero_strength() -> None:
    from utils.prompt.originality_augment import inject_originality_tokens

    rng = np.random.default_rng(0)
    assert inject_originality_tokens("", 0.5, rng, tokens=["x"]) == ""
    assert inject_originality_tokens("hello", 0.0, rng, tokens=["x"]) == "hello"


def test_default_originality_tokens_non_empty() -> None:
    from utils.prompt.originality_augment import default_originality_tokens

    t = default_originality_tokens()
    assert isinstance(t, list)
    assert len(t) >= 5
