"""Human-made anti-AI polish helpers."""

from __future__ import annotations

import numpy as np
from utils.quality.human_made import (
    HumanMadeConfig,
    apply_human_made_pipeline,
    apply_human_made_prompt_flags,
    human_made_preset,
    human_made_prompt_fragments,
    reduce_edge_halos,
)


def test_human_made_preset_order() -> None:
    lite = human_made_preset("lite")
    strong = human_made_preset("strong")
    assert lite.strength < strong.strength


def test_prompt_fragments_non_empty() -> None:
    pos, neg = human_made_prompt_fragments("standard")
    assert "plastic" in neg.lower()
    assert len(pos) > 10


def test_reduce_edge_halos_noop_at_zero() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert np.array_equal(reduce_edge_halos(img, strength=0.0), img)


def test_apply_pipeline_identity_when_disabled() -> None:
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert np.array_equal(apply_human_made_pipeline(img, HumanMadeConfig(strength=0.0)), img)


def test_pipeline_is_deterministic_and_adds_no_noise() -> None:
    flat = np.full((48, 48, 3), 120.0, dtype=np.float32)
    out = apply_human_made_pipeline(flat, human_made_preset("strong"))
    assert np.array_equal(out, apply_human_made_pipeline(flat, human_made_preset("strong")))
    assert float(out.std()) < 1e-3


def test_apply_human_made_prompt_flags() -> None:
    class Args:
        human_made = "standard"
        anti_ai_pack = "none"
        human_media_mode = "none"
        naturalize = False
        shortcomings_mitigation = "none"
        less_ai = False

    a = Args()
    apply_human_made_prompt_flags(a)
    assert a.naturalize is True
    assert a.anti_ai_pack != "none"
