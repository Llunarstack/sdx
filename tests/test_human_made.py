"""Human-made anti-AI polish helpers."""

from __future__ import annotations

import numpy as np
from utils.quality.human_made import (
    HumanMadeConfig,
    apply_human_made_pipeline,
    apply_human_made_prompt_flags,
    human_made_preset,
    human_made_prompt_fragments,
    remove_ai_speckles,
)


def test_human_made_preset_order() -> None:
    lite = human_made_preset("lite")
    strong = human_made_preset("strong")
    assert lite.strength < strong.strength


def test_prompt_fragments_non_empty() -> None:
    pos, neg = human_made_prompt_fragments("standard")
    assert "plastic" in neg.lower()
    assert len(pos) > 10


def test_remove_speckles_noop_at_zero() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = remove_ai_speckles(img, strength=0.0)
    assert out.shape == img.shape


def test_apply_pipeline_identity_when_disabled() -> None:
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    cfg = HumanMadeConfig(strength=0.0)
    out = apply_human_made_pipeline(img, cfg)
    assert out.shape == img.shape


def test_apply_human_made_prompt_flags() -> None:
    class Args:
        human_made = "standard"
        anti_ai_pack = "none"
        human_media_mode = "none"
        naturalize = False
        naturalize_grain = 0.0
        shortcomings_mitigation = "none"
        less_ai = False

    a = Args()
    apply_human_made_prompt_flags(a)
    assert a.naturalize is True
    assert a.anti_ai_pack != "none"
