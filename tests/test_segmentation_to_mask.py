"""Tests for text/vision-driven inpaint masks (heuristic paths; CI avoids heavy transformers)."""

from __future__ import annotations

import pytest
from PIL import Image
from utils.generation.segmentation_to_mask import (
    build_segmentation_mask_for_edit,
    phrase_to_fallback_region,
)


def test_phrase_to_fallback_regions():
    assert phrase_to_fallback_region("redo the BACKGROUND blur") == "background"
    assert phrase_to_fallback_region("fix her eyes") == "face"
    assert phrase_to_fallback_region("hands look wrong") == "hands"


def test_empty_phrase_raises():
    with pytest.raises(ValueError, match="non-empty"):
        build_segmentation_mask_for_edit(Image.new("RGB", (32, 32), (255, 0, 0)), "")


def test_heuristic_only_modes_skip_models():
    im = Image.new("RGB", (64, 64), (128, 128, 128))
    r = build_segmentation_mask_for_edit(
        im,
        "change only the backdrop",
        use_vision_models=False,
        feather_radius=2.0,
    )
    assert r.mode.startswith("heuristic")
    assert r.mask.size == (64, 64)


def test_unrecognized_words_no_models_behave():
    """No DINO/SAM: unknown nouns resolve to heuristic subject ellipse."""
    im = Image.new("RGB", (48, 48), (255, 255, 255))
    r = build_segmentation_mask_for_edit(im, "obscure thingamajig", use_vision_models=False)
    assert r.mask.size == (48, 48)
    assert "heuristic" in r.mode or "subject" in r.mode
