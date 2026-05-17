"""Package layout: subprocess/PIL helpers re-export from ``utils.generation``."""

from __future__ import annotations

import pytest

timm = pytest.importorskip("timm", reason="utils.generation pulls timm via models.dit_text")


def test_edit_masks_and_sample_runner_importable_via_generation_pkg():
    from utils import generation as g

    assert hasattr(g.edit_masks, "heuristic_inpaint_mask")
    assert hasattr(g.sample_edit_runner, "run_sample_inference")
    assert hasattr(g.segmentation_to_mask, "build_segmentation_mask_for_edit")


def test_generation_request_visual_design_defaults():
    from utils.generation.multimodal_generation import GenerationRequest

    req = GenerationRequest(prompt="test")
    assert req.visual_design_domain == "none"
    assert req.visual_design_preset == ""
    assert req.visual_design_negative_pack is False
