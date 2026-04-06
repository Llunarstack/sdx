"""Tests for extended AR macro-block traversal orders."""

import torch
from models.ar_masks_extended import create_block_causal_mask_2d
from utils.architecture.ar_block_layout import block_visit_order


def test_block_visit_order_supports_new_orders():
    b = 4
    for order in ("raster", "zorder", "snake", "spiral"):
        cells = block_visit_order(b, order)
        assert len(cells) == b * b
        assert len(set(cells)) == b * b


def test_create_block_causal_mask_supports_new_orders():
    h = w = 8
    b = 2
    masks = {
        o: create_block_causal_mask_2d(h, w, b, block_order=o)
        for o in ("raster", "zorder", "snake", "spiral")
    }
    for m in masks.values():
        assert m.shape == (h * w, h * w)
        assert torch.isfinite(m.diag()).all()
    # New orders should not collapse to identical masks in this setup.
    assert not torch.equal(masks["raster"], masks["snake"])
    assert not torch.equal(masks["raster"], masks["spiral"])
