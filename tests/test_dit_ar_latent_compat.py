"""Tests for ``utils.generation.dit_ar_latent_compat``."""

from __future__ import annotations

import pytest
import torch
from utils.generation import dit_ar_latent_compat as dac


def test_dit_latent_hw_from_model() -> None:
    class Emb:
        img_size = 64

    class M:
        x_embedder = Emb()

    assert dac.dit_latent_hw_from_model(M()) == (64, 64)


def test_validate_latent_edit_tensors_ok() -> None:
    class Emb:
        img_size = (8, 8)

    class M:
        x_embedder = Emb()

    z = torch.zeros(1, 4, 8, 8)
    dac.validate_latent_edit_tensors(z, None, M())
    m = torch.zeros(1, 1, 8, 8)
    dac.validate_latent_edit_tensors(z, m, M())


def test_validate_latent_edit_tensors_bad_hw() -> None:
    class Emb:
        img_size = 8

    class M:
        x_embedder = Emb()

    z = torch.zeros(1, 4, 7, 8)
    with pytest.raises(ValueError, match="spatial"):
        dac.validate_latent_edit_tensors(z, None, M())


def test_refresh_block_ar_mask_on_model() -> None:
    class M:
        num_patches = 16
        num_ar_blocks = 2
        ar_block_order = "raster"
        pos_embed = torch.nn.Parameter(torch.zeros(1, 16, 4))

    m = M()
    dac.refresh_block_ar_mask_on_model(m)
    assert m._ar_mask is not None
    dac.refresh_block_ar_mask_on_model(m, num_ar_blocks=0)
    assert m._ar_mask is None


def test_vit_scorer_ar_vector() -> None:
    v = dac.vit_scorer_ar_vector(4)
    assert v.shape == (4,)
    assert float(v.sum().item()) == pytest.approx(1.0)
    assert float(v[2].item()) == pytest.approx(1.0)


def test_generation_edit_metadata() -> None:
    d = dac.generation_edit_metadata(
        image_size_px=512,
        num_ar_blocks=2,
        strength=0.4,
        inpaint=True,
        inpaint_mode="mdm",
    )
    assert d["latent_hw"] == 64
    assert d["inpaint"] is True
    assert "block_ar" in d["ar_regime"]


def test_tag_row_for_vit_scorer() -> None:
    row = {"caption": "x"}
    out = dac.tag_row_for_vit_scorer(row, 2, overwrite=True)
    assert out["num_ar_blocks"] == 2
    assert out["dit_num_ar_blocks"] == 2
