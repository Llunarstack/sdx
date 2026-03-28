import random

import pytest
import torch

from data.t2i_dataset import collate_t2i
from utils.training.part_aware_training import (
    dit_patch_grid_hw,
    foveated_random_crop_box,
    foreground_attention_alignment_loss,
    hierarchical_caption_dropout,
    image_mask_to_patch_weights,
    merge_hierarchical_captions,
    merge_hierarchical_jsonl_record,
    token_coverage_loss,
    token_coverage_from_cross_attention,
)


def test_merge_hierarchical_captions_order():
    m = merge_hierarchical_captions(
        "base",
        caption_global="global",
        caption_local="local",
        entity_captions={"obj": "detail"},
    )
    assert "global" in m and "base" in m and "local" in m and "obj: detail" in m


def test_merge_hierarchical_jsonl_record():
    row = {"caption": "c", "caption_global": "g", "entity_captions": {"a": "b"}}
    out = merge_hierarchical_jsonl_record(row)
    assert "g" in out and "c" in out and "a: b" in out


def test_hierarchical_caption_dropout_deterministic():
    r = random.Random(0)
    s = "one | two | three"
    d = hierarchical_caption_dropout(s, p_drop_global=0.5, p_drop_local=0.5, rng=r)
    assert "|" in d or d == s


def test_image_mask_to_patch_weights():
    m = torch.zeros(2, 1, 8, 8)
    m[:, :, :4, :] = 1.0
    w = image_mask_to_patch_weights(m, num_patches_h=2, num_patches_w=2)
    assert w.shape == (2, 4)
    assert (w[0, :2] > 0.9).all() and (w[0, 2:] < 0.1).all()


def test_foreground_attention_alignment_loss_grad():
    b, h, n, l = 2, 4, 16, 8
    logits = torch.randn(b, h, n, l, requires_grad=True)
    attn = torch.softmax(logits, dim=-1)
    fg = torch.zeros(b, n)
    fg[:, :8] = 1.0
    loss = foreground_attention_alignment_loss(attn, fg)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_foreground_attention_alignment_loss_sample_valid_mask():
    b, h, n, l = 3, 2, 4, 5
    logits = torch.randn(b, h, n, l, requires_grad=True)
    attn = torch.softmax(logits, dim=-1)
    fg = torch.zeros(b, n)
    fg[0, :2] = 1.0
    fg[1, :2] = 1.0
    fg[2, :2] = 1.0
    valid = torch.tensor([True, False, False])
    loss = foreground_attention_alignment_loss(attn, fg, sample_valid=valid, min_fg_patch_mass=1e-5)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_token_coverage_loss_grad():
    b, h, n, l = 2, 4, 16, 10
    logits = torch.randn(b, h, n, l, requires_grad=True)
    attn = torch.softmax(logits, dim=-1)
    cov = token_coverage_from_cross_attention(attn)
    assert cov.shape == (b, l)
    loss = token_coverage_loss(attn, target_coverage=0.03)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_collate_t2i_partial_grounding_masks():
    b0 = {
        "pixel_values": torch.zeros(3, 8, 8),
        "caption": "a",
        "negative_caption": "",
        "style": "",
        "weight": 1.0,
        "grounding_mask": torch.ones(1, 8, 8),
    }
    b1 = {
        "pixel_values": torch.zeros(3, 8, 8),
        "caption": "b",
        "negative_caption": "",
        "style": "",
        "weight": 1.0,
    }
    out = collate_t2i([b0, b1])
    assert "grounding_mask" in out and "grounding_mask_valid" in out
    assert out["grounding_mask"].shape == (2, 1, 8, 8)
    assert out["grounding_mask_valid"].tolist() == [True, False]


def test_dit_patch_grid_hw():
    assert dit_patch_grid_hw(16) == (4, 4)
    with pytest.raises(ValueError):
        dit_patch_grid_hw(15)


def test_foveated_random_crop_box():
    y0, x0, y1, x1 = foveated_random_crop_box(100, 80, crop_frac=0.5, rng=random.Random(1))
    assert 0 <= y0 < y1 <= 100 and 0 <= x0 < x1 <= 80
