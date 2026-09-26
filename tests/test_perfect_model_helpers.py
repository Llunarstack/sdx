"""Tests for Dense Diffusion wiring, glyph projector, hand/count helpers."""

from __future__ import annotations

import numpy as np
import torch


def test_region_masks_and_bias():
    from frontier.attention.dense_diffusion import bias_cross_attention, region_masks_from_boxes
    from frontier.attention.layout_plan import AttentionLayoutPlan

    masks = region_masks_from_boxes(((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)), latent_h=8, latent_w=8)
    assert masks.shape == (2, 1, 8, 8)
    plan = AttentionLayoutPlan(
        region_names=("a", "b"),
        boxes=((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)),
        enforce_steps=(0, 1, 2),
        strength=0.9,
    )
    logits = torch.zeros(2, 4, 64, 16)
    out = bias_cross_attention(
        logits,
        region_masks=masks,
        step_index=0,
        plan=plan,
        cond_batch_size=1,
    )
    assert out.shape == logits.shape
    # Cond row biased; uncond (batch 1) unchanged
    assert not torch.allclose(out[0], logits[0])
    assert torch.allclose(out[1], logits[1])


def test_bias_pools_latent_masks_to_patch_grid():
    """Latent 8x8 masks vs 4x4 patch tokens must pool spatially (not flat-crop)."""
    from frontier.attention.dense_diffusion import bias_cross_attention, region_masks_from_boxes
    from frontier.attention.layout_plan import AttentionLayoutPlan

    # Region A: top-left quarter of 8x8 latent; region B: bottom-right quarter.
    masks = region_masks_from_boxes(((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)), latent_h=8, latent_w=8)
    plan = AttentionLayoutPlan(
        region_names=("a", "b"),
        boxes=((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)),
        enforce_steps=(0,),
        strength=1.0,
    )
    # 4x4 patch grid => 16 spatial tokens
    logits = torch.zeros(1, 1, 16, 8)
    out = bias_cross_attention(
        logits,
        region_masks=masks,
        step_index=0,
        plan=plan,
        num_patch_tokens=16,
    )
    # Text chunk 0 gets region A (top-left of 4x4); chunk 1 gets region B (bottom-right).
    # Token layout is row-major: indices 0..3 top row, 12..15 bottom row.
    bias_a = (out[0, 0, :, 0] - logits[0, 0, :, 0]).view(4, 4)
    bias_b = (out[0, 0, :, 4] - logits[0, 0, :, 4]).view(4, 4)
    assert bias_a[:2, :2].mean() > bias_a[2:, 2:].mean()
    assert bias_b[2:, 2:].mean() > bias_b[:2, :2].mean()
    # Opposite corners should stay near-zero for the mismatched region
    assert float(bias_a[3, 3]) < float(bias_a[0, 0])
    assert float(bias_b[0, 0]) < float(bias_b[3, 3])


def test_layout_attn_context_cross_attention():
    from frontier.attention.dense_diffusion import region_masks_from_boxes
    from frontier.attention.layout_plan import AttentionLayoutPlan
    from models.dit_text import CrossAttention
    from models.layout_attn_context import clear_layout_attn, set_layout_attn

    attn = CrossAttention(hidden_size=32, num_heads=4, text_dim=32, qk_norm=False)
    x = torch.randn(1, 16, 32)
    text = torch.randn(1, 8, 32)
    plan = AttentionLayoutPlan(
        region_names=("r0",),
        boxes=((0.0, 0.0, 1.0, 1.0),),
        enforce_steps=tuple(range(10)),
        strength=1.0,
    )
    masks = region_masks_from_boxes(plan.boxes, latent_h=4, latent_w=4)
    set_layout_attn(plan=plan, region_masks=masks, step_index=0, cond_batch_size=1, num_patch_tokens=16)
    try:
        out = attn(x, text, use_xformers=False)
        assert out.shape == x.shape
    finally:
        clear_layout_attn()


def test_persistent_glyph_projector():
    from utils.generation.quality_stack import get_persistent_glyph_projector, inject_glyph_residual

    p1 = get_persistent_glyph_projector(32, 64, device=torch.device("cpu"), dtype=torch.float32)
    p2 = get_persistent_glyph_projector(32, 64, device=torch.device("cpu"), dtype=torch.float32)
    assert p1 is p2
    enc = torch.randn(1, 16, 64)
    out = inject_glyph_residual(enc, 'sign says "OPEN"', strength=0.2, texts=["OPEN"])
    assert out.shape == enc.shape
    assert not torch.allclose(out, enc)


def test_hand_and_count_helpers():
    from utils.quality.count_binder_eval import analyze_prompt_counts, score_count_bind_match
    from utils.quality.hand_verifier import score_hand_structure

    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img[80:, :, :] = np.clip(
        img[80:, :, :].astype(np.int16) + np.random.randint(-40, 40, img[80:].shape),
        0,
        255,
    ).astype(np.uint8)
    s = score_hand_structure(img)
    assert 0.0 <= s <= 1.0
    report = analyze_prompt_counts("exactly three red birds on a fence")
    assert report.expected_counts or report.expected_subjects >= 1
    soft = score_count_bind_match(img, "exactly three red birds", estimated_people=3)
    assert 0.0 <= soft <= 1.0


def test_frontier_consume_sanitize_and_emphasis():
    from types import SimpleNamespace

    from utils.generation.frontier_consume import (
        apply_token_emphasis_to_args,
        merge_step_emphasis_into_noise_scales,
        sanitize_diffusion_feature_kwargs,
    )

    args = SimpleNamespace(prompt="a woman with detailed hands holding a sign", cfg_scale=4.0)
    meta = apply_token_emphasis_to_args(args)
    assert "weights" in meta
    assert "hands" in str(meta["weights"]).lower() or "(hands" in args.prompt.lower()
    assert args.cfg_scale > 4.0

    merged = merge_step_emphasis_into_noise_scales(
        step_emphasis=[2.0, 1.0, 0.5],
        step_noise_scales=[1.0, 1.0, 1.0],
    )
    assert merged is not None and len(merged) == 3
    assert merged[0] > merged[-1]

    clean = sanitize_diffusion_feature_kwargs(
        {
            "step_emphasis": [1.5, 1.0],
            "step_noise_scales": [1.0, 1.0],
            "frontier_token_emphasis": {"hands": 1.2},
            "attention_layout_plan": object(),
        }
    )
    assert "step_emphasis" not in clean
    assert "frontier_token_emphasis" not in clean
    assert "step_noise_scales" in clean
    assert "attention_layout_plan" in clean


def test_layout_attn_on_qk_norm():
    from frontier.attention.dense_diffusion import region_masks_from_boxes
    from frontier.attention.layout_plan import AttentionLayoutPlan
    from models.dit_text_variants import CrossAttentionQKNorm
    from models.layout_attn_context import clear_layout_attn, set_layout_attn

    attn = CrossAttentionQKNorm(hidden_size=32, num_heads=4, text_dim=32)
    x = torch.randn(1, 16, 32)
    text = torch.randn(1, 8, 32)
    plan = AttentionLayoutPlan(
        region_names=("r0",),
        boxes=((0.0, 0.0, 1.0, 1.0),),
        enforce_steps=tuple(range(10)),
        strength=1.0,
    )
    masks = region_masks_from_boxes(plan.boxes, latent_h=4, latent_w=4)
    set_layout_attn(plan=plan, region_masks=masks, step_index=0, cond_batch_size=1, num_patch_tokens=16)
    try:
        out = attn(x, text, use_xformers=False)
        assert out.shape == x.shape
    finally:
        clear_layout_attn()
