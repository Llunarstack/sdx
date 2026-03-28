import torch

from diffusion.holy_grail import (
    HolyGrailRecipe,
    adaptive_cfg_from_attention,
    apply_condition_noise,
    attention_entropy,
    attention_token_coverage,
    bounded_scale,
    build_holy_grail_step_plan,
    cads_noise_std,
    consistency_blend_latent,
    coverage_shortfall_loss,
    dynamic_percentile_clamp,
    fuse_condition_scales,
    style_detail_mix_for_progress,
    unsharp_mask_latent,
    weighted_patch_alignment_score,
)


def test_attention_entropy_shape_and_finite():
    logits = torch.randn(2, 4, 16, 12)
    attn = torch.softmax(logits, dim=-1)
    ent = attention_entropy(attn)
    assert ent.shape == (2,)
    assert torch.isfinite(ent).all()


def test_adaptive_cfg_returns_per_sample_scales():
    logits = torch.randn(3, 2, 8, 10)
    attn = torch.softmax(logits, dim=-1)
    cfg = adaptive_cfg_from_attention(7.5, attn)
    assert cfg.shape == (3,)
    assert (cfg > 0).all()


def test_prompt_coverage_ops():
    logits = torch.randn(2, 3, 9, 7)
    attn = torch.softmax(logits, dim=-1)
    cov = attention_token_coverage(attn)
    tw = torch.ones_like(cov)
    score = weighted_patch_alignment_score(attn, tw)
    loss = coverage_shortfall_loss(attn, target=0.1, token_weights=tw)
    assert cov.shape == (2, 7)
    assert score.shape == (2,)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_condition_scale_fusion_trend():
    c0, a0 = fuse_condition_scales(base_control_scale=1.0, base_adapter_scale=1.0, progress=0.0, frontload_control=True)
    c1, a1 = fuse_condition_scales(base_control_scale=1.0, base_adapter_scale=1.0, progress=1.0, frontload_control=True)
    assert c0 > c1
    assert a1 > a0


def test_latent_refiners_keep_shape():
    x = torch.randn(2, 4, 16, 16)
    y = unsharp_mask_latent(x, sigma=0.6, amount=0.2)
    z = dynamic_percentile_clamp(y, quantile=0.99)
    t = consistency_blend_latent(z, y, strength=0.1)
    assert y.shape == x.shape == z.shape == t.shape
    assert torch.isfinite(t).all()


def test_holy_grail_step_plan_progression():
    r = HolyGrailRecipe(base_cfg=7.0, control_base_scale=1.2, adapter_base_scale=1.0)
    p0 = build_holy_grail_step_plan(recipe=r, step_index=0, total_steps=20)
    p1 = build_holy_grail_step_plan(recipe=r, step_index=19, total_steps=20)
    assert p1.cfg_scale >= p0.cfg_scale
    assert p1.adapter_scale >= p0.adapter_scale
    assert p1.refine_strength > p0.refine_strength


def test_cads_noise_schedule_and_apply():
    s0 = cads_noise_std(progress=0.0, base_strength=0.04, min_strength=0.005, power=1.0)
    s1 = cads_noise_std(progress=1.0, base_strength=0.04, min_strength=0.005, power=1.0)
    assert s0 > s1 >= 0.0
    cond = torch.zeros(2, 8, 16)
    noisy = apply_condition_noise(cond, std=s0)
    assert noisy.shape == cond.shape
    assert not torch.equal(noisy, cond)


def test_style_router_progression():
    st0, dt0 = style_detail_mix_for_progress(0.0, style_strength=1.0, detail_strength=1.0)
    st1, dt1 = style_detail_mix_for_progress(1.0, style_strength=1.0, detail_strength=1.0)
    assert st1 > st0
    assert dt0 > dt1
    assert bounded_scale(5.0, lo=0.0, hi=2.0) == 2.0

