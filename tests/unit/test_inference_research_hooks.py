import torch

from utils.generation.inference_research_hooks import (
    NullGlyphEncoder,
    apply_size_embed_to_model_kwargs,
    highfreq_layout_prior,
    plan_dual_stage_latents,
    score_latent_prompt_alignment,
    should_rewind,
    spectral_latent_lowfreq_blend,
)


def test_plan_dual_stage_latents():
    p = plan_dual_stage_latents(256, layout_scale_div=2, layout_steps=10, detail_steps=12)
    assert p.layout_latent_hw == (16, 16)
    assert p.target_latent_hw == (32, 32)
    assert p.layout_steps == 10 and p.detail_steps == 12


def test_null_glyph_encoder():
    enc = NullGlyphEncoder()
    t = enc.encode_utf8(["hi"], device=torch.device("cpu"))
    assert t.shape == (1, 1, 64)


def test_rewind_stub():
    assert score_latent_prompt_alignment(torch.zeros(1, 4, 2, 2), "x") == 0.5
    assert should_rewind(0.2, threshold=0.35) is True
    assert should_rewind(0.9, threshold=0.35) is False


def test_apply_size_embed_noop_without_cfg():
    class Cfg:
        size_embed_dim = 0

    ce = torch.zeros(2, 3, 8)
    mc, mu = apply_size_embed_to_model_kwargs(
        {"a": 1}, {"b": 2}, cfg=Cfg(), cond_emb=ce, latent_h=8, latent_w=8, device=torch.device("cpu")
    )
    assert mc == {"a": 1} and mu == {"b": 2}


def test_highfreq_layout_prior():
    x = torch.randn(1, 4, 8, 8)
    y = highfreq_layout_prior(x, strength=0.05)
    assert y.shape == x.shape


def test_spectral_latent_lowfreq_blend():
    x = torch.randn(1, 4, 16, 16)
    assert torch.allclose(spectral_latent_lowfreq_blend(x, strength=0.0), x)
    y = spectral_latent_lowfreq_blend(x, strength=0.1, cutoff_frac=0.2)
    assert y.shape == x.shape
