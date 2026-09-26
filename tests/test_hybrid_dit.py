"""Hybrid DiT extras: encoder-decoder skips, MMDiT joint attn, Mix-FFN, late HF residual."""

from __future__ import annotations

import torch
from config.train_config import TrainConfig, get_dit_build_kwargs
from models import DiT_models_text
from models.dit_text import DiT_Text, SwiGLUMlp
from models.hybrid_dit import EncoderDecoderSkip, JointMMDiTAttention, MixFFN, TimeGatedHFResidual
from models.taca import TACA


def _tiny(**kwargs) -> DiT_Text:
    kw = dict(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=4,
        num_heads=4,
        text_dim=32,
        learn_sigma=False,
        use_xformers=False,
        qk_norm=True,
    )
    kw.update(kwargs)
    return DiT_Text(**kw)


def test_zero_init_modules_are_identity() -> None:
    torch.manual_seed(0)
    patches = torch.randn(2, 16, 32)
    text = torch.randn(2, 8, 32)
    skip = EncoderDecoderSkip(32)
    assert torch.allclose(skip(patches, torch.randn_like(patches)), patches)
    mix = MixFFN(32)
    assert torch.allclose(mix(patches, 4, 4), patches)
    joint = JointMMDiTAttention(32, 4)
    assert torch.allclose(joint(patches, text, use_xformers=False), patches)
    hf = TimeGatedHFResidual(32)
    t = torch.zeros(2)
    assert torch.allclose(hf(patches, 4, 4, t, num_timesteps=1000), patches)


def test_skip_path_identity_at_init() -> None:
    hourglass = _tiny(uvit_skips=True, hidit_hf=True, mix_ffn=True, mmdit_every_n=2)
    assert hourglass.hybrid is not None
    patches = torch.randn(1, 16, 32)
    text = torch.randn(1, 6, 32)
    hourglass.hybrid.begin_forward()
    y = patches
    for i in range(4):
        y = hourglass.hybrid.after_block(i, y, text, num_patches=16, h_patches=4, w_patches=4, use_xformers=False)
    y = hourglass.hybrid.after_all(y, torch.full((1,), 800.0), num_patches=16, h_patches=4, w_patches=4)
    assert torch.allclose(y, patches, atol=1e-5)


def test_skip_weights_change_output() -> None:
    torch.manual_seed(1)
    model = _tiny(uvit_skips=True)
    patches = torch.randn(1, 16, 32)
    text = torch.randn(1, 4, 32)
    model.hybrid.begin_forward()
    before = patches
    for i in range(4):
        before = model.hybrid.after_block(i, before, text, num_patches=16, h_patches=4, w_patches=4, use_xformers=False)
    with torch.no_grad():
        for skip in model.hybrid.skips:
            skip.proj.weight.add_(0.2)
    model.hybrid.begin_forward()
    after = patches
    for i in range(4):
        after = model.hybrid.after_block(i, after, text, num_patches=16, h_patches=4, w_patches=4, use_xformers=False)
    assert not torch.allclose(before, after)


def test_hybrid_dit_in_registry_and_kwargs() -> None:
    assert "HybridDiT-S/2" in DiT_models_text
    cfg = TrainConfig(model_name="HybridDiT-S/2", image_size=64)
    kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
    assert kw["uvit_skips"] is True
    assert kw["hidit_hf"] is True
    assert kw["mix_ffn"] is True
    assert int(kw["mmdit_every_n"]) == 4
    assert kw["qk_norm"] is True
    assert kw["use_taca"] is True
    assert kw["use_swiglu"] is True
    assert kw["use_rope"] is True
    assert int(kw["num_register_tokens"]) == 4
    model = DiT_models_text["HybridDiT-S/2"](
        input_size=8,
        text_dim=32,
        class_dropout_prob=0.0,
        use_xformers=False,
        learn_sigma=False,
    )
    assert model.use_taca and model.use_swiglu and model.use_rope
    assert int(model.num_register_tokens) == 4
    assert isinstance(model.blocks[0].cross_attn, TACA)
    assert isinstance(model.blocks[0].mlp, SwiGLUMlp)
    x = torch.randn(1, 4, 8, 8)
    t = torch.zeros(1)
    txt = torch.randn(1, 4, 32)
    with torch.no_grad():
        out = model(x, t, encoder_hidden_states=txt)
    assert out.shape[0] == 1 and out.shape[1] == 4


def test_taca_swiglu_tiny_forward_is_finite() -> None:
    model = _tiny(use_taca=True, use_swiglu=True, use_rope=True)
    assert isinstance(model.blocks[0].cross_attn, TACA)
    assert isinstance(model.blocks[0].mlp, SwiGLUMlp)
    assert torch.equal(
        model.blocks[0].cross_attn.out_proj.weight, torch.zeros_like(model.blocks[0].cross_attn.out_proj.weight)
    )
    x = torch.randn(1, 4, 8, 8)
    t = torch.zeros(1)
    txt = torch.randn(1, 4, 32)
    with torch.no_grad():
        out = model(x, t, encoder_hidden_states=txt)
    assert out.shape == (1, 4, 8, 8)
    assert torch.isfinite(out).all()
