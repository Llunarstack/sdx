"""
End-to-end runnability smoke test for the SuperiorViT generative backbone.

SuperiorViT used to be architecturally complete but dead-on-arrival: the shared
DiT build path (``get_dit_build_kwargs``) injected ~30 DiT-only kwargs its
constructor rejected, it had no ``enable_gradient_checkpointing`` method, and its
``forward`` used ``text_emb`` while the whole sampler/trainer stack calls
``model(x, t, encoder_hidden_states=..., **extra)``.

This test locks in the fix: SuperiorViT must build from the registry via the
shared config path, run a forward, tolerate the sampler's extra kwargs, support
gradient checkpointing, and back-propagate — exactly like a DiT-Text model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from models import DiT_models_text

from config import get_dit_build_kwargs


def _make_cfg(model_name: str = "SuperiorViT-S/2") -> SimpleNamespace:
    # A minimal config object; get_dit_build_kwargs only uses getattr with defaults.
    return SimpleNamespace(
        model_name=model_name,
        image_size=256,  # -> latent_size 32
        text_encoder="google/t5-v1_1-base",  # -> text_dim 768
        drop_path_rate=0.1,
        qk_norm=True,
        use_rope=True,
    )


def test_superior_vit_registered():
    assert "SuperiorViT-S/2" in DiT_models_text
    assert "SuperiorViT-XL/2" in DiT_models_text


def test_build_kwargs_are_constructor_clean():
    """The SuperiorViT branch must not leak DiT-only kwargs."""
    kw = get_dit_build_kwargs(_make_cfg(), class_dropout_prob=0.0)
    # None of the DiT feature kwargs should be present.
    for banned in ("num_ar_blocks", "control_cond_dim", "moe_num_experts", "token_routing_enabled"):
        assert banned not in kw, f"{banned} leaked into SuperiorViT build kwargs"
    assert kw["input_size"] == 32
    assert kw["text_dim"] == 768
    assert kw["uvit_skips"] is True
    assert kw["hidit_hf"] is True
    assert kw["mix_ffn"] is True
    assert int(kw["mmdit_every_n"]) == 4


def _build_model():
    cfg = _make_cfg()
    model_fn = DiT_models_text[cfg.model_name]
    return model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))


def test_forward_shape_and_extra_kwargs():
    torch.manual_seed(0)
    model = _build_model().eval()
    B, C, S = 2, 4, 32
    x = torch.randn(B, C, S, S)
    t = torch.randint(0, 1000, (B,))
    ehs = torch.randn(B, 8, 768)  # (B, L, text_dim)

    with torch.no_grad():
        # Pass the sampler's real call shape, including extras SuperiorViT must ignore.
        out = model(
            x,
            t,
            encoder_hidden_states=ehs,
            block_cache=None,
            style_embedding=None,
            control_image=None,
            negative_prompt_weight=0.0,
        )

    # learn_sigma=True doubles output channels; training slices [:, :C].
    assert out.shape == (B, 2 * C, S, S), out.shape
    assert torch.isfinite(out).all()
    assert out[:, :C].shape == x.shape


def test_gradient_checkpointing_and_backward():
    torch.manual_seed(0)
    model = _build_model().train()
    model.enable_gradient_checkpointing()
    assert model._grad_checkpointing is True

    B, C, S = 2, 4, 32
    x = torch.randn(B, C, S, S, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    ehs = torch.randn(B, 8, 768)

    out = model(x, t, encoder_hidden_states=ehs)
    target = torch.zeros_like(out)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    # At least one parameter received a finite gradient -> the graph is intact
    # through the checkpointed blocks.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert model.hybrid is not None


def test_superior_vit_hybrid_extras_identity_at_init():
    """Prefix-aware HybridDiT extras are identity at init, matching a no-extras twin."""
    from models.superior_vit import SuperiorViT

    common = dict(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=4,
        num_heads=4,
        text_dim=32,
        learn_sigma=False,
        num_registers=2,
        use_jumbo=True,
        use_dynamic_patch=False,
        drop_path_rate=0.0,
        layer_scale_init=0.0,
    )
    torch.manual_seed(0)
    with_ex = SuperiorViT(**common, uvit_skips=True, hidit_hf=True, mix_ffn=True, mmdit_every_n=2).eval()
    torch.manual_seed(0)
    plain = SuperiorViT(**common, uvit_skips=False, hidit_hf=False, mix_ffn=False, mmdit_every_n=0).eval()
    assert with_ex.hybrid is not None
    assert plain.hybrid is None

    x = torch.randn(1, 4, 8, 8)
    t = torch.zeros(1)
    txt = torch.randn(1, 4, 32)
    with torch.no_grad():
        a = with_ex(x, t, encoder_hidden_states=txt)
        b = plain(x, t, encoder_hidden_states=txt)
    assert a.shape == b.shape == (1, 4, 8, 8)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_superior_vit_registered()
    print("[ok] registered")
    test_build_kwargs_are_constructor_clean()
    print("[ok] build kwargs clean")
    test_forward_shape_and_extra_kwargs()
    print("[ok] forward + extra kwargs tolerated")
    test_gradient_checkpointing_and_backward()
    print("[ok] gradient checkpointing + backward")
    print("\nSuperiorViT is runnable end-to-end.")
