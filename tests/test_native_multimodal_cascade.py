from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _DummyDiT(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, t, encoder_hidden_states=None, **kwargs):
        del t, encoder_hidden_states, kwargs
        return self.proj(x)


def test_native_multimodal_transformer_shapes() -> None:
    from models.native_multimodal_transformer import NativeMultimodalTransformer

    mm = NativeMultimodalTransformer(vision_dim=256, text_dim=768, model_dim=512, num_layers=2, num_heads=8)
    v = torch.randn(2, 64, 256)
    t = torch.randn(2, 77, 768)
    out = mm(v, t)
    assert out["fused_vision_tokens"].shape == (2, 64, 512)
    assert out["fused_text_tokens"].shape == (2, 77, 512)
    assert out["fused_all_tokens"].shape == (2, 64 + 77, 512)
    assert out["fused_all_tokens"].shape[0] == 2


def test_native_multimodal_padding_mask_and_extra() -> None:
    from models.native_multimodal_transformer import NativeMultimodalTransformer

    mm = NativeMultimodalTransformer(
        vision_dim=32,
        text_dim=48,
        extra_dim=16,
        model_dim=128,
        num_layers=1,
        num_heads=4,
        proj_dropout=0.1,
    )
    b, nv, nt, ne = 2, 5, 7, 3
    v = torch.randn(b, nv, 32)
    te = torch.randn(b, nt, 48)
    ex = torch.randn(b, ne, 16)
    vm = torch.zeros(b, nv, dtype=torch.bool)
    vm[:, -2:] = True  # pad last two vision tokens
    out = mm(v, te, extra_tokens=ex, vision_padding_mask=vm)
    assert out["fused_extra_tokens"].shape == (b, ne, 128)
    assert out["fused_all_tokens"].shape[1] == nv + nt + ne


def test_native_multimodal_no_modality_embeddings() -> None:
    from models.native_multimodal_transformer import NativeMultimodalTransformer

    mm = NativeMultimodalTransformer(
        vision_dim=16,
        text_dim=16,
        model_dim=64,
        num_layers=1,
        num_heads=4,
        use_modality_embeddings=False,
    )
    assert mm.modality_embed is None
    y = mm(torch.randn(1, 3, 16), torch.randn(1, 4, 16))
    assert y["fused_text_tokens"].shape == (1, 4, 64)


def test_native_multimodal_model_dim_must_divide_heads() -> None:
    from models.native_multimodal_transformer import NativeMultimodalTransformer

    try:
        NativeMultimodalTransformer(vision_dim=8, text_dim=8, model_dim=50, num_heads=8, num_layers=1)
    except ValueError as e:
        assert "divisible" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_concat_padding_masks_helper() -> None:
    from models.native_multimodal_transformer import concat_padding_masks

    device = torch.device("cpu")
    m = torch.tensor([[False, True], [False, False]], dtype=torch.bool)
    full = concat_padding_masks(2, [m, None], [2, 3], device=device)
    assert full is not None
    assert full.shape == (2, 5)
    assert full[:, 1].tolist() == [True, False]
    assert full[:, 2:].sum() == 0


def test_native_multimodal_cross_attn_rms_film() -> None:
    from models.native_multimodal_transformer import NativeMultimodalTransformer

    mm = NativeMultimodalTransformer(
        vision_dim=16,
        text_dim=16,
        model_dim=64,
        num_layers=1,
        num_heads=4,
        cross_attn_heads=4,
        output_norm="rmsnorm",
        film_cond_dim=8,
    )
    b, nv, nt = 2, 3, 4
    v = torch.randn(b, nv, 16)
    t = torch.randn(b, nt, 16)
    c = torch.randn(b, 8)
    out = mm(v, t, film_cond=c)
    assert out["fused_vision_tokens"].shape == (b, nv, 64)
    assert out["fused_all_tokens"].shape == (b, nv + nt, 64)


def test_native_multimodal_output_norm_invalid() -> None:
    from models.native_multimodal_transformer import NativeMultimodalTransformer

    try:
        NativeMultimodalTransformer(
            vision_dim=8, text_dim=8, model_dim=32, num_heads=4, num_layers=1, output_norm="batchnorm"
        )
    except ValueError as e:
        assert "rmsnorm" in str(e).lower() or "layernorm" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_cascaded_blend_and_detach() -> None:
    from models.cascaded_multimodal_diffusion import CascadedMultimodalDiffusion

    base = _DummyDiT(in_channels=4)
    refine = _DummyDiT(in_channels=4)
    pipe = CascadedMultimodalDiffusion(base, refine, blend_base_refine=0.5)
    x = torch.randn(1, 4, 8, 8)
    t0 = torch.tensor([100])
    t1 = torch.tensor([200])
    out = pipe(x, t0, t1)
    # blend = 0.5 * base + 0.5 * refine; refine is Conv on base_out so not trivially equal to either
    assert out["final_output"].shape == out["refine_output"].shape
    out2 = pipe(x, t0, t1, detach_base_for_refine=True)
    assert out2["final_output"].shape == (1, 4, 8, 8)


def test_rae_bridge_learnable_scale() -> None:
    from models.rae_latent_bridge import RAELatentBridge

    b = RAELatentBridge(8, 4, learnable_output_scale=True)
    z = torch.randn(1, 8, 4, 4)
    y = b.rae_to_dit(z)
    assert y.shape == (1, 4, 4, 4)
    loss = b.cycle_loss(z)
    assert loss.ndim == 0


def test_cascaded_multimodal_diffusion_bridge_flow() -> None:
    from models.cascaded_multimodal_diffusion import CascadedMultimodalDiffusion
    from models.rae_latent_bridge import RAELatentBridge

    base = _DummyDiT(in_channels=4)
    refine = _DummyDiT(in_channels=4)
    bridge = RAELatentBridge(rae_channels=16, dit_channels=4)
    pipe = CascadedMultimodalDiffusion(base, refine, bridge=bridge)

    x_rae = torch.randn(2, 16, 32, 32)
    t0 = torch.randint(0, 1000, (2,))
    t1 = torch.randint(0, 1000, (2,))
    out = pipe(x_rae, t0, t1, output_rae_latents=True)
    assert out["base_output"].shape == (2, 4, 32, 32)
    assert out["refine_output"].shape == (2, 4, 32, 32)
    assert out["final_output"].shape == (2, 16, 32, 32)
