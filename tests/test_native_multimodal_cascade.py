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
    assert out["fused_all_tokens"].shape[0] == 2


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
