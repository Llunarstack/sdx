"""
Tests for model construction and forward pass correctness.

Verifies that DiT variants build without error, produce correct output shapes,
and that the config → build kwargs pipeline is consistent.

Run with:
    pytest tests/test_model_forward.py -v
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Config → build kwargs
# ---------------------------------------------------------------------------

class TestBuildKwargs:
    def test_default_config_builds(self):
        from config.train_config import TrainConfig, get_dit_build_kwargs
        cfg = TrainConfig()
        kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
        assert "input_size" in kw
        assert "text_dim" in kw
        assert kw["input_size"] == cfg.image_size // 8

    def test_text_dim_xxl(self):
        from config.train_config import TrainConfig, get_dit_build_kwargs
        cfg = TrainConfig(text_encoder="google/t5-v1_1-xxl")
        kw = get_dit_build_kwargs(cfg)
        assert kw["text_dim"] == 4096

    def test_text_dim_xl(self):
        from config.train_config import TrainConfig, get_dit_build_kwargs
        cfg = TrainConfig(text_encoder="google/t5-v1_1-xl")
        kw = get_dit_build_kwargs(cfg)
        assert kw["text_dim"] == 1024

    def test_class_dropout_override(self):
        from config.train_config import TrainConfig, get_dit_build_kwargs
        cfg = TrainConfig(caption_dropout_prob=0.1)
        kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
        assert kw["class_dropout_prob"] == 0.0

    def test_latent_size_256(self):
        from config.train_config import TrainConfig, get_dit_build_kwargs
        cfg = TrainConfig(image_size=256)
        kw = get_dit_build_kwargs(cfg)
        assert kw["input_size"] == 32

    def test_latent_size_512(self):
        from config.train_config import TrainConfig, get_dit_build_kwargs
        cfg = TrainConfig(image_size=512)
        kw = get_dit_build_kwargs(cfg)
        assert kw["input_size"] == 64


# ---------------------------------------------------------------------------
# DiT-XL/2-Text forward pass (CPU, small latent)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dit_xl_model():
    """Build a DiT-XL/2-Text on CPU with minimal config."""
    from config.train_config import TrainConfig, get_dit_build_kwargs
    from models import DiT_models_text

    cfg = TrainConfig(image_size=64)  # 8×8 latent — fast on CPU
    kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
    model = DiT_models_text["DiT-XL/2-Text"](**kw)
    model.eval()
    return model, kw


class TestDiTForward:
    def test_output_shape(self, dit_xl_model):
        model, kw = dit_xl_model
        B, C_in, H, W = 2, model.in_channels, kw["input_size"], kw["input_size"]
        x = torch.randn(B, C_in, H, W)
        t = torch.randint(0, 1000, (B,))
        enc = torch.randn(B, 32, kw["text_dim"])
        with torch.no_grad():
            out = model(x, t, encoder_hidden_states=enc)
        assert out.shape == (B, model.out_channels, H, W), f"unexpected output shape {out.shape}"

    def test_output_finite(self, dit_xl_model):
        model, kw = dit_xl_model
        B, C, H, W = 2, model.in_channels, kw["input_size"], kw["input_size"]
        x = torch.randn(B, C, H, W)
        t = torch.randint(0, 1000, (B,))
        enc = torch.randn(B, 32, kw["text_dim"])
        with torch.no_grad():
            out = model(x, t, encoder_hidden_states=enc)
        assert torch.isfinite(out).all(), "model output contains NaN or Inf"

    def test_batch_size_1(self, dit_xl_model):
        model, kw = dit_xl_model
        C_in, H, W = model.in_channels, kw["input_size"], kw["input_size"]
        x = torch.randn(1, C_in, H, W)
        t = torch.randint(0, 1000, (1,))
        enc = torch.randn(1, 32, kw["text_dim"])
        with torch.no_grad():
            out = model(x, t, encoder_hidden_states=enc)
        assert out.shape == (1, model.out_channels, H, W)

    def test_deterministic_eval(self, dit_xl_model):
        """Same input → same output in eval mode (no dropout)."""
        model, kw = dit_xl_model
        B, C, H, W = 2, model.in_channels, kw["input_size"], kw["input_size"]
        x = torch.randn(B, C, H, W)
        t = torch.randint(0, 1000, (B,))
        enc = torch.randn(B, 32, kw["text_dim"])
        with torch.no_grad():
            out1 = model(x, t, encoder_hidden_states=enc)
            out2 = model(x, t, encoder_hidden_states=enc)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# RAELatentBridge
# ---------------------------------------------------------------------------

class TestRAELatentBridge:
    def test_rae_to_dit_shape(self):
        from models.rae_latent_bridge import RAELatentBridge
        bridge = RAELatentBridge(rae_channels=16, dit_channels=4)
        z = torch.randn(2, 16, 8, 8)
        out = bridge.rae_to_dit(z)
        assert out.shape == (2, 4, 8, 8)

    def test_dit_to_rae_shape(self):
        from models.rae_latent_bridge import RAELatentBridge
        bridge = RAELatentBridge(rae_channels=16, dit_channels=4)
        z = torch.randn(2, 4, 8, 8)
        out = bridge.dit_to_rae(z)
        assert out.shape == (2, 16, 8, 8)

    def test_cycle_loss_positive(self):
        from models.rae_latent_bridge import RAELatentBridge
        bridge = RAELatentBridge(rae_channels=16, dit_channels=4)
        z = torch.randn(2, 16, 8, 8)
        loss = bridge.cycle_loss(z)
        assert loss.ndim == 0  # scalar
        assert loss >= 0

    def test_invalid_channels_raises(self):
        from models.rae_latent_bridge import RAELatentBridge
        with pytest.raises(ValueError):
            RAELatentBridge(rae_channels=0, dit_channels=4)


# ---------------------------------------------------------------------------
# Latent geometry helpers
# ---------------------------------------------------------------------------

class TestLatentGeometry:
    def test_num_patch_tokens(self):
        from sdx_native.latent_geometry import num_patch_tokens
        # 256px image, VAE scale 8, patch size 2 → 32×32 latent → 16×16 patches → 256 tokens
        assert num_patch_tokens(256, 8, 2) == 256

    def test_num_patch_tokens_512(self):
        from sdx_native.latent_geometry import num_patch_tokens
        assert num_patch_tokens(512, 8, 2) == 1024

    def test_invalid_returns_zero(self):
        from sdx_native.latent_geometry import num_patch_tokens
        assert num_patch_tokens(255, 8, 2) == 0  # not divisible
