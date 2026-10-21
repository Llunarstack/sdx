"""Tests for model construction and forward pass correctness.

Verifies that DiT variants build without error, produce correct output shapes,
and that the config → build kwargs pipeline is consistent.

Run with:
    pytest tests/test_model.py -v

Also holds what used to be: test_model_forward, test_model_paths_gen_searcher, test_model_presets.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch
from config.defaults.model_presets import OP_MODES, PRESETS, apply_op_mode_to_args, apply_preset_to_args
from utils.generation.sample_cli_parser import build_sample_parser
from utils.modeling.model_paths import pretrained_catalog, verify_gen_searcher_8b_local

# ==================================================================================================
# test_model_forward: Tests for model construction and forward pass correctness.
#
# Verifies that DiT variants build without error, produce correct output shapes,
# and that the config → build kwargs pipeline is consistent.
#
# Run with:
#     pytest tests/test_model.py -v
# ==================================================================================================

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


@pytest.mark.parametrize("name", ["DiT_Text", "DiT_Supreme_Text"])
def test_every_block_branch_trains_with_default_layerscale(name):
    """adaLN-Zero gates start at 0; a LayerScale gamma of 0 on top would leave both with zero gradient forever."""
    import models.dit_text as dt

    torch.manual_seed(0)
    model = getattr(dt, name)(input_size=8, hidden_size=32, depth=2, num_heads=4, text_dim=16, class_dropout_prob=0.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        out = model(torch.randn(2, 4, 8, 8), torch.tensor([10, 500]), encoder_hidden_states=torch.randn(2, 5, 16))
        opt.zero_grad()
        (out - torch.randn_like(out)).pow(2).mean().backward()
        opt.step()
    block = model.blocks[0]
    for p in (block.attn.qkv.weight, block.adaLN_modulation[-1].weight, block.ls_attn.gamma, block.ls_mlp.gamma):
        assert p.grad is not None and p.grad.abs().sum() > 0


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


# ==================================================================================================
# test_model_paths_gen_searcher: test_model_paths_gen_searcher
# ==================================================================================================


def test_verify_gen_searcher_local_missing(tmp_path: Path):
    out = verify_gen_searcher_8b_local(str(tmp_path))
    assert out["is_local_dir"] is True
    assert out["all_required_present"] is False
    assert out["missing"]


def test_verify_gen_searcher_local_complete(tmp_path: Path):
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    for name in required:
        (tmp_path / name).write_text("x", encoding="utf-8")
    out = verify_gen_searcher_8b_local(str(tmp_path))
    assert out["is_local_dir"] is True
    assert out["all_required_present"] is True
    assert out["missing"] == []
    assert len(out["found_shards"]) == 4


def test_pretrained_catalog_has_core_rows():
    rows = pretrained_catalog()
    names = {str(r["name"]) for r in rows}
    assert "T5-XXL" in names
    assert "StableCascade-Prior" in names
    assert "StableCascade-Decoder" in names
    assert "GenSearcher-8B" in names
    assert "ImageReward" in names
    assert "PickScore_v1" in names
    assert "GroundingDINO-Base" in names
    assert "CountGD" in names
    assert "TrOCR-Large-Printed" in names
    assert "PerceptCLIP_IQA" in names
    assert "Depth-Anything-V2-Large" in names
    assert "SAM2-Hiera-Large" in names
    assert "Real-ESRGAN" in names
    assert "LongCLIP-L" in names
    assert "moondream2" in names
    assert "Marigold-Depth-v1-1" in names
    assert "Marigold-Normals-v1-1" in names
    assert "TAESD" in names
    assert "TAESDXL" in names
    assert "CodeFormer" in names
    assert "Consistency-Decoder" in names
    assert "ConvNeXtV2-Large" in names
    assert "LAION-Aesthetic-v2" in names
    assert "AnyDoor-Ref" in names
    assert "Florence-2-base" in names
    assert "BLIP2-opt-2.7b" in names
    assert "Qwen2-VL-2B-Instruct" in names
    assert "Depth-Anything-V2-Small" in names
    assert "ControlNet-Canny" in names
    assert "SmolVLM2-2B-Instruct" in names
    assert "GIT-base-coco" in names
    assert "MUSIQ" in names
    assert "LLaVA-1.5-7B" in names
    assert "ZoeDepth" in names
    assert "GFPGAN" in names
    assert "UMT5-XXL" in names


# ==================================================================================================
# test_model_presets: Sampler presets / OP modes: CLI exposure and soft-default application.
# ==================================================================================================


@pytest.mark.parametrize("flag, names", [("--preset", PRESETS), ("--op-mode", OP_MODES)])
def test_cli_accepts_every_table_entry(flag, names):
    # utils/agentic/tools.py launches sample.py with ``--preset superior``.
    parser = build_sample_parser()
    for name in names:
        args = parser.parse_args(["--ckpt", "x.pt", "--prompt", "p", flag, name])
        assert getattr(args, flag[2:].replace("-", "_")) == name


def test_preset_fills_parser_defaults_only():
    args = Namespace(cfg_scale=7.5, steps=50, solver="heun", naturalize=False, anti_bleed=False)
    apply_preset_to_args(args, "sdxl")
    assert args.cfg_scale == 6.5 and args.steps == 28
    assert args.solver == "heun"
    assert args.naturalize and args.anti_bleed


def test_op_mode_values_override_defaults():
    args = Namespace(solver="dpmpp_2m", scheduler="ays_dit", steps=50, naturalize_grain=0.015, naturalize=False)
    apply_op_mode_to_args(args, "fast")
    assert (args.solver, args.steps) == ("unipc", 15)
    apply_op_mode_to_args(args, "portrait")
    assert args.naturalize and args.naturalize_grain == 0.02
