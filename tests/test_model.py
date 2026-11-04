"""Tests for model construction and forward pass correctness.

Verifies that DiT variants build without error, produce correct output shapes,
and that the config → build kwargs pipeline is consistent.

Run with:
    pytest tests/test_model.py -v

Also holds what used to be: test_model_forward, test_model_paths_gen_searcher, test_model_presets, test_models_regressions, test_ar_masks_extended, test_hybrid_dit, test_enhanced_spatial_control, test_superior_vit_runnable, test_lora_train, test_checkpoint_analysis, test_hf, test_multi_encoder_encode, test_text_encoder_penta, test_text_encoder_stack, test_ultimate_pipeline.
"""

from __future__ import annotations

import copy
import warnings
from argparse import Namespace
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.presets import OP_MODES, PRESETS, apply_op_mode_to_args, apply_preset_to_args
from config.train_config import TrainConfig, get_dit_build_kwargs
from models import DiT_models_text
from models.attention import (
    TACA,
    LocalWindowAttention,
    block_visit_order,
    build_2d_rope_freqs,
    create_block_causal_mask_2d,
)
from models.conditioning import PromptParser
from models.dit_text import DiT_Text, EncoderDecoderSkip, JointMMDiTAttention, MixFFN, SwiGLUMlp, TimeGatedHFResidual
from models.dit_variants import EnhancedDiT, SpatialControlModule
from models.lora import TrainableLoRALinear, apply_loras, inject_trainable_lora, lora_state_dict
from utils.captioning import merge_wd_tags_into_caption, merge_wd_tags_into_row
from utils.generation.sample_cli_parser import build_sample_parser
from utils.modeling.checkpoints import LoRABank, LoRAEntry, resolve_lora_specs_from_prompt, slugify_lora_key
from utils.modeling.encoders import (
    PENTA_CATALOG,
    TRIPLE_CATALOG,
    PentaTextFusion,
    TripleTextFusion,
    catalog_for_mode,
    checkpoint_text_stack,
    clip_and_long_captions_for_layout,
    encode_kwargs_for_captions,
    prepare_multi_encoder_kwargs,
    stack_status,
    stack_status_lines,
    text_encoder_mode_label,
)
from utils.modeling.hf_loaders import (
    CONTROLNET_REGISTRY,
    HFRewardPanel,
    HFRewardWeights,
    caption_image_chain,
    extract_pil_proxy,
    list_controlnet_types,
    resolve_controlnet_path,
    score_hpsv2,
)
from utils.modeling.model_paths import (
    CONFIG_ONLY_IGNORE,
    FACE_RESTORE_MODELS,
    UPSCALE_MODELS,
    has_local_weights,
    pretrained_catalog,
    registry_summary,
    resolve_entries,
    role_counts,
    scaffold_registry,
    verify_gen_searcher_8b_local,
)

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
        from models.conditioning import RAELatentBridge

        bridge = RAELatentBridge(rae_channels=16, dit_channels=4)
        z = torch.randn(2, 16, 8, 8)
        out = bridge.rae_to_dit(z)
        assert out.shape == (2, 4, 8, 8)

    def test_dit_to_rae_shape(self):
        from models.conditioning import RAELatentBridge

        bridge = RAELatentBridge(rae_channels=16, dit_channels=4)
        z = torch.randn(2, 4, 8, 8)
        out = bridge.dit_to_rae(z)
        assert out.shape == (2, 16, 8, 8)

    def test_cycle_loss_positive(self):
        from models.conditioning import RAELatentBridge

        bridge = RAELatentBridge(rae_channels=16, dit_channels=4)
        z = torch.randn(2, 16, 8, 8)
        loss = bridge.cycle_loss(z)
        assert loss.ndim == 0  # scalar
        assert loss >= 0

    def test_invalid_channels_raises(self):
        from models.conditioning import RAELatentBridge

        with pytest.raises(ValueError):
            RAELatentBridge(rae_channels=0, dit_channels=4)


# ---------------------------------------------------------------------------
# Latent geometry helpers
# ---------------------------------------------------------------------------


class TestLatentGeometry:
    def test_num_patch_tokens(self):
        from sdx_native.native_tools import num_patch_tokens

        # 256px image, VAE scale 8, patch size 2 → 32×32 latent → 16×16 patches → 256 tokens
        assert num_patch_tokens(256, 8, 2) == 256

    def test_num_patch_tokens_512(self):
        from sdx_native.native_tools import num_patch_tokens

        assert num_patch_tokens(512, 8, 2) == 1024

    def test_invalid_returns_zero(self):
        from sdx_native.native_tools import num_patch_tokens

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
    # utils/agentic/agent.py launches sample.py with ``--preset superior``.
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


# ---- 16-channel latent space (FLUX / SD3 / REPA-E VAEs) -------------------------------------------


def _tiny_vae(path: Path, *, channels: int, scale: float, shift: float | None) -> Path:
    """Random-weight AutoencoderKL with the FLUX VAE's layout (8x downsample) at toy width."""
    from diffusers import AutoencoderKL

    kw = {"shift_factor": shift} if shift is not None else {}
    vae = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 16, 16),
        latent_channels=channels,
        norm_num_groups=4,
        scaling_factor=scale,
        **kw,
    )
    vae.save_pretrained(path)
    return path


def test_load_autoencoder_reads_latent_space_and_applies_shift(tmp_path):
    from types import SimpleNamespace

    from utils.modeling.encoders import ShiftedAutoencoder, apply_latent_spec, load_autoencoder

    vae = load_autoencoder("kl", str(_tiny_vae(tmp_path / "flux", channels=16, scale=0.3611, shift=0.1159)))
    assert isinstance(vae, ShiftedAutoencoder)
    cfg = SimpleNamespace()
    spec = apply_latent_spec(cfg, vae)
    assert (spec.channels, spec.scale, spec.shift) == (16, 0.3611, 0.1159)
    assert (cfg.latent_channels, cfg.latent_scale, cfg.latent_shift) == (16, 0.3611, 0.1159)
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        z_inner = vae.vae.encode(x).latent_dist.mode()
        z = vae.encode(x).latent_dist.mode()
        assert z.shape == (1, 16, 4, 4)
        assert torch.allclose(z, z_inner - 0.1159)  # latents are (z - shift) * scale
        lat = torch.randn(1, 16, 4, 4)
        assert torch.allclose(vae.decode(lat).sample, vae.vae.decode(lat + 0.1159).sample)
    plain = load_autoencoder("kl", str(_tiny_vae(tmp_path / "sd", channels=4, scale=0.18215, shift=None)))
    assert not isinstance(plain, ShiftedAutoencoder)  # SD-style VAEs are untouched


@pytest.mark.parametrize(
    "name", ["DiT-B/2-Text", "DiT-P/2-Text", "DiT-Supreme/2-Text", "SuperiorViT-S/2", "HybridDiT-S/2"]
)
def test_every_model_family_trains_on_16_channel_latents(name):
    from types import SimpleNamespace

    from config.train_config import get_dit_build_kwargs
    from diffusion.flow_matching import flow_matching_per_sample_losses
    from models import DiT_models_text

    cfg = SimpleNamespace(model_name=name, image_size=64, latent_channels=16, text_encoder="google/t5-v1_1-base")
    model = DiT_models_text[name](**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    assert model.in_channels == 16
    x0 = torch.randn(2, 16, 8, 8)
    mk = {"encoder_hidden_states": torch.randn(2, 6, get_dit_build_kwargs(cfg)["text_dim"])}
    loss = flow_matching_per_sample_losses(model, x0, torch.randn_like(x0), 1000, mk).mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_old_checkpoint_configs_still_build_4_channel_models():
    from types import SimpleNamespace

    from config.train_config import get_dit_build_kwargs

    old = SimpleNamespace(model_name="DiT-B/2-Text", image_size=128)  # saved before latent_channels existed
    assert get_dit_build_kwargs(old)["in_channels"] == 4


def test_latent_cache_from_another_vae_is_refused(tmp_path):
    import json

    from utils.modeling.encoders import LatentSpec

    from training import loop

    (tmp_path / "latent_spec.json").write_text(json.dumps({"channels": 4, "scale": 0.18215, "shift": 0.0}))
    loop.check_latent_cache_spec(str(tmp_path), LatentSpec(4, 0.18215, 0.0))
    with pytest.raises(ValueError, match="Re-run"):
        loop.check_latent_cache_spec(str(tmp_path), LatentSpec(16, 0.3611, 0.1159))


# ---- instruction editing (in-context source tokens) -------------------------------------------------


def _edit_model(edit: bool = True, channels: int = 16, name: str = "DiT-B/2-Text"):
    from types import SimpleNamespace

    from config.train_config import get_dit_build_kwargs
    from models import DiT_models_text

    cfg = SimpleNamespace(
        model_name=name,
        image_size=64,
        latent_channels=channels,
        text_encoder="google/t5-v1_1-base",
        edit_conditioning=edit,
    )
    model = DiT_models_text[name](**get_dit_build_kwargs(cfg, class_dropout_prob=0.0)).eval()
    # adaLN-Zero makes every block an identity at init (no token mixing at all); open the gates the way the
    # first training steps would, so these tests see tokens attend to each other.
    for pname, p in model.named_parameters():
        if "adaLN_modulation" in pname or pname.startswith("final_layer"):
            torch.nn.init.normal_(p, std=0.02)
    return model


def test_edit_model_conditions_on_the_source_image():
    torch.manual_seed(0)
    model = _edit_model()
    x, t = torch.randn(2, 16, 8, 8), torch.tensor([100, 700])
    mk = {"encoder_hidden_states": torch.randn(2, 5, 768)}
    with torch.no_grad():
        plain = model(x, t, **mk)
        src_a = model(x, t, edit_latents=torch.randn(2, 16, 8, 8), **mk)
        src_b = model(x, t, edit_latents=torch.randn(2, 16, 8, 8), **mk)
    assert plain.shape == src_a.shape == (2, 32, 8, 8)  # only the noisy tokens are predicted
    assert not torch.allclose(src_a, src_b)  # the source image matters
    with pytest.raises(ValueError, match="match the noisy latents"):
        model(x, t, edit_latents=torch.randn(2, 16, 4, 4), **mk)
    with pytest.raises(ValueError, match="without edit_conditioning"):
        _edit_model(edit=False)(x, t, edit_latents=x, **mk)


@pytest.mark.parametrize("name", ["DiT-P/2-Text", "DiT-Supreme/2-Text"])
def test_edit_conditioning_trains_on_qknorm_and_supreme_variants(name):
    from diffusion.flow_matching import flow_matching_per_sample_losses

    model = _edit_model(name=name).train()
    x0 = torch.randn(2, 16, 8, 8)
    mk = {"encoder_hidden_states": torch.randn(2, 5, 768), "edit_latents": torch.randn(2, 16, 8, 8)}
    flow_matching_per_sample_losses(model, x0, torch.randn_like(x0), 1000, mk).mean().backward()
    assert model.edit_type_embed.grad is not None and model.edit_type_embed.grad.abs().sum() > 0


def test_edit_rows_load_source_and_target_pixel_aligned(tmp_path):
    import json

    import numpy as np
    from data import Text2ImageDataset, collate_t2i
    from PIL import Image

    img = tmp_path / "a.png"
    Image.fromarray((np.random.rand(96, 128, 3) * 255).astype("uint8")).save(img)
    manifest = tmp_path / "edits.jsonl"
    row = {"image_path": str(img), "source_image_path": str(img), "instruction": "make the sky purple"}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ds = Text2ImageDataset(str(manifest), image_size=64, crop_mode="random")
    batch = collate_t2i([ds[0], ds[0]])
    assert "make the sky purple" in batch["captions"][0]
    # Same file as source and target + one shared random crop -> identical tensors.
    assert torch.equal(batch["pixel_values"], batch["init_pixel_values"])


def test_edit_training_step_encodes_source_and_respects_dropout(tmp_path):
    from types import SimpleNamespace

    from diffusion.flow_matching import flow_matching_per_sample_losses
    from utils.modeling.encoders import apply_latent_spec, load_autoencoder

    from training import loop

    vae = load_autoencoder("kl", str(_tiny_vae(tmp_path / "vae", channels=16, scale=0.3611, shift=0.1159)))
    cfg = SimpleNamespace(edit_conditioning=True)
    apply_latent_spec(cfg, vae)
    cpu = torch.device("cpu")
    batch = {"init_pixel_values": torch.rand(2, 3, 64, 64) * 2 - 1}
    src = loop.edit_source_latents(cfg, batch, vae, cfg.latent_scale, cpu, dropout=0.0)
    assert src.shape == (2, 16, 8, 8)
    assert loop.edit_source_latents(cfg, batch, vae, cfg.latent_scale, cpu, dropout=1.0) is None
    assert loop.edit_source_latents(cfg, {}, vae, cfg.latent_scale, cpu, dropout=0.0) is None
    model = _edit_model().train()
    kw = loop.conditioning_kwargs(
        cfg,
        {},
        src,
        cpu,
        encoder_hidden=torch.randn(2, 5, 768),
        encoder_hidden_neg=None,
        style_embedding=None,
        token_weights=None,
        edit_latents=src,
    )
    assert kw["edit_latents"] is src
    loss = flow_matching_per_sample_losses(model, torch.randn_like(src), torch.randn_like(src), 1000, kw).mean()
    loss.backward()
    assert torch.isfinite(loss) and model.edit_type_embed.grad is not None


def test_text_to_image_checkpoint_seeds_an_edit_model_but_resume_stays_strict():
    from training import loop

    t2i, edit = _edit_model(edit=False), _edit_model(edit=True)
    assert loop.load_dit_weights(edit, t2i.state_dict(), allow_new_edit_embed=True) is True
    assert torch.equal(edit.x_embedder.proj.weight, t2i.x_embedder.proj.weight)
    with pytest.raises(RuntimeError):
        loop.load_dit_weights(edit, t2i.state_dict(), allow_new_edit_embed=False)  # --resume
    broken = {k: v for k, v in t2i.state_dict().items() if not k.startswith("final_layer")}
    with pytest.raises(RuntimeError, match="does not fit"):
        loop.load_dit_weights(edit, broken, allow_new_edit_embed=True)


def test_sample_edit_image_feeds_both_cfg_branches(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    from diffusion import create_diffusion
    from PIL import Image
    from utils.generation.sample_core import _edit_kwargs
    from utils.modeling.encoders import load_autoencoder

    src = tmp_path / "photo.png"
    Image.fromarray((np.random.rand(70, 90, 3) * 255).astype("uint8")).save(src)
    vae = load_autoencoder("kl", str(_tiny_vae(tmp_path / "vae", channels=16, scale=0.3611, shift=0.1159)))
    model = _edit_model()
    m = SimpleNamespace(
        model=model, vae=vae, ae_type="kl", latent_scale=0.3611, device=torch.device("cpu"), canvas_w=64, canvas_h=64
    )
    emb = torch.randn(1, 5, 768)
    cond = {"encoder_hidden_states": emb.expand(2, -1, -1)}
    uncond = {"encoder_hidden_states": torch.zeros_like(emb).expand(2, -1, -1)}
    _edit_kwargs(SimpleNamespace(edit_image=str(src)), m, cond, uncond, 2)
    assert cond["edit_latents"].shape == uncond["edit_latents"].shape == (2, 16, 8, 8)
    out = create_diffusion(timestep_respacing="", num_timesteps=1000).sample_loop(
        model,
        (2, 16, 8, 8),
        model_kwargs_cond=cond,
        model_kwargs_uncond=uncond,
        num_inference_steps=3,
        cfg_scale=4.0,
        flow_matching_sample=True,
        device="cpu",
    )
    assert out.shape == (2, 16, 8, 8) and torch.isfinite(out).all()
    m.model = _edit_model(edit=False)
    with pytest.raises(SystemExit, match="edit-conditioning"):
        _edit_kwargs(SimpleNamespace(edit_image=str(src)), m, {}, {}, 1)


def test_edit_conditioning_config_guards():
    from config.train_config import TrainConfig
    from utils.training.monitoring import validate_train_config

    def errors(**kw):
        name = kw.pop("model_name", "DiT-B/2-Text")
        cfg = TrainConfig(data_path="x", model_name=name, edit_conditioning=True, **kw)
        return " | ".join(str(e) for e in validate_train_config(cfg, require_cuda=False))

    assert "edit_conditioning" not in errors()
    assert "img2img_prob" in errors(img2img_prob=0.2)
    assert "DiT-*-Text" in errors(model_name="SuperiorViT-S/2")


# ==================================================================================================
# test_models_regressions: Regression tests for bugs fixed in ``models/``.
# ==================================================================================================


def test_token_routing_with_register_tokens_backpropagates():
    # The register-token gate used to be written in place into the sigmoid output, breaking backward.
    model = DiT_Text(
        input_size=8,
        patch_size=2,
        hidden_size=32,
        depth=1,
        num_heads=4,
        text_dim=16,
        num_register_tokens=2,
        token_routing_enabled=True,
        use_xformers=False,
    ).train()
    out = model(torch.randn(2, 4, 8, 8), torch.tensor([10, 500]), encoder_hidden_states=torch.randn(2, 3, 16))
    out.sum().backward()
    assert model.blocks[0].token_router[1].weight.grad is not None


def test_trained_dora_adapter_loads_with_same_output():
    # Inference used to multiply the low-rank delta by the DoRA magnitude instead of renormalizing W + dW.
    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(8, 6))
    trained = TrainableLoRALinear(copy.deepcopy(base[0]), rank=2, alpha=4.0, use_dora=True)
    with torch.no_grad():
        trained.lora_up.normal_()
        trained.dora_magnitude.mul_(1.7)
    x = torch.randn(3, 8)
    _, touched = apply_loras(base, [(trained.adapter_state("0"), 1.0)], normalize_scales=False)
    assert touched == 1
    assert torch.allclose(base(x), trained(x), atol=1e-5)


def test_prompt_parser_two_word_negation_and_attribute_positions():
    parsed = PromptParser().parse("a man free of armor wearing a red and blue cloak")
    armor, cloak = parsed.triples
    # "free of" / "devoid of" never matched the single-word negation regex.
    assert armor.negated and parsed.negation_indices == [2]
    # token_indices assumed attributes sit right before the subject, picking "and" instead of "red".
    assert cloak.token_indices == [7, 9, 10]


def test_rope2d_half_precision_tables_match_float32():
    # Angles used to be computed in the requested dtype; bf16 angles near 60 rad were ~0.1 off in cos.
    ref_cos, _ = build_2d_rope_freqs(64, 64, 64)
    bf16_cos, _ = build_2d_rope_freqs(64, 64, 64, dtype=torch.bfloat16)
    assert bf16_cos.dtype == torch.bfloat16
    assert (bf16_cos.float() - ref_cos).abs().max() < 0.01


def test_local_window_attention_masks_padding():
    # Zero-padded keys of partial windows used to take part in the softmax and dilute real attention.
    torch.manual_seed(0)
    attn = LocalWindowAttention(32, 4, window_size=8, num_global_tokens=0, qk_norm=False)
    torch.nn.init.normal_(attn.out_proj.weight, std=0.2)
    x = torch.randn(2, 25, 32)  # 5x5 grid inside one padded 8x8 window == plain full attention
    q, k, v = (t.reshape(2, 25, 4, 8).transpose(1, 2) for t in attn.qkv(x).chunk(3, dim=-1))
    ref = attn.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 25, 32))
    assert torch.allclose(attn(x), ref, atol=1e-5)


# ==================================================================================================
# test_ar_masks_extended: Tests for extended AR macro-block traversal orders.
# ==================================================================================================


def test_block_visit_order_supports_new_orders():
    b = 4
    for order in ("raster", "zorder", "snake", "spiral"):
        cells = block_visit_order(b, order)
        assert len(cells) == b * b
        assert len(set(cells)) == b * b


def test_create_block_causal_mask_supports_new_orders():
    h = w = 8
    b = 2
    masks = {o: create_block_causal_mask_2d(h, w, b, block_order=o) for o in ("raster", "zorder", "snake", "spiral")}
    for m in masks.values():
        assert m.shape == (h * w, h * w)
        assert torch.isfinite(m.diag()).all()
    # New orders should not collapse to identical masks in this setup.
    assert not torch.equal(masks["raster"], masks["snake"])
    assert not torch.equal(masks["raster"], masks["spiral"])


# ==================================================================================================
# test_hybrid_dit: Hybrid DiT extras: encoder-decoder skips, MMDiT joint attn, Mix-FFN, late HF residual.
# ==================================================================================================


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


# ==================================================================================================
# test_enhanced_spatial_control: Tests for EnhancedDiT: count-modulated layout attention and gradient checkpointing.
# ==================================================================================================


class _FixedCount(nn.Module):
    def __init__(self, value: float, num_objects: int) -> None:
        super().__init__()
        self.value = value
        self.num_objects = num_objects

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled.new_full((pooled.shape[0], self.num_objects), self.value)


def test_spatial_control_preserves_shape() -> None:
    torch.manual_seed(0)
    mod = SpatialControlModule(hidden_size=32, num_objects=4)
    x = torch.randn(2, 16, 32)
    layout = torch.rand(2, 4, 4)
    out = mod(x, layout)
    assert out.shape == (2, 16, 32)


def test_count_constraint_modulates_layout_attention() -> None:
    torch.manual_seed(0)
    mod = SpatialControlModule(hidden_size=32, num_objects=4)
    x = torch.randn(2, 16, 32)
    layout = torch.rand(2, 4, 4)

    mod.count_constraint = _FixedCount(0.0, num_objects=4)
    out_zero = mod(x, layout)
    mod.count_constraint = _FixedCount(1.0, num_objects=4)
    out_one = mod(x, layout)

    assert out_zero.shape == (2, 16, 32)
    assert out_one.shape == (2, 16, 32)
    assert not torch.allclose(out_zero, out_one)


def test_gradient_checkpointing_recomputes_each_block() -> None:
    torch.manual_seed(0)
    plain = EnhancedDiT(input_size=8, hidden_size=32, depth=3, num_heads=4, num_classes=4, class_dropout_prob=0.0)
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ckpt = copy.deepcopy(plain)
    ckpt.enable_gradient_checkpointing()
    x, t, y = torch.randn(2, 4, 8, 8), torch.tensor([10, 500]), torch.tensor([1, 2])
    for m in (plain, ckpt):
        m(x, t, y).pow(2).mean().backward()
    for a, b in zip(plain.parameters(), ckpt.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


# ==================================================================================================
# test_superior_vit_runnable: End-to-end runnability smoke test for the SuperiorViT generative backbone.
# ==================================================================================================


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
    from models.dit_variants import SuperiorViT

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


# ==================================================================================================
# test_lora_train: Trainable LoRA/DoRA injection: freezing, grad flow, and adapter round-trip.
# ==================================================================================================


class _TinyDiT(nn.Module):
    """Stand-in with the module-name substrings the default targets match."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(16, 16)  # not a target substring -> stays frozen-trainable base

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 48)
        self.proj = nn.Linear(16, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        _ = self.qkv(x)
        x = self.proj(x)
        x = self.fc2(torch.relu(self.fc1(x)))
        return x


def test_injection_wraps_targets_and_freezes_base():
    m = _TinyDiT()
    trainable, wrapped = inject_trainable_lora(m, rank=4, alpha=8.0)
    # 2 blocks x (qkv, proj, fc1, fc2) = 8 wrapped Linears.
    assert wrapped == 8
    # Base weights frozen; only adapter params require grad.
    for name, p in m.named_parameters():
        if "lora_down" in name or "lora_up" in name:
            assert p.requires_grad, name
        else:
            assert not p.requires_grad, name
    assert len(trainable) == 2 * 4 * 2  # 8 layers x (down, up)


def test_forward_and_backward_updates_only_adapters():
    torch.manual_seed(0)
    m = _TinyDiT()
    inject_trainable_lora(m, rank=4, alpha=8.0)
    base_before = m.blocks[0].proj.linear.weight.clone()

    x = torch.randn(2, 16)
    loss = m(x).pow(2).mean()
    loss.backward()
    # Adapter got grad; frozen base did not.
    assert m.blocks[0].proj.lora_down.grad is not None
    assert m.blocks[0].proj.linear.weight.grad is None
    torch.testing.assert_close(m.blocks[0].proj.linear.weight, base_before)


def test_lora_identity_at_init():
    # up is zero-init, so the wrapped layer == base layer at step 0.
    torch.manual_seed(1)
    lin = nn.Linear(16, 16)
    wrapped = TrainableLoRALinear(lin, rank=4, alpha=8.0)
    x = torch.randn(3, 16)
    torch.testing.assert_close(wrapped(x), lin(x))


def test_dora_has_magnitude_and_runs():
    lin = nn.Linear(16, 16)
    w = TrainableLoRALinear(lin, rank=4, alpha=8.0, use_dora=True)
    assert w.dora_magnitude is not None and w.dora_magnitude.requires_grad
    out = w(torch.randn(2, 16))
    assert out.shape == (2, 16)


def test_adapter_state_dict_keys_match_inference_loader():
    from models.lora import _extract_adapters

    m = _TinyDiT()
    inject_trainable_lora(m, rank=4, alpha=8.0, use_dora=True)
    sd = lora_state_dict(m)
    # Keys use the .lora_down/.lora_up/.alpha/.dora_magnitude_vector convention.
    assert any(k.endswith(".lora_down.weight") for k in sd)
    assert any(k.endswith(".lora_up.weight") for k in sd)
    assert any(k.endswith(".dora_magnitude_vector") for k in sd)
    # The inference loader parses them into adapters with down+up populated.
    adapters = _extract_adapters(sd)
    assert adapters
    a = next(iter(adapters.values()))
    assert a.down is not None and a.up is not None


def test_no_target_match_reports_zero():
    solo = nn.Linear(8, 8)  # name "" has no target substring
    _, wrapped = inject_trainable_lora(solo, rank=2, alpha=2.0)
    assert wrapped == 0


def test_layer_group_last_matches_lora_stage_buckets():
    m = nn.Module()
    m.blocks = nn.ModuleList([nn.ModuleDict({"attn": nn.Linear(8, 8), "mlp": nn.Linear(8, 8)}) for _ in range(9)])
    params, n = inject_trainable_lora(m, rank=4, alpha=4.0, targets=("attn", "mlp"), layer_group="last")
    # Depth 0..8 splits into non-overlapping thirds; "last" is blocks 6..8: 3 blocks x 2 linears.
    assert n == 6
    assert all(p.requires_grad for p in params)
    assert [type(b["attn"]).__name__ for b in m.blocks] == ["Linear"] * 6 + ["TrainableLoRALinear"] * 3


def test_layer_groups_partition_depth_like_inference_loader():
    from models.lora import layer_in_group

    for depth in (2, 9, 28):
        groups = [[i for i in range(depth) if layer_in_group(i, g, 0, depth - 1)] for g in ("first", "middle", "last")]
        assert sorted(sum(groups, [])) == list(range(depth)), (depth, groups)


# ==================================================================================================
# test_checkpoint_analysis: Tests for ``utils.modeling.checkpoints.analyze_checkpoint_differences``.
# ==================================================================================================


@dataclass(slots=True)
class _SlottedTrainStub:
    lr: float


def test_analyze_checkpoint_differences_slotted_config(tmp_path: Path) -> None:
    """Embedded checkpoint configs may be slotted dataclasses (no ``__dict__``)."""
    torch = pytest.importorskip("torch")
    from utils.modeling.checkpoints import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save(
        {"model": {"w": torch.tensor([1.0, 1.0])}, "config": _SlottedTrainStub(lr=0.01), "step": 0},
        p1,
    )
    torch.save(
        {"model": {"w": torch.tensor([2.0, 2.0])}, "config": _SlottedTrainStub(lr=0.02), "step": 1},
        p2,
    )
    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["statistics"]["step_difference"] == 1
    assert out["statistics"]["tensors_compared"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0
    assert out["statistics"]["tensor_keys_checked"] == 1
    assert out["statistics"]["config_keys_differing"] == 1
    assert "lr" in out["config_differences"]
    assert out["config_differences"]["lr"]["checkpoint1"] == 0.01
    assert out["config_differences"]["lr"]["checkpoint2"] == 0.02


def test_analyze_checkpoint_single_el_tensor_no_std_warning(tmp_path: Path) -> None:
    """Single-element weight diffs must not trigger PyTorch unbiased-std warnings."""
    torch = pytest.importorskip("torch")
    from utils.modeling.checkpoints import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"model": {"w": torch.tensor([1.0])}, "config": {}, "step": 0}, p1)
    torch.save({"model": {"w": torch.tensor([2.0])}, "config": {}, "step": 0}, p2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = analyze_checkpoint_differences(str(p1), str(p2))

    assert out["statistics"]["tensors_compared"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0
    assert out["parameter_differences"]["w"]["std_diff"] == 0.0
    dof_msgs = [str(w.message).lower() for w in caught if "degrees of freedom" in str(w.message).lower()]
    assert not dof_msgs, dof_msgs


def test_analyze_checkpoint_missing_model_dict(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from utils.modeling.checkpoints import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"step": 0, "config": {}}, p1)
    torch.save({"model": {"w": torch.tensor([1.0, 2.0])}, "step": 0, "config": {}}, p2)

    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["statistics"]["model_dict_checkpoint1"] == "missing_or_empty"
    assert out["statistics"]["model_dict_checkpoint2"] == "ok"
    assert out["parameter_differences"]["w"]["status"] == "only_in_checkpoint2"
    assert out["statistics"]["total_parameters"] == 0
    assert out["statistics"]["tensors_only_checkpoint2"] == 1
    assert out["statistics"]["tensors_compared"] == 0
    assert out["statistics"]["tensor_keys_checked"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0


def test_analyze_checkpoint_shape_mismatch_flagged(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from utils.modeling.checkpoints import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"model": {"w": torch.zeros(2)}, "step": 0}, p1)
    torch.save({"model": {"w": torch.zeros(3)}, "step": 0}, p2)

    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["parameter_differences"]["w"]["status"] == "shape_mismatch"
    assert out["statistics"]["total_parameters"] == 0
    assert out["statistics"]["tensors_shape_mismatch"] == 1
    assert out["statistics"]["tensors_compared"] == 0
    assert out["statistics"]["tensor_keys_checked"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0


def test_analyze_checkpoint_non_tensor_weight_skipped(tmp_path: Path) -> None:
    """Model dicts may contain non-tensor values; diff must not call ``torch`` ops on them."""
    torch = pytest.importorskip("torch")
    from utils.modeling.checkpoints import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"model": {"w": torch.zeros(2), "meta": 1}, "step": 0}, p1)
    torch.save({"model": {"w": torch.ones(2), "meta": 2}, "step": 0}, p2)

    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["parameter_differences"]["meta"]["status"] == "skipped_non_tensor"
    assert out["parameter_differences"]["meta"]["type_checkpoint1"] == "int"
    assert out["statistics"]["tensors_skipped_non_tensor"] == 1
    assert out["statistics"]["tensors_compared"] == 1
    assert out["statistics"]["tensor_keys_checked"] == 2
    assert "w" in out["parameter_differences"]
    assert "mean_diff" in out["parameter_differences"]["w"]


# ==================================================================================================
# test_hf: Also holds what used to be: test_hf_control, test_hf_loaders, test_hf_reward, test_hf_scaffold.
# ==================================================================================================


# ==================================================================================================
# test_hf_control: test_hf_control
# ==================================================================================================


def test_controlnet_registry_has_extended_types():
    types = list_controlnet_types()
    for name in ("mlsd", "softedge", "seg", "normal", "hed", "canny_sdxl"):
        assert name in types
    assert len(CONTROLNET_REGISTRY) >= 12


def test_resolve_controlnet_path_returns_string():
    p = resolve_controlnet_path("canny")
    assert isinstance(p, str)
    assert p


def test_pil_softedge_proxy(tmp_path: Path):
    from PIL import Image

    src = tmp_path / "in.png"
    out = tmp_path / "soft.png"
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(src)
    written = extract_pil_proxy(str(src), str(out), "softedge")
    assert written == str(out)
    assert out.is_file()


# ==================================================================================================
# test_hf_loaders: test_hf_loaders
# ==================================================================================================


def test_caption_chain_heuristic_fallback(tmp_path: Path):
    img = tmp_path / "ref.png"
    from PIL import Image

    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(img)
    cap, backend = caption_image_chain(str(img), backends=[])
    assert cap == ""
    assert backend == ""


def test_score_hpsv2_returns_none_without_model():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    assert score_hpsv2(rgb, "a cat", device="cpu") is None


# ==================================================================================================
# test_hf_reward: test_hf_reward
# ==================================================================================================


def test_hf_reward_panel_empty_backends():
    # Zero every backend weight (including ones added after this test was
    # written) so the panel really has no active backends.
    panel = HFRewardPanel(weights=HFRewardWeights(**{f.name: 0.0 for f in fields(HFRewardWeights)}))
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    bd = panel.score_breakdown(rgb, prompt="test")
    assert bd.composite == 0.5
    assert bd.parts == {}


def test_hf_reward_panel_score_without_models():
    panel = HFRewardPanel(device="cpu")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    s = panel.score(rgb, prompt="a cat")
    assert 0.0 <= s <= 1.0


# ==================================================================================================
# test_hf_scaffold: test_hf_scaffold
# ==================================================================================================


def test_scaffold_registry_has_new_boost_models():
    names = {e.name for e in scaffold_registry()}
    for expected in (
        "Florence-2-base",
        "BLIP2-opt-2.7b",
        "Qwen2-VL-2B-Instruct",
        "Depth-Anything-V2-Small",
        "CLIP-ViT-H-14",
        "OWLv2-base-patch16-ensemble",
        "ControlNet-Canny",
        "SmolVLM-256M-Instruct",
        "OneAlign",
        "Florence-2-large",
        "ControlNet-Depth",
        "MetaCLIP-B16",
        "SmolVLM2-2B-Instruct",
        "GIT-base-coco",
        "GroundingDINO-SwinT",
        "ControlNet-SoftEdge",
        "MUSIQ",
        "EVA02-CLIP-L-14",
        "LLaVA-1.5-7B",
        "InternVL2-2B",
        "Qwen2.5-VL-3B-Instruct",
        "ControlNet-HED",
        "ZoeDepth",
        "NSFW-Detector",
        "LLaVA-v1.6-Mistral-7B",
        "PaliGemma2-3B",
        "DINOv3-ViT-S16",
        "GFPGAN",
        "Watermark-Detector",
        "UMT5-XXL",
    ):
        assert expected in names


def test_resolve_entries_by_role():
    vlms = resolve_entries(roles=["vlm"])
    assert vlms
    assert all(e.role == "vlm" for e in vlms)
    assert "Florence-2-base" in {e.name for e in vlms}


def test_config_only_ignore_blocks_weights():
    assert any("safetensors" in p for p in CONFIG_ONLY_IGNORE)


def test_has_local_weights(tmp_path: Path):
    assert has_local_weights(tmp_path) is False
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert has_local_weights(tmp_path) is False
    (tmp_path / "model.safetensors").write_text("x", encoding="utf-8")
    assert has_local_weights(tmp_path) is True


def test_pretrained_catalog_includes_new_rows():
    names = {str(r["name"]) for r in pretrained_catalog()}
    assert "Florence-2-base" in names
    assert "BLIP2-opt-2.7b" in names
    assert "Qwen2-VL-2B-Instruct" in names
    assert "Depth-Anything-V2-Small" in names
    assert "ControlNet-Canny" in names


def test_registry_summary_and_roles():
    s = registry_summary()
    assert int(s["total_registry"]) == len(scaffold_registry()) >= 140
    rc = role_counts()
    for role in ("vlm", "reward", "control", "depth", "text_encoder"):
        assert rc.get(role, 0) >= 1


def test_post_process_model_lists():
    assert {"Real-ESRGAN", "SwinIR-classical-x2"} <= set(UPSCALE_MODELS)
    assert {"CodeFormer", "GFPGAN"} <= set(FACE_RESTORE_MODELS)


# ==================================================================================================
# test_multi_encoder_encode: Tests for multi-encoder encode kwargs helper.
# ==================================================================================================


@dataclass
class _FakeFusion:
    extra_token_count: int = 2


@dataclass
class _FakeBundle:
    mode: str
    fusion: _FakeFusion | None = None


def test_encode_kwargs_t5_only():
    assert encode_kwargs_for_captions(["hello"], None) == {}
    assert encode_kwargs_for_captions(["hello"], _FakeBundle(mode="t5")) == {}


def test_encode_kwargs_triple():
    b = _FakeBundle(mode="triple", fusion=_FakeFusion())
    kw = encode_kwargs_for_captions(["a cat"], b)
    assert kw == {"clip_captions": ["a cat"]}
    assert "long_clip_captions" not in kw


def test_encode_kwargs_penta():
    b = _FakeBundle(mode="penta", fusion=_FakeFusion(extra_token_count=4))
    kw = encode_kwargs_for_captions(["long prompt text"], b)
    assert kw["clip_captions"] == ["long prompt text"]
    assert kw["long_clip_captions"] == ["long prompt text"]


def test_layout_aware_clip_caption():
    layout = {
        "preset_order": "subject_first",
        "subjects": ["red sports car"],
        "environment": "mountain road at sunset",
    }
    flat = "red sports car, mountain road at sunset, cinematic"
    clip_s, long_s = clip_and_long_captions_for_layout(flat, layout)
    assert "SUBJECT" in clip_s or "red sports car" in clip_s
    assert long_s == flat

    b = _FakeBundle(mode="penta", fusion=_FakeFusion(extra_token_count=4))
    kw = prepare_multi_encoder_kwargs(
        [flat],
        b,
        layout_specs=[layout],
        flat_full_prompts=[flat],
    )
    assert kw["clip_captions"][0] != flat or "SUBJECT" in kw["clip_captions"][0]
    assert kw["long_clip_captions"][0] == flat


def test_layout_specs_none_falls_back():
    b = _FakeBundle(mode="triple", fusion=_FakeFusion())
    kw = prepare_multi_encoder_kwargs(["plain"], b, layout_specs=[None])
    assert kw["clip_captions"] == ["plain"]


# ==================================================================================================
# test_text_encoder_penta: Unit tests for penta / triple text encoder fusion (no HF weights required).
# ==================================================================================================


def test_triple_fusion_appends_two_tokens():
    fusion = TripleTextFusion(768, 1280, out_dim=4096)
    t5 = torch.randn(2, 16, 4096)
    clip_l = torch.randn(2, 768)
    clip_bg = torch.randn(2, 1280)
    out = fusion(t5, clip_l, clip_bg)
    assert out.shape == (2, 18, 4096)
    assert fusion.extra_token_count == 2


def test_penta_fusion_appends_four_tokens():
    fusion = PentaTextFusion(768, 1280, 1024, 768, out_dim=4096)
    t5 = torch.randn(2, 16, 4096)
    clip_l = torch.randn(2, 768)
    clip_bg = torch.randn(2, 1280)
    clip_h = torch.randn(2, 1024)
    clip_long = torch.randn(2, 768)
    out = fusion(t5, clip_l, clip_bg, clip_h, clip_long)
    assert out.shape == (2, 20, 4096)
    assert fusion.extra_token_count == 4


# ==================================================================================================
# test_text_encoder_stack: Tests for text encoder stack readiness and checkpoint introspection.
# ==================================================================================================


def test_catalog_for_mode():
    assert catalog_for_mode("penta") == PENTA_CATALOG
    assert catalog_for_mode("triple") == TRIPLE_CATALOG
    assert catalog_for_mode("t5") == ("T5-XXL",)


def test_stack_status_returns_slots():
    st = stack_status("penta")
    assert st.mode == "penta"
    assert len(st.slots) == 5
    assert {s.name for s in st.slots} == set(PENTA_CATALOG)


def test_stack_status_lines_markdown():
    lines = stack_status_lines("triple")
    assert any("triple" in ln for ln in lines)
    assert any("CLIP-ViT-L-14" in ln for ln in lines)


def test_text_encoder_mode_label():
    assert "LongCLIP-L" in text_encoder_mode_label("penta")
    assert "CLIP-ViT-L-14" in text_encoder_mode_label("triple")
    assert text_encoder_mode_label("t5") == "T5-XXL"


def test_checkpoint_text_stack(tmp_path):
    ckpt = tmp_path / "mini.pt"
    torch.save(
        {"config": TrainConfig(text_encoder_mode="penta"), "model": {}, "text_encoder_fusion": {"w": 1}}, str(ckpt)
    )
    assert checkpoint_text_stack(str(ckpt)) == ("penta", True)
    assert checkpoint_text_stack(str(tmp_path / "missing.pt")) == ("t5", False)


# ==================================================================================================
# test_ultimate_pipeline: Tests for WD tag merge and LoRA bank resolution.
# ==================================================================================================


def test_merge_wd_tags_preserves_identity():
    cap = merge_wd_tags_into_caption(
        "1girl, solo",
        ["long hair", "blue eyes", "1girl"],
        identity_tags=["1girl", "wlop"],
    )
    assert "long hair" in cap
    assert "blue eyes" in cap
    assert cap.count("1girl") == 1


def test_merge_wd_tags_into_row():
    row = merge_wd_tags_into_row(
        {"caption": "1girl", "artist_tags": ["wlop"], "character_tags": ["hatsune miku"]},
        ["twintails", "skirt"],
    )
    assert "twintails" in row["caption"]
    assert "wd_tagger" in row["tag_sources"]


def test_slugify_lora_key():
    assert slugify_lora_key("WLOP") == "wlop"
    assert slugify_lora_key("kantoku") == "kantoku"


def test_resolve_lora_specs_from_prompt(tmp_path: Path):
    bank = LoRABank(root=tmp_path)
    fake = tmp_path / "artist" / "wlop" / "best_lora.pt"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"x")
    bank.artists["wlop"] = LoRAEntry(lora=str(fake), default_scale=0.8, role="style")

    specs = resolve_lora_specs_from_prompt("@wlop 1girl", bank, artist_strength=1.25)
    assert len(specs) == 1
    assert "best_lora.pt" in specs[0]
    assert ":1.000:" in specs[0] or specs[0].endswith(":style")


def test_lora_bank_save_load(tmp_path: Path):
    bank = LoRABank(root=tmp_path)
    fake = tmp_path / "style" / "anime" / "best_lora.pt"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"x")
    bank.styles["anime"] = LoRAEntry(lora="style/anime/best_lora.pt", default_scale=0.6)
    idx = tmp_path / "index.json"
    bank.save(idx)
    loaded = LoRABank.load(idx)
    assert "anime" in loaded.styles
    assert loaded.styles["anime"].default_scale == 0.6
