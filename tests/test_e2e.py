"""End-to-end smoke: train a tiny DiT a few steps on CPU, save a checkpoint, sample it through sample.py's entry point.

Everything is built locally (tiny T5 + word-level tokenizer, tiny KL VAE, 2-block DiT), so nothing downloads.
``train.py``'s ``main`` requires CUDA, so the steps run through the same training helpers it calls.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")
tokenizers = pytest.importorskip("tokenizers")

TINY = "DiT-E2E/2-Text"
WORDS = "a red cube blue sphere on the table green tree under sky photo of".split()


def _tiny_text_encoder(root: Path) -> Path:
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {w: i for i, w in enumerate(["<pad>", "</s>", "<unk>", *WORDS])}
    tok = tokenizers.Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    out = root / "tiny_t5"
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", model_max_length=300
    ).save_pretrained(out)
    cfg = transformers.T5Config(
        vocab_size=len(vocab), d_model=768, d_kv=16, d_ff=32, num_layers=1, num_heads=2, decoder_start_token_id=0
    )
    transformers.T5EncoderModel(cfg).save_pretrained(out)
    return out


def _tiny_vae(root: Path) -> Path:
    out = root / "tiny_vae"
    diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=64,
    ).save_pretrained(out)
    return out


def test_train_tiny_dit_then_sample(tmp_path, monkeypatch):
    from config.train_config import TrainConfig, get_dit_build_kwargs
    from diffusion import create_diffusion
    from models import DiT_models_text
    from models.dit_text import DiT_Text
    from training.loop import (
        diffusion_step_loss,
        encode_images_vae,
        get_t5_and_vae,
        step_timesteps,
        training_noise,
    )
    from utils.modeling.encoders import apply_latent_spec, encode_text

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setitem(DiT_models_text, TINY, partial(DiT_Text, depth=2, hidden_size=64, patch_size=2, num_heads=2))
    torch.manual_seed(0)
    device = torch.device("cpu")

    cfg = TrainConfig(
        model_name=TINY,
        text_encoder=str(_tiny_text_encoder(tmp_path)),
        vae_model=str(_tiny_vae(tmp_path)),
        image_size=64,
        results_dir=str(tmp_path / "results"),
    )
    tokenizer, text_encoder, vae = get_t5_and_vae(device, cfg)
    apply_latent_spec(cfg, vae)
    model = DiT_models_text[cfg.model_name](**get_dit_build_kwargs(cfg, class_dropout_prob=0.0)).to(device)
    diffusion = create_diffusion(
        timestep_respacing=cfg.timestep_respacing,
        num_timesteps=cfg.num_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.rand(4, 3, 64, 64) * 2 - 1
    captions = ["a red cube", "a blue sphere on the table", "a green tree under sky", "photo of a cube"]
    losses = []
    for step in range(3):
        with torch.no_grad():
            latents = encode_images_vae(images, vae, cfg.latent_scale)
            enc = encode_text(captions, tokenizer, text_encoder, device, max_length=16)
        t = step_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], device, global_step=step)
        loss = diffusion_step_loss(
            cfg,
            diffusion,
            model,
            latents,
            t,
            {"encoder_hidden_states": enc},
            None,
            training_noise(cfg, latents),
            refinement_prob=0.0,
            refinement_max_t=0,
        ).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))

    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "steps": 3, "config": cfg}, ckpt)

    from PIL import Image
    from utils.generation.sample_run import main as sample_main

    out = tmp_path / "out.png"
    sample_main(
        [
            "--ckpt",
            str(ckpt),
            "--prompt",
            "a red cube on the table",
            "--out",
            str(out),
            "--device",
            "cpu",
            "--width",
            "64",
            "--height",
            "64",
            "--steps",
            "2",
            "--num",
            "1",
            "--seed",
            "1",
            "--no-quality-defaults",
            "--pick-best",
            "none",
        ]
    )
    assert out.is_file()
    with Image.open(out) as img:
        img.verify()
    with Image.open(out) as img:
        assert img.size[0] > 0 and img.size[1] > 0
