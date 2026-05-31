#!/usr/bin/env python3
"""
**LADD-style** latent adversarial distillation (teacher MSE + optional patch discriminator).

Example::

    python -m scripts.tools train_ladd_distill \\
        --teacher-ckpt results/best.pt \\
        --data data/manifest.jsonl \\
        --out results/ladd_student.pt \\
        --steps 1000 --adversarial 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from config.train_config import get_dit_build_kwargs
from data import Text2ImageDataset, collate_t2i
from diffusion import create_diffusion
from models import DiT_models_text
from torch.utils.data import DataLoader
from utils.checkpoint.checkpoint_loading import load_dit_text_checkpoint
from utils.modeling.multi_encoder_encode import encode_kwargs_for_captions, load_text_bundle_for_training
from utils.training.ladd_distillation import (
    LADDConfig,
    LatentPatchDiscriminator,
    ladd_discriminator_step,
    ladd_generator_step,
)

import train as train_mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="ladd_student.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adversarial", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-caption-length", type=int, default=300)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    teacher, cfg, rae_bridge, _, fusion_sd = load_dit_text_checkpoint(
        args.teacher_ckpt, device=device, reject_enhanced=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    name = getattr(cfg, "model_name", "DiT-XL/2-Text")
    fn = DiT_models_text.get(name) or DiT_models_text["DiT-XL/2-Text"]
    student = fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    student.load_state_dict(teacher.state_dict())
    student = student.to(device)
    student.train()

    in_ch = int(getattr(cfg, "in_channels", 4))
    D = LatentPatchDiscriminator(in_ch).to(device)
    ladd_cfg = LADDConfig(mse_teacher=1.0, adversarial=float(args.adversarial))
    opt_g = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=0.01)
    opt_d = torch.optim.AdamW(D.parameters(), lr=float(args.lr) * 2.0, weight_decay=0.0)

    tokenizer, text_encoder, vae = train_mod.get_t5_and_vae(device, cfg)
    text_bundle = load_text_bundle_for_training(cfg, torch.device(device), fusion_sd)
    image_size = int(getattr(cfg, "image_size", 256))
    ds = Text2ImageDataset(str(Path(args.data)), image_size=image_size, max_caption_length=int(args.max_caption_length))
    if len(ds) == 0:
        print("No samples in --data.", file=sys.stderr)
        return 1

    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_t2i)
    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=int(getattr(cfg, "num_timesteps", 1000)),
        beta_schedule=str(getattr(cfg, "beta_schedule", "linear")),
        prediction_type=str(getattr(cfg, "prediction_type", "epsilon")),
    )
    diffusion._to_device(torch.device(device))
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    latent_scale = float(getattr(cfg, "latent_scale", 0.18215)) if ae_type == "kl" else 1.0
    use_amp = device == "cuda" and not args.no_amp

    it = iter(dl)
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        imgs, prompts = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            px = train_mod.encode_images_vae(imgs, vae, latent_scale)
            if rae_bridge is not None:
                px = rae_bridge.rae_to_dit(px)
        enc = train_mod.encode_text(
            prompts,
            tokenizer,
            text_encoder,
            device,
            max_length=int(args.max_caption_length),
            dtype=torch.bfloat16 if use_amp else torch.float32,
            text_bundle=text_bundle,
            train_fusion=False,
            **encode_kwargs_for_captions(prompts, text_bundle),
        )
        mk = {"encoder_hidden_states": enc}
        t = torch.randint(0, diffusion.num_timesteps, (px.shape[0],), device=device, dtype=torch.long)
        noise = torch.randn_like(px)
        x_t = diffusion.q_sample(px, t, noise=noise)

        with torch.no_grad():
            t_out = teacher(x_t, t, **mk)
            if t_out.shape[1] > px.shape[1]:
                t_out = t_out[:, : px.shape[1]]
        s_out = student(x_t, t, **mk)
        if s_out.shape[1] > px.shape[1]:
            s_out = s_out[:, : px.shape[1]]

        if ladd_cfg.adversarial > 0.0:
            ladd_discriminator_step(D, opt_d, px, s_out.detach(), cfg=ladd_cfg)
        stats = ladd_generator_step(
            D,
            student,
            opt_g,
            x_t,
            t,
            teacher,
            cfg=ladd_cfg,
            latent_for_d=s_out if ladd_cfg.adversarial > 0.0 else None,
            **mk,
        )
        if step % 50 == 0:
            print(f"step {step} {stats}")

    torch.save({"model": student.state_dict(), "ema": student.state_dict(), "config": cfg}, args.out)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
