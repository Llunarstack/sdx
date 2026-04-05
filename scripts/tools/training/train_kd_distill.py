#!/usr/bin/env python3
"""
Teacher–student **output matching** (MSE on DiT prediction at shared ``x_t``, ``t``).

Loads a checkpoint as the frozen teacher, builds a student with the same architecture (weights
copied from teacher by default), and minimizes ``MSE(student(x_t,t), teacher(x_t,t))`` with
standard VP ``q_sample`` noise. Useful for slight smoothing / same-arch distillation experiments.

Example::

    python scripts/tools/training/train_kd_distill.py \\
        --teacher-ckpt results/best.pt \\
        --data data/manifest.jsonl \\
        --out results/kd_student.pt \\
        --steps 2000 --batch-size 4
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
from utils.modeling.text_encoder_bundle import load_text_encoder_bundle

import train as train_mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-ckpt", type=str, required=True)
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Training image root (subdir layout) or a manifest .jsonl (see Text2ImageDataset).",
    )
    ap.add_argument("--out", type=str, default="kd_student.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-caption-length", type=int, default=300)
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="Random student init instead of copying teacher weights.",
    )
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        device = "cpu"

    teacher, cfg, rae_bridge, _, _ = load_dit_text_checkpoint(
        args.teacher_ckpt, device=device, reject_enhanced=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    name = getattr(cfg, "model_name", "DiT-XL/2-Text")
    fn = DiT_models_text.get(name) or DiT_models_text["DiT-XL/2-Text"]
    student = fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    if not args.from_scratch:
        student.load_state_dict(teacher.state_dict())
    student = student.to(device)
    student.train()

    tokenizer, text_encoder, vae = train_mod.get_t5_and_vae(device, cfg)
    text_bundle = load_text_encoder_bundle(cfg, torch.device(device))

    image_size = int(getattr(cfg, "image_size", 256))
    ds = Text2ImageDataset(
        str(Path(args.data)),
        image_size=image_size,
        max_caption_length=int(args.max_caption_length),
        crop_mode="center",
    )
    if len(ds) == 0:
        print("No samples found in --data.", file=sys.stderr)
        return 1

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_t2i,
    )

    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=int(getattr(cfg, "num_timesteps", 1000)),
        beta_schedule=str(getattr(cfg, "beta_schedule", "linear")),
        prediction_type=str(getattr(cfg, "prediction_type", "epsilon")),
    )
    diffusion._to_device(torch.device(device))

    opt = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=0.01)
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    latent_scale = float(getattr(cfg, "latent_scale", 0.18215)) if ae_type == "kl" else 1.0

    use_amp = device == "cuda" and not args.no_amp
    autocast_device = "cuda" if device == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

    it = iter(dl)
    for step in range(int(args.steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        px = batch["pixel_values"].to(device, non_blocking=True)
        caps = batch["captions"]

        with torch.no_grad():
            latents = train_mod.encode_images_vae(px, vae, latent_scale)
            if rae_bridge is not None and latents.shape[1] != 4:
                latents = rae_bridge.rae_to_dit(latents)

        enc = train_mod.encode_text(
            caps,
            tokenizer,
            text_encoder,
            device,
            max_length=int(args.max_caption_length),
            dtype=torch.bfloat16 if use_amp else torch.float32,
            text_bundle=text_bundle,
            train_fusion=False,
        )
        mk: dict = {"encoder_hidden_states": enc}
        if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
            bsz, lh, lw = latents.shape[0], latents.shape[-2], latents.shape[-1]
            mk["size_embed"] = torch.tensor([[float(lh), float(lw)]], device=device, dtype=torch.float32).expand(
                bsz, -1
            )

        t = torch.randint(0, diffusion.num_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
        noise = torch.randn_like(latents)
        x_t = diffusion.q_sample(latents, t, noise=noise)

        with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            with torch.no_grad():
                tea_o = teacher(x_t, t, **mk)
            stu_o = student(x_t, t, **mk)
            if tea_o.shape != latents.shape and tea_o.shape[1] > latents.shape[1]:
                tea_o = tea_o[:, : latents.shape[1]]
            if stu_o.shape != latents.shape and stu_o.shape[1] > latents.shape[1]:
                stu_o = stu_o[:, : latents.shape[1]]
            loss = torch.nn.functional.mse_loss(stu_o, tea_o)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step} loss {loss.detach().float().item():.6f}", flush=True)

    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "teacher_ckpt": str(args.teacher_ckpt),
            "steps": int(args.steps),
        },
        args.out,
    )
    print(f"Saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
