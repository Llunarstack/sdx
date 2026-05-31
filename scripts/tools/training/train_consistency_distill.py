#!/usr/bin/env python3
"""
**Consistency distillation** (LCM-style) for SDX DiT using ``ConsistencyFlowLoss``.

Distills a student DiT from a frozen teacher so the student matches teacher predictions
across adjacent flow timesteps — enables fewer-step inference.

Example::

    python -m scripts.tools train_consistency_distill \\
        --teacher-ckpt results/best.pt \\
        --data data/manifest.jsonl \\
        --out results/lcm_student.pt \\
        --steps 2000 --consistency-weight 1.0
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
from diffusion.flow_rectified import ConsistencyFlowLoss, LogitNormalTimeSampler
from models import DiT_models_text
from torch.utils.data import DataLoader
from utils.checkpoint.checkpoint_loading import load_dit_text_checkpoint
from utils.modeling.multi_encoder_encode import encode_kwargs_for_captions, load_text_bundle_for_training

import train as train_mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="consistency_student.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--consistency-weight", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--delta-t", type=float, default=0.05)
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

    teacher_ema = fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    teacher_ema.load_state_dict(teacher.state_dict())
    teacher_ema = teacher_ema.to(device)
    teacher_ema.eval()
    for p in teacher_ema.parameters():
        p.requires_grad = False

    cf = ConsistencyFlowLoss(
        teacher_ema,
        student,
        time_sampler=LogitNormalTimeSampler(),
        consistency_weight=float(args.consistency_weight),
        ema_decay=float(args.ema_decay),
    )

    tokenizer, text_encoder, vae = train_mod.get_t5_and_vae(device, cfg)
    text_bundle = load_text_bundle_for_training(cfg, torch.device(device), fusion_sd)
    image_size = int(getattr(cfg, "image_size", 256))
    ds = Text2ImageDataset(str(Path(args.data)), image_size=image_size, max_caption_length=int(args.max_caption_length))
    if len(ds) == 0:
        print("No samples in --data.", file=sys.stderr)
        return 1

    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_t2i)
    opt = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=0.01)
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    latent_scale = float(getattr(cfg, "latent_scale", 0.18215)) if ae_type == "kl" else 1.0
    use_amp = device == "cuda" and not args.no_amp
    autocast_device = "cuda" if device == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

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
        noise = torch.randn_like(px)
        with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            out = cf.compute(px, noise, mk, delta_t=float(args.delta_t))
            loss = out["loss_mean"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        cf.update_teacher_ema()
        if step % 50 == 0:
            print(f"step {step} consistency_loss={float(loss.detach().cpu()):.6f}")

    torch.save({"model": student.state_dict(), "ema": student.state_dict(), "config": cfg}, args.out)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
