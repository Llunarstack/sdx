#!/usr/bin/env python3
"""
Stage-2 **Diffusion-DPO** training: pairwise preferences (win/lose images + shared prompt).

Loads a DiT-Text checkpoint, freezes a **reference** copy of the weights, and optimizes the
**policy** with a DPO term built from per-sample VP diffusion losses at a **shared** timestep
``t`` and **shared** Gaussian noise ``epsilon`` (Wallace-style coupling).

Uses ``train.encode_text`` with ``load_text_encoder_bundle`` when the checkpoint config has
``text_encoder_mode == "triple"`` (T5 + CLIP-L + bigG + fusion); otherwise T5-only.

Example::

    python scripts/tools/training/train_diffusion_dpo.py \\
        --ckpt results/best.pt \\
        --preference-jsonl data/prefs.jsonl \\
        --image-root data/images \\
        --out results/dpo_policy.pt \\
        --steps 500 --batch-size 2 --dpo-beta 300
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from config.train_config import get_dit_build_kwargs
from diffusion import create_diffusion
from models import DiT_models_text
from utils.checkpoint.checkpoint_loading import load_dit_text_checkpoint
from utils.modeling.text_encoder_bundle import load_text_encoder_bundle
from utils.training.diffusion_dpo_loss import dpo_preference_loss
from utils.training.preference_image_dataset import PreferenceImageDataset, collate_preference_batch

import train as train_mod


def _build_reference(policy: torch.nn.Module, cfg, device: str) -> torch.nn.Module:
    name = getattr(cfg, "model_name", "DiT-XL/2-Text")
    fn = DiT_models_text.get(name) or DiT_models_text["DiT-XL/2-Text"]
    ref = fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    ref.load_state_dict(policy.state_dict())
    ref = ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def _spectral_sfp_kwargs(cfg) -> dict:
    if not bool(getattr(cfg, "spectral_sfp_loss", False)):
        return {
            "use_spectral_sfp_loss": False,
            "spectral_sfp_low_sigma": 0.22,
            "spectral_sfp_high_sigma": 0.22,
            "spectral_sfp_tau_power": 1.0,
        }
    return {
        "use_spectral_sfp_loss": True,
        "spectral_sfp_low_sigma": float(getattr(cfg, "spectral_sfp_low_sigma", 0.22)),
        "spectral_sfp_high_sigma": float(getattr(cfg, "spectral_sfp_high_sigma", 0.22)),
        "spectral_sfp_tau_power": float(getattr(cfg, "spectral_sfp_tau_power", 1.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="Policy checkpoint with embedded config")
    ap.add_argument("--preference-jsonl", type=str, required=True)
    ap.add_argument("--image-root", type=str, default="", help="Base directory for relative win/lose paths")
    ap.add_argument("--out", type=str, default="dpo_policy.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--dpo-beta", type=float, default=500.0)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-caption-length", type=int, default=300)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        device = "cpu"

    policy, cfg, rae_bridge, _name, _fusion = load_dit_text_checkpoint(
        args.ckpt, device=device, reject_enhanced=True
    )
    policy.train()
    for p in policy.parameters():
        p.requires_grad = True

    ref = _build_reference(policy, cfg, device)
    tokenizer, text_encoder, ae = train_mod.get_t5_and_vae(device, cfg)
    text_bundle = load_text_encoder_bundle(cfg, torch.device(device))

    image_size = int(getattr(cfg, "image_size", 256))
    ds = PreferenceImageDataset(
        args.preference_jsonl,
        image_size=image_size,
        image_root=args.image_root or None,
    )
    if args.batch_size < 2:
        print("batch-size must be >= 2 (drop_last=True).", file=sys.stderr)
        return 1
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_preference_batch,
        drop_last=True,
    )

    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=int(getattr(cfg, "num_timesteps", 1000)),
        beta_schedule=str(getattr(cfg, "beta_schedule", "linear")),
        prediction_type=str(getattr(cfg, "prediction_type", "epsilon")),
    )
    diffusion._to_device(torch.device(device))

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    latent_scale = float(getattr(cfg, "latent_scale", 0.18215)) if ae_type == "kl" else 1.0
    sk = _spectral_sfp_kwargs(cfg)

    use_amp = device == "cuda" and not args.no_amp
    autocast_device = "cuda" if device == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    noise_off = float(getattr(cfg, "noise_offset", 0.0))
    min_snr = float(getattr(cfg, "min_snr_gamma", 5.0))
    lw = str(getattr(cfg, "loss_weighting", "min_snr"))
    lwsd = float(getattr(cfg, "loss_weighting_sigma_data", 0.5))

    global_step = 0
    cycler = itertools.cycle(dl)
    while global_step < args.steps:
        batch = next(cycler)
        imgs_w = batch["win"].to(device)
        imgs_l = batch["lose"].to(device)
        prompts = batch["prompt"]
        bsz = imgs_w.shape[0]

        with torch.no_grad():
            px_w = train_mod.encode_images_vae(imgs_w, ae, latent_scale)
            px_l = train_mod.encode_images_vae(imgs_l, ae, latent_scale)
            if rae_bridge is not None:
                px_w = rae_bridge.rae_to_dit(px_w)
                px_l = rae_bridge.rae_to_dit(px_l)

        enc = train_mod.encode_text(
            prompts,
            tokenizer,
            text_encoder,
            device,
            max_length=int(args.max_caption_length),
            dtype=torch.bfloat16 if use_amp else torch.float32,
            text_bundle=text_bundle,
            train_fusion=False,
        )
        mk = {"encoder_hidden_states": enc}
        if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
            lh, lw_ = px_w.shape[-2], px_w.shape[-1]
            mk["size_embed"] = torch.tensor([[float(lh), float(lw_)]], device=device, dtype=torch.float32).expand(
                bsz, -1
            )

        t = torch.randint(0, diffusion.num_timesteps, (bsz,), device=device, dtype=torch.long)
        noise = torch.randn_like(px_w)

        with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            loss_w_p = diffusion.per_sample_training_losses(
                policy,
                px_w,
                t,
                model_kwargs=mk,
                noise=noise,
                refinement_prob=0.0,
                refinement_max_t=0,
                noise_offset=noise_off,
                min_snr_gamma=min_snr,
                loss_weighting=lw,
                loss_weighting_sigma_data=lwsd,
                **sk,
            )
            loss_l_p = diffusion.per_sample_training_losses(
                policy,
                px_l,
                t,
                model_kwargs=mk,
                noise=noise,
                refinement_prob=0.0,
                refinement_max_t=0,
                noise_offset=noise_off,
                min_snr_gamma=min_snr,
                loss_weighting=lw,
                loss_weighting_sigma_data=lwsd,
                **sk,
            )

        with torch.no_grad():
            with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
                loss_w_r = diffusion.per_sample_training_losses(
                    ref,
                    px_w,
                    t,
                    model_kwargs=mk,
                    noise=noise,
                    refinement_prob=0.0,
                    refinement_max_t=0,
                    noise_offset=noise_off,
                    min_snr_gamma=min_snr,
                    loss_weighting=lw,
                    loss_weighting_sigma_data=lwsd,
                    **sk,
                )
                loss_l_r = diffusion.per_sample_training_losses(
                    ref,
                    px_l,
                    t,
                    model_kwargs=mk,
                    noise=noise,
                    refinement_prob=0.0,
                    refinement_max_t=0,
                    noise_offset=noise_off,
                    min_snr_gamma=min_snr,
                    loss_weighting=lw,
                    loss_weighting_sigma_data=lwsd,
                    **sk,
                )

        dpo = dpo_preference_loss(
            -loss_w_p.float(),
            -loss_l_p.float(),
            -loss_w_r.float(),
            -loss_l_r.float(),
            beta=float(args.dpo_beta),
        )

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(dpo).backward()
            scaler.step(opt)
            scaler.update()
        else:
            dpo.backward()
            opt.step()

        global_step += 1
        if global_step % 50 == 0:
            print(f"step {global_step} dpo_loss={float(dpo.detach().cpu()):.6f}")
        if args.save_every > 0 and global_step % args.save_every == 0:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "ema": policy.state_dict(),
                    "config": cfg,
                    "dpo_step": global_step,
                },
                args.out,
            )
            print(f"checkpoint {args.out} (step {global_step})")

    torch.save(
        {
            "model": policy.state_dict(),
            "ema": policy.state_dict(),
            "config": cfg,
            "dpo_step": global_step,
        },
        args.out,
    )
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
