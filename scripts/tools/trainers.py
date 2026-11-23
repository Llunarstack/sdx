#!/usr/bin/env python3
"""Fine-tuning / distillation trainers and training-prep commands.

    python -m scripts.tools trainers dpo --ckpt best.pt --preference-jsonl prefs.jsonl --image-root data/images \\
        --out dpo_policy.pt --steps 500 --batch-size 2 --dpo-beta 300 --dpo-logit-clip 40
    python -m scripts.tools trainers flow-grpo --ckpt best.pt --prompts prompts.txt --out grpo.pt --num-samples 4
    python -m scripts.tools trainers consistency-distill --teacher-ckpt best.pt --data manifest.jsonl --out lcm.pt
    python -m scripts.tools trainers kd-distill --teacher-ckpt best.pt --data manifest.jsonl --out kd.pt
    python -m scripts.tools trainers ladd-distill --teacher-ckpt best.pt --data manifest.jsonl --adversarial 0.1
    python -m scripts.tools trainers reference-adapter --data-dir data/smoke --out reference_adapter.pt --steps 200
    python -m scripts.tools trainers style-lora-oneshot --images style.png --out-dir data/style_tok --trigger styletok
    python -m scripts.tools trainers video --latents-dir latents/ --out results/video_dit/best.pt
    python -m scripts.tools trainers precompute-latents --manifest-jsonl data/train.jsonl --out-dir latent_cache
    python -m scripts.tools trainers expandable-train <train.py args>   # train.py with expandable CUDA segments

Multi-GPU: ``python -m torch.distributed.run --standalone --nproc_per_node=N train.py <args>``.

Old names (``train_diffusion_dpo``, ``train_flow_grpo``, ``train_consistency_distill``, ``train_kd_distill``,
``train_ladd_distill``, ``train_reference_adapter``, ``train_style_lora_oneshot``, ``train_video``,
``precompute_latents``, ``train_with_expandable_segments``) still work as ``python -m scripts.tools <old_name>``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.safety.minor_guard import reasons

_PROG = "python -m scripts.tools trainers"


# --- shared trainer plumbing -------------------------------------------------------------------------------------


def _pick_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        return "cpu"
    return device


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _same_arch(cfg, device: str, state_dict: dict | None = None) -> torch.nn.Module:
    """Fresh DiT built from the checkpoint ``cfg`` (optionally loaded with ``state_dict``) on ``device``."""
    from config.train_config import get_dit_build_kwargs
    from models import DiT_models_text

    fn = DiT_models_text.get(getattr(cfg, "model_name", "DiT-XL/2-Text")) or DiT_models_text["DiT-XL/2-Text"]
    model = fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model.to(device)


def _diffusion_for(cfg, device: str):
    from diffusion import create_diffusion

    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=int(getattr(cfg, "num_timesteps", 1000)),
        beta_schedule=str(getattr(cfg, "beta_schedule", "linear")),
        prediction_type=str(getattr(cfg, "prediction_type", "epsilon")),
    )
    diffusion._to_device(torch.device(device))
    return diffusion


def _latent_scale(cfg) -> tuple[str, float]:
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    return ae_type, float(getattr(cfg, "latent_scale", 0.18215)) if ae_type == "kl" else 1.0


def _forever(loader):
    """Endless batches, reshuffling every pass (unlike ``itertools.cycle``, which replays and caches pass one)."""
    while True:
        yield from loader


def _text_encoder(cfg, device: str, fusion_sd, max_length: int, dtype: torch.dtype):
    """``(tokenizer, text_encoder, vae, encode)`` with ``encode(captions) -> encoder_hidden_states``."""
    import training.loop as train_mod
    from utils.modeling.encoders import encode_kwargs_for_captions, encode_text, load_text_bundle_for_training

    tokenizer, text_encoder, vae = train_mod.get_t5_and_vae(device, cfg)
    bundle = load_text_bundle_for_training(cfg, torch.device(device), fusion_sd)

    def encode(captions: list[str]) -> torch.Tensor:
        return encode_text(
            captions,
            tokenizer,
            text_encoder,
            device,
            max_length=max_length,
            dtype=dtype,
            text_bundle=bundle,
            train_fusion=False,
            **encode_kwargs_for_captions(captions, bundle),
        )

    return vae, encode


def _encode_latents(imgs: torch.Tensor, vae, latent_scale: float, rae_bridge) -> torch.Tensor:
    import training.loop as train_mod

    with torch.no_grad():
        px = train_mod.encode_images_vae(imgs, vae, latent_scale)
        return rae_bridge.rae_to_dit(px) if rae_bridge is not None else px


def _size_embed(cfg, latents: torch.Tensor, device: str) -> dict:
    if int(getattr(cfg, "size_embed_dim", 0) or 0) <= 0:
        return {}
    h, w = latents.shape[-2], latents.shape[-1]
    emb = torch.tensor([[float(h), float(w)]], device=device, dtype=torch.float32)
    return {"size_embed": emb.expand(latents.shape[0], -1)}


def _load_teacher(ckpt: str, device: str):
    from utils.modeling.checkpoints import load_dit_text_checkpoint

    teacher, cfg, rae_bridge, _, fusion_sd = load_dit_text_checkpoint(ckpt, device=device, reject_enhanced=True)
    return _freeze(teacher), cfg, rae_bridge, fusion_sd


def _t2i_loader(cfg, data: str, batch_size: int, max_caption_length: int, **ds_kw):
    from data import Text2ImageDataset, collate_t2i

    image_size = int(getattr(cfg, "image_size", 256))
    ds = Text2ImageDataset(str(Path(data)), image_size=image_size, max_caption_length=max_caption_length, **ds_kw)
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_t2i)


def _amp(device: str, no_amp: bool) -> tuple[bool, str, torch.dtype]:
    use_amp = device == "cuda" and not no_amp
    return use_amp, ("cuda" if device == "cuda" else "cpu"), (torch.bfloat16 if use_amp else torch.float32)


# --- dpo -----------------------------------------------------------------------------------------------------------


def _spectral_sfp_kwargs(cfg) -> dict:
    on = bool(getattr(cfg, "spectral_sfp_loss", False))
    return {
        "use_spectral_sfp_loss": on,
        "spectral_sfp_low_sigma": float(getattr(cfg, "spectral_sfp_low_sigma", 0.22)) if on else 0.22,
        "spectral_sfp_high_sigma": float(getattr(cfg, "spectral_sfp_high_sigma", 0.22)) if on else 0.22,
        "spectral_sfp_tau_power": float(getattr(cfg, "spectral_sfp_tau_power", 1.0)) if on else 1.0,
    }


def _dpo_main(argv: list[str]) -> int:
    """Stage-2 Diffusion-DPO: pairwise preferences (win/lose images + shared prompt).

    The policy is optimised against a frozen reference copy with a DPO term built from per-sample denoising
    losses of the checkpoint's own objective (VP eps/v/x0, or rectified-flow velocity) at a shared ``t`` and
    shared noise (noise offset included) for win/lose and policy/reference (Wallace et al. 2023 coupling).
    Triple/penta text-encoder checkpoints use their fused CLIP + T5 bundle; otherwise T5 only.
    """
    ap = argparse.ArgumentParser(prog=f"{_PROG} dpo", description=_dpo_main.__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="Policy checkpoint with embedded config")
    ap.add_argument("--preference-jsonl", type=str, required=True)
    ap.add_argument("--image-root", type=str, default="", help="Base directory for relative win/lose paths")
    ap.add_argument("--out", type=str, default="dpo_policy.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--dpo-beta", type=float, default=500.0)
    ap.add_argument(
        "--dpo-logit-clip",
        type=float,
        default=40.0,
        help="Clamp |β·Δimplicit_logp| before logsigmoid (0=off). Reduces exploding grads with large beta.",
    )
    ap.add_argument(
        "--sync-ref-every",
        type=int,
        default=0,
        help="If >0: copy policy weights into the frozen reference every n steps (keeps DPO margin meaningful).",
    )
    ap.add_argument(
        "--ref-ema-alpha",
        type=float,
        default=0.0,
        help="If >0: soft EMA update of reference toward policy each step (alternative to --sync-ref-every).",
    )
    ap.add_argument(
        "--timestep-dpo-weight",
        type=str,
        default="high_noise",
        choices=["high_noise", "low_noise", "uniform"],
        help="Reweight DPO loss by diffusion timestep (high_noise recommended for diffusion-DPO).",
    )
    ap.add_argument("--timestep-dpo-power", type=float, default=0.5, help="Exponent for --timestep-dpo-weight curve.")
    ap.add_argument(
        "--safeguarded-dpo",
        type=float,
        default=0.85,
        help="SDPO-style safeguard strength (0=off, 0.85=default). Caps destructive loser pushes.",
    )
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-caption-length", type=int, default=300)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)
    if args.batch_size < 2:
        print("batch-size must be >= 2 (drop_last=True).", file=sys.stderr)
        return 1

    import training.loop as train_mod
    from diffusion.flow_matching import flow_matching_per_sample_losses, shift_timestep_indices
    from diffusion.gaussian_diffusion import offset_noise
    from utils.modeling.checkpoints import finetuned_checkpoint, load_dit_text_checkpoint
    from utils.training.preference import (
        PreferenceImageDataset,
        collate_preference_batch,
        ema_update_reference,
        safeguarded_dpo_preference_loss,
        timestep_dpo_weight,
    )

    device = _pick_device(args.device)
    policy, cfg, rae_bridge, _name, fusion_sd = load_dit_text_checkpoint(args.ckpt, device=device, reject_enhanced=True)
    policy.train()
    for p in policy.parameters():
        p.requires_grad = True
    ref = _freeze(_same_arch(cfg, device, policy.state_dict()))

    use_amp, autocast_device, autocast_dtype = _amp(device, args.no_amp)
    ae, encode = _text_encoder(cfg, device, fusion_sd, int(args.max_caption_length), autocast_dtype)

    ds = PreferenceImageDataset(
        args.preference_jsonl, image_size=int(getattr(cfg, "image_size", 256)), image_root=args.image_root or None
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_preference_batch, drop_last=True
    )
    diffusion = _diffusion_for(cfg, device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    _, latent_scale = _latent_scale(cfg)
    sk = _spectral_sfp_kwargs(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    noise_off = float(getattr(cfg, "noise_offset", 0.0))
    min_snr = float(getattr(cfg, "min_snr_gamma", 5.0))
    lw = str(getattr(cfg, "loss_weighting", "min_snr"))
    lwsd = float(getattr(cfg, "loss_weighting_sigma_data", 0.5))
    is_flow = bool(getattr(cfg, "flow_matching_training", False))
    flow_sk = train_mod.flow_spectral_kwargs(cfg)
    lc = float(args.dpo_logit_clip)
    sg = max(0.0, float(args.safeguarded_dpo))

    def save(step: int) -> None:
        torch.save(finetuned_checkpoint(policy, cfg, rae_bridge, fusion_sd, dpo_step=step), args.out)

    batches = _forever(dl)
    global_step = 0
    while global_step < args.steps:
        batch = next(batches)
        px_w = _encode_latents(batch["win"].to(device), ae, latent_scale, rae_bridge)
        px_l = _encode_latents(batch["lose"].to(device), ae, latent_scale, rae_bridge)
        bsz = px_w.shape[0]
        mk = {"encoder_hidden_states": encode(batch["prompt"]), **_size_embed(cfg, px_w, device)}

        t = torch.randint(0, diffusion.num_timesteps, (bsz,), device=device, dtype=torch.long)
        # One draw (offset included) shared by win/lose and policy/reference: the DPO coupling.
        noise = offset_noise(torch.randn_like(px_w), noise_off)
        if is_flow:
            t = shift_timestep_indices(t, diffusion.num_timesteps, train_mod.flow_shift_for_latents(cfg, px_w))

        def per_sample(model, x0, t=t, noise=noise, mk=mk):
            if is_flow:
                return flow_matching_per_sample_losses(
                    model, x0, noise, diffusion.num_timesteps, mk, t_idx=t, spectral_kwargs=flow_sk
                )
            return diffusion.per_sample_training_losses(
                model,
                x0,
                t,
                model_kwargs=mk,
                noise=noise,
                refinement_prob=0.0,
                refinement_max_t=0,
                min_snr_gamma=min_snr,
                loss_weighting=lw,
                loss_weighting_sigma_data=lwsd,
                **sk,
            )

        with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            loss_w_p = per_sample(policy, px_w)
            loss_l_p = per_sample(policy, px_l)
            with torch.no_grad():
                loss_w_r = per_sample(ref, px_w)
                loss_l_r = per_sample(ref, px_l)

        tw = None
        if str(args.timestep_dpo_weight) != "uniform":
            tw = timestep_dpo_weight(
                t, diffusion.num_timesteps, mode=str(args.timestep_dpo_weight), power=float(args.timestep_dpo_power)
            )
        dpo = safeguarded_dpo_preference_loss(
            -loss_w_p.float(),
            -loss_l_p.float(),
            -loss_w_r.float(),
            -loss_l_r.float(),
            beta=float(args.dpo_beta),
            logit_clip=lc if lc > 0.0 else None,
            safeguard_strength=sg,
            timestep_weights=tw,
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
        if float(args.ref_ema_alpha) > 0.0:
            ema_update_reference(ref, policy, alpha=float(args.ref_ema_alpha))
        elif int(args.sync_ref_every) > 0 and global_step % int(args.sync_ref_every) == 0:
            ref.load_state_dict(policy.state_dict())
            _freeze(ref)
        if global_step % 50 == 0:
            print(f"step {global_step} dpo_loss={float(dpo.detach().cpu()):.6f}")
        if args.save_every > 0 and global_step % args.save_every == 0:
            save(global_step)
            print(f"checkpoint {args.out} (step {global_step})")

    save(global_step)
    print(f"Saved {args.out}")
    return 0


# --- flow-grpo -----------------------------------------------------------------------------------------------------


def _load_prompts(path: str) -> list[str]:
    p = Path(path)
    prompts = [path]
    if p.is_file():
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            prompts = lines
    kept = [prompt for prompt in prompts if not reasons(prompt, training=True)]
    if len(kept) < len(prompts):
        print(f"train_flow_grpo: dropped {len(prompts) - len(kept)} prompt(s) involving minors.", file=sys.stderr)
    if not kept:
        raise ValueError("No GRPO prompts left after the minor-safety filter.")
    return kept


def _flow_grpo_main(argv: list[str]) -> int:
    """Flow-GRPO online alignment: sample -> reward -> group-relative weighted fine-tune.

    Rollouts use Flow-GRPO's SDE (flow checkpoints) or ancestral DDIM (VP); rewards come from
    ``CompositeRanker('online').score_one``; the update is reward-weighted regression of the checkpoint's own
    denoising objective over each rollout group plus a KL-style anchor to the reference model.
    """
    ap = argparse.ArgumentParser(prog=f"{_PROG} flow-grpo", description=_flow_grpo_main.__doc__)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True, help="Prompt string or .txt file (one per line).")
    ap.add_argument("--out", type=str, default="grpo_policy.pt")
    ap.add_argument("--steps", type=int, default=200, help="Training optimizer steps.")
    ap.add_argument("--num-samples", type=int, default=4, help="Rollouts per prompt for GRPO group.")
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--cfg-scale", type=float, default=7.0)
    ap.add_argument("--rollout-steps", type=int, default=8)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--vit-ckpt", type=str, default="")
    ap.add_argument("--dense-grpo", action="store_true", help="Use DenseGRPO step-wise reward gains (2026).")
    ap.add_argument("--dense-ode-steps", type=int, default=4, help="ODE refine steps for dense reward estimation.")
    ap.add_argument(
        "--flash-grpo",
        action="store_true",
        help="Iso-temporal GRPO advantages + temporal gradient rectification (Flash-GRPO 2025).",
    )
    ap.add_argument(
        "--branch-grpo",
        action="store_true",
        help="Branch-relative advantages over multiple rollout paths (BranchGRPO scaffold).",
    )
    ap.add_argument("--branch-factor", type=int, default=2, help="Children per branch split when --branch-grpo.")
    ap.add_argument(
        "--tp-grpo",
        action="store_true",
        help="TurningPoint-GRPO: incremental + turning-point long-term rewards (arXiv:2602.06422).",
    )
    ap.add_argument(
        "--grpo-guard",
        action="store_true",
        help="GRPO-Guard: ratio-norm advantages + timestep gradient reweight (arXiv:2510.22319).",
    )
    ap.add_argument(
        "--srpo",
        action="store_true",
        help="SRPO realism: add a semantic-relative reward (natural vs waxy AI-look control phrases) and mask "
        "updates on late (near-clean) timesteps, where reward tuning over-polishes skin.",
    )
    ap.add_argument("--srpo-weight", type=float, default=1.0, help="Scale of the SRPO relative reward term.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)

    from utils.agentic.quality_loop import CompositeRanker
    from utils.modeling.checkpoints import finetuned_checkpoint, load_dit_text_checkpoint
    from utils.training.preference import (
        BranchGRPOConfig,
        DenseGRPOConfig,
        FlowGRPOConfig,
        GRPOGuardConfig,
        SRPOConfig,
        TurningPointGRPOConfig,
        branch_relative_advantages,
        branch_rollout_flow_samples,
        decode_latent_to_rgb_uint8,
        enumerate_branch_paths,
        flow_fraction_from_t_index,
        group_relative_advantages,
        grpo_guard_weighted_loss,
        grpo_weighted_loss,
        iso_temporal_group_advantages,
        rectify_policy_gradient,
        reference_kl_penalty,
        rollout_flow_sample,
        sample_iso_temporal_index,
        score_latent_reward,
        semantic_relative_reward,
        short_ode_refine_x0,
        step_advantages_from_gains,
        timestep_reward_mask,
        tp_grpo_step_weights,
    )

    device = _pick_device(args.device)
    policy, cfg, rae_bridge, _, fusion_sd = load_dit_text_checkpoint(args.ckpt, device=device, reject_enhanced=True)
    ref = _freeze(_same_arch(cfg, device, policy.state_dict()))
    policy.train()
    for p in policy.parameters():
        p.requires_grad = True

    # Rollouts run the fp32 policy outside autocast.
    vae, encode = _text_encoder(cfg, device, fusion_sd, 300, torch.float32)
    diffusion = _diffusion_for(cfg, device)
    is_flow = bool(getattr(cfg, "flow_matching_training", False))
    num_t = int(diffusion.num_timesteps)

    def noisy_and_target(z: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """``x_t`` and the regression target of the objective this checkpoint was trained with."""
        if is_flow:
            s = flow_fraction_from_t_index(t, num_t).view(-1, *([1] * (z.dim() - 1)))
            return (1.0 - s) * z + s * noise, noise - z
        return diffusion.q_sample(z, t, noise=noise), diffusion.training_target(z, noise, t)

    ae_type, latent_scale = _latent_scale(cfg)
    latent_h = int(getattr(cfg, "image_size", 256)) // 8
    shape = (1, int(getattr(cfg, "in_channels", 4)), latent_h, latent_h)
    ranker = CompositeRanker("online")
    srpo_cfg = SRPOConfig()

    def score(rgb, text: str) -> float:
        return ranker.score_one(rgb, prompt=text, device=str(device), vit_ckpt=str(args.vit_ckpt or ""))

    def reward(rgb, prompt: str) -> float:
        r = score(rgb, prompt)
        if args.srpo:
            r += float(args.srpo_weight) * semantic_relative_reward(score, rgb, prompt, config=srpo_cfg)
        return r

    def latent_reward(x0_hat: torch.Tensor, prompt: str) -> float:
        return score_latent_reward(
            x0_hat,
            prompt,
            vae=vae,
            score_fn=reward,
            latent_scale=latent_scale,
            ae_type=ae_type,
            rae_bridge=rae_bridge,
        )

    grpo_cfg = FlowGRPOConfig(
        num_samples=int(args.num_samples), denoise_steps=int(args.rollout_steps), kl_coef=float(args.kl_coef)
    )
    branch_cfg = BranchGRPOConfig(branch_factor=int(args.branch_factor))
    dense_cfg = DenseGRPOConfig(ode_refine_steps=int(args.dense_ode_steps))
    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=0.0)
    use_amp = device == "cuda" and not args.no_amp
    prompts = _load_prompts(args.prompts)
    # The uncond the model was trained on with caption dropout (and that sample.py uses): the empty prompt.
    mk_u = {"encoder_hidden_states": encode([""])}

    for global_step in range(1, int(args.steps) + 1):
        prompt = prompts[(global_step - 1) % len(prompts)]
        enc = encode([prompt])
        mk_c = {"encoder_hidden_states": enc}

        def one_rollout(mk_c=mk_c) -> torch.Tensor:
            return rollout_flow_sample(
                policy,
                diffusion,
                shape,
                model_kwargs_cond=mk_c,
                model_kwargs_uncond=mk_u,
                cfg_scale=float(args.cfg_scale),
                steps=grpo_cfg.denoise_steps,
                device=device,
                sde_noise_scale=grpo_cfg.sde_noise_scale,
                flow=is_flow,
                flow_shift=float(getattr(cfg, "flow_shift", 1.0) or 1.0),
            )

        def rollouts(one_rollout=one_rollout, global_step=global_step):
            if args.branch_grpo:
                with torch.no_grad():
                    yield from branch_rollout_flow_samples(
                        one_rollout,
                        num_paths=grpo_cfg.num_samples,
                        steps=grpo_cfg.denoise_steps,
                        branch_factor=int(args.branch_factor),
                        base_seed=int(global_step),
                    )
            else:
                for _ in range(grpo_cfg.num_samples):
                    with torch.no_grad():
                        z = one_rollout()
                    yield z

        rewards: list[float] = []
        latents: list[torch.Tensor] = []
        for z in rollouts():
            rgb = decode_latent_to_rgb_uint8(z, vae, latent_scale=latent_scale, ae_type=ae_type, rae_bridge=rae_bridge)
            rewards.append(reward(rgb, prompt))
            latents.append(z.detach())

        reward_t = torch.tensor(rewards, device=device)
        adv = group_relative_advantages(reward_t)
        shared_t: torch.Tensor | None = None
        if args.flash_grpo:
            shared_t = sample_iso_temporal_index(diffusion.num_timesteps, device=device)
            t_idx = torch.full((len(rewards),), int(shared_t.item()), device=device, dtype=torch.long)
            adv = iso_temporal_group_advantages(reward_t, t_idx, num_timesteps=diffusion.num_timesteps)
        if args.branch_grpo:
            paths = enumerate_branch_paths(
                grpo_cfg.denoise_steps,
                branch_factor=branch_cfg.branch_factor,
                split_fractions=branch_cfg.split_step_fractions,
            )
            if len(paths) > len(rewards):
                rewards = (rewards * ((len(paths) + len(rewards) - 1) // len(rewards)))[: len(paths)]
                reward_t = torch.tensor(rewards, device=device)
            adv = branch_relative_advantages(reward_t, [str(i) for i in range(len(rewards))])
        if args.tp_grpo and args.dense_grpo:
            step_tables: list[list[float]] = []
            for z in latents:
                t_d = torch.randint(0, num_t, (1,), device=device, dtype=torch.long)
                x_t_d, _ = noisy_and_target(z, t_d, torch.randn_like(z))
                with torch.no_grad():
                    row = [
                        latent_reward(
                            short_ode_refine_x0(
                                policy, diffusion, x_t_d, t_d, model_kwargs=mk_c, steps=1, flow=is_flow
                            ),
                            prompt,
                        )
                        for _ in range(max(1, dense_cfg.ode_refine_steps))
                    ]
                step_tables.append(row if row else [float(rewards[len(step_tables)])])
            # One advantage per rollout: its TP-GRPO return (incremental + turning-point step rewards).
            returns = [
                sum(tp_grpo_step_weights(traj, float(term), config=TurningPointGRPOConfig()))
                for traj, term in zip(step_tables, rewards, strict=False)
            ]
            adv = group_relative_advantages(torch.tensor(returns, device=device))
        elif args.dense_grpo:
            gains: list[float] = []
            for z in latents:
                t_d = torch.randint(0, num_t, (1,), device=device, dtype=torch.long)
                x_t_d, _ = noisy_and_target(z, t_d, torch.randn_like(z))
                with torch.no_grad():
                    x0_hat = short_ode_refine_x0(
                        policy,
                        diffusion,
                        x_t_d,
                        t_d,
                        model_kwargs=mk_c,
                        steps=dense_cfg.ode_refine_steps,
                        flow=is_flow,
                    )
                    gains.append(latent_reward(x0_hat, prompt))
            adv = step_advantages_from_gains(gains).to(device)
        # One forward over the whole group: the advantage weights are normalised across it.
        z = torch.cat(latents, dim=0)
        g = z.shape[0]
        adv = adv.reshape(-1)[:g].to(device)
        if shared_t is not None:
            t = shared_t.reshape(1).expand(g).to(device)
        else:
            t = torch.randint(0, num_t, (g,), device=device, dtype=torch.long)
        x_t, target = noisy_and_target(z, t, torch.randn_like(z))
        mk_g = {"encoder_hidden_states": enc.expand(g, -1, -1)}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
            pred_p = policy(x_t, t, **mk_g)[:, : z.shape[1]]
            with torch.no_grad():
                pred_r = ref(x_t, t, **mk_g)[:, : z.shape[1]]
            per = (pred_p.float() - target.float()).pow(2).flatten(1).mean(dim=1)
            t_frac = flow_fraction_from_t_index(t, num_t)
            if args.srpo:
                # t_frac is the noise fraction; the mask takes the clean-side trajectory fraction.
                per = per * timestep_reward_mask(1.0 - t_frac, cutoff=srpo_cfg.late_timestep_cutoff)
            if args.flash_grpo:
                per = rectify_policy_gradient(per, t_frac)
            if args.grpo_guard:
                loss = grpo_guard_weighted_loss(per, adv, t_frac, config=GRPOGuardConfig())
            else:
                loss = grpo_weighted_loss(
                    per, adv, clip=grpo_cfg.advantage_clip, temperature=grpo_cfg.reward_temperature
                )
            loss = loss + reference_kl_penalty(pred_p, pred_r, coef=grpo_cfg.kl_coef)
        loss.backward()
        opt.step()
        if global_step % 10 == 0:
            print(f"step {global_step} mean_reward={sum(rewards) / len(rewards):.4f} adv={adv.tolist()}")

    torch.save(finetuned_checkpoint(policy, cfg, rae_bridge, fusion_sd, grpo_step=int(args.steps)), args.out)
    print(f"Saved {args.out}")
    return 0


# --- distillation --------------------------------------------------------------------------------------------------


def _distill_parser(cmd: str, doc: str, out: str, steps: int, batch_size: int) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"{_PROG} {cmd}", description=doc)
    ap.add_argument("--teacher-ckpt", type=str, required=True)
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Training image root (subdir layout) or a manifest .jsonl (see Text2ImageDataset).",
    )
    ap.add_argument("--out", type=str, default=out)
    ap.add_argument("--steps", type=int, default=steps)
    ap.add_argument("--batch-size", type=int, default=batch_size)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-caption-length", type=int, default=300)
    return ap


def _consistency_distill_main(argv: list[str]) -> int:
    """Consistency distillation (LCM-style): the student's velocity matches the EMA teacher's across adjacent flow
    timesteps of a rectified-flow (``--flow-matching-training``) teacher, enabling fewer-step inference."""
    ap = _distill_parser("consistency-distill", _consistency_distill_main.__doc__, "consistency_student.pt", 2000, 4)
    ap.add_argument("--consistency-weight", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--delta-t", type=float, default=0.05)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)

    from diffusion.flow_matching import ConsistencyFlowLoss
    from utils.modeling.checkpoints import finetuned_checkpoint

    device = _pick_device(args.device)
    teacher, cfg, rae_bridge, fusion_sd = _load_teacher(args.teacher_ckpt, device)
    if not bool(getattr(cfg, "flow_matching_training", False)):
        print("Consistency distillation needs a --flow-matching-training teacher checkpoint.", file=sys.stderr)
        return 1
    student = _same_arch(cfg, device, teacher.state_dict()).train()
    teacher_ema = _freeze(_same_arch(cfg, device, teacher.state_dict()))
    cf = ConsistencyFlowLoss(
        teacher_ema,
        student,
        consistency_weight=float(args.consistency_weight),
        ema_decay=float(args.ema_decay),
        num_timesteps=int(getattr(cfg, "num_timesteps", 1000)),
    )

    use_amp, autocast_device, autocast_dtype = _amp(device, args.no_amp)
    vae, encode = _text_encoder(cfg, device, fusion_sd, int(args.max_caption_length), autocast_dtype)
    dl = _t2i_loader(cfg, args.data, int(args.batch_size), int(args.max_caption_length))
    if dl is None:
        print("No samples in --data.", file=sys.stderr)
        return 1
    opt = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=0.01)
    _, latent_scale = _latent_scale(cfg)

    batches = _forever(dl)
    for step in range(1, int(args.steps) + 1):
        batch = next(batches)
        px = _encode_latents(batch["pixel_values"].to(device), vae, latent_scale, rae_bridge)
        mk = {"encoder_hidden_states": encode(batch["captions"])}
        with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            loss = cf.compute(px, torch.randn_like(px), mk, delta_t=float(args.delta_t))["loss_mean"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        cf.update_teacher_ema()
        if step % 50 == 0:
            print(f"step {step} consistency_loss={float(loss.detach().cpu()):.6f}")

    torch.save(finetuned_checkpoint(student, cfg, rae_bridge, fusion_sd), args.out)
    print(f"Saved {args.out}")
    return 0


def _kd_distill_main(argv: list[str]) -> int:
    """Teacher-student output matching: MSE(student(x_t, t), teacher(x_t, t)) at shared VP ``q_sample`` inputs.
    The student copies the teacher's weights unless --from-scratch (same-arch smoothing / distillation)."""
    ap = _distill_parser("kd-distill", _kd_distill_main.__doc__, "kd_student.pt", 2000, 4)
    ap.add_argument(
        "--from-scratch", action="store_true", help="Random student init instead of copying teacher weights."
    )
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)

    device = _pick_device(args.device)
    teacher, cfg, rae_bridge, fusion_sd = _load_teacher(args.teacher_ckpt, device)
    student = _same_arch(cfg, device, None if args.from_scratch else teacher.state_dict()).train()

    use_amp, autocast_device, autocast_dtype = _amp(device, args.no_amp)
    vae, encode = _text_encoder(cfg, device, fusion_sd, int(args.max_caption_length), autocast_dtype)
    dl = _t2i_loader(cfg, args.data, int(args.batch_size), int(args.max_caption_length), crop_mode="center")
    if dl is None:
        print("No samples found in --data.", file=sys.stderr)
        return 1
    diffusion = _diffusion_for(cfg, device)
    opt = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=0.01)
    _, latent_scale = _latent_scale(cfg)

    import training.loop as train_mod

    batches = _forever(dl)
    for step in range(int(args.steps)):
        batch = next(batches)
        with torch.no_grad():
            latents = train_mod.encode_images_vae(
                batch["pixel_values"].to(device, non_blocking=True), vae, latent_scale
            )
            if rae_bridge is not None and latents.shape[1] != 4:
                latents = rae_bridge.rae_to_dit(latents)
        mk = {"encoder_hidden_states": encode(batch["captions"]), **_size_embed(cfg, latents, device)}
        t = torch.randint(0, diffusion.num_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
        x_t = diffusion.q_sample(latents, t, noise=torch.randn_like(latents))

        c = latents.shape[1]
        with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            with torch.no_grad():
                tea_o = teacher(x_t, t, **mk)[:, :c]
            loss = F.mse_loss(student(x_t, t, **mk)[:, :c], tea_o)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.detach().float().item():.6f}", flush=True)

    torch.save(
        {"student_state_dict": student.state_dict(), "teacher_ckpt": str(args.teacher_ckpt), "steps": int(args.steps)},
        args.out,
    )
    print(f"Saved {args.out}", flush=True)
    return 0


def _ladd_distill_main(argv: list[str]) -> int:
    """LADD-style latent adversarial distillation: teacher MSE + optional patch discriminator comparing real clean
    latents with the student's one-step clean estimate (VP ``q_sample`` or rectified-flow inputs)."""
    ap = _distill_parser("ladd-distill", _ladd_distill_main.__doc__, "ladd_student.pt", 1000, 2)
    ap.add_argument("--adversarial", type=float, default=0.1)
    args = ap.parse_args(argv)

    from utils.modeling.checkpoints import finetuned_checkpoint
    from utils.training.preference import (
        LADDConfig,
        LatentPatchDiscriminator,
        ladd_discriminator_step,
        ladd_generator_step,
    )

    device = _pick_device(args.device)
    teacher, cfg, rae_bridge, fusion_sd = _load_teacher(args.teacher_ckpt, device)
    student = _same_arch(cfg, device, teacher.state_dict()).train()
    disc = LatentPatchDiscriminator(int(getattr(cfg, "in_channels", 4))).to(device)
    ladd_cfg = LADDConfig(mse_teacher=1.0, adversarial=float(args.adversarial))
    opt_g = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=0.01)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=float(args.lr) * 2.0, weight_decay=0.0)

    vae, encode = _text_encoder(cfg, device, fusion_sd, int(args.max_caption_length), torch.float32)
    dl = _t2i_loader(cfg, args.data, int(args.batch_size), int(args.max_caption_length))
    if dl is None:
        print("No samples in --data.", file=sys.stderr)
        return 1
    diffusion = _diffusion_for(cfg, device)
    is_flow = bool(getattr(cfg, "flow_matching_training", False))
    num_t = int(diffusion.num_timesteps)
    _, latent_scale = _latent_scale(cfg)

    batches = _forever(dl)
    for step in range(1, int(args.steps) + 1):
        batch = next(batches)
        px = _encode_latents(batch["pixel_values"].to(device), vae, latent_scale, rae_bridge)
        mk = {"encoder_hidden_states": encode(batch["captions"])}
        t = torch.randint(0, num_t, (px.shape[0],), device=device, dtype=torch.long)
        noise = torch.randn_like(px)
        if is_flow:
            s = (t.float() / max(num_t - 1, 1)).view(-1, 1, 1, 1)
            x_t = (1.0 - s) * px + s * noise
        else:
            x_t = diffusion.q_sample(px, t, noise=noise)

        x0_fake = None
        if ladd_cfg.adversarial > 0.0:
            s_out = student(x_t, t, **mk)[:, : px.shape[1]]
            x0_fake = x_t - s * s_out if is_flow else diffusion._predict_x0_and_noise(s_out, x_t, t)[0]
            ladd_discriminator_step(disc, opt_d, px, x0_fake.detach(), cfg=ladd_cfg)
        stats = ladd_generator_step(disc, student, opt_g, x_t, t, teacher, cfg=ladd_cfg, latent_for_d=x0_fake, **mk)
        if step % 50 == 0:
            print(f"step {step} {stats}")

    torch.save(finetuned_checkpoint(student, cfg, rae_bridge, fusion_sd), args.out)
    print(f"Saved {args.out}")
    return 0


# --- reference adapter ---------------------------------------------------------------------------------------------


class _PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, str]], size: int = 224):
        self.pairs = pairs
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import numpy as np
        from PIL import Image

        path, caption = self.pairs[idx]
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        return torch.from_numpy(np.asarray(img).astype("float32") / 255.0).permute(2, 0, 1), caption


def _collect_pairs(data_dir: Path | None, jsonl: Path | None) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    if jsonl and jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            p = Path(str(row.get("image_path") or row.get("image") or ""))
            cap = str(row.get("caption") or row.get("prompt") or "").strip()
            if p.is_file() and cap:
                pairs.append((p, cap))
    if data_dir and data_dir.is_dir():
        for img in sorted(data_dir.rglob("*")):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            txt = img.with_suffix(".txt")
            if not txt.is_file():
                continue
            cap = txt.read_text(encoding="utf-8").strip()
            if cap:
                pairs.append((img, cap))
    return [(img, cap) for img, cap in pairs if not reasons(cap, training=True)]


_CLIP_CACHE: dict = {}


def _clip(device: torch.device):
    """``(model, processor)`` for openai/clip-vit-base-patch32, or ``None`` when unavailable; loaded once."""
    if device not in _CLIP_CACHE:
        try:
            from transformers import CLIPModel, CLIPProcessor

            name = "openai/clip-vit-base-patch32"
            _CLIP_CACHE[device] = (
                CLIPModel.from_pretrained(name).to(device).eval(),
                CLIPProcessor.from_pretrained(name),
            )
        except Exception:
            _CLIP_CACHE[device] = None
    return _CLIP_CACHE[device]


@torch.no_grad()
def _clip_embed(images: torch.Tensor, captions: list[str], device: torch.device, dim: int):
    """``(image_embeds, text_embeds, dim)``; falls back to a deterministic pooled stub when CLIP is unavailable."""
    clip = _clip(device)
    if clip is not None:
        import numpy as np
        from PIL import Image

        model, proc = clip
        pil = [Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for im in images]
        inputs = proc(text=captions, images=pil, return_tensors="pt", padding=True, truncation=True)
        out = model(**{k: v.to(device) for k, v in inputs.items()})
        return out.image_embeds, out.text_embeds, out.image_embeds.shape[-1]
    x = images.to(device)
    rgb = x.mean(dim=(2, 3)).mean(dim=1, keepdim=True)
    ramp = torch.linspace(0, 1, dim, device=device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)
    img_e = torch.tanh(ramp * 3.0 + rgb * 2.0)
    rows = []
    for c in captions:
        g = torch.Generator(device=device)
        g.manual_seed(zlib.crc32(c.encode("utf-8")) % (2**31 - 1))
        rows.append(torch.randn(dim, generator=g, device=device))
    return F.normalize(img_e, dim=-1), F.normalize(torch.stack(rows, dim=0), dim=-1), dim


def _reference_adapter_main(argv: list[str]) -> int:
    """Train ``ReferenceTokenProjector`` (IP-Adapter-style tokens) on (image, caption) pairs; only the small
    projector is optimised. The .pt works with ``sample.py --reference-adapter-pt``. Input: image files with
    matching ``.txt`` captions and/or a JSONL with ``image_path`` / ``caption`` keys."""
    ap = argparse.ArgumentParser(prog=f"{_PROG} reference-adapter", description=_reference_adapter_main.__doc__)
    ap.add_argument("--data-dir", type=str, default="", help="Folder of image+txt pairs")
    ap.add_argument("--jsonl", type=str, default="", help="JSONL with image_path/caption")
    ap.add_argument("--out", type=str, required=True, help="Output .pt path")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-dim", type=int, default=512)
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-tokens", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args(argv)

    pairs = _collect_pairs(Path(args.data_dir) if args.data_dir else None, Path(args.jsonl) if args.jsonl else None)
    if len(pairs) < 2:
        print("Need at least 2 (image, caption) pairs.", file=sys.stderr)
        return 2

    from models.conditioning import ReferenceTokenProjector

    device = torch.device(args.device)
    loader = DataLoader(_PairDataset(pairs), batch_size=min(args.batch_size, len(pairs)), shuffle=True)
    sample_imgs, sample_caps = next(iter(loader))
    _, _, clip_dim = _clip_embed(sample_imgs, list(sample_caps), device, args.clip_dim)

    proj = ReferenceTokenProjector(clip_dim, args.hidden_size, num_tokens=args.num_tokens).to(device)
    # Caption side gets its own projector into the same token space (contrastive pairing).
    text_proj = ReferenceTokenProjector(clip_dim, args.hidden_size, num_tokens=args.num_tokens).to(device)
    opt = torch.optim.AdamW(list(proj.parameters()) + list(text_proj.parameters()), lr=args.lr)

    batches = _forever(loader)
    for step in range(1, args.steps + 1):
        imgs, caps = next(batches)
        img_e, txt_e, _ = _clip_embed(imgs.to(device), list(caps), device, clip_dim)
        # InfoNCE on mean-pooled tokens
        a = F.normalize(proj(img_e).mean(dim=1), dim=-1)
        b = F.normalize(text_proj(txt_e).mean(dim=1), dim=-1)
        logits = a @ b.T / 0.07
        labels = torch.arange(a.shape[0], device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == 1:
            print(f"step={step}/{args.steps} loss={float(loss):.4f}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": proj.state_dict(),
            "clip_dim": int(clip_dim),
            "hidden_size": int(args.hidden_size),
            "num_tokens": int(args.num_tokens),
        },
        out,
    )
    print(f"Wrote {out}", file=sys.stderr)
    return 0


# --- style lora one-shot -------------------------------------------------------------------------------------------


def _style_lora_oneshot_main(argv: list[str]) -> int:
    """One-shot / few-shot style LoRA recipe (B-LoRA last-block targeting): builds a tiny captioned image folder
    from 1-N reference images and prints the recommended ``train.py --lora-train --lora-style-preset`` command."""
    p = argparse.ArgumentParser(prog=f"{_PROG} style-lora-oneshot", description=_style_lora_oneshot_main.__doc__)
    p.add_argument(
        "--images", type=str, nargs="+", required=True, help="One or more style reference images (PNG/JPG/WEBP)."
    )
    p.add_argument("--out-dir", type=str, required=True, help="Output dataset directory (images + .txt captions).")
    p.add_argument(
        "--trigger",
        type=str,
        default="styletok",
        help="Trigger token used in captions (e.g. 'in the style of styletok').",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="How many copies of each image to write (augments effective epoch length).",
    )
    p.add_argument(
        "--class-prompt",
        type=str,
        default="artwork",
        help="Generic class noun for captions (avoids locking the reference subject).",
    )
    p.add_argument("--rank", type=int, default=8, help="Suggested LoRA rank for the printed train command.")
    p.add_argument("--print-only", action="store_true", help="Do not copy images; only print the train recipe.")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    images = [Path(x) for x in args.images]
    for img in images:
        if not img.is_file():
            print(f"Missing image: {img}", file=sys.stderr)
            return 2

    caption = f"in the style of {args.trigger}, {args.class_prompt}"
    if not args.print_only:
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in images:
            for r in range(max(1, int(args.repeats))):
                stem = f"{img.stem}_r{r:03d}"
                shutil.copy2(img, out / f"{stem}{img.suffix.lower()}")
                (out / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
                n += 1
        print(f"Wrote {n} image/caption pairs under {out}")

    rank = max(4, int(args.rank))
    print("\nRecommended train (B-LoRA last-third style blocks):")
    print(
        f"python train.py --data-dir {out.as_posix()} --lora-train --lora-style-preset "
        f"--lora-rank {rank} --lora-alpha {rank} --epochs 1 "
        f"# ~200–400 effective steps depending on batch; use trigger '{args.trigger}' at sample time"
    )
    print("\nSample with style LoRA on last layers only:")
    print(
        f'python sample.py --prompt "a cat, in the style of {args.trigger}" '
        f"--lora path/to/ckpt_lora.pt:1.0:style --lora-layers last"
    )
    print("\nZero-shot alternative (no train): python sample.py --reference-image STYLE.png")
    return 0


# --- video ---------------------------------------------------------------------------------------------------------


def _video_main(argv: list[str]) -> int:
    """Train VideoDiT on short precomputed latent clips ``(C, T, H, W)`` (.pt files) with flow-matching v2.
    Without clips in --latents-dir it runs on a synthetic smoke batch."""
    p = argparse.ArgumentParser(prog=f"{_PROG} video", description=_video_main.__doc__)
    p.add_argument("--latents-dir", type=str, required=True, help="Directory of .pt latent clips (C,T,H,W)")
    p.add_argument("--out", type=str, default="results/video_dit/best.pt")
    p.add_argument("--model", type=str, default="PermanentVideoDiT-S/2")
    p.add_argument("--steps", type=int, default=100, help="Optimizer steps (smoke default)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--context-dim", type=int, default=768)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--permanence-weight", type=float, default=0.1, help="Slot-memory consistency loss weight")
    p.add_argument("--dry-run", action="store_true", help="Build model + one loss step then exit")
    args = p.parse_args(argv)

    from diffusion.flow_matching import flow_matching_per_sample_losses
    from models.video_dit import permanence_consistency_loss
    from pipelines.video.neural_sample import load_video_dit

    device = torch.device(args.device)
    permanent = "Permanent" in str(args.model)
    model = load_video_dit(
        None, model_name=args.model, context_dim=args.context_dim, device=device, permanent=permanent
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    latent_paths = sorted(Path(args.latents_dir).glob("*.pt"))
    if latent_paths:
        clips = [torch.load(p, map_location=device, weights_only=True) for p in latent_paths[: args.batch]]
        x0 = torch.stack([c if c.ndim == 4 else c[0] for c in clips], dim=0).to(device)
    else:
        x0 = torch.randn(args.batch, 4, 8, 16, 16, device=device)
    context = torch.randn(x0.shape[0], args.context_dim, device=device)

    for step in range(1 if args.dry_run else max(1, int(args.steps))):
        opt.zero_grad(set_to_none=True)
        loss = flow_matching_per_sample_losses(
            model, x0, torch.randn_like(x0), 1000, {"context": context}, logit_normal=True, shift=3.0
        ).mean()
        if permanent and float(args.permanence_weight) > 0:
            # Extra pass on clean latents for slot-trajectory smoothness.
            out = model(x0, torch.zeros(x0.shape[0], device=device), context=context, return_slots=True)
            if isinstance(out, tuple):
                loss = loss + float(args.permanence_weight) * permanence_consistency_loss(out[1])
        loss.backward()
        opt.step()
        if step % 10 == 0 or args.dry_run:
            print(f"step={step} loss={float(loss):.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_name": args.model}, out)
    print(f"Wrote {out}")
    return 0


# --- precompute latents --------------------------------------------------------------------------------------------


class _ImagePaths(Dataset):
    def __init__(self, data_path, image_size=256, *, data_root: str | None = None, out_dir: Path | None = None):
        from data.manifest_utils import first_str, iter_manifest_rows

        self.data_path = Path(data_path)
        self.data_root = Path(data_root) if data_root else None
        self.image_size = image_size
        self.out_dir = out_dir
        self.paths: list[str] = []
        if self.data_path.suffix.lower() == ".jsonl":
            rows = iter_manifest_rows(self.data_path)
            self.paths = [p for p in (first_str(r, ("image_path", "path", "image")) for r in rows) if p]
        else:
            for subdir in self.data_path.iterdir():
                if not subdir.is_dir():
                    continue
                images_dir = subdir / "images"
                scan = images_dir if images_dir.is_dir() else subdir
                self.paths += [str(p) for p in scan.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import numpy as np
        from data.manifest_utils import resolve_image_path
        from data.t2i_dataset import center_crop
        from PIL import Image

        rel = self.paths[idx]
        manifest_dir = self.data_path.parent if self.data_path.suffix.lower() == ".jsonl" else None
        resolved = (resolve_image_path(rel, self.data_root, manifest_dir) or Path(rel)).resolve()
        if self.out_dir is not None and (self.out_dir / (resolved.stem + ".pt")).is_file():
            return None, str(resolved)
        pil = center_crop(Image.open(resolved).convert("RGB"), self.image_size)
        img = (np.array(pil).astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(img).permute(2, 0, 1), str(resolved)


def _collate_skip(batch):
    """``(stacked images or None, their paths, number of already-cached items)``."""
    kept = [(img, path) for img, path in batch if img is not None]
    imgs = torch.stack([img for img, _ in kept]) if kept else None
    return imgs, [path for _, path in kept], len(batch) - len(kept)


def _precompute_latents_main(argv: list[str]) -> int:
    """Precompute VAE latents for a dataset; then train with --latent-cache-dir for faster training."""
    parser = argparse.ArgumentParser(prog=f"{_PROG} precompute-latents", description=_precompute_latents_main.__doc__)
    parser.add_argument("--data-path", type=str, default=None, help="Image folder or legacy alias.")
    parser.add_argument("--manifest-jsonl", type=str, default=None, help="Training manifest (preferred).")
    parser.add_argument(
        "--data-root", type=str, default=None, help="Resolve manifest image_path (e.g. /workspace/data)."
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Latent cache directory (e.g. latent_cache)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--vae",
        type=str,
        default="REPA-E/e2e-flux-vae",
        help="VAE or RAE model id/path (uses --autoencoder-type); match train.py --vae-model",
    )
    parser.add_argument("--vae-subfolder", type=str, default="", help="e.g. 'vae' for a full pipeline repo")
    parser.add_argument("--scale", type=float, default=None, help="Override the VAE config's scaling_factor")
    parser.add_argument(
        "--autoencoder-type",
        type=str,
        default="kl",
        choices=["kl", "rae"],
        help="kl=AutoencoderKL, rae=AutoencoderRAE",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers; -1 = min(16, CPU count). Use 0 on Windows if workers hang.",
    )
    args = parser.parse_args(argv)
    manifest = args.manifest_jsonl or args.data_path
    if not manifest:
        parser.error("Provide --manifest-jsonl or --data-path")

    import training.loop as train_mod
    from utils.modeling.encoders import get_autoencoder_class, latent_spec, load_autoencoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = args.num_workers if args.num_workers >= 0 else min(16, os.cpu_count() or 1)

    spec_out: dict = {"vae": args.vae, "subfolder": args.vae_subfolder, "autoencoder_type": args.autoencoder_type}
    if args.autoencoder_type == "rae":
        vae = get_autoencoder_class("rae").from_pretrained(args.vae).to(device).eval()
        latent_scale = 1.0
        ch = getattr(getattr(vae, "config", None), "encoder_hidden_size", None)
        if ch is not None and int(ch) != 4:
            print(
                f"Warning: RAE latents have {ch} channels, but this repo's DiT expects 4-channel SD latents. "
                "Precomputing may not be usable until the DiT/diffusion latent dimensions are updated.",
                file=sys.stderr,
            )
    else:
        vae = load_autoencoder("kl", args.vae, subfolder=args.vae_subfolder, device=device)
        spec = latent_spec(vae)
        latent_scale = spec.scale if args.scale is None else float(args.scale)
        spec_out.update(channels=spec.channels, scale=latent_scale, shift=spec.shift)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # train.py compares this with its own VAE so a cache from another latent space fails loudly.
    (out_dir / "latent_spec.json").write_text(json.dumps(spec_out, indent=2), encoding="utf-8")
    dataset = _ImagePaths(manifest, args.image_size, data_root=args.data_root, out_dir=out_dir)

    pin = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        collate_fn=_collate_skip,
    )
    processed = skipped = 0
    for i, (imgs, paths, n_cached) in enumerate(loader):
        skipped += n_cached
        if imgs is None:
            continue
        with torch.inference_mode():
            latents = train_mod.encode_images_vae(imgs.to(device, non_blocking=pin), vae, latent_scale).cpu()
        for lat, p in zip(latents, paths, strict=True):
            torch.save(lat.clone(), out_dir / (Path(p).stem + ".pt"))
        processed += len(paths)
        if (i + 1) % 100 == 0:
            print(f"Encoded {processed} new, skipped {skipped} cached / {len(dataset)} total")
    print(f"Done. Encoded {processed} new ({skipped} already cached). Train with: --latent-cache-dir {out_dir}")
    return 0


# --- train.py wrapper ----------------------------------------------------------------------------------------------


def _expandable_train_main(argv: list[str]) -> int:
    """Run ``train.py <args>`` with CUDA allocator expandable segments (reduces fragmentation OOM).
    The allocator env var must be set before torch loads, hence a child process."""
    if argv[:1] in (["-h"], ["--help"]):
        print(f"usage: {_PROG} expandable-train <train.py args>\n\n{_expandable_train_main.__doc__}")
        return 0
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    return subprocess.call([sys.executable, "train.py", *argv], cwd=_REPO, env=env)


COMMANDS = {
    "dpo": _dpo_main,
    "flow-grpo": _flow_grpo_main,
    "consistency-distill": _consistency_distill_main,
    "kd-distill": _kd_distill_main,
    "ladd-distill": _ladd_distill_main,
    "reference-adapter": _reference_adapter_main,
    "style-lora-oneshot": _style_lora_oneshot_main,
    "video": _video_main,
    "precompute-latents": _precompute_latents_main,
    "expandable-train": _expandable_train_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
