#!/usr/bin/env python3
"""
**Flow-GRPO** online alignment: sample → reward → group-relative weighted fine-tune.

Rollouts use Flow-GRPO's SDE (flow checkpoints) or ancestral DDIM (VP), rewards come from
``CompositeRanker('online').score_one``, and the update is reward-weighted regression of the checkpoint's own
denoising objective (velocity for flow, eps/v/x0 for VP) over each rollout group, plus a KL-style anchor to the
reference model. Practical single-GPU scaffold inspired by Flow-GRPO (NeurIPS 2025).

Example::

    python -m scripts.tools train_flow_grpo \\
        --ckpt results/best.pt \\
        --prompts prompts.txt \\
        --out results/grpo_policy.pt \\
        --steps 200 --num-samples 4
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
from diffusion import create_diffusion
from models import DiT_models_text
from utils.checkpoint.checkpoint_loading import finetuned_checkpoint, load_dit_text_checkpoint
from utils.modeling.multi_encoder_encode import encode_kwargs_for_captions, load_text_bundle_for_training
from utils.safety.minor_guard import reasons
from utils.superior.inference_pipeline import CompositeRanker
from utils.training.flow_grpo import (
    FlowGRPOConfig,
    SRPOConfig,
    decode_latent_to_rgb_uint8,
    group_relative_advantages,
    grpo_weighted_loss,
    reference_kl_penalty,
    rollout_flow_sample,
    semantic_relative_reward,
    timestep_reward_mask,
)
from utils.training.grpo import (
    BranchGRPOConfig,
    DenseGRPOConfig,
    GRPOGuardConfig,
    TurningPointGRPOConfig,
    branch_relative_advantages,
    branch_rollout_flow_samples,
    enumerate_branch_paths,
    flow_fraction_from_t_index,
    grpo_guard_weighted_loss,
    iso_temporal_group_advantages,
    rectify_policy_gradient,
    sample_iso_temporal_index,
    score_latent_reward,
    short_ode_refine_x0,
    step_advantages_from_gains,
    tp_grpo_step_weights,
)

import train as train_mod


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
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
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    policy, cfg, rae_bridge, _, fusion_sd = load_dit_text_checkpoint(args.ckpt, device=device, reject_enhanced=True)
    name = getattr(cfg, "model_name", "DiT-XL/2-Text")
    fn = DiT_models_text.get(name) or DiT_models_text["DiT-XL/2-Text"]
    ref = fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    ref.load_state_dict(policy.state_dict())
    ref = ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    policy.train()
    for p in policy.parameters():
        p.requires_grad = True

    tokenizer, text_encoder, vae = train_mod.get_t5_and_vae(device, cfg)
    text_bundle = load_text_bundle_for_training(cfg, torch.device(device), fusion_sd)
    diffusion = create_diffusion(
        timestep_respacing="",
        num_timesteps=int(getattr(cfg, "num_timesteps", 1000)),
        beta_schedule=str(getattr(cfg, "beta_schedule", "linear")),
        prediction_type=str(getattr(cfg, "prediction_type", "epsilon")),
    )
    diffusion._to_device(torch.device(device))
    is_flow = bool(getattr(cfg, "flow_matching_training", False))
    num_t = int(diffusion.num_timesteps)

    def _noisy_and_target(z: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """``x_t`` and the regression target of the objective this checkpoint was trained with."""
        if is_flow:
            s = flow_fraction_from_t_index(t, num_t).view(-1, *([1] * (z.dim() - 1)))
            return (1.0 - s) * z + s * noise, noise - z
        return diffusion.q_sample(z, t, noise=noise), diffusion.training_target(z, noise, t)

    ae_type = getattr(cfg, "autoencoder_type", "kl")
    latent_scale = float(getattr(cfg, "latent_scale", 0.18215)) if ae_type == "kl" else 1.0
    image_size = int(getattr(cfg, "image_size", 256))
    latent_h = image_size // 8
    in_ch = int(getattr(cfg, "in_channels", 4))
    shape = (1, in_ch, latent_h, latent_h)

    ranker = CompositeRanker("online")

    srpo_cfg = SRPOConfig()

    def _score(rgb, text: str) -> float:
        return ranker.score_one(rgb, prompt=text, device=str(device), vit_ckpt=str(args.vit_ckpt or ""))

    def reward(rgb, prompt: str) -> float:
        r = _score(rgb, prompt)
        if args.srpo:
            r += float(args.srpo_weight) * semantic_relative_reward(_score, rgb, prompt, config=srpo_cfg)
        return r

    grpo_cfg = FlowGRPOConfig(
        num_samples=int(args.num_samples),
        denoise_steps=int(args.rollout_steps),
        kl_coef=float(args.kl_coef),
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=0.0)
    use_amp = device == "cuda" and not args.no_amp
    prompts = _load_prompts(args.prompts)

    for global_step in range(1, int(args.steps) + 1):
        prompt = prompts[(global_step - 1) % len(prompts)]

        def _enc(text: str) -> torch.Tensor:
            return train_mod.encode_text(
                [text],
                tokenizer,
                text_encoder,
                device,
                max_length=300,
                dtype=torch.float32,  # rollouts run the fp32 policy outside autocast
                text_bundle=text_bundle,
                train_fusion=False,
                **encode_kwargs_for_captions([text], text_bundle),
            )

        enc = _enc(prompt)
        mk_c = {"encoder_hidden_states": enc}
        # The uncond the model was trained on with caption dropout (and that sample.py uses): the empty prompt.
        mk_u = {"encoder_hidden_states": _enc("")}

        rewards: list[float] = []
        latents: list[torch.Tensor] = []

        def _one_rollout() -> torch.Tensor:
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

        if bool(getattr(args, "branch_grpo", False)):
            with torch.no_grad():
                branch_latents = branch_rollout_flow_samples(
                    _one_rollout,
                    num_paths=grpo_cfg.num_samples,
                    steps=grpo_cfg.denoise_steps,
                    branch_factor=int(args.branch_factor),
                    base_seed=int(global_step),
                )
            for z in branch_latents:
                rgb = decode_latent_to_rgb_uint8(
                    z, vae, latent_scale=latent_scale, ae_type=ae_type, rae_bridge=rae_bridge
                )
                rewards.append(reward(rgb, prompt))
                latents.append(z.detach())
        else:
            for _ in range(grpo_cfg.num_samples):
                with torch.no_grad():
                    z = _one_rollout()
                rgb = decode_latent_to_rgb_uint8(
                    z, vae, latent_scale=latent_scale, ae_type=ae_type, rae_bridge=rae_bridge
                )
                rewards.append(reward(rgb, prompt))
                latents.append(z.detach())

        reward_t = torch.tensor(rewards, device=device)
        adv = group_relative_advantages(reward_t)
        branch_cfg = BranchGRPOConfig(branch_factor=int(args.branch_factor))
        shared_t: torch.Tensor | None = None
        if bool(getattr(args, "flash_grpo", False)):
            shared_t = sample_iso_temporal_index(diffusion.num_timesteps, device=device)
            t_idx = torch.full((len(rewards),), int(shared_t.item()), device=device, dtype=torch.long)
            adv = iso_temporal_group_advantages(reward_t, t_idx, num_timesteps=diffusion.num_timesteps)
        if bool(getattr(args, "branch_grpo", False)):
            paths = enumerate_branch_paths(
                grpo_cfg.denoise_steps,
                branch_factor=branch_cfg.branch_factor,
                split_fractions=branch_cfg.split_step_fractions,
            )
            if len(paths) > len(rewards):
                rewards = rewards * ((len(paths) + len(rewards) - 1) // len(rewards))
                rewards = rewards[: len(paths)]
                reward_t = torch.tensor(rewards, device=device)
            adv = branch_relative_advantages(reward_t, [str(i) for i in range(len(rewards))])
        dense_cfg = DenseGRPOConfig(ode_refine_steps=int(args.dense_ode_steps))
        if bool(getattr(args, "tp_grpo", False)) and bool(getattr(args, "dense_grpo", False)):
            step_tables: list[list[float]] = []
            for z in latents:
                t_d = torch.randint(0, num_t, (1,), device=device, dtype=torch.long)
                x_t_d, _ = _noisy_and_target(z, t_d, torch.randn_like(z))
                row: list[float] = []
                with torch.no_grad():
                    for _ in range(max(1, dense_cfg.ode_refine_steps)):
                        x0_hat = short_ode_refine_x0(
                            policy,
                            diffusion,
                            x_t_d,
                            t_d,
                            model_kwargs=mk_c,
                            steps=1,
                            flow=is_flow,
                        )
                        row.append(
                            score_latent_reward(
                                x0_hat,
                                prompt,
                                vae=vae,
                                score_fn=reward,
                                latent_scale=latent_scale,
                                ae_type=ae_type,
                                rae_bridge=rae_bridge,
                            )
                        )
                step_tables.append(row if row else [float(rewards[len(step_tables)])])
            # One advantage per rollout: its TP-GRPO return (incremental + turning-point step rewards).
            returns = [
                sum(tp_grpo_step_weights(traj, float(term), config=TurningPointGRPOConfig()))
                for traj, term in zip(step_tables, rewards)
            ]
            adv = group_relative_advantages(torch.tensor(returns, device=device))
        elif bool(getattr(args, "dense_grpo", False)):
            gains: list[float] = []
            for z in latents:
                t_d = torch.randint(0, num_t, (1,), device=device, dtype=torch.long)
                x_t_d, _ = _noisy_and_target(z, t_d, torch.randn_like(z))
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
                    gains.append(
                        score_latent_reward(
                            x0_hat,
                            prompt,
                            vae=vae,
                            score_fn=reward,
                            latent_scale=latent_scale,
                            ae_type=ae_type,
                            rae_bridge=rae_bridge,
                        )
                    )
            adv = step_advantages_from_gains(gains).to(device)
        # One forward over the whole group: the advantage weights are normalised across it.
        z = torch.cat(latents, dim=0)
        g = z.shape[0]
        adv = adv.reshape(-1)[:g].to(device)
        if shared_t is not None:
            t = shared_t.reshape(1).expand(g).to(device)
        else:
            t = torch.randint(0, num_t, (g,), device=device, dtype=torch.long)
        x_t, target = _noisy_and_target(z, t, torch.randn_like(z))
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
            if bool(getattr(args, "flash_grpo", False)):
                per = rectify_policy_gradient(per, t_frac)
            if bool(getattr(args, "grpo_guard", False)):
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


if __name__ == "__main__":
    raise SystemExit(main())
