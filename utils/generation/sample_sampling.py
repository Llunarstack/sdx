"""Sampling phase of ``sample.py``: sampler settings, the denoising passes, and latent-space refinement."""

from __future__ import annotations

import sys
from typing import Any

import torch
from diffusion.pixel_perfect import dit_patch_size, dit_rgb_stride_px, snap_to_multiple
from diffusion.sampling import sanitize_holy_grail_kwargs

from utils.generation.sample_helpers import _refine_gate_score, renoise_for_grid

# Model kwargs that carry one row per sample.
_PER_SAMPLE_KWARGS = (
    "encoder_hidden_states",
    "style_embedding",
    "token_weights",
    "creativity",
    "reference_tokens",
    "control_image",
    "control_type",
    "control_scale",
)


def resolve_inference_amp(args, device) -> tuple[bool, torch.dtype | None]:
    """Map ``--inference-amp`` to ``(enabled, autocast dtype)`` for ``sample_loop``."""
    mode = str(getattr(args, "inference_amp", "auto") or "auto").strip().lower()
    if mode == "fp16":
        return True, torch.float16
    if mode == "bf16":
        return True, torch.bfloat16
    if mode == "auto" and getattr(device, "type", None) == "cuda":
        return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return False, None


def _with_batch(kw: dict, n: int) -> dict:
    """Repeat batch-1 per-sample kwargs to ``n`` rows (beam branches share one prompt)."""
    return {
        k: v.expand(n, *v.shape[1:]) if k in _PER_SAMPLE_KWARGS and torch.is_tensor(v) and v.shape[0] == 1 else v
        for k, v in kw.items()
    }


def _f(args: Any, name: str, default: float) -> float:
    """Float knob; an explicit 0 is honoured (only a missing / None value takes the default)."""
    value = getattr(args, name, None)
    return float(default if value is None else value)


class Sampler:
    """One run's sampler settings; every pass goes through ``run`` so they share the same guidance stack."""

    def __init__(self, args: Any, m: Any, plan: Any, cond: Any, init: Any, feature_kw: dict):
        self.args, self.m, self.plan, self.c, self.init = args, m, plan, cond, init
        self.feature_kw = feature_kw
        cfg_scale = getattr(args, "cfg_scale", 7.5)
        cfg_rescale = getattr(args, "cfg_rescale", 0.0)
        dyn_p = getattr(args, "dynamic_threshold_percentile", 0.0)
        if getattr(args, "originality", 0.0):
            # Lower CFG a little so the model can explore beyond the most literal token match.
            strength = max(0.0, min(1.0, float(args.originality)))
            cfg_scale = max(1.0, cfg_scale - strength * 2.0)
            if cfg_rescale == 0.0 and cfg_scale > 6.0:
                cfg_rescale = 0.6
        # High CFG without rescale oversaturates; Imagen thresholding is not auto-enabled because on VAE latents
        # (|x| well above 1) it rescales the whole latent toward grey.
        if cfg_scale > 10 and cfg_rescale == 0.0 and dyn_p == 0.0:
            cfg_rescale = 0.7
            print(f"High CFG ({cfg_scale}): auto-enabled cfg_rescale=0.7 to reduce oversaturation.", file=sys.stderr)
        self.cfg_scale = cfg_scale
        holy = sanitize_holy_grail_kwargs(_holy_grail_kwargs(args))
        if holy["holy_grail_enable"]:
            print("Holy-grail diffusion policy enabled.", file=sys.stderr)
        # --per-region-cads: CADS condition noise keeps a box layout's regions from collapsing into one another early.
        cads_layout = getattr(args, "per_region_cads", False) and getattr(args, "_box_layout_spec", None) is not None
        if cads_layout and holy["holy_grail_cads_strength"] <= 0.0:
            holy["holy_grail_cads_strength"] = 0.15
        schedule = getattr(args, "guidance_schedule", None)
        schedule = schedule.strip() if isinstance(schedule, str) and schedule.strip() else None
        if holy["holy_grail_enable"] and schedule:
            print("--guidance-schedule ignored when holy-grail is enabled.", file=sys.stderr)
            schedule = None
        if init.flow:
            print(
                f"Using flow-matching sampler (solver={getattr(args, 'flow_solver', 'dpmpp_2m') or 'dpmpp_2m'}); "
                "pair with checkpoints trained using --flow-matching-training.",
                file=sys.stderr,
            )
        flow_shift = getattr(args, "flow_shift", None)
        if flow_shift is None:
            flow_shift = getattr(m.cfg, "flow_shift", None) or 1.0
        amp_on, amp_dtype = resolve_inference_amp(args, m.device)
        self.common = dict(
            model_kwargs_cond=cond.cond,
            model_kwargs_uncond=cond.uncond,
            cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            eta=getattr(args, "eta", None),
            device=m.device,
            dtype=torch.float32,
            dynamic_threshold_percentile=dyn_p,
            dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
            dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
            solver=getattr(args, "solver", "dpmpp_2m"),
            karras_rho=float(getattr(args, "karras_rho", 7.0)),
            volatile_cfg_boost=_f(args, "volatile_cfg_boost", 0.0),
            volatile_cfg_quantile=_f(args, "volatile_cfg_quantile", 0.72),
            volatile_cfg_window=int(getattr(args, "volatile_cfg_window", 6) or 6),
            flow_matching_sample=bool(init.flow),
            flow_solver=str(getattr(args, "flow_solver", "dpmpp_2m") or "dpmpp_2m"),
            flow_schedule=str(getattr(args, "flow_schedule", "ays") or "ays"),
            flow_karras_rho=_f(args, "flow_karras_rho", 7.0),
            # SD3 resolution shift: the trained shift at the native grid, rescaled for each sampled grid.
            flow_shift=float(flow_shift),
            flow_shift_ref_hw=(m.latent_size, m.latent_size),
            sde_band=getattr(args, "sde_band", None),
            eps_scale=float(getattr(args, "eps_scale", 1.0)),
            eps_scale_k=float(getattr(args, "eps_scale_k", 0.0)),
            control_guidance_start=_f(args, "control_guidance_start", 0.0),
            control_guidance_end=_f(args, "control_guidance_end", 1.0),
            control_guidance_decay=_f(args, "control_guidance_decay", 1.0),
            inference_amp=amp_on,
            inference_amp_dtype=amp_dtype,
            cfg_guidance_schedule=schedule,
            cfg_guidance_linear_start_multiplier=float(getattr(args, "guidance_schedule_linear_start", 0.7)),
            cfg_guidance_linear_end_multiplier=float(getattr(args, "guidance_schedule_linear_end", 1.0)),
            cfg_guidance_cosine_min_multiplier=float(getattr(args, "guidance_schedule_cosine_min", 0.65)),
            cfg_guidance_cosine_max_multiplier=float(getattr(args, "guidance_schedule_cosine_max", 1.0)),
            regional_cfg_plan=getattr(args, "_regional_cfg_plan", None),
            **holy,
            **_guidance_stack_kwargs(args),
        )
        # Trajectories that start from noise (or an img2img start) use the user's schedule and SAG / CLIP monitor.
        self.trajectory = dict(
            scheduler=getattr(args, "scheduler", "ays_dit"),
            timestep_schedule=getattr(args, "timestep_schedule", None),
            sag_blur_sigma=_f(args, "sag_blur_sigma", 0.0),
            sag_scale=_f(args, "sag_scale", 0.0),
            **self._clip_monitor_kwargs(),
        )
        early_exit = bool(getattr(args, "ada_early_exit", False))
        self.early_exit = dict(
            ada_early_exit_delta_threshold=getattr(args, "ada_exit_delta_threshold", 0.0) if early_exit else 0.0,
            ada_early_exit_patience=int(getattr(args, "ada_exit_patience", 0)) if early_exit else 0,
            ada_early_exit_min_steps=int(getattr(args, "ada_exit_min_steps", 0)),
        )

    def _clip_monitor_kwargs(self) -> dict:
        """``--clip-monitor-every``: mid-loop CLIP cosine check that boosts CFG (and optionally rewinds) when low."""
        args, m = self.args, self.m
        every = int(getattr(args, "clip_monitor_every", 0) or 0)
        if every <= 0:
            return {}
        from utils.generation.clip_alignment import latent_x0_clip_cosine

        model_id = str(getattr(args, "clip_guard_model", "openai/clip-vit-base-patch32"))

        def clip_fn(_step: int, x0p: torch.Tensor) -> float:
            return latent_x0_clip_cosine(
                x0p,
                prompt=self.plan.prompt,
                model_id=model_id,
                device=m.device,
                vae=m.vae,
                latent_scale=m.latent_scale,
                ae_type=m.ae_type,
                rae_bridge=m.rae_bridge,
            )

        return dict(
            periodic_alignment_interval=every,
            periodic_alignment_threshold=_f(args, "clip_monitor_threshold", 0.22),
            periodic_alignment_cfg_boost=_f(args, "clip_monitor_cfg_boost", 0.12),
            periodic_alignment_fn=clip_fn,
            periodic_alignment_rewind_strength=_f(args, "clip_monitor_rewind", 0.0),
        )

    def run(self, shape: tuple, **kw: Any) -> Any:
        with torch.no_grad():
            return self.m.diffusion.sample_loop(self.m.model, shape, **{**self.common, **kw})

    def resolve_from(self, x_t: torch.Tensor, t_s: int, n_steps: int, cfg: float | None = None) -> torch.Tensor:
        """Backward solve from the noised latent ``x_t`` at timestep ``t_s`` with the run's sampler settings."""
        return self.run(
            x_t.shape,
            cfg_scale=self.cfg_scale if cfg is None else cfg,
            num_inference_steps=max(4, int(n_steps)),
            x_init=x_t,
            start_timestep=t_s,
            scheduler="euler",
        )

    def renoise_resolve(self, x0: torch.Tensor, t_s: int, n_steps: int, cycles: int = 1) -> torch.Tensor:
        """Forward-noise a clean latent to ``t_s`` and solve back (one Restart cycle, VP or flow)."""
        return self.m.diffusion.restart(
            x0, lambda x_t, t: self.resolve_from(x_t, t, n_steps), t_max=t_s, cycles=cycles, flow=self.init.flow
        )

    def renoise_refine(self, x_lo: torch.Tensor, lh: int, lw: int, strength: float, n_steps: int, cfg: float):
        """Upscale + re-noise a clean latent for the ``lh x lw`` grid, then denoise it (hires-fix, dual-stage)."""
        x_t, t0 = renoise_for_grid(
            self.m.diffusion,
            x_lo,
            lh,
            lw,
            strength,
            native_latent_hw=(self.m.latent_size, self.m.latent_size),
            flow=self.init.flow,
        )
        print(f"Refine pass at latent {lh}x{lw}: t_start={t0}, {n_steps} steps, cfg={cfg}.", file=sys.stderr)
        return self.resolve_from(x_t, t0, n_steps, cfg)

    def pick_preview(self, x0_pred: torch.Tensor, metric: str) -> tuple[int, list[float]]:
        from utils.generation.sample_output import pick_indices, to_rgb_u8

        with torch.no_grad():
            return pick_indices(self.args, self.plan, to_rgb_u8(self.m.decode_unit(x0_pred)), metric, self.m.device)


def _holy_grail_kwargs(args: Any) -> dict:
    return dict(
        holy_grail_enable=bool(getattr(args, "holy_grail", False)),
        holy_grail_cfg_early_ratio=_f(args, "holy_grail_cfg_early_ratio", 0.72),
        holy_grail_cfg_late_ratio=_f(args, "holy_grail_cfg_late_ratio", 1.0),
        holy_grail_control_mult=_f(args, "holy_grail_control_mult", 1.0),
        holy_grail_adapter_mult=_f(args, "holy_grail_adapter_mult", 1.0),
        holy_grail_frontload_control=not bool(getattr(args, "holy_grail_no_frontload_control", False)),
        holy_grail_late_adapter_boost=_f(args, "holy_grail_late_adapter_boost", 1.15),
        holy_grail_cads_strength=_f(args, "holy_grail_cads_strength", 0.0),
        holy_grail_cads_min_strength=_f(args, "holy_grail_cads_min_strength", 0.0),
        holy_grail_cads_power=_f(args, "holy_grail_cads_power", 1.0),
        holy_grail_unsharp_sigma=_f(args, "holy_grail_unsharp_sigma", 0.0),
        holy_grail_unsharp_amount=_f(args, "holy_grail_unsharp_amount", 0.0),
        holy_grail_clamp_quantile=_f(args, "holy_grail_clamp_quantile", 0.0),
        holy_grail_clamp_floor=_f(args, "holy_grail_clamp_floor", 1.0),
    )


def _guidance_stack_kwargs(args: Any) -> dict:
    """APG / FDG / CFG-Zero* / interval / TCFG / PAG / SLG, feature + block caches, QSilk, dynamic width."""
    return dict(
        fdg_low=float(getattr(args, "fdg_low", 1.0)),
        fdg_high=float(getattr(args, "fdg_high", 1.0)),
        feature_cache_delta_threshold=_f(args, "feature_cache_delta", 0.0),
        feature_cache_max_reuse=int(getattr(args, "feature_cache_max_reuse", 2) or 2),
        block_cache_threshold=_f(args, "block_cache_thresh", 0.0),
        block_cache_recompute_every=int(getattr(args, "block_cache_recompute_every", 4) or 4),
        taylor_cache=bool(getattr(args, "taylor_cache", False)),
        taylor_cache_order=int(getattr(args, "taylor_cache_order", 1) or 1),
        taylor_cache_interval=int(getattr(args, "block_cache_recompute_every", 4) or 4),
        apg_parallel_eta=float(getattr(args, "apg_parallel_eta", -1.0)),
        apg_norm_threshold=float(getattr(args, "apg_norm_threshold", 0.0)),
        cfg_zero_star=bool(getattr(args, "cfg_zero_star", False)),
        cfg_zero_init_frac=float(getattr(args, "cfg_zero_init_frac", 0.04)),
        qsilk_micrograin=_f(args, "qsilk_micrograin", 0.0),
        dynamic_dit_width=bool(getattr(args, "dynamic_dit_width", False)),
        dynamic_dit_early=_f(args, "dynamic_dit_early", 0.88),
        dynamic_sdt=bool(getattr(args, "dynamic_sdt", False)),
        apg_momentum_beta=float(getattr(args, "apg_momentum_beta", 0.0)),
        cfg_interval=getattr(args, "cfg_interval", None) or None,
        tcfg_damping=_f(args, "tcfg_damping", 0.0),
        pag_scale=_f(args, "pag_scale", 0.0),
        pag_blocks=str(getattr(args, "pag_blocks", "auto") or "auto"),
        slg_scale=_f(args, "slg_scale", 0.0),
        slg_skip_blocks=str(getattr(args, "slg_skip_blocks", "auto") or "auto"),
        cfg_rejection_rerank=bool(getattr(args, "cfg_rejection_rerank", False))
        and int(getattr(args, "num", 1) or 1) > 1,
        dbc_separate_cfg=bool(getattr(args, "dbc_separate_cfg", False)) or _f(args, "block_cache_thresh", 0.0) > 0.0,
    )


def _beam_metric(args: Any, *names: str) -> str:
    """First of ``names`` that is set (not auto / none), else the ViT combo critic."""
    for name in names:
        value = str(getattr(args, name, "") or "").strip().lower()
        if value and value not in ("auto", "none"):
            return value
    return "combo_vit_hq" if str(getattr(args, "pick_vit_ckpt", "") or "").strip() else "combo_vit"


def _dual_stage(s: Sampler) -> torch.Tensor | None:
    """``--dual-stage-layout``: compose on a coarse grid, then a detail pass at the canvas grid."""
    args, m, init = s.args, s.m, s.init
    if not bool(getattr(args, "dual_stage_layout", False)):
        return None
    if m.ae_type != "kl" or args.mask or args.init_image or args.init_latent or init.x_init is not None:
        print("Dual-stage layout disabled (need KL VAE, no inpaint/img2img/from-z/init).", file=sys.stderr)
        return None
    div = max(2, int(getattr(args, "dual_stage_div", 2) or 2))
    patch = dit_patch_size(m.model)
    ch, cw = snap_to_multiple(round(m.lat_h / div), patch), snap_to_multiple(round(m.lat_w / div), patch)
    layout_steps = int(getattr(args, "dual_layout_steps", 24) or 24)
    print(f"Dual-stage: layout latent {ch}x{cw} ({layout_steps} steps) -> {m.lat_h}x{m.lat_w}...", file=sys.stderr)
    x0 = s.run((s.plan.num_gen, 4, ch, cw), num_inference_steps=layout_steps, **s.trajectory, **s.feature_kw)
    with torch.no_grad():
        return s.renoise_refine(
            x0,
            m.lat_h,
            m.lat_w,
            float(getattr(args, "dual_detail_strength", 0.38)),
            int(getattr(args, "dual_detail_steps", 20) or 20),
            s.cfg_scale,
        )


def _single_trajectory_ok(s: Sampler) -> bool:
    return s.plan.num_gen == 1 and not s.init.flow and s.init.inpaint_mask is None


def _beam_search(s: Sampler, report: dict) -> int:
    """``--beam-width/--beam-steps``: denoise several seeds a few steps, keep the best preview; returns steps used."""
    args, init = s.args, s.init
    width, steps = int(getattr(args, "beam_width", 0) or 0), int(getattr(args, "beam_steps", 0) or 0)
    if not (width > 1 and steps > 0 and _single_trajectory_ok(s) and init.x_init is None):
        return 0
    try:
        metric = _beam_metric(args, "beam_metric", "pick_best")
        print(f"Beam search: width={width} steps={steps} metric={metric}", file=sys.stderr)
        x_t, x0_pred, t_resume = s.run(
            (width, 4, s.m.lat_h, s.m.lat_w),
            num_inference_steps=steps,
            model_kwargs_cond=_with_batch(s.c.cond, width),
            model_kwargs_uncond=_with_batch(s.c.uncond, width),
            return_intermediate_state=True,
            **s.trajectory,
            **s.early_exit,
        )
        best, scores = s.pick_preview(x0_pred, metric)
    except Exception as e:
        print(f"Beam search disabled (fallback): {e}", file=sys.stderr)
        return 0
    print(f"beam scores={scores} -> keep {best} resume_t={t_resume}", file=sys.stderr)
    report["beam1"] = {
        "metric": metric,
        "scores": [float(x) for x in scores],
        "best_index": int(best),
        "resume_t": int(t_resume),
        "beam_width": width,
        "beam_steps": steps,
    }
    init.x_init, init.start_timestep = x_t[best : best + 1].contiguous(), int(t_resume)
    return steps


def _micro_beam(s: Sampler, steps_main: int, report: dict) -> int:
    """``--beam2-*``: branch from a mid-trajectory latent with small noise and re-pick; returns steps left."""
    args, init = s.args, s.init
    width, steps = int(getattr(args, "beam2_width", 0) or 0), int(getattr(args, "beam2_steps", 0) or 0)
    at, noise = _f(args, "beam2_at_frac", 0.65), _f(args, "beam2_noise", 0.03)
    if not (width > 1 and steps > 0 and _single_trajectory_ok(s) and steps_main >= steps + 2):
        return steps_main
    try:
        metric = _beam_metric(args, "beam2_metric", "beam_metric", "pick_best")
        split = int(round(float(steps_main) * min(0.95, max(0.05, at))))
        split = max(1, min(split, steps_main - (steps + 1)))
        shape = (1, 4, s.m.lat_h, s.m.lat_w)
        x_mid, _, t_mid = s.run(
            shape,
            num_inference_steps=split,
            x_init=init.x_init,
            start_timestep=init.start_timestep,
            return_intermediate_state=True,
            **s.trajectory,
            **s.early_exit,
        )
        x_seed = x_mid.expand(width, -1, -1, -1).contiguous()
        if noise > 0.0:
            x_seed = x_seed + noise * torch.randn_like(x_seed)
        x2, x0_2, t2 = s.run(
            (width, *shape[1:]),
            num_inference_steps=steps,
            x_init=x_seed,
            start_timestep=int(t_mid),
            model_kwargs_cond=_with_batch(s.c.cond, width),
            model_kwargs_uncond=_with_batch(s.c.uncond, width),
            return_intermediate_state=True,
            **s.trajectory,
            **s.early_exit,
        )
        best, scores = s.pick_preview(x0_2, metric)
    except Exception as e:
        print(f"Micro-beam disabled (fallback): {e}", file=sys.stderr)
        return steps_main
    print(
        f"Micro-beam: split={split}/{steps_main} width={width} steps={steps} "
        f"metric={metric} scores={scores} -> keep {best} resume_t={t2}",
        file=sys.stderr,
    )
    report["beam2"] = {
        "metric": metric,
        "scores": [float(x) for x in scores],
        "best_index": int(best),
        "resume_t": int(t2),
        "beam2_width": width,
        "beam2_steps": steps,
        "beam2_at_frac": at,
        "beam2_noise": noise,
        "split_steps": split,
    }
    init.x_init, init.start_timestep = x2[best : best + 1].contiguous(), int(t2)
    return max(1, steps_main - split - steps)


def _denoise(s: Sampler, report: dict) -> torch.Tensor:
    """Main trajectory, after the optional beam searches."""
    args, init = s.args, s.init
    print(f"Sampling (steps={args.steps}, num={s.plan.num_gen}, cfg_scale={s.cfg_scale})...")
    steps_main = int(args.steps) - _beam_search(s, report)
    steps_main = _micro_beam(s, max(1, steps_main), report)
    return s.run(
        (s.plan.num_gen, 4, s.m.lat_h, s.m.lat_w),
        num_inference_steps=steps_main,
        x_init=init.x_init,
        start_timestep=init.start_timestep,
        inpaint_mask=init.inpaint_mask,
        inpaint_x0=init.inpaint_x0,
        inpaint_noise=init.inpaint_noise,
        inpaint_freeze_known=bool(args.mask and args.inpaint_mode == "mdm"),
        **s.trajectory,
        **s.early_exit,
        **s.feature_kw,
    )


def _refine_latents(s: Sampler, x0: torch.Tensor) -> torch.Tensor:
    """Post-trajectory passes: CFG-rejection rerank, CLIP guard, self-correct, latent priors, hires-fix, Restart."""
    args, m = s.args, s.m
    if bool(getattr(args, "cfg_rejection_rerank", False)) and s.plan.num_gen > 1:
        probe = getattr(m.diffusion, "_last_guidance_probe", None)
        order = probe.rerank_indices() if probe is not None else None
        if order and len(order) == int(x0.shape[0]):
            x0 = x0[order]
            print(f"CFG-rejection rerank order={order}", file=sys.stderr)

    if _f(args, "clip_guard_threshold", 0.0) > 0.0:
        try:
            from utils.generation.clip_alignment import maybe_clip_refine_latent

            x0 = maybe_clip_refine_latent(
                x0,
                prompt=s.plan.prompt,
                sim_threshold=float(args.clip_guard_threshold),
                model_id=str(getattr(args, "clip_guard_model", "openai/clip-vit-base-patch32")),
                device=m.device,
                vae=m.vae,
                latent_scale=m.latent_scale,
                ae_type=m.ae_type,
                rae_bridge=m.rae_bridge,
                num_timesteps=s.init.num_timesteps,
                t_frac=float(getattr(args, "clip_guard_t_frac", 0.22)),
                refine_steps=int(getattr(args, "clip_guard_steps", 12)),
                refiner=s.renoise_resolve,
            )
        except Exception as e:
            print(f"CLIP guard refine skipped: {e}", file=sys.stderr)
    elif getattr(args, "superior_self_correct", False):
        try:
            from utils.superior.inference_pipeline import SelfCorrectConfig, SelfCorrectPolicy

            policy = SelfCorrectPolicy(
                SelfCorrectConfig(clip_model_id=str(getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32")))
            )
            score = policy.score(
                x0,
                s.plan.prompt,
                vae=m.vae,
                latent_scale=m.latent_scale,
                ae_type=m.ae_type,
                rae_bridge=m.rae_bridge,
                device=m.device,
            )
            if policy.needs_correction(score):
                t_s, n_st = policy.refine_plan(s.init.num_timesteps)
                x0 = s.renoise_resolve(x0, t_s, n_st)
                policy.note_refinement_done()
                print(f"Superior self-correct: refined (score={score:.3f}, t={t_s})", file=sys.stderr)
        except Exception as e:
            print(f"Superior self-correct skipped: {e}", file=sys.stderr)

    if _f(args, "domain_prior_latent", 0.0) > 0.0:
        from utils.generation.latent_filters import highfreq_layout_prior

        x0 = highfreq_layout_prior(x0, strength=float(args.domain_prior_latent))
    if _f(args, "spectral_coherence_latent", 0.0) > 0.0:
        from utils.generation.latent_filters import spectral_latent_lowfreq_blend

        x0 = spectral_latent_lowfreq_blend(
            x0,
            strength=float(args.spectral_coherence_latent),
            cutoff_frac=_f(args, "spectral_coherence_cutoff", 0.15),
        )
    x0 = _hires_fix(s, x0)
    if str(getattr(args, "refine", "off") or "off") == "restart" and not getattr(args, "no_refine", False):
        if _refine_gate_open(s, x0):
            x0 = s.renoise_resolve(
                x0,
                int(getattr(args, "refine_t", 200)),
                int(getattr(args, "restart_steps", 6) or 6),
                cycles=int(getattr(args, "restart_cycles", 2) or 2),
            )
    return x0


def _hires_fix(s: Sampler, x0: torch.Tensor) -> torch.Tensor:
    """``--hires-fix``: upscale the clean latent to the requested size, re-noise and denoise it there."""
    args, m = s.args, s.m
    if not getattr(args, "hires_fix", False):
        return x0
    if m.ae_type != "kl":
        print("Hires-fix skipped: only supported for KL VAE checkpoints.", file=sys.stderr)
        return x0
    if args.mask or args.init_image or args.init_latent:
        print("Hires-fix skipped: not combined with inpaint / img2img / from-z.", file=sys.stderr)
        return x0
    if int(getattr(args, "width", 0) or 0) > 0 or int(getattr(args, "height", 0) or 0) > 0:
        th_px, tw_px = m.req_h, m.req_w
    else:
        scale = max(_f(args, "hires_scale", 1.5), 1.0)
        th_px, tw_px = round(m.canvas_h * scale), round(m.canvas_w * scale)
    stride = dit_rgb_stride_px(m.model)
    tlh, tlw = snap_to_multiple(th_px, stride) // 8, snap_to_multiple(tw_px, stride) // 8
    if tlh * tlw <= x0.shape[-2] * x0.shape[-1]:
        print(f"Hires-fix skipped: latent already {x0.shape[-2]}x{x0.shape[-1]} (target {tlh}x{tlw}).", file=sys.stderr)
        return x0
    cfg_h = float(getattr(args, "hires_cfg_scale", -1.0))
    with torch.no_grad():
        return s.renoise_refine(
            x0,
            tlh,
            tlw,
            float(getattr(args, "hires_strength", 0.35)),
            max(1, int(getattr(args, "hires_steps", 15))),
            float(s.cfg_scale) if cfg_h < 0 else cfg_h,
        )


def _refine_gate_open(s: Sampler, x0: torch.Tensor) -> bool:
    """``--refine-gate auto``: skip the Restart refine when a cheap preview already scores well."""
    args = s.args
    if str(getattr(args, "refine_gate", "off") or "off").lower() != "auto":
        return True
    try:
        from utils.generation.sample_output import to_rgb_u8

        with torch.no_grad():
            rgb = to_rgb_u8(s.m.decode_unit(x0[:1]))[0]
        score, details = _refine_gate_score(image_rgb_u8=rgb, expected_texts=s.plan.expected_texts)
    except Exception as e:
        print(f"refine-gate fallback: {e}", file=sys.stderr)
        return True
    thr = _f(args, "refine_gate_threshold", 0.62)
    print(f"refine-gate: score={score:.3f} thr={thr:.3f} run_refine={score < thr} details={details}", file=sys.stderr)
    return score < thr


def export_attention(args: Any, m: Any, cond: Any, x0: torch.Tensor) -> None:
    """``--save-attn`` / ``--explain-adherence``: cross-attention maps at t=0 for the first sample."""
    adherence_path = str(getattr(args, "explain_adherence", "") or "").strip()
    if not (args.save_attn or adherence_path):
        return
    with torch.no_grad():
        out = m.model(x0[:1], torch.zeros(1, device=m.device, dtype=torch.long), return_attn=True, **cond.cond)
    if not isinstance(out, tuple):
        print("Model does not support attention export; skipping.", file=sys.stderr)
        return
    _, attn = out
    if args.save_attn:
        torch.save({"attn": attn.cpu(), "prompt": args.prompt}, args.save_attn)
        print(f"Saved attention: {args.save_attn}")
    if adherence_path:
        from utils.generation.sample_features import export_adherence_heatmap

        export_adherence_heatmap(
            attn,
            args.prompt or "",
            adherence_path,
            latent_h=int(x0.shape[-2]),
            latent_w=int(x0.shape[-1]),
            box_spec=getattr(args, "_box_layout_spec", None),
        )
        print(f"Saved adherence heatmap: {adherence_path}")


def _injects_noise(args: Any, flow: bool) -> bool:
    """Whether the solver adds noise each step; frontier ``step_noise_scales`` curves only scale that noise."""
    from diffusion.gaussian_diffusion import canonicalize_flow_solver, canonicalize_vp_solver

    if flow:
        solver = canonicalize_flow_solver(str(getattr(args, "flow_solver", "dpmpp_2m") or "dpmpp_2m"))
        stochastic = ("dpmpp_2m_sde",)
    else:
        solver = canonicalize_vp_solver(str(getattr(args, "solver", "dpmpp_2m") or "dpmpp_2m"))
        stochastic = ("ddim", "heun", "dpmpp_2m_sde")
    eta = getattr(args, "eta", None)
    eta = (1.0 if solver == "dpmpp_2m_sde" else 0.0) if eta is None else float(eta)
    return solver in stochastic and eta > 0.0


def sample_latents(args: Any, m: Any, plan: Any, cond: Any, init: Any, report: dict) -> torch.Tensor:
    """Denoise (dual-stage, beam or plain), then refine in latent space; returns clean DiT latents."""
    from utils.generation.frontier_consume import sanitize_diffusion_feature_kwargs
    from utils.generation.sample_features import apply_sample_feature_diffusion_phase

    feature_kw = sanitize_diffusion_feature_kwargs(
        apply_sample_feature_diffusion_phase(args, steps=int(getattr(args, "steps", 50) or 50))
    )
    if "step_noise_scales" in feature_kw and not _injects_noise(args, init.flow):
        del feature_kw["step_noise_scales"]
        print(
            "Frontier noise curve skipped: the solver injects no noise (use --eta > 0 with ddim, heun or "
            "dpmpp_2m_sde).",
            file=sys.stderr,
        )
    s = Sampler(args, m, plan, cond, init, feature_kw)
    x0 = _dual_stage(s)
    if x0 is None:
        x0 = _denoise(s, report)
    return _refine_latents(s, x0)


__all__ = ["Sampler", "export_attention", "resolve_inference_amp", "sample_latents"]
