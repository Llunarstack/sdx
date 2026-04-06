"""CLI argument parsing helpers for train.py."""

from __future__ import annotations

from typing import Optional

from config.train_config import TrainConfig
from utils.modeling.model_paths import default_t5_path


def parse_caption_dropout_schedule(s: Optional[str]):
    """Parse '0,0.2,10000,0.05' -> [(0, 0.2), (10000, 0.05)]. Returns None if s is None/empty."""
    if not s or not str(s).strip():
        return None
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    if len(parts) % 2 != 0:
        return None
    out = []
    for i in range(0, len(parts), 2):
        out.append((int(parts[i]), float(parts[i + 1])))
    return out if out else None


def parse_resolution_buckets(s: Optional[str]):
    """Parse ``256,384`` or ``512x768,256x512`` into list[(H, W)] or None."""
    if not s or not str(s).strip():
        return None
    out = []
    for part in str(s).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" in part:
            a, b = part.split("x", 1)
            out.append((int(a.strip()), int(b.strip())))
        else:
            z = int(part)
            out.append((z, z))
    return out or None


def parse_mdm_mask_schedule(s: Optional[str]):
    """Parse '0,0.05,500,0.25' -> [(0,0.05),(500,0.25)] or None."""
    if not s or not str(s).strip():
        return None
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    if len(parts) % 2 != 0:
        return None
    out = []
    for i in range(0, len(parts), 2):
        out.append((int(float(parts[i])), float(parts[i + 1])))
    return out if out else None


def build_train_config_from_args(args) -> TrainConfig:
    """Build TrainConfig from argparse namespace (train.py CLI)."""
    return TrainConfig(
        data_path=args.data_path,
        manifest_jsonl=args.manifest_jsonl,
        results_dir=args.results_dir,
        model_name=args.model,
        text_encoder=(args.text_encoder or default_t5_path()),
        text_encoder_mode=str(getattr(args, "text_encoder_mode", "t5")),
        clip_text_encoder_l=str(getattr(args, "clip_text_encoder_l", "") or ""),
        clip_text_encoder_bigg=str(getattr(args, "clip_text_encoder_bigg", "") or ""),
        image_size=args.image_size,
        resolution_buckets=parse_resolution_buckets(args.resolution_buckets or None),
        vae_model=args.vae_model,
        autoencoder_type=args.autoencoder_type,
        rae_use_latent_bridge=not getattr(args, "no_rae_latent_bridge", False),
        rae_bridge_cycle_weight=float(getattr(args, "rae_bridge_cycle_weight", 0.01)),
        global_batch_size=args.global_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        use_bf16=not args.no_bf16,
        use_compile=not args.no_compile,
        grad_checkpointing=not args.no_grad_checkpoint,
        global_seed=args.seed,
        num_ar_blocks=args.num_ar_blocks,
        ar_block_order=str(getattr(args, "ar_block_order", "raster") or "raster"),
        ar_curriculum_mode=str(getattr(args, "ar_curriculum_mode", "none") or "none"),
        ar_curriculum_warmup_steps=int(getattr(args, "ar_curriculum_warmup_steps", 0)),
        ar_curriculum_ramp_start=int(getattr(args, "ar_curriculum_ramp_start", 0)),
        ar_curriculum_ramp_end=int(getattr(args, "ar_curriculum_ramp_end", 0)),
        ar_curriculum_start_blocks=int(getattr(args, "ar_curriculum_start_blocks", -1)),
        ar_curriculum_target_blocks=int(getattr(args, "ar_curriculum_target_blocks", -1)),
        ar_order_mix=str(getattr(args, "ar_order_mix", "") or ""),
        use_xformers=not args.no_xformers,
        passes=args.passes,
        max_steps=args.max_steps,
        min_lr=args.min_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        refinement_prob=args.refinement_prob,
        refinement_max_t=args.refinement_max_t,
        img2img_prob=args.img2img_prob,
        mdm_mask_ratio=args.mdm_mask_ratio,
        mdm_mask_schedule=parse_mdm_mask_schedule(getattr(args, "mdm_mask_schedule", None)),
        mdm_patch_size=args.mdm_patch_size,
        mdm_min_mask_patches=args.mdm_min_mask_patches,
        mdm_loss_only_masked=not args.no_mdm_loss_only_masked,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_balance_loss_weight=args.moe_balance_loss_weight,
        save_best=not args.no_save_best,
        negative_prompt_weight=args.negative_prompt_weight,
        style_embed_dim=args.style_embed_dim,
        style_strength=args.style_strength,
        control_cond_dim=args.control_cond_dim,
        control_num_types=int(getattr(args, "control_num_types", 0) or 0),
        control_scale=args.control_scale,
        creativity_embed_dim=args.creativity_embed_dim,
        creativity_max=args.creativity_max,
        creativity_jitter_std=float(getattr(args, "creativity_jitter_std", 0.0)),
        train_originality_augment_prob=float(getattr(args, "train_originality_prob", 0.0)),
        train_originality_strength=float(getattr(args, "train_originality_strength", 0.5)),
        size_embed_dim=getattr(args, "size_embed_dim", 0),
        patch_se=getattr(args, "patch_se", False),
        patch_se_reduction=getattr(args, "patch_se_reduction", 8),
        curriculum_difficulty_steps=[int(x.strip()) for x in args.curriculum_difficulty_steps.split(",") if x.strip()]
        if (getattr(args, "curriculum_difficulty_steps", None) and str(args.curriculum_difficulty_steps).strip())
        else None,
        curriculum_difficulty_easy_first=not getattr(args, "no_difficulty_easy_first", False),
        rule_loss_weight=args.rule_loss_weight,
        repa_weight=args.repa_weight,
        repa_encoder_model=args.repa_encoder_model,
        repa_out_dim=args.repa_out_dim,
        repa_projector_hidden_dim=args.repa_projector_hidden_dim,
        ssm_every_n=args.ssm_every_n,
        ssm_kernel_size=args.ssm_kernel_size,
        num_register_tokens=args.num_register_tokens,
        use_rope=args.use_rope,
        rope_base=args.rope_base,
        kv_merge_factor=args.kv_merge_factor,
        token_routing_enabled=args.token_routing_enabled,
        token_routing_strength=args.token_routing_strength,
        token_keep_ratio=float(getattr(args, "token_keep_ratio", 1.0)),
        token_keep_min_value=float(getattr(args, "token_keep_min_value", 0.0)),
        drop_path_rate=float(getattr(args, "drop_path_rate", 0.0)),
        layerscale_init=float(getattr(args, "layerscale_init", 0.0)),
        qk_norm=bool(getattr(args, "qk_norm", False)),
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        noise_offset=args.noise_offset,
        min_snr_gamma=args.min_snr_gamma,
        loss_weighting=getattr(args, "loss_weighting", "min_snr"),
        loss_weighting_sigma_data=getattr(args, "loss_weighting_sigma_data", 0.5),
        spectral_sfp_loss=bool(getattr(args, "spectral_sfp_loss", False)),
        spectral_sfp_low_sigma=float(getattr(args, "spectral_sfp_low_sigma", 0.22)),
        spectral_sfp_high_sigma=float(getattr(args, "spectral_sfp_high_sigma", 0.22)),
        spectral_sfp_tau_power=float(getattr(args, "spectral_sfp_tau_power", 1.0)),
        ot_noise_pair_reg=float(getattr(args, "ot_noise_pair_reg", 0.0)),
        ot_noise_pair_iters=int(getattr(args, "ot_noise_pair_iters", 40)),
        ot_noise_pair_mode=str(getattr(args, "ot_noise_pair_mode", "soft")),
        flow_matching_training=bool(getattr(args, "flow_matching_training", False)),
        bridge_aux_weight=float(getattr(args, "bridge_aux_weight", 0.0)),
        bridge_aux_lambda=float(getattr(args, "bridge_aux_lambda", 0.2)),
        timestep_sample_mode=getattr(args, "timestep_sample_mode", "uniform"),
        timestep_logit_mean=getattr(args, "timestep_logit_mean", 0.0),
        timestep_logit_std=getattr(args, "timestep_logit_std", 1.0),
        resume=args.resume,
        val_split=args.val_split,
        val_every=args.val_every,
        early_stopping_patience=args.early_stopping_patience,
        val_max_batches=args.val_max_batches,
        deterministic=args.deterministic,
        latent_cache_dir=args.latent_cache_dir,
        caption_dropout_schedule=parse_caption_dropout_schedule(getattr(args, "caption_dropout_schedule", None)),
        crop_mode=getattr(args, "crop_mode", "center"),
        region_caption_mode=getattr(args, "region_caption_mode", "append"),
        region_layout_tag=getattr(args, "region_layout_tag", "[layout]"),
        boost_adherence_caption=bool(getattr(args, "boost_adherence_caption", False)),
        train_shortcomings_mitigation=str(getattr(args, "train_shortcomings_mitigation", "none") or "none"),
        train_shortcomings_2d=bool(getattr(args, "train_shortcomings_2d", False)),
        train_art_guidance_mode=str(getattr(args, "train_art_guidance_mode", "none") or "none"),
        train_art_guidance_photography=bool(getattr(args, "train_art_guidance_photography", True)),
        train_anatomy_guidance=str(getattr(args, "train_anatomy_guidance", "none") or "none"),
        train_style_guidance_mode=str(getattr(args, "train_style_guidance_mode", "none") or "none"),
        train_style_guidance_artists=bool(getattr(args, "train_style_guidance_artists", True)),
        caption_unicode_normalize=bool(getattr(args, "caption_unicode_normalize", False)),
        train_prompt_emphasis=bool(getattr(args, "train_prompt_emphasis", False)),
        attn_grounding_loss_weight=float(getattr(args, "attn_grounding_loss_weight", 0.0)),
        attn_grounding_token_start=int(getattr(args, "attn_grounding_token_start", 0)),
        attn_grounding_token_end=int(getattr(args, "attn_grounding_token_end", 0)),
        attn_grounding_min_fg_patch_mass=float(getattr(args, "attn_grounding_min_fg_patch_mass", 1e-4)),
        attn_token_coverage_loss_weight=float(getattr(args, "attn_token_coverage_loss_weight", 0.0)),
        attn_token_coverage_target=float(getattr(args, "attn_token_coverage_target", 0.025)),
        use_hierarchical_captions=bool(getattr(args, "use_hierarchical_captions", False)),
        hierarchical_caption_separator=str(getattr(args, "hierarchical_caption_separator", " | ")),
        hierarchical_caption_drop_global_p=float(getattr(args, "hierarchical_caption_drop_global_p", 0.0)),
        hierarchical_caption_drop_local_p=float(getattr(args, "hierarchical_caption_drop_local_p", 0.0)),
        foveated_train_prob=float(getattr(args, "foveated_train_prob", 0.0)),
        foveated_crop_frac=float(getattr(args, "foveated_crop_frac", 0.55)),
        grounding_mask_soft=bool(getattr(args, "grounding_mask_soft", False)),
        prompt_reinject_every_n=int(getattr(args, "prompt_reinject_every_n", 0)),
        prompt_reinject_alpha=float(getattr(args, "prompt_reinject_alpha", 0.0)),
        prompt_reinject_decay=float(getattr(args, "prompt_reinject_decay", 1.0)),
        prompt_timestep_schedule_enabled=bool(getattr(args, "prompt_timestep_schedule_enabled", False)),
        prompt_early_scale=float(getattr(args, "prompt_early_scale", 1.1)),
        prompt_late_scale=float(getattr(args, "prompt_late_scale", 1.0)),
        save_polyak=getattr(args, "save_polyak", 0),
        wandb_project=getattr(args, "wandb_project", None),
        tensorboard_dir=getattr(args, "tensorboard_dir", None),
        dry_run=getattr(args, "dry_run", False),
        save_run_manifest=not bool(getattr(args, "no_save_run_manifest", False)),
        strict_warnings=bool(getattr(args, "strict_warnings", False)),
        log_images_every=getattr(args, "log_images_every", 0),
        log_images_prompt=getattr(args, "log_images_prompt", "a photo of a cat"),
    )

