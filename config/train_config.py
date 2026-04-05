"""Training configuration — single source of truth for all hyperparameters."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TrainConfig:
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    data_path: str = ""
    manifest_jsonl: Optional[str] = None
    image_size: int = 256
    # Optional list of (H, W) targets for multi-resolution / aspect-ratio bucketing.
    # None = single square crop at --image-size.
    resolution_buckets: Optional[List[Tuple[int, int]]] = None
    num_workers: int = 8
    global_batch_size: int = 128
    caption_dropout_prob: float = 0.1
    # Step-dependent caption dropout schedule: list of (step, prob) breakpoints.
    # Example: [(0, 0.2), (10000, 0.05)] decays from 0.2 to 0.05 over 10k steps.
    caption_dropout_schedule: Optional[List[tuple]] = None
    crop_mode: str = "center"  # "center" | "random" | "largest_center"
    # Merge JSONL `parts` / `region_captions` into the T5 caption string.
    region_caption_mode: str = "append"  # "append" | "prefix" | "off"
    region_layout_tag: str = "[layout]"  # prefix before regional block; "" to disable
    # Prepend adherence tags to training captions (see data/caption_utils.py).
    boost_adherence_caption: bool = False
    # Append failure-mode prompt/negative hints per caption (see config.defaults.ai_image_shortcomings).
    train_shortcomings_mitigation: str = "none"  # "none" | "auto" | "all"
    train_shortcomings_2d: bool = False
    # Artist-first medium guidance per caption (see config.defaults.art_mediums).
    train_art_guidance_mode: str = "none"  # "none" | "auto" | "all"
    train_art_guidance_photography: bool = True
    train_anatomy_guidance: str = "none"  # "none" | "lite" | "strong"
    train_style_guidance_mode: str = "none"  # "none" | "auto" | "all"
    train_style_guidance_artists: bool = True
    # NFKC + zero-width strip on training captions (see sdx_native.text_hygiene).
    caption_unicode_normalize: bool = False
    # Strip ( ) / [ ] emphasis brackets for T5 and pass token_weights to DiT
    # (see utils/prompt/prompt_emphasis.py).
    train_prompt_emphasis: bool = False

    # -------------------------------------------------------------------------
    # Part-aware / grounding
    # JSONL fields: grounding_mask, caption_global, caption_local, entity_captions
    # See utils/training/part_aware_training.py
    # -------------------------------------------------------------------------
    attn_grounding_loss_weight: float = 0.0  # VP-DDPM only; 0 = off
    attn_grounding_token_start: int = 0
    attn_grounding_token_end: int = 0  # 0 = all text positions
    attn_grounding_min_fg_patch_mass: float = 1e-4  # skip near-empty masks (0 = off)
    use_hierarchical_captions: bool = False
    hierarchical_caption_separator: str = " | "
    hierarchical_caption_drop_global_p: float = 0.0
    hierarchical_caption_drop_local_p: float = 0.0
    foveated_train_prob: float = 0.0
    foveated_crop_frac: float = 0.55
    grounding_mask_soft: bool = False  # keep grayscale 0–1 instead of binarising at 0.5
    # Attend-and-Excite style token coverage auxiliary loss.
    attn_token_coverage_loss_weight: float = 0.0
    attn_token_coverage_target: float = 0.025
    # Prompt reinjection + timestep-aware text scaling (see DiT_Text).
    prompt_reinject_every_n: int = 0
    prompt_reinject_alpha: float = 0.0
    prompt_reinject_decay: float = 1.0
    prompt_timestep_schedule_enabled: bool = False
    prompt_early_scale: float = 1.1
    prompt_late_scale: float = 1.0

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model_name: str = "DiT-XL/2-Text"
    # T5-XXL like PixArt/ReVe; accepts HF id or local path (e.g. pretrained/T5-XXL).
    text_encoder: str = "google/t5-v1_1-xxl"
    # "t5" = T5 only (default).
    # "triple" = T5 + CLIP-ViT-L/14 + CLIP-ViT-bigG/14 pooled tokens fused
    # (see utils/modeling/text_encoder_bundle.py).
    text_encoder_mode: str = "t5"
    clip_text_encoder_l: str = ""    # empty = resolve via utils/modeling/model_paths.py
    clip_text_encoder_bigg: str = "" # empty = resolve via utils/modeling/model_paths.py
    vae_model: str = "stabilityai/sd-vae-ft-mse"
    # "kl"  = AutoencoderKL (classic Stable Diffusion VAE)
    # "rae" = AutoencoderRAE (Representation Autoencoder)
    autoencoder_type: str = "kl"
    latent_scale: float = 0.18215
    # When RAE latent channels != 4: train a 1×1 bridge (RAELatentBridge) to/from
    # DiT latents (see models/rae_latent_bridge.py).
    rae_use_latent_bridge: bool = True
    rae_bridge_cycle_weight: float = 0.01  # cycle loss z ≈ to_rae(to_dit(z)); 0 = off

    # -------------------------------------------------------------------------
    # REPA — Representation Alignment auxiliary loss
    # Aligns DiT internal features with a frozen DINOv2 or CLIP encoder.
    # -------------------------------------------------------------------------
    repa_weight: float = 0.0  # 0 = off
    # Recommended encoders:
    #   "facebook/dinov2-base"          (768-dim)
    #   "openai/clip-vit-large-patch14" (768-dim)
    repa_encoder_model: str = "facebook/dinov2-base"
    # Must match the frozen encoder embedding dim:
    #   CLIP ViT-L/14 → 768, DINOv2-base → 768, DINOv2-large → 1024.
    repa_out_dim: int = 768
    repa_projector_hidden_dim: int = 0  # 0 = linear projection head

    # -------------------------------------------------------------------------
    # Hybrid SSM token mixer
    # Replace every Nth self-attention block with a lightweight SSM-like mixer.
    # -------------------------------------------------------------------------
    ssm_every_n: int = 0  # 0 = off
    ssm_kernel_size: int = 7

    # -------------------------------------------------------------------------
    # ViT architectural features
    # -------------------------------------------------------------------------
    # Register tokens (scratchpad tokens appended to patch sequence).
    num_register_tokens: int = 0
    # RoPE (rotary positional embeddings) for self-attention.
    use_rope: bool = False
    rope_base: float = 10000.0
    # KV pooling factor for self-attention (1 = disabled; 2 = merge 2×2 blocks).
    kv_merge_factor: int = 1
    # Soft per-token routing gate (does not change compute graph).
    token_routing_enabled: bool = False
    token_routing_strength: float = 1.0
    token_keep_ratio: float = 1.0       # top-k keep over patch tokens (1.0 = disabled)
    token_keep_min_value: float = 0.0   # residual gate floor for dropped tokens
    drop_path_rate: float = 0.0         # stochastic depth (0 = off)
    layerscale_init: float = 0.0        # > 0 enables LayerScale residual gains (e.g. 1e-5)
    # QK-norm (SD3.5-style): normalise Q and K per head before attention.
    # Improves training stability at high resolution and large batch sizes.
    # Note: changes model architecture — not compatible with checkpoints trained without it.
    qk_norm: bool = False
    # Block-wise AR mask (ACDiT-style): 0 = full bidirectional attention.
    # 2 = 2×2 blocks, 4 = 4×4 blocks in raster order. See docs/AR.md.
    num_ar_blocks: int = 0
    ar_block_order: str = "raster"  # "raster" | "zorder"
    use_xformers: bool = True
    negative_prompt_weight: float = 0.5
    style_embed_dim: int = 0        # > 0 enables T5-encoded style conditioning
    style_strength: float = 0.7     # blend strength for style (0.6–0.8 recommended)
    control_cond_dim: int = 0       # 1 = enable ControlNet; 0 = off
    control_num_types: int = 0      # 0 = off; e.g. 9 for canny/depth/pose/seg/…
    control_scale: float = 0.85     # ControlNet strength (0.7–1.0 recommended)
    # Creativity/diversity scalar conditioning (0 = off; set to hidden dim, e.g. 64).
    creativity_embed_dim: int = 0
    creativity_max: float = 1.0
    creativity_jitter_std: float = 0.0  # extra noise on creativity scalar during training
    # Randomly inject originality tokens into captions during training.
    train_originality_augment_prob: float = 0.0  # 0 = off; try 0.1–0.25
    train_originality_strength: float = 0.5      # 0–1 controls token insertion density
    # PixArt-style (h, w) latent grid → timestep conditioning (0 = off).
    size_embed_dim: int = 0
    # Channel gate on patch tokens after embed (zero-init = identity at start).
    patch_se: bool = False
    patch_se_reduction: int = 8

    # -------------------------------------------------------------------------
    # Diffusion
    # -------------------------------------------------------------------------
    num_timesteps: int = 1000
    timestep_respacing: str = ""
    beta_schedule: str = "linear"      # "linear" | "cosine" | "sigmoid" | "squaredcos_cap_v2"
    prediction_type: str = "epsilon"   # "epsilon" | "v" | "x0"
    noise_offset: float = 0.0          # SD-style noise offset for light/dark balance (e.g. 0.1)
    min_snr_gamma: float = 5.0         # min-SNR loss cap (0 = off; 5 is typical)
    # "min_snr" | "min_snr_soft" | "unit" | "edm" | "v" | "eps"
    loss_weighting: str = "min_snr"
    loss_weighting_sigma_data: float = 0.5  # used when loss_weighting="edm"
    # Spectral Flow Prediction: FFT-weighted MSE on (pred - target) in latent space.
    # Ignored when MDM masked training is active.
    spectral_sfp_loss: bool = False
    spectral_sfp_low_sigma: float = 0.22
    spectral_sfp_high_sigma: float = 0.22
    spectral_sfp_tau_power: float = 1.0
    # OT noise–latent mini-batch coupling (see utils/training/ot_noise_pairing.py).
    ot_noise_pair_reg: float = 0.0   # 0 = off; Sinkhorn regulariser (e.g. 0.05)
    ot_noise_pair_iters: int = 40
    ot_noise_pair_mode: str = "soft"  # "soft" | "hungarian" (hungarian requires scipy)
    # Rectified-flow training (see diffusion/flow_matching.py).
    # Mutually exclusive with MDM masked training.
    flow_matching_training: bool = False
    # VP bridge auxiliary regulariser (shuffle-pair latent mix).
    bridge_aux_weight: float = 0.0   # 0 = off; try 0.02–0.15
    bridge_aux_lambda: float = 0.2   # mix x0 = (1-λ)x + λ·shuffle(x); in (0, 1]
    # Timestep sampling distribution (see diffusion/timestep_sampling.py).
    timestep_sample_mode: str = "uniform"  # "uniform" | "logit_normal" | "high_noise"
    timestep_logit_mean: float = 0.0
    timestep_logit_std: float = 1.0

    # -------------------------------------------------------------------------
    # Training length
    # Priority: passes > max_steps > epochs
    # -------------------------------------------------------------------------
    passes: int = 0      # full passes over dataset (steps = passes × steps_per_epoch)
    max_steps: int = 0   # hard step cap (0 = use epochs)
    epochs: int = 100    # used only when passes == 0 and max_steps == 0
    lr: float = 1e-4
    min_lr: float = 1e-6        # cosine schedule floor
    lr_warmup_steps: int = 500  # linear warmup then cosine
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_bf16: bool = True
    use_compile: bool = True
    grad_accum_steps: int = 1
    grad_checkpointing: bool = True
    save_best: bool = True  # save checkpoint whenever train loss improves

    # -------------------------------------------------------------------------
    # Validation + early stopping
    # -------------------------------------------------------------------------
    val_split: float = 0.0          # fraction held out for validation (0 = off)
    val_every: int = 2000           # evaluate every N steps
    early_stopping_patience: int = 0  # stop after N val checks with no improvement (0 = off)
    val_max_batches: Optional[int] = None  # cap val batches per eval (None = full set)

    # -------------------------------------------------------------------------
    # Refinement
    # -------------------------------------------------------------------------
    refinement_prob: float = 0.25  # probability of sampling small-t refinement steps
    refinement_max_t: int = 150    # upper bound for refinement timestep range

    # -------------------------------------------------------------------------
    # Img2img training
    # -------------------------------------------------------------------------
    img2img_prob: float = 0.0  # 0 = off; e.g. 0.2 to learn image-to-image editing

    # -------------------------------------------------------------------------
    # MDM — Masked Diffusion Models style training
    # Randomly mask latent patches; model learns to inpaint masked regions.
    # -------------------------------------------------------------------------
    mdm_mask_ratio: float = 0.0  # 0 = off; e.g. 0.2–0.5
    # Step-dependent mask ratio schedule: list of (t_step, mask_ratio) breakpoints.
    # Example: [(0, 0.05), (500, 0.25), (999, 0.35)]
    mdm_mask_schedule: Optional[List[tuple]] = None
    mdm_patch_size: int = 2         # must match DiT patch embed size (typically 2)
    mdm_loss_only_masked: bool = True
    mdm_min_mask_patches: int = 1   # minimum masked patches per sample

    # -------------------------------------------------------------------------
    # Mixture-of-Experts (MoE) FFN upgrade
    # -------------------------------------------------------------------------
    moe_num_experts: int = 0        # 0 = off; replaces dense FFN with sparse MoE
    moe_top_k: int = 2
    moe_balance_loss_weight: float = 0.0  # router load-balancing auxiliary loss

    allow_imperfect_output: bool = False

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------
    # Increase max caption length at these step thresholds.
    curriculum_caption_steps: Optional[List[int]] = None   # e.g. [5000, 15000, 30000]
    curriculum_max_lengths: Optional[List[int]] = None     # e.g. [77, 150, 300]
    # Difficulty curriculum: prefer easy/hard samples at different training stages.
    curriculum_difficulty_steps: Optional[List[int]] = None  # e.g. [0, 5000, 10000]
    curriculum_difficulty_easy_first: bool = True

    # -------------------------------------------------------------------------
    # Rule-based auxiliary loss
    # -------------------------------------------------------------------------
    rule_loss_weight: float = 0.0  # 0 = off

    # -------------------------------------------------------------------------
    # EMA
    # -------------------------------------------------------------------------
    ema_decay: float = 0.9999

    # -------------------------------------------------------------------------
    # Logging and checkpointing
    # -------------------------------------------------------------------------
    results_dir: str = "results"
    log_every: int = 50
    ckpt_every: int = 5000
    global_seed: int = 42
    resume: Optional[str] = None  # path to checkpoint to resume from
    wandb_project: Optional[str] = None    # e.g. "sdx" to enable WandB logging
    tensorboard_dir: Optional[str] = None  # e.g. "runs" to enable TensorBoard
    log_images_every: int = 0              # 0 = off; log a sample image every N steps
    log_images_prompt: str = "a photo of a cat"
    dry_run: bool = False          # run 1 step then exit (verify setup)
    save_run_manifest: bool = True # persist run_manifest.json + config.train.json
    strict_warnings: bool = False  # escalate project warnings to errors
    save_polyak: int = 0           # > 0: keep running avg of weights, save as polyak.pt

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    deterministic: bool = False
    latent_cache_dir: Optional[str] = None  # precomputed latents for faster training

    # -------------------------------------------------------------------------
    # Distributed (set by launcher, not CLI)
    # -------------------------------------------------------------------------
    local_rank: int = 0
    world_size: int = 1

    @property
    def latent_size(self) -> int:
        """Spatial size of VAE latents (image_size // 8)."""
        return self.image_size // 8

    @property
    def per_device_batch_size(self) -> int:
        """Effective per-GPU batch size."""
        return max(1, self.global_batch_size // self.world_size)


def get_dit_build_kwargs(cfg: object, *, class_dropout_prob: Optional[float] = None) -> dict:
    """Build the keyword-argument dict for DiT model constructors from a config object.

    Works with both ``TrainConfig`` instances and checkpoint config objects (any object
    supporting ``getattr``).

    Args:
        cfg: Configuration object.
        class_dropout_prob: Override caption dropout probability.
            ``None`` uses ``cfg.caption_dropout_prob`` (training default).
            Pass ``0.0`` for inference (no dropout).

    Returns:
        Dict of keyword arguments suitable for ``DiT_models_text[name](**kw)``.
    """
    latent_size = getattr(cfg, "image_size", 256) // 8
    te = getattr(cfg, "text_encoder", "google/t5-v1_1-xxl").lower()
    text_dim = 4096 if "xxl" in te else (1024 if "xl" in te and "xxl" not in te else 768)
    dropout = class_dropout_prob if class_dropout_prob is not None else getattr(cfg, "caption_dropout_prob", 0.1)
    model_name = str(getattr(cfg, "model_name", ""))
    include_moe = not model_name.startswith("EnhancedDiT")

    kw: dict = {
        "input_size": latent_size,
        "text_dim": text_dim,
        "class_dropout_prob": dropout,
        "num_ar_blocks": getattr(cfg, "num_ar_blocks", 0),
        "ar_block_order": str(getattr(cfg, "ar_block_order", "raster") or "raster"),
        "use_xformers": getattr(cfg, "use_xformers", True),
        "style_embed_dim": getattr(cfg, "style_embed_dim", 0),
        "control_cond_dim": getattr(cfg, "control_cond_dim", 0),
        "control_num_types": int(getattr(cfg, "control_num_types", 0) or 0),
        "creativity_embed_dim": getattr(cfg, "creativity_embed_dim", 0),
        "size_embed_dim": getattr(cfg, "size_embed_dim", 0),
        "patch_se": bool(getattr(cfg, "patch_se", False)),
        "patch_se_reduction": int(getattr(cfg, "patch_se_reduction", 8)),
    }

    repa_w = float(getattr(cfg, "repa_weight", 0.0))
    is_enhanced = model_name.startswith("EnhancedDiT")
    if not is_enhanced:
        kw["repa_out_dim"] = int(getattr(cfg, "repa_out_dim", 768)) if repa_w > 0 else 0
        kw["repa_projector_hidden_dim"] = int(getattr(cfg, "repa_projector_hidden_dim", 0)) if repa_w > 0 else 0

        ssm_every_n = int(getattr(cfg, "ssm_every_n", 0))
        kw["ssm_every_n"] = ssm_every_n if ssm_every_n > 0 else 0
        kw["ssm_kernel_size"] = int(getattr(cfg, "ssm_kernel_size", 7))

        kw["num_register_tokens"] = int(getattr(cfg, "num_register_tokens", 0))
        kw["use_rope"] = bool(getattr(cfg, "use_rope", False))
        kw["rope_base"] = float(getattr(cfg, "rope_base", 10000.0))
        kw["kv_merge_factor"] = int(getattr(cfg, "kv_merge_factor", 1))
        kw["token_routing_enabled"] = bool(getattr(cfg, "token_routing_enabled", False))
        kw["token_routing_strength"] = float(getattr(cfg, "token_routing_strength", 1.0))
        kw["token_keep_ratio"] = float(getattr(cfg, "token_keep_ratio", 1.0))
        kw["token_keep_min_value"] = float(getattr(cfg, "token_keep_min_value", 0.0))
        kw["drop_path_rate"] = float(getattr(cfg, "drop_path_rate", 0.0))
        kw["layerscale_init"] = float(getattr(cfg, "layerscale_init", 0.0))
        kw["qk_norm"] = bool(getattr(cfg, "qk_norm", False))
        kw["prompt_reinject_every_n"] = int(getattr(cfg, "prompt_reinject_every_n", 0))
        kw["prompt_reinject_alpha"] = float(getattr(cfg, "prompt_reinject_alpha", 0.0))
        kw["prompt_reinject_decay"] = float(getattr(cfg, "prompt_reinject_decay", 1.0))
        kw["prompt_timestep_schedule_enabled"] = bool(getattr(cfg, "prompt_timestep_schedule_enabled", False))
        kw["prompt_early_scale"] = float(getattr(cfg, "prompt_early_scale", 1.1))
        kw["prompt_late_scale"] = float(getattr(cfg, "prompt_late_scale", 1.0))
        kw["prompt_schedule_num_timesteps"] = int(getattr(cfg, "num_timesteps", 1000))

    if include_moe:
        kw["moe_num_experts"] = getattr(cfg, "moe_num_experts", 0)
        kw["moe_top_k"] = getattr(cfg, "moe_top_k", 2)

    return kw


# ---------------------------------------------------------------------------
# Convenience instance for quick scripts.
# Training CLIs should build TrainConfig from parsed args via train_args.py.
# ---------------------------------------------------------------------------
cfg = TrainConfig(
    model_name="EnhancedDiT-XL/2",
    image_size=512,
    global_batch_size=32,
    lr=5e-5,
    epochs=50,
    data_path="./data",
    num_workers=4,
    use_bf16=True,
    grad_checkpointing=True,
    grad_accum_steps=2,
    max_grad_norm=1.0,
    results_dir="./runs",
    log_every=25,
    ckpt_every=2500,
)
