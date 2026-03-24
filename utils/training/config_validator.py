"""
Configuration validation utilities for SDX training and inference.
"""

from pathlib import Path
from typing import List, Optional

import torch
from config.train_config import TrainConfig


class ConfigValidationError(Exception):
    """Configuration validation error."""

    pass


def validate_train_config(cfg: TrainConfig) -> List[str]:
    """Validate training configuration and return list of warnings/errors."""
    warnings = []
    errors = []

    # Data validation
    if not cfg.data_path and not cfg.manifest_jsonl:
        errors.append("Either data_path or manifest_jsonl must be specified")

    rcm = getattr(cfg, "region_caption_mode", "append")
    if rcm not in {"append", "prefix", "off"}:
        errors.append(f"region_caption_mode must be append|prefix|off, got: {rcm}")

    if cfg.data_path and not Path(cfg.data_path).exists():
        errors.append(f"Data path does not exist: {cfg.data_path}")

    if cfg.manifest_jsonl and not Path(cfg.manifest_jsonl).exists():
        errors.append(f"Manifest JSONL does not exist: {cfg.manifest_jsonl}")

    # Model validation
    from models import DiT_models_text

    if cfg.model_name not in DiT_models_text:
        errors.append(f"Unknown model: {cfg.model_name}. Available: {list(DiT_models_text.keys())}")

    # Autoencoder validation
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    if ae_type not in {"kl", "rae"}:
        errors.append(f"autoencoder_type must be 'kl' or 'rae', got: {ae_type}")

    # REPA validation
    repa_w = float(getattr(cfg, "repa_weight", 0.0))
    if repa_w < 0:
        errors.append(f"repa_weight must be >= 0, got: {repa_w}")
    repa_out_dim = int(getattr(cfg, "repa_out_dim", 768))
    if repa_out_dim <= 0:
        errors.append(f"repa_out_dim must be > 0, got: {repa_out_dim}")
    repa_proj_hidden = int(getattr(cfg, "repa_projector_hidden_dim", 0))
    if repa_proj_hidden < 0:
        errors.append(f"repa_projector_hidden_dim must be >= 0, got: {repa_proj_hidden}")

    # SSM swap validation (hybrid SSM token mixer)
    ssm_every_n = int(getattr(cfg, "ssm_every_n", 0))
    if ssm_every_n < 0:
        errors.append(f"ssm_every_n must be >= 0, got: {ssm_every_n}")
    ssm_kernel = int(getattr(cfg, "ssm_kernel_size", 7))
    if ssm_kernel < 3:
        errors.append(f"ssm_kernel_size must be >= 3, got: {ssm_kernel}")

    # ViT feature validations
    num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
    if num_register_tokens < 0:
        errors.append(f"num_register_tokens must be >= 0, got: {num_register_tokens}")

    kv_merge_factor = int(getattr(cfg, "kv_merge_factor", 1))
    if kv_merge_factor < 1:
        errors.append(f"kv_merge_factor must be >= 1, got: {kv_merge_factor}")

    token_routing_enabled = bool(getattr(cfg, "token_routing_enabled", False))
    if token_routing_enabled:
        token_routing_strength = float(getattr(cfg, "token_routing_strength", 1.0))
        if token_routing_strength <= 0:
            errors.append(f"token_routing_strength must be > 0, got: {token_routing_strength}")

    # Compatibility: our AR masks are defined over patch tokens only; register tokens and KV-merge
    # change the effective token/kv lengths, so we disallow with num_ar_blocks>0 for now.
    if getattr(cfg, "num_ar_blocks", 0) and int(getattr(cfg, "num_ar_blocks", 0)) > 0:
        if num_register_tokens > 0:
            errors.append("num_register_tokens is not supported together with num_ar_blocks>0 (AR mask mismatch).")
        if kv_merge_factor > 1:
            errors.append("kv_merge_factor>1 is not supported together with num_ar_blocks>0 (AR mask mismatch).")

    # Training length validation
    if cfg.passes <= 0 and cfg.max_steps <= 0 and cfg.epochs <= 0:
        errors.append("Must specify either passes > 0, max_steps > 0, or epochs > 0")

    if cfg.passes > 0 and cfg.max_steps > 0:
        warnings.append("Both passes and max_steps specified. max_steps will cap total training.")

    # Batch size validation
    if cfg.global_batch_size <= 0:
        errors.append("global_batch_size must be positive")

    if cfg.global_batch_size % cfg.world_size != 0:
        warnings.append(f"global_batch_size ({cfg.global_batch_size}) not divisible by world_size ({cfg.world_size})")

    # Learning rate validation
    if cfg.lr <= 0:
        errors.append("Learning rate must be positive")

    if cfg.min_lr >= cfg.lr:
        warnings.append("min_lr should be less than lr for cosine schedule")

    # Memory optimization warnings
    if cfg.image_size > 512 and not cfg.grad_checkpointing:
        warnings.append("Consider enabling grad_checkpointing for large image sizes")

    if cfg.global_batch_size > 64 and not cfg.use_bf16:
        warnings.append("Consider enabling bf16 for large batch sizes")

    # Validation split warnings
    if cfg.val_split > 0 and cfg.val_split >= 0.5:
        warnings.append("Validation split is very large (>= 50%)")

    if cfg.early_stopping_patience > 0 and cfg.val_split <= 0:
        warnings.append("early_stopping_patience requires val_split > 0")

    # AR blocks validation
    if cfg.num_ar_blocks not in [0, 2, 4]:
        errors.append("num_ar_blocks must be 0, 2, or 4")

    # Caption dropout schedule validation
    if cfg.caption_dropout_schedule:
        for step, prob in cfg.caption_dropout_schedule:
            if not (0 <= prob <= 1):
                errors.append(f"Caption dropout probability must be in [0,1], got {prob}")

    # Refinement validation
    if cfg.refinement_prob < 0 or cfg.refinement_prob > 1:
        errors.append("refinement_prob must be in [0,1]")

    if cfg.refinement_max_t >= cfg.num_timesteps:
        warnings.append("refinement_max_t should be less than num_timesteps")

    # MDM validation
    if getattr(cfg, "mdm_mask_ratio", 0.0) < 0 or getattr(cfg, "mdm_mask_ratio", 0.0) > 1:
        errors.append("mdm_mask_ratio must be in [0,1]")
    if getattr(cfg, "mdm_patch_size", 2) <= 0:
        errors.append("mdm_patch_size must be > 0")
    if getattr(cfg, "mdm_min_mask_patches", 1) < 0:
        errors.append("mdm_min_mask_patches must be >= 0")
    if getattr(cfg, "mdm_mask_schedule", None) is not None:
        for t_step, r in cfg.mdm_mask_schedule:
            if not (0 <= float(r) <= 1):
                errors.append(f"mdm_mask_schedule mask_ratio must be in [0,1], got {r}")
                break
            if int(t_step) < 0:
                errors.append(f"mdm_mask_schedule t_step must be >=0, got {t_step}")
                break
            if int(t_step) >= cfg.num_timesteps:
                warnings.append(
                    f"mdm_mask_schedule contains t_step={t_step} >= num_timesteps={cfg.num_timesteps} (it will be clamped)."
                )

    # MoE validation
    moe_num_experts = getattr(cfg, "moe_num_experts", 0)
    moe_top_k = getattr(cfg, "moe_top_k", 2)
    if moe_num_experts is not None and int(moe_num_experts) < 0:
        errors.append("moe_num_experts must be >= 0")
    if int(moe_top_k) <= 0:
        errors.append("moe_top_k must be > 0")
    if moe_num_experts is not None and int(moe_num_experts) > 0 and int(moe_top_k) > int(moe_num_experts):
        errors.append("moe_top_k cannot exceed moe_num_experts")
    if getattr(cfg, "moe_balance_loss_weight", 0.0) < 0:
        errors.append("moe_balance_loss_weight must be >= 0")

    # Hardware validation
    if not torch.cuda.is_available():
        errors.append("CUDA is required but not available")

    # Combine errors and warnings
    issues = []
    for error in errors:
        issues.append(f"ERROR: {error}")
    for warning in warnings:
        issues.append(f"WARNING: {warning}")

    return issues


def estimate_memory_usage(cfg: TrainConfig) -> dict:
    """Estimate GPU memory usage for training configuration."""
    # Rough estimates based on model size and batch size
    model_memory_map = {
        "DiT-B/2-Text": 1.2,  # GB
        "DiT-L/2-Text": 2.8,  # GB
        "DiT-XL/2-Text": 4.5,  # GB
        "DiT-P/2-Text": 5.2,  # GB
        "DiT-P-L/2-Text": 6.8,  # GB
        "DiT-Supreme/2-Text": 5.8,  # GB
        "DiT-Supreme-L/2-Text": 7.5,  # GB
    }

    base_model_memory = model_memory_map.get(cfg.model_name, 4.5)

    # EMA doubles model memory
    model_memory = base_model_memory * 2

    # Batch memory (rough estimate)
    batch_size_per_gpu = cfg.global_batch_size // cfg.world_size
    image_memory_per_sample = (cfg.image_size**2 * 3 * 4) / (1024**3)  # RGB float32
    latent_memory_per_sample = image_memory_per_sample / 64  # VAE compression ~8x8

    batch_memory = batch_size_per_gpu * (image_memory_per_sample + latent_memory_per_sample)

    # T5 encoder memory
    t5_memory = 2.5 if "xxl" in cfg.text_encoder.lower() else 1.0

    # VAE memory
    vae_memory = 0.5

    # Optimizer states (AdamW: 2x model params)
    optimizer_memory = base_model_memory * 2

    total_estimated = model_memory + batch_memory + t5_memory + vae_memory + optimizer_memory

    return {
        "model_memory_gb": model_memory,
        "batch_memory_gb": batch_memory,
        "t5_memory_gb": t5_memory,
        "vae_memory_gb": vae_memory,
        "optimizer_memory_gb": optimizer_memory,
        "total_estimated_gb": total_estimated,
        "recommended_vram_gb": total_estimated * 1.3,  # 30% buffer
    }


def suggest_optimizations(cfg: TrainConfig, available_vram_gb: Optional[float] = None) -> List[str]:
    """Suggest configuration optimizations based on available resources."""
    suggestions = []

    memory_est = estimate_memory_usage(cfg)

    if available_vram_gb and memory_est["total_estimated_gb"] > available_vram_gb:
        suggestions.append(
            f"Estimated memory usage ({memory_est['total_estimated_gb']:.1f}GB) exceeds available VRAM ({available_vram_gb}GB)"
        )
        suggestions.append("Consider: reducing global_batch_size, enabling grad_checkpointing, using smaller model")

    # Performance suggestions
    if not cfg.use_xformers:
        suggestions.append("Enable use_xformers for better memory efficiency")

    if not cfg.use_compile:
        suggestions.append("Enable use_compile for faster training (PyTorch 2.0+)")

    if cfg.num_workers < 4:
        suggestions.append("Consider increasing num_workers for faster data loading")

    # Training efficiency
    if cfg.grad_accum_steps == 1 and cfg.global_batch_size < 128:
        suggestions.append("Consider using gradient accumulation to increase effective batch size")

    if cfg.lr_warmup_steps < 100:
        suggestions.append("Consider longer warmup for stable training")

    return suggestions


def validate_inference_args(args) -> List[str]:
    """Validate inference arguments."""
    issues = []

    if not Path(args.ckpt).exists():
        issues.append(f"ERROR: Checkpoint not found: {args.ckpt}")

    if args.width <= 0 or args.height <= 0:
        issues.append("ERROR: Width and height must be positive")

    if args.width % 8 != 0 or args.height % 8 != 0:
        issues.append("WARNING: Width and height should be multiples of 8 for VAE")

    if args.steps <= 0:
        issues.append("ERROR: Number of steps must be positive")

    if args.cfg_scale < 1.0:
        issues.append("WARNING: CFG scale < 1.0 may produce poor results")

    if args.cfg_scale > 20.0:
        issues.append("WARNING: Very high CFG scale may cause artifacts")

    return issues
