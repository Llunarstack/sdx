# Training config: single place for hyperparameters (fast + good defaults).
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainConfig:
    # Data
    data_path: str = ""
    manifest_jsonl: Optional[str] = None
    image_size: int = 256
    num_workers: int = 8
    global_batch_size: int = 128
    caption_dropout_prob: float = 0.1
    # IMPROVEMENTS 1.3: caption dropout schedule (step -> prob); e.g. [(0, 0.2), (10000, 0.05)] = decay from 0.2 to 0.05
    caption_dropout_schedule: Optional[List[tuple]] = None  # list of (step, prob)
    # IMPROVEMENTS 1.2: crop mode for training images
    crop_mode: str = "center"  # "center" | "random" | "largest_center"

    # Model
    model_name: str = "DiT-XL/2-Text"
    text_encoder: str = "google/t5-v1_1-xxl"  # T5-XXL like PixArt/ReVe
    vae_model: str = "stabilityai/sd-vae-ft-mse"
    # Autoencoder type:
    # - "kl": AutoencoderKL (classic Stable Diffusion VAE)
    # - "rae": AutoencoderRAE (Representation Autoencoder; diffusion works on its encode()/decode() latents)
    autoencoder_type: str = "kl"
    latent_scale: float = 0.18215
    # When RAE latent channels != 4: train a 1x1 bridge (RAELatentBridge) to/from DiT latents (see models/rae_latent_bridge.py).
    rae_use_latent_bridge: bool = True
    rae_bridge_cycle_weight: float = 0.01  # Auxiliary cycle loss z ≈ to_rae(to_dit(z)); 0=off

    # --- REPA (Representation Alignment) ---
    # Optional auxiliary loss to align DiT internal features with a frozen vision encoder
    # (DINOv2/CLIP). This is the "fast hint" upgrade.
    repa_weight: float = 0.0  # 0=off
    # Recommended: one of:
    # - "facebook/dinov2-base" (768-dim)
    # - "openai/clip-vit-large-patch14" (768-dim)
    repa_encoder_model: str = "facebook/dinov2-base"
    # The model's projection head output dim (must match the frozen encoder embedding dim).
    # For common models: CLIP ViT-L/14 -> 768, DINOv2 base -> 768, DINOv2 large -> 1024.
    repa_out_dim: int = 768
    repa_projector_hidden_dim: int = 0  # 0=linear head

    # --- ViT-Gen / Hybrid SSM swap (token mixer) ---
    # Replace every Nth self-attention block with a lightweight SSM-like token mixer.
    ssm_every_n: int = 0  # 0=off
    ssm_kernel_size: int = 7

    # --- ViT-Gen / Elysium-Flow ViT features ---
    # 1) Register tokens ("sandwich" scratchpad, simplified: patches + N register tokens)
    num_register_tokens: int = 0
    # 2) RoPE (rotary positional embeddings) for self-attention (1D index RoPE implementation)
    use_rope: bool = False
    rope_base: float = 10000.0
    # 3) Hierarchical Patch Merging 2.0 (KV pooling factor for self-attention keys/values)
    #    kv_merge_factor=1 disables; e.g. 2 merges each 2x2 patch block into one KV token.
    kv_merge_factor: int = 1
    # 4) Cross-scale token routing (soft gating per token; does not change compute graph yet)
    token_routing_enabled: bool = False
    token_routing_strength: float = 1.0
    # Block-wise AR (ACDiT-style): 0 = full bidirectional; 2 = 2×2 blocks, 4 = 4×4 blocks (raster order).
    # See docs/AR.md for when to use AR and how it affects structure/fixability.
    num_ar_blocks: int = 0
    use_xformers: bool = True
    # Negative prompt: try really hard not to add those features
    negative_prompt_weight: float = 0.5
    # No reference image — model excels on dataset only.
    style_embed_dim: int = 0  # same as text_dim if style from T5; enables style conditioning
    style_strength: float = 0.7  # blend strength for style (0.6-0.8 recommended)
    control_cond_dim: int = 0  # 1 = enable ControlNet (control image); 0 = off
    control_scale: float = 0.85  # ControlNet strength (0.7-1.0 recommended)
    # Creativity/diversity knob (IMPROVEMENTS 8.7): 0 = off; else hidden dim for scalar conditioning (e.g. 64)
    creativity_embed_dim: int = 0
    creativity_max: float = 1.0  # Training: sample creativity in [0, creativity_max]
    size_embed_dim: int = 0  # PixArt-style (h, w) latent grid -> timestep conditioning; 0 = off (still can infer H,W from x)
    # Optional channel gate on patch tokens after embed (zero-init = identity at start).
    patch_se: bool = False
    patch_se_reduction: int = 8

    # Diffusion (SD/SDXL-style options)
    num_timesteps: int = 1000
    timestep_respacing: str = ""
    beta_schedule: str = "linear"  # "linear" or "cosine"
    prediction_type: str = "epsilon"  # "epsilon" or "v" (velocity, SD2-style)
    noise_offset: float = 0.0  # SD/SDXL: shift noise for better light/dark balance (e.g. 0.1)
    min_snr_gamma: float = 5.0  # Min-SNR weighting: cap SNR for loss (0 = off, 5 typical)
    # Alternative loss weighting (generative-models-style): "unit" | "edm" | "v" | "eps". If set, min_snr_gamma is ignored for timestep weight.
    loss_weighting: str = "min_snr"  # "min_snr" (default) | "unit" | "edm" | "v" | "eps"
    loss_weighting_sigma_data: float = 0.5  # For loss_weighting="edm"

    # Training length: prefer passes (N full passes over data), then max_steps, then epochs
    passes: int = 0  # If > 0, train for this many full passes over dataset (steps = passes * steps_per_epoch)
    max_steps: int = 0  # Cap when using passes; or raw step limit when passes==0 (0 = use epochs)
    epochs: int = 100  # Used only when passes==0 and max_steps==0
    lr: float = 1e-4
    min_lr: float = 1e-6  # Cosine schedule decays to this (avoids collapse)
    lr_warmup_steps: int = 500  # Linear warmup then cosine
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_bf16: bool = True
    use_compile: bool = True
    grad_accum_steps: int = 1
    grad_checkpointing: bool = True
    save_best: bool = True  # Save checkpoint when loss is best (so more steps = better saved model)

    # Validation + early stopping: avoid overtraining; "best" = best by val loss
    val_split: float = 0.0  # Fraction of data for validation (e.g. 0.05); 0 = off
    val_every: int = 2000  # Evaluate val loss every N steps (when val_split > 0)
    early_stopping_patience: int = 0  # Stop after this many val checks with no improvement; 0 = off
    val_max_batches: Optional[int] = None  # Cap val batches per eval (None = full val set)

    # Refinement: train model to fix imperfections during generation (unless user wants raw output)
    refinement_prob: float = 0.25  # Prob of training on "fix small problems" (small t)
    refinement_max_t: int = 150  # For refinement, t in [0, refinement_max_t]

    # Img2img training (FLUX/NoobAI/illust): when init_image in data, use it as x_start with this prob
    img2img_prob: float = 0.0  # 0 = off; e.g. 0.2 to learn image-to-image editing

    # MDM (Masked Diffusion Models)-style training:
    # Randomly mask latent patches, keep unmasked patches as x0 (clean context),
    # and compute denoising loss only on masked patches.
    # This trains the model to better "fill in blanks" during inference inpaint.
    mdm_mask_ratio: float = 0.0  # 0 = off; e.g. 0.2-0.5
    # Optional state-dependent schedule for mdm_mask_ratio.
    # Format: list of (t_step, mask_ratio) where t_step is an integer diffusion timestep index.
    # Example: [(0, 0.05), (500, 0.25), (999, 0.35)]
    mdm_mask_schedule: Optional[List[tuple]] = None
    mdm_patch_size: int = 2  # Latent patch size that corresponds to DiT patch embed (typically 2)
    mdm_loss_only_masked: bool = True  # If True: loss is averaged only over masked pixels
    mdm_min_mask_patches: int = 1  # Ensure each sample has at least N masked patches (avoid empty-mask)

    # Mixture-of-Experts (MoE) DiT upgrade (MLP-only MoE).
    # When moe_num_experts > 0, the FFN/MLP inside DiT blocks is replaced by a sparse MoE FFN.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_balance_loss_weight: float = 0.0  # if >0, adds router aux loss into training objective

    # Inference: allow_imperfect_output=True means skip refinement pass (user wants raw/fucked look)
    allow_imperfect_output: bool = False

    # Optional curriculum: increase max caption length at these step thresholds (e.g. [5000, 15000, 30000])
    curriculum_caption_steps: Optional[List[int]] = None
    curriculum_max_lengths: Optional[List[int]] = None  # e.g. [77, 150, 300]
    # Difficulty curriculum (IMPROVEMENTS 8.12): steps when to prefer easy vs hard (JSONL "difficulty" 0-1)
    curriculum_difficulty_steps: Optional[List[int]] = None  # e.g. [0, 5000, 10000]
    curriculum_difficulty_easy_first: bool = True  # True = early steps prefer low difficulty

    # Rule-based auxiliary loss (IMPROVEMENTS 8.2): weight for constitutional/rule loss (0 = off)
    rule_loss_weight: float = 0.0

    # EMA
    ema_decay: float = 0.9999

    # Log / ckpt
    results_dir: str = "results"
    log_every: int = 50
    ckpt_every: int = 5000
    global_seed: int = 42
    resume: Optional[str] = None  # Path to checkpoint to resume from
    # IMPROVEMENTS 5.1: optional WandB / TensorBoard
    wandb_project: Optional[str] = None  # e.g. "sdx" to enable WandB
    tensorboard_dir: Optional[str] = None  # e.g. "runs" to enable TensorBoard
    log_images_every: int = 0  # 0 = off; when > 0 and wandb/tb enabled, log a sample image every N steps
    log_images_prompt: str = "a photo of a cat"  # prompt used for log sample image
    # IMPROVEMENTS 1.5: Polyak (running average of last N steps); 0 = off
    save_polyak: int = 0  # if > 0, keep running avg of weights and save as polyak.pt every ckpt_every

    # Quality / reproducibility
    deterministic: bool = False  # Reproducible training (worker seeds, etc.)
    latent_cache_dir: Optional[str] = None  # Precomputed latents dir for faster training (optional)

    # Distributed (set by launcher)
    local_rank: int = 0
    world_size: int = 1

    @property
    def latent_size(self) -> int:
        return self.image_size // 8

    @property
    def per_device_batch_size(self) -> int:
        return max(1, self.global_batch_size // self.world_size)


def get_dit_build_kwargs(cfg, *, class_dropout_prob=None):
    """Single place for DiT build kwargs from config. Used by train.py, sample.py, inference.py, self_improve.py.
    cfg: TrainConfig or checkpoint config (any object with getattr).
    class_dropout_prob: None = use cfg.caption_dropout_prob (training); 0.0 for inference.
    """
    latent_size = getattr(cfg, "image_size", 256) // 8
    te = getattr(cfg, "text_encoder", "google/t5-v1_1-xxl").lower()
    text_dim = 4096 if "xxl" in te else (1024 if "xl" in te and "xxl" not in te else 768)
    dropout = class_dropout_prob if class_dropout_prob is not None else getattr(cfg, "caption_dropout_prob", 0.1)
    model_name = str(getattr(cfg, "model_name", ""))
    include_moe = not model_name.startswith("EnhancedDiT")

    kw = {
        "input_size": latent_size,
        "text_dim": text_dim,
        "class_dropout_prob": dropout,
        "num_ar_blocks": getattr(cfg, "num_ar_blocks", 0),
        "use_xformers": getattr(cfg, "use_xformers", True),
        "style_embed_dim": getattr(cfg, "style_embed_dim", 0),
        "control_cond_dim": getattr(cfg, "control_cond_dim", 0),
        "creativity_embed_dim": getattr(cfg, "creativity_embed_dim", 0),
        "size_embed_dim": getattr(cfg, "size_embed_dim", 0),
        "patch_se": bool(getattr(cfg, "patch_se", False)),
        "patch_se_reduction": int(getattr(cfg, "patch_se_reduction", 8)),
    }

    # REPA (Representation Alignment) upgrade:
    # If enabled, model constructors will create a small projector head.
    repa_w = float(getattr(cfg, "repa_weight", 0.0))
    is_enhanced = model_name.startswith("EnhancedDiT")
    if not is_enhanced:
        kw["repa_out_dim"] = int(getattr(cfg, "repa_out_dim", 768)) if repa_w > 0 else 0
        kw["repa_projector_hidden_dim"] = int(getattr(cfg, "repa_projector_hidden_dim", 0)) if repa_w > 0 else 0

        # SSM swap settings (token mixer)
        ssm_every_n = int(getattr(cfg, "ssm_every_n", 0))
        kw["ssm_every_n"] = ssm_every_n if ssm_every_n > 0 else 0
        kw["ssm_kernel_size"] = int(getattr(cfg, "ssm_kernel_size", 7))

        # ViT-Gen features
        kw["num_register_tokens"] = int(getattr(cfg, "num_register_tokens", 0))
        kw["use_rope"] = bool(getattr(cfg, "use_rope", False))
        kw["rope_base"] = float(getattr(cfg, "rope_base", 10000.0))
        kw["kv_merge_factor"] = int(getattr(cfg, "kv_merge_factor", 1))
        kw["token_routing_enabled"] = bool(getattr(cfg, "token_routing_enabled", False))
        kw["token_routing_strength"] = float(getattr(cfg, "token_routing_strength", 1.0))

    # MoE DiT upgrade: MLP-only MoE FFN.
    # Only pass MoE kwargs to models that accept them (skip EnhancedDiT).
    if include_moe:
        kw["moe_num_experts"] = getattr(cfg, "moe_num_experts", 0)
        kw["moe_top_k"] = getattr(cfg, "moe_top_k", 2)

    return kw


# Default configuration instance
cfg = TrainConfig(
    # Enhanced model settings
    model_name="EnhancedDiT-XL/2",
    image_size=512,
    global_batch_size=32,  # Smaller for 3B model
    lr=5e-5,  # Lower learning rate for large model
    epochs=50,
    
    # Data settings
    data_path="./data",
    num_workers=4,
    
    # Training settings
    use_bf16=True,
    grad_checkpointing=True,
    grad_accum_steps=2,
    max_grad_norm=1.0,
    
    # Logging
    results_dir="./enhanced_results",
    log_every=25,
    ckpt_every=2500,
    
    # Enhanced features (will be added by train script)
    # These will be set by the training script based on command line args
)
