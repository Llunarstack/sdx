"""Train CLI parser builder (extracted from train.py)."""

from __future__ import annotations

import argparse


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="", help="Image folder or manifest JSONL")
    parser.add_argument("--manifest-jsonl", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="DiT-XL/2-Text")
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="",
        help="T5 encoder path or HF id (empty = use pretrained/T5-XXL if present else google/t5-v1_1-xxl)",
    )
    parser.add_argument(
        "--text-encoder-mode",
        type=str,
        default="t5",
        choices=["t5", "triple"],
        help="t5=T5 only; triple=T5+CLIP-L+CLIP-bigG with trainable fusion (match downloaded model/ stack)",
    )
    parser.add_argument(
        "--clip-text-encoder-l",
        type=str,
        default="",
        help="CLIP-ViT-L/14 folder or HF id (triple mode; empty = default)",
    )
    parser.add_argument(
        "--clip-text-encoder-bigg",
        type=str,
        default="",
        help="CLIP-ViT-bigG/14 folder or HF id (triple mode; empty = default)",
    )
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Autoencoder model id/path (VAE=AutoencoderKL or RAE=AutoencoderRAE)",
    )
    parser.add_argument(
        "--autoencoder-type",
        type=str,
        default="kl",
        choices=["kl", "rae"],
        help="Autoencoder type: kl=AutoencoderKL, rae=AutoencoderRAE",
    )
    parser.add_argument(
        "--no-rae-latent-bridge",
        action="store_true",
        help="When using RAE with C!=4, error out instead of training RAELatentBridge",
    )
    parser.add_argument(
        "--rae-bridge-cycle-weight", type=float, default=0.01, help="Cycle loss weight for RAELatentBridge (0=off)"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--resolution-buckets",
        type=str,
        default="",
        help="IMPROVEMENTS §1.1: comma-separated sizes, e.g. 256,384,512 or 512x768,256x512 (single-GPU only; no val-split)",
    )
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--num-ar-blocks", type=int, default=0, help="Block-wise AR (0=off, 2 or 4)")
    parser.add_argument(
        "--ar-block-order",
        type=str,
        default="raster",
        choices=["raster", "zorder"],
        help="AR macro-block order: raster (row-major) or zorder (Morton). See docs/AR_EXTENSIONS.md",
    )
    parser.add_argument("--no-xformers", action="store_true", help="Disable xformers attention")
    parser.add_argument(
        "--passes",
        type=int,
        default=0,
        help="Train for N full passes over the dataset (recommended). Overrides epochs; use with cosine LR.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap steps when using --passes, or raw step limit when passes=0 (0=use epochs).",
    )
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Min LR for cosine schedule")
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--refinement-prob", type=float, default=0.25, help="Prob of training on fix-imperfection (small t)"
    )
    parser.add_argument("--refinement-max-t", type=int, default=150)
    parser.add_argument(
        "--img2img-prob", type=float, default=0.0, help="Img2img training: prob to use init_image as x_start (0=off)"
    )
    parser.add_argument(
        "--mdm-mask-ratio", type=float, default=0.0, help="MDM training: fraction of latent patches to mask (0=off)"
    )
    parser.add_argument(
        "--mdm-mask-schedule",
        type=str,
        default=None,
        help="MDM training: state-dependent mask ratio schedule as comma pairs: t_step,mask_ratio (e.g. 0,0.05,500,0.25,999,0.35)",
    )
    parser.add_argument(
        "--mdm-patch-size",
        type=int,
        default=2,
        help="MDM training: latent patch size (typically 2, matches DiT patch embed)",
    )
    parser.add_argument(
        "--mdm-min-mask-patches", type=int, default=1, help="MDM training: ensure at least N patches masked per sample"
    )
    parser.add_argument(
        "--no-mdm-loss-only-masked",
        action="store_true",
        help="MDM training: include unmasked regions in loss (default is masked-only)",
    )
    parser.add_argument("--moe-num-experts", type=int, default=0, help="MoE training: number of FFN experts (0=off)")
    parser.add_argument("--moe-top-k", type=int, default=2, help="MoE routing: top-k experts per token")
    parser.add_argument(
        "--moe-balance-loss-weight", type=float, default=0.0, help="MoE: auxiliary router balance loss weight (0=off)"
    )
    parser.add_argument("--no-save-best", action="store_true", help="Disable saving best checkpoint by loss")
    parser.add_argument(
        "--negative-prompt-weight", type=float, default=0.5, help="Weight for subtracting negative prompt"
    )
    parser.add_argument(
        "--style-embed-dim", type=int, default=0, help="Style conditioning (same as text_dim, e.g. 4096); 0=off"
    )
    parser.add_argument("--style-strength", type=float, default=0.7, help="Style blend strength in training (0.6-0.8)")
    parser.add_argument("--control-cond-dim", type=int, default=0, help="1=enable ControlNet; 0=off")
    parser.add_argument(
        "--control-num-types",
        type=int,
        default=0,
        help="Control type embedding count (0=off, e.g. 9 for unknown/canny/depth/pose/seg/lineart/scribble/normal/hed)",
    )
    parser.add_argument("--control-scale", type=float, default=0.85, help="ControlNet strength in training (0.7-1.0)")
    parser.add_argument(
        "--creativity-embed-dim", type=int, default=0, help="Creativity/diversity knob (0=off; e.g. 64)"
    )
    parser.add_argument("--creativity-max", type=float, default=1.0, help="Training: sample creativity in [0, this]")
    parser.add_argument(
        "--size-embed-dim",
        type=int,
        default=0,
        dest="size_embed_dim",
        help="PixArt-style latent (H,W) -> timestep embed dim (0=off; DiT still sees native res via pos embed)",
    )
    parser.add_argument(
        "--patch-se",
        action="store_true",
        dest="patch_se",
        help="Zero-init patch channel gate after patch embed (identity at init)",
    )
    parser.add_argument(
        "--patch-se-reduction",
        type=int,
        default=8,
        dest="patch_se_reduction",
        help="Bottleneck divisor for patch SE MLP",
    )
    parser.add_argument(
        "--curriculum-difficulty-steps",
        type=str,
        default=None,
        help="Comma-sep steps for difficulty curriculum (e.g. 0,5000,10000); use with JSONL 'difficulty' 0-1",
    )
    parser.add_argument(
        "--no-difficulty-easy-first", action="store_true", help="If set, late steps prefer easy (default: early=easy)"
    )
    parser.add_argument(
        "--rule-loss-weight", type=float, default=0.0, help="Constitutional/rule auxiliary loss weight (0=off)"
    )
    parser.add_argument("--repa-weight", type=float, default=0.0, help="REPA auxiliary loss weight (0=off)")
    parser.add_argument(
        "--repa-encoder-model",
        type=str,
        default="facebook/dinov2-base",
        help="Frozen vision encoder: dinov2* or clip* (HF id)",
    )
    parser.add_argument(
        "--repa-out-dim", type=int, default=768, help="Projection output dim; must match encoder embedding dim"
    )
    parser.add_argument(
        "--repa-projector-hidden-dim", type=int, default=0, help="REPA projector hidden dim (0=linear head)"
    )
    parser.add_argument(
        "--ssm-every-n",
        type=int,
        default=0,
        help="Replace every Nth self-attention block with SSM-like token mixer (0=off).",
    )
    parser.add_argument(
        "--ssm-kernel-size", type=int, default=7, help="SSM token mixer depthwise conv kernel size (odd >=3)."
    )
    parser.add_argument(
        "--num-register-tokens",
        type=int,
        default=0,
        help="Append N learnable register tokens to the patch token stream.",
    )
    parser.add_argument(
        "--use-rope", action="store_true", help="Enable RoPE (rotary positional embeddings) in self-attention."
    )
    parser.add_argument("--rope-base", type=float, default=10000.0, help="RoPE base frequency (theta).")
    parser.add_argument(
        "--kv-merge-factor",
        type=int,
        default=1,
        help="KV pooling factor for hierarchical patch merging in self-attention (1=off).",
    )
    parser.add_argument(
        "--token-routing-enabled", action="store_true", help="Enable soft per-token routing (gating) in DiT blocks."
    )
    parser.add_argument(
        "--token-routing-strength",
        type=float,
        default=1.0,
        help="Token routing strength in [0,1] (higher = more gating).",
    )
    parser.add_argument(
        "--token-keep-ratio",
        type=float,
        default=1.0,
        help="Top-k patch token keep ratio (1.0=off); compute-preserving adaptive token gating.",
    )
    parser.add_argument(
        "--token-keep-min-value",
        type=float,
        default=0.0,
        help="Gate floor for non-top-k patch tokens when --token-keep-ratio < 1.0.",
    )
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=0.0,
        help="Max stochastic-depth drop rate across DiT blocks (0=off).",
    )
    parser.add_argument(
        "--layerscale-init",
        type=float,
        default=0.0,
        help="LayerScale init (e.g. 1e-5). 0 disables LayerScale.",
    )
    parser.add_argument(
        "--qk-norm",
        action="store_true",
        help="Enable QK-norm (SD3.5-style per-head normalisation before attention). "
             "Improves stability at high resolution. Incompatible with checkpoints trained without it.",
    )
    parser.add_argument(
        "--lora-layers",
        type=str,
        default="all",
        choices=["all", "first", "middle", "last"],
        help="Restrict LoRA application to a layer group: all (default), first third, middle third, or last third.",
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "sigmoid", "squaredcos_cap_v2"],
    )
    parser.add_argument(
        "--prediction-type",
        type=str,
        default="epsilon",
        choices=["epsilon", "v", "x0"],
        help="epsilon=noise, v=velocity (SD2-style), x0=direct clean latent (train+sample must match)",
    )
    parser.add_argument("--noise-offset", type=float, default=0.0, help="SD-style noise offset (e.g. 0.1)")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0, help="Min-SNR loss weighting (0=off)")
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default="min_snr",
        choices=["min_snr", "min_snr_soft", "unit", "edm", "v", "eps"],
        help="Timestep loss weight: min_snr | min_snr_soft (smooth) | unit | edm | v | eps",
    )
    parser.add_argument(
        "--loss-weighting-sigma-data", type=float, default=0.5, help="Sigma_data for loss_weighting=edm"
    )
    parser.add_argument(
        "--spectral-sfp-loss",
        action="store_true",
        help="Frequency-weighted FFT loss (prototype): emphasize low freqs at high t, high freqs at low t. "
        "VP-DDPM only; not used with MDM masked loss.",
    )
    parser.add_argument(
        "--spectral-sfp-low-sigma",
        type=float,
        default=0.22,
        help="Radial falloff for low-frequency bin emphasis (see diffusion/spectral_sfp.py).",
    )
    parser.add_argument(
        "--spectral-sfp-high-sigma",
        type=float,
        default=0.22,
        help="Radial falloff for high-frequency bin emphasis.",
    )
    parser.add_argument(
        "--spectral-sfp-tau-power",
        type=float,
        default=1.0,
        help="Exponent on normalized timestep when blending low vs high frequency weights.",
    )
    parser.add_argument(
        "--ot-noise-pair-reg",
        type=float,
        default=0.0,
        help="Sinkhorn OT coupling between batch latents and Gaussian noise (0=off; try 0.03–0.1). Experimental.",
    )
    parser.add_argument(
        "--ot-noise-pair-iters",
        type=int,
        default=40,
        help="Sinkhorn iterations for --ot-noise-pair-reg.",
    )
    parser.add_argument(
        "--ot-noise-pair-mode",
        type=str,
        default="soft",
        choices=["soft", "hungarian"],
        help="soft = P @ noise (GPU); hungarian = min-cost permutation (CPU scipy).",
    )
    parser.add_argument(
        "--flow-matching-training",
        action="store_true",
        help="Train with rectified-flow-style velocity loss (incompatible with MDM masked training; not VP sample_loop-compatible).",
    )
    parser.add_argument(
        "--bridge-aux-weight",
        type=float,
        default=0.0,
        help="Add VP auxiliary loss on shuffle-paired latent mix (0=off). See diffusion/bridge_training.py.",
    )
    parser.add_argument(
        "--bridge-aux-lambda",
        type=float,
        default=0.2,
        help="Endpoint mix lambda for --bridge-aux-weight (0–1].",
    )
    parser.add_argument(
        "--timestep-sample-mode",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal", "high_noise"],
        help="Training t distribution: uniform (classic) | logit_normal (SD3-style discrete) | high_noise (Beta bias to large t)",
    )
    parser.add_argument(
        "--timestep-logit-mean",
        type=float,
        default=0.0,
        help="Gaussian mean on logit axis for --timestep-sample-mode logit_normal",
    )
    parser.add_argument(
        "--timestep-logit-std",
        type=float,
        default=1.0,
        help="Gaussian std on logit axis for --timestep-sample-mode logit_normal",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of data for validation (e.g. 0.05); 0=off. Enables best-by-val and early stopping.",
    )
    parser.add_argument(
        "--val-every", type=int, default=2000, help="Evaluate val loss every N steps (when val-split > 0)"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0, help="Stop after N val checks with no improvement; 0=off"
    )
    parser.add_argument(
        "--val-max-batches", type=int, default=None, help="Max val batches per eval (default: full val set)"
    )
    parser.add_argument("--deterministic", action="store_true", help="Reproducible training (worker seeds)")
    parser.add_argument("--latent-cache-dir", type=str, default=None, help="Use precomputed latents for faster training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--caption-dropout-schedule",
        type=str,
        default=None,
        help="Comma-sep pairs step,prob e.g. 0,0.2,10000,0.05 (decay caption dropout over training)",
    )
    parser.add_argument(
        "--crop-mode",
        type=str,
        default="center",
        choices=["center", "random", "largest_center"],
        help="Crop strategy for training images (1.2)",
    )
    parser.add_argument(
        "--region-caption-mode",
        type=str,
        default="append",
        choices=["append", "prefix", "off"],
        help="Merge JSONL parts/region_captions into training caption: append|prefix|off (see docs/REGION_CAPTIONS.md)",
    )
    parser.add_argument(
        "--region-layout-tag",
        type=str,
        default="[layout]",
        help="Tag before regional block in merged caption (empty string to omit)",
    )
    parser.add_argument(
        "--boost-adherence-caption",
        action="store_true",
        help="Prepend adherence tags to each training caption (literal prompt following; see caption_utils.prepend_adherence_boost)",
    )
    parser.add_argument(
        "--train-shortcomings-mitigation",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        dest="train_shortcomings_mitigation",
        help="Per-caption failure-mode hints (docs/COMMON_SHORTCOMINGS_AI_IMAGES.md); auto=keyword match, all=photoreal+digital+CG base pack",
    )
    parser.add_argument(
        "--train-shortcomings-2d",
        action="store_true",
        dest="train_shortcomings_2d",
        help="With --train-shortcomings-mitigation auto|all: include stylized 2D packs (anime/manga/cel)",
    )
    parser.add_argument(
        "--train-art-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        dest="train_art_guidance_mode",
        help="Artist-first medium guidance per training caption (traditional/digital/photography)",
    )
    parser.set_defaults(train_art_guidance_photography=True)
    parser.add_argument(
        "--no-train-art-guidance-photography",
        action="store_false",
        dest="train_art_guidance_photography",
        help="Disable photography packs for --train-art-guidance-mode auto|all",
    )
    parser.add_argument(
        "--train-anatomy-guidance",
        type=str,
        default="none",
        choices=["none", "lite", "strong"],
        dest="train_anatomy_guidance",
        help="Add anatomy/proportion constraints to training captions",
    )
    parser.add_argument(
        "--train-style-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        dest="train_style_guidance_mode",
        help="Style-domain guidance for training captions (anime/comic/concept/game/photo language)",
    )
    parser.set_defaults(train_style_guidance_artists=True)
    parser.add_argument(
        "--no-train-style-guidance-artists",
        action="store_false",
        dest="train_style_guidance_artists",
        help="Disable artist/game-name stabilization cues in train style guidance",
    )
    parser.add_argument(
        "--caption-unicode-normalize",
        action="store_true",
        help="NFKC + zero-width strip per caption segment before emphasis/boost (see sdx_native.text_hygiene)",
    )
    parser.add_argument(
        "--train-prompt-emphasis",
        action="store_true",
        help="Strip ( )/[ ] from captions for T5 (same as sample.py) and pass DiT token_weights (1.2 / 0.8); triple text appends two 1.0 weights for CLIP tokens",
    )
    parser.add_argument(
        "--attn-grounding-loss-weight",
        type=float,
        default=0.0,
        help="Aux loss: align DiT block-0 cross-attention with JSONL grounding_mask (VP-DDPM only; mixed masked/unmasked batches supported)",
    )
    parser.add_argument("--attn-grounding-token-start", type=int, default=0, help="Token slice start for grounding loss")
    parser.add_argument(
        "--attn-grounding-token-end",
        type=int,
        default=0,
        help="Token slice end for grounding loss (0 = all positions)",
    )
    parser.add_argument(
        "--attn-grounding-min-fg-patch-mass",
        type=float,
        default=1e-4,
        help="Skip rows whose downsampled grounding mask mass is below this threshold (0=off)",
    )
    parser.add_argument(
        "--attn-token-coverage-loss-weight",
        type=float,
        default=0.0,
        help="Aux loss: encourage each prompt token to receive sufficient cross-attention (0=off).",
    )
    parser.add_argument(
        "--attn-token-coverage-target",
        type=float,
        default=0.025,
        help="Target max patch-attention per token for --attn-token-coverage-loss-weight.",
    )
    parser.add_argument(
        "--prompt-reinject-every-n",
        type=int,
        default=0,
        help="Reinject base text conditioning every N DiT blocks (0=off).",
    )
    parser.add_argument(
        "--prompt-reinject-alpha",
        type=float,
        default=0.0,
        help="Strength of prompt reinjection residual (0=off).",
    )
    parser.add_argument(
        "--prompt-reinject-decay",
        type=float,
        default=1.0,
        help="Per-reinjection decay multiplier for prompt residual.",
    )
    parser.add_argument(
        "--prompt-timestep-schedule-enabled",
        action="store_true",
        help="Enable timestep-aware text scaling (stronger at high noise, softer at low noise).",
    )
    parser.add_argument("--prompt-early-scale", type=float, default=1.1, help="Text scale at high-noise timesteps.")
    parser.add_argument("--prompt-late-scale", type=float, default=1.0, help="Text scale at low-noise timesteps.")
    parser.add_argument(
        "--use-hierarchical-captions",
        action="store_true",
        help="Merge caption_global / caption_local / entity_captions from JSONL (see part_aware_training.py)",
    )
    parser.add_argument(
        "--hierarchical-caption-separator",
        type=str,
        default=" | ",
        help="Separator between merged caption segments",
    )
    parser.add_argument(
        "--hierarchical-caption-drop-global-p",
        type=float,
        default=0.0,
        help="Randomly drop first merged segment (global) with this probability",
    )
    parser.add_argument(
        "--hierarchical-caption-drop-local-p",
        type=float,
        default=0.0,
        help="Randomly drop last merged segment (local) with this probability",
    )
    parser.add_argument(
        "--foveated-train-prob",
        type=float,
        default=0.0,
        help="Probability of random zoom crop then resize to train resolution (skipped when init/control present)",
    )
    parser.add_argument(
        "--foveated-crop-frac",
        type=float,
        default=0.55,
        help="Short-side fraction for --foveated-train-prob crop",
    )
    parser.add_argument(
        "--grounding-mask-soft",
        action="store_true",
        help="Use grayscale mask values in [0,1] instead of thresholding at 0.5",
    )
    parser.add_argument(
        "--train-originality-prob",
        type=float,
        default=0.0,
        help="Per-sample prob to inject novelty phrases (like sample.py --originality); 0=off, try 0.1–0.25",
    )
    parser.add_argument(
        "--train-originality-strength",
        type=float,
        default=0.5,
        help="0–1: how many originality tokens to insert when augment triggers (default 0.5)",
    )
    parser.add_argument(
        "--creativity-jitter-std",
        type=float,
        default=0.0,
        help="Gaussian noise on training creativity scalar when --creativity-embed-dim > 0 (0=off; try 0.05–0.15)",
    )
    parser.add_argument(
        "--save-polyak",
        type=int,
        default=0,
        help="Running avg of last N steps; save as polyak.pt every ckpt-every (0=off)",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name (enables WandB logging)")
    parser.add_argument("--tensorboard-dir", type=str, default=None, help="TensorBoard log dir (enables TensorBoard)")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step and exit (verify setup)")
    parser.add_argument(
        "--no-save-run-manifest",
        action="store_true",
        help="Disable writing run_manifest.json/config.train.json in results dir.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat project UserWarning/FutureWarning as errors during training.",
    )
    parser.add_argument(
        "--log-images-every", type=int, default=0, help="Log a sample image to WandB/TB every N steps (0=off)"
    )
    parser.add_argument(
        "--log-images-prompt", type=str, default="a photo of a cat", help="Prompt for --log-images-every sample"
    )
    return parser

