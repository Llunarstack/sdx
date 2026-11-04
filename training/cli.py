"""``train.py`` command line: the argparse parser, its book/comic preset and the mapping to ``TrainConfig``.

Imports only argparse at module level, so ``train.py --help`` never loads torch.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.train_config import TrainConfig


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
        choices=["t5", "triple", "penta"],
        help="t5=T5 only; triple=T5+CLIP-L+CLIP-bigG; penta=T5+CLIP-L+bigG+H+LongCLIP-L with trainable fusion",
    )
    parser.add_argument(
        "--clip-text-encoder-l",
        type=str,
        default="",
        help="CLIP-ViT-L/14 folder or HF id (triple/penta; empty = default)",
    )
    parser.add_argument(
        "--clip-text-encoder-bigg",
        type=str,
        default="",
        help="CLIP-ViT-bigG/14 folder or HF id (triple/penta; empty = default)",
    )
    parser.add_argument(
        "--clip-text-encoder-h",
        type=str,
        default="",
        help="CLIP-ViT-H/14 folder or HF id (penta mode; empty = default)",
    )
    parser.add_argument(
        "--clip-text-encoder-long",
        type=str,
        default="",
        help="LongCLIP-L folder or HF id (penta mode; empty = default)",
    )
    parser.add_argument(
        "--vae-model",
        type=str,
        default="REPA-E/e2e-flux-vae",
        help="Autoencoder model id/path (VAE=AutoencoderKL or RAE=AutoencoderRAE). Default: REPA-E's 16-channel "
        "FLUX VAE. Latent channels / scale / shift come from its config. SD 1.x: stabilityai/sd-vae-ft-mse",
    )
    parser.add_argument(
        "--vae-subfolder", type=str, default="", help="Subfolder holding the VAE (e.g. 'vae' in a pipeline repo)"
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
        help="IMPROVEMENTS sec 1.1: comma-separated sizes, e.g. 256,384,512 or 512x768,256x512 (single-GPU only; no val-split)",
    )
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader workers (-1 = auto heuristic)")
    parser.add_argument(
        "--auto-num-workers",
        action="store_true",
        help="Pick num_workers from CPU count and dataset size (overrides --num-workers)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor per worker (0 = disable explicit prefetch)",
    )
    parser.add_argument(
        "--no-auto-num-workers",
        action="store_true",
        help="Use --num-workers literally instead of the CPU/dataset heuristic",
    )
    parser.add_argument(
        "--no-fused-adamw",
        action="store_true",
        help="Disable fused CUDA AdamW optimizer kernel",
    )
    parser.add_argument(
        "--no-batch-text-encode",
        action="store_true",
        help="Separate T5 forwards for negative/style (slower; debug only)",
    )
    parser.add_argument(
        "--no-cuda-stream-prefetch",
        action="store_true",
        help="Disable CUDA-stream batch prefetch (H2D/compute overlap)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode when compile is enabled",
    )
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
        choices=["raster", "zorder", "snake", "spiral"],
        help="AR macro-block order: raster, zorder (Morton), snake (boustrophedon), spiral. See docs/AR_EXTENSIONS.md",
    )
    parser.add_argument(
        "--ar-curriculum-mode",
        type=str,
        default="none",
        choices=["none", "step", "linear"],
        help="Runtime AR block curriculum: none=fixed, step=warmup switch, linear=ramp.",
    )
    parser.add_argument(
        "--ar-curriculum-warmup-steps",
        type=int,
        default=0,
        help="Warmup for AR curriculum (used by step mode; fallback ramp length for linear).",
    )
    parser.add_argument(
        "--ar-curriculum-ramp-start",
        type=int,
        default=0,
        help="Linear curriculum ramp start step.",
    )
    parser.add_argument(
        "--ar-curriculum-ramp-end",
        type=int,
        default=0,
        help="Linear curriculum ramp end step (<=0 uses warmup length fallback).",
    )
    parser.add_argument(
        "--ar-curriculum-start-blocks",
        type=int,
        default=-1,
        choices=[-1, 0, 2, 4],
        help="Start AR blocks for curriculum (-1 defaults to 0).",
    )
    parser.add_argument(
        "--ar-curriculum-target-blocks",
        type=int,
        default=-1,
        choices=[-1, 0, 2, 4],
        help="Target AR blocks for curriculum (-1 defaults to --num-ar-blocks).",
    )
    parser.add_argument(
        "--ar-order-mix",
        type=str,
        default="",
        help="Comma list of per-step AR traversal orders (subset of raster,zorder,snake,spiral).",
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
        "--edit-conditioning",
        action="store_true",
        help="Train an instruction-edit model: manifest rows with source_image (or init_image) + an instruction "
        "caption condition on the source image's tokens (FLUX Kontext style). Fine-tune with --init-from.",
    )
    parser.add_argument(
        "--edit-source-dropout",
        type=float,
        default=0.1,
        help="With --edit-conditioning: fraction of steps trained without the source image (keeps text-to-image)",
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
    parser.add_argument("--repa-weight", type=float, default=0.0, help="REPA auxiliary loss weight (0=off)")
    parser.add_argument(
        "--repa-encoder-model",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Frozen vision encoder: dinov3* / dinov2* / clip* (HF id)",
    )
    parser.add_argument(
        "--repa-out-dim",
        type=int,
        default=1024,
        help="Projection output dim; must match encoder embedding dim (DINOv3-L=1024)",
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
        "--lora-train",
        action="store_true",
        help="Train LoRA/DoRA adapters only (freeze the base DiT). Saves small adapter-only checkpoints "
        "compatible with `sample.py --lora`.",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA/DoRA rank (with --lora-train).")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA/DoRA alpha scaling (with --lora-train).")
    parser.add_argument(
        "--lora-dora",
        action="store_true",
        help="Use DoRA (weight-decomposed LoRA) instead of plain LoRA when --lora-train is set.",
    )
    parser.add_argument(
        "--lora-target",
        type=str,
        default="",
        help="Comma-separated module-name substrings to wrap with adapters (default: attention + MLP linears).",
    )
    parser.add_argument(
        "--live-dashboard",
        action="store_true",
        help="Show a real-time terminal dashboard (progress, loss trend, throughput, ETA, GPU) during training.",
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "sigmoid", "squaredcos_cap_v2"],
    )
    parser.add_argument(
        "--zero-terminal-snr",
        action="store_true",
        help="Rescale the beta schedule to zero terminal SNR (Lin et al. 2024): the last step is pure noise, so "
        "the model can make true blacks/whites and very dark or bright images. Requires --prediction-type v (or x0) "
        "and a sampler schedule that starts at t=T-1 (e.g. euler/karras_rho, not the leading 'ddim' spacing).",
    )
    parser.add_argument(
        "--prediction-type",
        type=str,
        default="epsilon",
        choices=["epsilon", "v", "x0"],
        help="epsilon=noise, v=velocity (SD2-style), x0=direct clean latent (train+sample must match)",
    )
    parser.add_argument(
        "--noise-offset",
        type=float,
        default=0.0,
        help="Per-channel noise offset (e.g. 0.05-0.1): lets the model shift overall brightness/colour; VP and flow.",
    )
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=5.0,
        help="Min-SNR gamma (Hang et al. 2023), converted to the --prediction-type target (0=off)",
    )
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default="min_snr",
        choices=["min_snr", "min_snr_soft", "unit", "edm", "v", "eps"],
        help="Timestep loss weight (defined on the x0 error, converted to the prediction target): "
        "min_snr | min_snr_soft (smooth) | unit | edm | v | eps",
    )
    parser.add_argument(
        "--loss-weighting-sigma-data", type=float, default=0.5, help="Sigma_data for loss_weighting=edm"
    )
    parser.add_argument(
        "--spectral-sfp-loss",
        action="store_true",
        help="Frequency-weighted FFT loss: emphasize low freqs at high t, high freqs at low t (same scale as MSE). "
        "VP and flow matching; not used with MDM masked loss.",
    )
    parser.add_argument(
        "--spectral-sfp-low-sigma",
        type=float,
        default=0.22,
        help="Radial falloff for low-frequency bin emphasis (see diffusion/flow_matching.py).",
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
        help="Minibatch OT pairing of latents and noise rows (0=off). For soft mode the entropic regulariser in units "
        "of the batch cost std: 0.05 ~ exact OT, >=1 ~ random pairing.",
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
        help="soft = permutation sampled from the Sinkhorn plan; hungarian = exact min-cost permutation (scipy).",
    )
    parser.add_argument(
        "--flow-matching-training",
        action="store_true",
        help="Train with the rectified-flow velocity loss (sample with --flow-matching-sample). Honors "
        "--timestep-sample-mode (logit_normal = SD3), --flow-shift, --noise-offset and --spectral-sfp-loss. "
        "Incompatible with MDM masked training.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=1.0,
        help="SD3 timestep shift for flow training at the native --image-size grid; other resolution buckets scale "
        "it by sqrt(tokens / native tokens). Stored in the checkpoint and reused by sample.py (e.g. 3.0 at 1024px).",
    )
    parser.add_argument(
        "--bridge-aux-weight",
        type=float,
        default=0.0,
        help="Add VP auxiliary loss on shuffle-paired latent mix (0=off). See diffusion/flow_matching.py.",
    )
    parser.add_argument(
        "--bridge-aux-lambda",
        type=float,
        default=0.2,
        help="Endpoint mix lambda for --bridge-aux-weight (0-1].",
    )
    parser.add_argument(
        "--timestep-sample-mode",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal", "high_noise", "low_noise"],
        help="Training t distribution (VP and flow): uniform | logit_normal (SD3-style) | high_noise (Beta->large t) "
        "| low_noise (Beta->small t)",
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
    parser.add_argument(
        "--timestep-curriculum-schedule",
        type=str,
        default="",
        help="Override t-distribution by global step: segments 'start:mode|start:mode:mu:sig' "
        "(see utils/training/objectives.py). Example: 0:high_noise|40000:logit_normal:0:1|80000:low_noise",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Load weights from checkpoint (same as resume) but ignore optimizer/steps  -  use for fine-tuning "
        "(e.g. book/comic) from a general SDX checkpoint. Mutually exclusive with --resume.",
    )
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
    parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="Disable cuDNN autotune (slower convs; shapes already stable)",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 for matmul/cuDNN on Ampere+ (stricter FP32, slower)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="torch.set_num_threads(N); 0=default. Lower if DataLoader workers starve.",
    )
    parser.add_argument(
        "--cpu-interop-threads",
        type=int,
        default=0,
        help="torch.set_num_interop_threads(N); 0=default.",
    )
    parser.add_argument(
        "--latent-cache-dir", type=str, default=None, help="Use precomputed latents for faster training"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--caption-dropout-prob",
        type=float,
        default=0.1,
        help="Per-sample probability of replacing the caption by the empty prompt (the CFG uncond).",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay cap; the decay warms up as min(cap, (1+n)/(10+n)) over optimizer steps n.",
    )
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
        help="Merge JSONL parts/region_captions into training caption: append|prefix|off (see docs/PROMPT_STACK.md)",
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
        help="Strip ( )/[ ] from captions for T5 (same as sample.py) and pass DiT token_weights (1.2 / 0.8); triple/penta append 1.0 weights for fused CLIP tokens",
    )
    parser.add_argument(
        "--attn-grounding-loss-weight",
        type=float,
        default=0.0,
        help="Aux loss: align DiT block-0 cross-attention with JSONL grounding_mask (VP-DDPM only; mixed masked/unmasked batches supported)",
    )
    parser.add_argument(
        "--attn-grounding-token-start", type=int, default=0, help="Token slice start for grounding loss"
    )
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
        help="Per-sample prob to inject novelty phrases (like sample.py --originality); 0=off, try 0.1-0.25",
    )
    parser.add_argument(
        "--train-originality-strength",
        type=float,
        default=0.5,
        help="0-1: how many originality tokens to insert when augment triggers (default 0.5)",
    )
    parser.add_argument(
        "--creativity-jitter-std",
        type=float,
        default=0.0,
        help="Gaussian noise on training creativity scalar when --creativity-embed-dim > 0 (0=off; try 0.05-0.15)",
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
    parser.add_argument(
        "--book-train-preset",
        type=str,
        default="",
        choices=["", "fast", "balanced", "production"],
        help=(
            "Book/comic/manga training bundle on the same DiT-Text stack (AR, hierarchical captions, "
            "guidance packs). Empty=off. Same checkpoint family as sample.py  -  not a separate model type."
        ),
    )
    parser.add_argument(
        "--book-ar-profile",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "layout",
            "strong",
            "zorder",
            "vit_layout",
            "vit_strong",
            "comic_snake",
            "cinema_spiral",
        ],
        help="When --book-train-preset is set: one-flag AR block layout (see pipelines/book_comic/README.md).",
    )
    return parser


def parse_caption_dropout_schedule(s: str | None):
    """Parse '0,0.2,10000,0.05' -> [(0, 0.2), (10000, 0.05)]. Returns None if s is None/empty."""
    if not s or not str(s).strip():
        return None
    values = [value.strip() for value in str(s).split(",") if value.strip()]
    if len(values) % 2 != 0:
        return None
    schedule = []
    for index in range(0, len(values), 2):
        schedule.append((int(values[index]), float(values[index + 1])))
    return schedule if schedule else None


def parse_resolution_buckets(s: str | None):
    """Parse ``256,384`` or ``512x768,256x512`` into list[(H, W)] or None."""
    if not s or not str(s).strip():
        return None
    buckets = []
    for raw_bucket in str(s).split(","):
        bucket_value = raw_bucket.strip().lower()
        if not bucket_value:
            continue
        if "x" in bucket_value:
            height_str, width_str = bucket_value.split("x", 1)
            buckets.append((int(height_str.strip()), int(width_str.strip())))
        else:
            edge = int(bucket_value)
            buckets.append((edge, edge))
    return buckets or None


def parse_mdm_mask_schedule(s: str | None):
    """Parse '0,0.05,500,0.25' -> [(0,0.05),(500,0.25)] or None."""
    if not s or not str(s).strip():
        return None
    values = [value.strip() for value in str(s).split(",") if value.strip()]
    if len(values) % 2 != 0:
        return None
    schedule = []
    for index in range(0, len(values), 2):
        schedule.append((int(float(values[index])), float(values[index + 1])))
    return schedule if schedule else None


def with_zero_terminal_snr(beta_schedule: str, enabled: bool) -> str:
    """``--zero-terminal-snr`` as the ``_ztsnr`` schedule suffix, so every consumer of ``cfg.beta_schedule`` (sampling,
    DPO, GRPO, distillation) rebuilds the same rescaled schedule from the checkpoint."""
    from diffusion.schedules import ZTSNR_SUFFIX

    if enabled and not beta_schedule.endswith(ZTSNR_SUFFIX):
        return beta_schedule + ZTSNR_SUFFIX
    return beta_schedule


def build_train_config_from_args(args) -> TrainConfig:
    """Build TrainConfig from argparse namespace (train.py CLI)."""
    from config.train_config import TrainConfig
    from utils.modeling.model_paths import default_t5_path

    return TrainConfig(
        data_path=args.data_path,
        manifest_jsonl=args.manifest_jsonl,
        results_dir=args.results_dir,
        model_name=args.model,
        text_encoder=(args.text_encoder or default_t5_path()),
        text_encoder_mode=str(getattr(args, "text_encoder_mode", "t5")),
        clip_text_encoder_l=str(getattr(args, "clip_text_encoder_l", "") or ""),
        clip_text_encoder_bigg=str(getattr(args, "clip_text_encoder_bigg", "") or ""),
        clip_text_encoder_h=str(getattr(args, "clip_text_encoder_h", "") or ""),
        clip_text_encoder_long=str(getattr(args, "clip_text_encoder_long", "") or ""),
        image_size=args.image_size,
        resolution_buckets=parse_resolution_buckets(args.resolution_buckets or None),
        vae_model=args.vae_model,
        vae_subfolder=getattr(args, "vae_subfolder", ""),
        autoencoder_type=args.autoencoder_type,
        rae_use_latent_bridge=not getattr(args, "no_rae_latent_bridge", False),
        rae_bridge_cycle_weight=float(getattr(args, "rae_bridge_cycle_weight", 0.01)),
        global_batch_size=args.global_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_train=bool(getattr(args, "lora_train", False)),
        lora_rank=int(getattr(args, "lora_rank", 16)),
        lora_alpha=float(getattr(args, "lora_alpha", 16.0)),
        lora_dora=bool(getattr(args, "lora_dora", False)),
        lora_target=str(getattr(args, "lora_target", "") or ""),
        live_dashboard=bool(getattr(args, "live_dashboard", False)),
        num_workers=args.num_workers,
        prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
        cuda_stream_prefetch=not bool(getattr(args, "no_cuda_stream_prefetch", False)),
        auto_num_workers=(
            not bool(getattr(args, "no_auto_num_workers", False)) or bool(getattr(args, "auto_num_workers", False))
        ),
        fused_adamw=not bool(getattr(args, "no_fused_adamw", False)),
        batch_text_encode=not bool(getattr(args, "no_batch_text_encode", False)),
        compile_mode=str(getattr(args, "compile_mode", "reduce-overhead") or "reduce-overhead"),
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
        edit_conditioning=bool(getattr(args, "edit_conditioning", False)),
        edit_source_dropout=float(getattr(args, "edit_source_dropout", 0.1)),
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
        beta_schedule=with_zero_terminal_snr(args.beta_schedule, bool(getattr(args, "zero_terminal_snr", False))),
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
        flow_shift=float(getattr(args, "flow_shift", 1.0)),
        bridge_aux_weight=float(getattr(args, "bridge_aux_weight", 0.0)),
        bridge_aux_lambda=float(getattr(args, "bridge_aux_lambda", 0.2)),
        timestep_sample_mode=getattr(args, "timestep_sample_mode", "uniform"),
        timestep_logit_mean=getattr(args, "timestep_logit_mean", 0.0),
        timestep_logit_std=getattr(args, "timestep_logit_std", 1.0),
        timestep_curriculum_schedule=str(getattr(args, "timestep_curriculum_schedule", "") or ""),
        resume=args.resume,
        init_from=getattr(args, "init_from", None),
        val_split=args.val_split,
        val_every=args.val_every,
        early_stopping_patience=args.early_stopping_patience,
        val_max_batches=args.val_max_batches,
        deterministic=args.deterministic,
        cudnn_benchmark=not bool(getattr(args, "no_cudnn_benchmark", False)),
        enable_tf32=not bool(getattr(args, "no_tf32", False)),
        torch_cpu_num_threads=int(getattr(args, "cpu_threads", 0) or 0),
        torch_cpu_num_interop_threads=int(getattr(args, "cpu_interop_threads", 0) or 0),
        latent_cache_dir=args.latent_cache_dir,
        caption_dropout_prob=float(getattr(args, "caption_dropout_prob", 0.1)),
        caption_dropout_schedule=parse_caption_dropout_schedule(getattr(args, "caption_dropout_schedule", None)),
        ema_decay=float(getattr(args, "ema_decay", 0.9999)),
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


def apply_book_train_preset_to_args(args: Any) -> None:
    """
    When ``--book-train-preset`` is set, fill unset train flags from book presets.

    Book training uses the **same** ``train.py`` / DiT-*-Text stack as general image gen;
    only defaults (AR blocks, caption hierarchy, guidance packs) differ.
    """
    preset_name = str(getattr(args, "book_train_preset", "") or "").strip().lower()
    if not preset_name:
        return

    from pipelines.book_comic.training import resolve_book_train_settings

    book_ar = str(getattr(args, "book_ar_profile", "") or "").strip()
    if book_ar and not str(getattr(args, "ar_profile", "") or "").strip():
        args.ar_profile = book_ar

    settings = resolve_book_train_settings(args)

    args.model = settings.model
    args.image_size = settings.image_size
    args.global_batch_size = settings.global_batch_size
    args.lr = settings.lr
    args.passes = settings.passes
    if settings.max_steps > 0:
        args.max_steps = settings.max_steps

    args.train_shortcomings_mitigation = settings.train_shortcomings_mitigation
    args.train_shortcomings_2d = settings.train_shortcomings_2d
    args.train_art_guidance_mode = settings.train_art_guidance_mode
    args.train_anatomy_guidance = settings.train_anatomy_guidance
    args.train_style_guidance_mode = settings.train_style_guidance_mode
    args.region_caption_mode = settings.region_caption_mode
    args.num_ar_blocks = settings.num_ar_blocks
    args.ar_block_order = settings.ar_block_order
    args.use_hierarchical_captions = settings.use_hierarchical_captions
    args.boost_adherence_caption = settings.boost_adherence_caption
