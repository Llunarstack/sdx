"""
Generate an image from a text prompt using a trained checkpoint.
Supports: prompt, negative prompt, steps, width, height, CFG, timestep schedules (ddim, euler, karras_rho, …) and solvers (ddim, heun).
Optional: style, control-image, lora, img2img, inpainting, dual-stage layout (coarse latent then detail pass),
hires-fix (latent upscale + refine), volatile CFG (spike-aware guidance), CLIP-guard extra denoise,
CLIP monitor (mid-loop CFG boost on low cosine), spectral-coherence latent (FFT lowfreq blend),
domain latent prior, sharpen, contrast, saturation, clarity / tone-punch / chroma-smooth / polish /
finishing-preset (cross-style post), emphasis (word)/[word].

Presets and OP modes:
- --preset sdxl|flux|anime|zit: apply a model-style preset from config.defaults.model_presets.
- --op-mode portrait|fullbody|anime_char: apply a high-level OP bundle on top.

Profiling (optional): pass ``--profile-out PATH`` (plus ``--profile-sort cumulative|tottime|...``,
``--profile-top N``) to write cProfile ``.prof`` and a text summary next to PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_sample_parser() -> "argparse.ArgumentParser":
    """Build the sample.py CLI parser. Kept import-light so ``--help``
    works without importing the heavy GPU stack (test_cli_entrypoints.py)."""
    parser = argparse.ArgumentParser(
        description="Generate image: prompt, negative prompt, steps, width, height, CFG, and scheduler."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (e.g. results/.../best.pt)")
    parser.add_argument(
        "--prompt", type=str, default="", help="Positive prompt (optional if --tags or --tags-file provided)"
    )
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt (what to avoid)")
    parser.add_argument(
        "--out", type=str, default="output.png", help="Output image path (with --num N: stem_0.png, stem_1.png, ...)"
    )
    parser.add_argument(
        "--num", type=int, default=1, help="Number of images to generate (batch); saved as out_0.png, out_1.png, ..."
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--width", type=int, default=0, help="Output width (0 = use model image_size)")
    parser.add_argument("--height", type=int, default=0, help="Output height (0 = use model image_size)")
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="stretch",
        choices=["stretch", "center_crop", "saliency_crop"],
        help="When --width/--height differ from model native: stretch (default), center crop+resize, or saliency crop+resize.",
    )
    parser.add_argument(
        "--resize-saliency-face-bias",
        type=float,
        default=0.0,
        help="Extra face priority for --resize-mode saliency_crop (0 disables face boost).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    # Optional (kept; only CFG/sampler/scheduler removed)
    parser.add_argument("--style", type=str, default="", help="Style prompt (e.g. oil painting, artist name)")
    parser.add_argument("--style-strength", type=float, default=0.7, help="Style blend strength")
    parser.add_argument(
        "--auto-style-from-prompt",
        action="store_true",
        help="Extract style/artist from prompt when --style empty (e.g. 'by X', 'style of X', artist tags)",
    )
    parser.add_argument("--control-image", type=str, default="", help="Path to control image (depth/edge/pose)")
    parser.add_argument(
        "--control",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Stack multiple controls: path, path:scale, path:type, path:type:scale, or path:scale:type. "
            "Example: --control canny.png:canny:0.8 depth.png:depth:0.6"
        ),
    )
    parser.add_argument(
        "--control-type",
        type=str,
        default="auto",
        help="Control type: auto|unknown|canny|depth|pose|seg|lineart|scribble|normal|hed",
    )
    parser.add_argument("--control-scale", type=float, default=0.85, help="ControlNet strength")
    parser.add_argument(
        "--control-guidance-start",
        type=float,
        default=0.0,
        help="Control schedule start as denoise progress fraction (0.0 = first step).",
    )
    parser.add_argument(
        "--control-guidance-end",
        type=float,
        default=1.0,
        help="Control schedule end as denoise progress fraction (1.0 = last step).",
    )
    parser.add_argument(
        "--control-guidance-decay",
        type=float,
        default=1.0,
        help="Control decay power in [start,end]: 1=linear, >1 faster fade, <1 slower fade.",
    )
    parser.add_argument(
        "--holy-grail",
        action="store_true",
        help="Enable holy-grail adaptive guidance stack (CFG/control scheduling + optional condition annealing/refine).",
    )
    parser.add_argument(
        "--holy-grail-cfg-early-ratio",
        type=float,
        default=0.72,
        help="Holy-grail CFG multiplier ratio at first denoise step.",
    )
    parser.add_argument(
        "--holy-grail-cfg-late-ratio",
        type=float,
        default=1.0,
        help="Holy-grail CFG multiplier ratio at final denoise step.",
    )
    parser.add_argument(
        "--holy-grail-control-mult",
        type=float,
        default=1.0,
        help="Holy-grail multiplier for control scaling policy.",
    )
    parser.add_argument(
        "--holy-grail-adapter-mult",
        type=float,
        default=1.0,
        help="Holy-grail multiplier for adapter scaling policy.",
    )
    parser.add_argument(
        "--holy-grail-no-frontload-control",
        action="store_true",
        help="Disable holy-grail control frontloading (use flatter control schedule).",
    )
    parser.add_argument(
        "--holy-grail-late-adapter-boost",
        type=float,
        default=1.15,
        help="Late-step boost factor for adapter scale in holy-grail policy.",
    )
    parser.add_argument(
        "--holy-grail-cads-strength",
        type=float,
        default=0.0,
        help="CADS-style condition noise strength for prompt embeddings (0=off).",
    )
    parser.add_argument(
        "--holy-grail-cads-min-strength",
        type=float,
        default=0.0,
        help="Minimum CADS condition-noise strength near final steps.",
    )
    parser.add_argument(
        "--holy-grail-cads-power",
        type=float,
        default=1.0,
        help="Power for CADS decay curve (higher => faster late-stage decay).",
    )
    parser.add_argument(
        "--holy-grail-unsharp-sigma",
        type=float,
        default=0.0,
        help="Final latent unsharp blur sigma for holy-grail refine (0=off).",
    )
    parser.add_argument(
        "--holy-grail-unsharp-amount",
        type=float,
        default=0.0,
        help="Final latent unsharp amount for holy-grail refine (0=off).",
    )
    parser.add_argument(
        "--holy-grail-clamp-quantile",
        type=float,
        default=0.0,
        help="Final latent dynamic percentile clamp quantile in [0,1] (0=off).",
    )
    parser.add_argument(
        "--holy-grail-clamp-floor",
        type=float,
        default=1.0,
        help="Lower bound for holy-grail dynamic clamp scale.",
    )
    parser.add_argument(
        "--lora",
        type=str,
        nargs="*",
        default=[],
        help=(
            "LoRA/DoRA/LyCORIS specs: path, path:scale, or path:scale:role "
            "(role examples: character/style/detail/composition)."
        ),
    )
    parser.add_argument(
        "--no-lora-normalize-scales",
        action="store_true",
        help="Disable per-layer multi-LoRA scale normalization (enabled by default for style stability).",
    )
    parser.add_argument(
        "--lora-max-total-scale",
        type=float,
        default=1.5,
        help="Max total absolute adapter scale per layer when stacking LoRA/DoRA/LyCORIS.",
    )
    parser.add_argument(
        "--lora-default-role",
        type=str,
        default="style",
        help="Default adapter role when --lora spec has no :role suffix.",
    )
    parser.add_argument(
        "--lora-role-budgets",
        type=str,
        default="character=1.8,style=1.0,detail=0.8,composition=1.0,other=0.8",
        help="Per-role scale caps used before global cap, e.g. 'character=1.8,style=1.0,detail=0.8'.",
    )
    parser.add_argument(
        "--lora-stage-policy",
        type=str,
        default="auto",
        choices=["off", "auto", "character_focus", "style_focus", "balanced"],
        help="Depth-aware role routing policy for stacked adapters (early/mid/late layer weighting).",
    )
    parser.add_argument(
        "--lora-layers",
        type=str,
        default="all",
        choices=["all", "first", "middle", "last"],
        help=(
            "Restrict LoRA application to a layer group: all (default), first third (structure/layout), "
            "middle third (fine detail), or last third (aesthetics/style)."
        ),
    )
    parser.add_argument(
        "--lora-role-stage-weights",
        type=str,
        default="",
        help=("Override per-role early/mid/late multipliers, e.g. 'character=1.15/1.0/0.85,style=0.9/1.0/1.1'."),
    )
    parser.add_argument(
        "--lora-trigger",
        type=str,
        default="",
        help="Trigger word(s) to prepend to prompt when using LoRAs (e.g. style or character trigger)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated tags; prepended to prompt with subject-first order (Danbooru-style)",
    )
    parser.add_argument(
        "--tags-file",
        type=str,
        default="",
        help="Path to file with tags (one per line or comma-separated); used like --tags",
    )
    parser.add_argument("--init-image", type=str, default="", help="Img2img: path to initial image")
    parser.add_argument("--strength", type=float, default=0.75, help="Img2img strength 0-1")
    parser.add_argument("--init-latent", type=str, default="", help="Start from saved latent .pt (from-z)")
    parser.add_argument("--mask", type=str, default="", help="Inpainting: path to mask (white=inpaint)")
    parser.add_argument(
        "--dissect-refs",
        type=str,
        default="",
        help=(
            "Comma-separated reference image paths for prompt-driven part extraction/compositing. "
            "Example: --dissect-refs \"ref1.png,ref2.png\" and prompt 'use the hat from image 1 and background from image 2'."
        ),
    )
    parser.add_argument(
        "--auto-init-from-dissection",
        action="store_true",
        help="If set (and no --init-image/--mask provided), auto-build init+mask from --dissect-refs + prompt dissection.",
    )
    parser.add_argument(
        "--dissection-lock-background",
        action="store_true",
        help="When background is requested from a reference image, preserve it (mask black everywhere).",
    )
    parser.add_argument(
        "--inpaint-mode",
        type=str,
        default="legacy",
        choices=["legacy", "mdm"],
        help="Inpainting behavior: legacy (old hack) or mdm (freeze known regions each step).",
    )
    parser.add_argument(
        "--sharpen", type=float, default=0.0, help="Post-process: unsharp strength 0-1 (0=off; needs scipy)"
    )
    parser.add_argument("--contrast", type=float, default=1.0, help="Post-process: contrast factor (1=off)")
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="Post-process: color saturation (1=off; 1.05–1.15 adds pop via PIL Color enhance)",
    )
    parser.add_argument(
        "--clarity",
        type=float,
        default=0.0,
        help="Post-process: luminance-only unsharp (0–1; sharper micro-detail, fewer RGB halos; needs scipy).",
    )
    parser.add_argument(
        "--tone-punch",
        type=float,
        default=0.0,
        help="Post-process: gentle S-curve on luminance only (0–0.35; depth without crushing color).",
    )
    parser.add_argument(
        "--chroma-smooth",
        type=float,
        default=0.0,
        help="Post-process: light chroma blur to calm noise in flats/skin/cel fills (0–0.45).",
    )
    parser.add_argument(
        "--polish",
        type=float,
        default=0.0,
        help="Post-process: one-knob combo (S-curve + chroma smooth + luma clarity + tiny grain); 0.4–0.65 typical.",
    )
    parser.add_argument(
        "--finishing-preset",
        type=str,
        default="none",
        choices=["none", "photo", "anime", "illustration", "characters", "painterly"],
        help="Adds baseline clarity/tone/chroma-smooth amounts on top of explicit flags (style-aware defaults).",
    )
    # Artistic post-processing (compositional director, value structure, asymmetry, SSS, etc.)
    parser.add_argument(
        "--composition-guide",
        type=str,
        default="none",
        choices=["none", "rule_of_thirds", "golden_ratio", "dynamic_symmetry"],
        help=(
            "Nudge visual weight toward compositional guide points. "
            "Counteracts AI center-bias. 'rule_of_thirds' is the most natural starting point."
        ),
    )
    parser.add_argument(
        "--composition-guide-strength",
        type=float,
        default=0.15,
        help="Strength of compositional guide nudge (0.1-0.25 is subtle; default 0.15).",
    )
    parser.add_argument(
        "--value-structure",
        action="store_true",
        help="Enforce value discipline: lift shadows, roll off highlights, boost midtone separation.",
    )
    parser.add_argument(
        "--value-shadow-lift",
        type=float,
        default=0.0,
        help="Raise the black point slightly (0.0-0.12) to prevent crushed shadows.",
    )
    parser.add_argument(
        "--value-highlight-roll",
        type=float,
        default=0.0,
        help="Compress highlights (0.0-0.12) to prevent blown-out whites.",
    )
    parser.add_argument(
        "--value-midtone-contrast",
        type=float,
        default=0.0,
        help="Boost midtone separation (0.0-0.25) for more visual depth.",
    )
    parser.add_argument(
        "--asymmetry",
        type=float,
        default=0.0,
        help=(
            "Introduce subtle organic asymmetry to break AI's perfect bilateral symmetry "
            "(uncanny valley fix). 0.1-0.35 is subliminal; above 0.5 becomes visible."
        ),
    )
    parser.add_argument(
        "--lost-found-edges",
        type=float,
        default=0.0,
        help=(
            "Vary edge sharpness to mimic human mark-making ('lost and found' edges). "
            "0.2-0.45 is natural; above 0.6 is painterly."
        ),
    )
    parser.add_argument(
        "--sss",
        type=float,
        default=0.0,
        help=(
            "Simulate subsurface scattering for skin/wax/translucent materials. "
            "0.15-0.35 for subtle skin; 0.5-0.7 for wax/candle. Needs scipy."
        ),
    )
    parser.add_argument(
        "--sss-radius",
        type=float,
        default=3.0,
        help="SSS blur radius in pixels (2-6 typical; default 3.0).",
    )
    parser.add_argument(
        "--chromatic-aberration",
        type=float,
        default=0.0,
        help="Subtle lens chromatic aberration (0.1-0.3 is natural; adds lens character).",
    )
    parser.add_argument(
        "--vignette",
        type=float,
        default=0.0,
        help="Radial vignette strength (0.15-0.4 is natural; frames the composition).",
    )
    parser.add_argument(
        "--micro-detail",
        type=float,
        default=0.0,
        help="Luminance-only micro-detail recovery (0.2-0.5; no RGB halos unlike sharpen).",
    )
    parser.add_argument(
        "--face-enhance",
        action="store_true",
        help="Post-process: OpenCV Haar frontal-face detection + local sharpen/contrast (needs opencv-python, scipy).",
    )
    parser.add_argument(
        "--face-enhance-sharpen",
        type=float,
        default=0.35,
        help="Unsharp strength on detected face patches when --face-enhance.",
    )
    parser.add_argument(
        "--face-enhance-contrast",
        type=float,
        default=1.04,
        help="Micro-contrast factor on face patches (1.0 = off).",
    )
    parser.add_argument(
        "--face-enhance-padding",
        type=float,
        default=0.25,
        help="Expand each face bbox by this fraction of max(w,h).",
    )
    parser.add_argument(
        "--face-enhance-max",
        type=int,
        default=4,
        help="Maximum faces to enhance per output image.",
    )
    parser.add_argument(
        "--post-reference-image",
        type=str,
        default="",
        help="Optional reference image: whole-frame linear RGB blend (weak color/style pull; not identity lock).",
    )
    parser.add_argument(
        "--post-reference-alpha",
        type=float,
        default=0.0,
        help="Blend weight 0–0.5 for --post-reference-image (0 = off).",
    )
    parser.add_argument(
        "--face-restore-shell",
        type=str,
        default="",
        help="After final save, run via shell; substitute {src} and {dst} with the output PNG path (e.g. GFPGAN/ADetailer CLI).",
    )
    parser.add_argument(
        "--creativity",
        type=float,
        default=None,
        help="Creativity/diversity 0-1 (only if model was trained with --creativity-embed-dim)",
    )
    parser.add_argument(
        "--creativity-jitter",
        type=float,
        default=0.0,
        help="Std dev of Gaussian noise added to creativity per image (0-1); use with --num >1 for varied batches",
    )
    parser.add_argument(
        "--originality",
        type=float,
        default=0.0,
        help="0-1; inject novelty tokens and tune sampling/creativity for less templated results",
    )
    parser.add_argument(
        "--save-attn",
        type=str,
        default="",
        help="Save cross-attention weights to path (e.g. attn.pt) for explanation/heatmap",
    )
    parser.add_argument("--no-refine", action="store_true", help="Disable refinement pass (raw/imperfect look, faster)")
    parser.add_argument(
        "--refine-t", type=int, default=50, help="Refinement noise level t (small t fixes imperfections; e.g. 50)"
    )
    parser.add_argument(
        "--refine-gate",
        type=str,
        default="off",
        choices=["off", "auto"],
        help="Run refinement only when quick quality score is below threshold.",
    )
    parser.add_argument(
        "--refine-gate-threshold",
        type=float,
        default=0.62,
        help="Threshold for --refine-gate auto (higher => refine less often).",
    )
    parser.add_argument(
        "--hires-fix",
        action="store_true",
        help="After main sample: bicubic upscale latent then short denoise (A1111-style). Best with SD KL VAE; "
        "needs variable-res DiT or size_embed. Skipped for RAE, img2img, from-z, inpaint.",
    )
    parser.add_argument(
        "--hires-scale",
        type=float,
        default=1.5,
        help="When --hires-fix and no --width/--height: target side = round(image_size * this).",
    )
    parser.add_argument(
        "--hires-steps",
        type=int,
        default=15,
        help="Denoising steps for the hires latent pass.",
    )
    parser.add_argument(
        "--hires-strength",
        type=float,
        default=0.35,
        help="Noise level 0–1 for hires pass (forward noise from upscaled latent); ~0.3–0.5 typical.",
    )
    parser.add_argument(
        "--hires-cfg-scale",
        type=float,
        default=-1.0,
        help="CFG during hires pass; <0 means use same as --cfg-scale.",
    )
    parser.add_argument(
        "--dynamic-threshold-percentile",
        type=float,
        default=0.0,
        help="If > 0, clamp x0 to this percentile (e.g. 99.5); use with --dynamic-threshold-type percentile",
    )
    parser.add_argument(
        "--dynamic-threshold-type",
        type=str,
        default="percentile",
        choices=["percentile", "norm", "spatial_norm"],
        help="x0 thresholding: percentile | norm | spatial_norm (ControlNet-style)",
    )
    parser.add_argument(
        "--dynamic-threshold-value",
        type=float,
        default=0.0,
        help="For norm/spatial_norm: min norm (e.g. 1.0); ignored for percentile",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (lower 3-5 = softer, 7-10 = stronger; use with --cfg-rescale if oversaturated)",
    )
    parser.add_argument(
        "--cfg-rescale", type=float, default=0.0, help="ComfyUI-style CFG rescale to reduce oversaturation (e.g. 0.7)"
    )
    # AdaGen (adaptive sampling) early-exit: stop when latent deltas get small.
    parser.add_argument(
        "--ada-early-exit",
        action="store_true",
        help="Enable AdaGen-style early exit during sampling (faster, may slightly reduce detail).",
    )
    parser.add_argument(
        "--ada-exit-delta-threshold",
        type=float,
        default=1e-3,
        help="Early-exit threshold for average latent delta magnitude.",
    )
    parser.add_argument(
        "--ada-exit-patience", type=int, default=3, help="Number of consecutive steps below threshold before exiting."
    )
    parser.add_argument(
        "--ada-exit-min-steps", type=int, default=5, help="Minimum sampling steps before early-exit is allowed."
    )
    # PBFM-style guidance (lightweight edge/high-pass drift in latent update)
    parser.add_argument(
        "--pbfm-edge-boost", type=float, default=0.0, help="PBFM heuristic: add high-pass drift to x0_pred (0=off)."
    )
    parser.add_argument("--pbfm-edge-kernel", type=int, default=3, help="PBFM high-pass kernel size (odd >=3).")
    parser.add_argument(
        "--reference-image",
        type=str,
        default="",
        help="Path to reference image: CLIP vision -> extra cross-attn tokens (IP-Adapter-style; projector is untrained unless --reference-adapter-pt).",
    )
    parser.add_argument(
        "--reference-strength",
        type=float,
        default=1.0,
        help="Scale injected reference tokens (0 disables even if --reference-image is set).",
    )
    parser.add_argument("--reference-tokens", type=int, default=4, help="Number of reference tokens to inject.")
    parser.add_argument(
        "--reference-clip-model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face model id for CLIP vision encoding of --reference-image.",
    )
    parser.add_argument(
        "--reference-adapter-pt",
        type=str,
        default="",
        help="Optional .pt state_dict for ReferenceTokenProjector (train separately; strict=False load).",
    )
    parser.add_argument(
        "--sag-blur-sigma",
        type=float,
        default=0.0,
        help="Blur-based self-attention guidance: Gaussian blur sigma in latent pixels (0=off; try 0.35-0.9).",
    )
    parser.add_argument(
        "--sag-scale",
        type=float,
        default=0.0,
        help="SAG heuristic strength: pred += scale*(pred-pred_on_blurred_latent). Typical 0.12-0.35; ~2× sampling cost.",
    )
    parser.add_argument(
        "--volatile-cfg-boost",
        type=float,
        default=0.0,
        help="When latent update spikes vs recent steps, multiply CFG on following steps by (1+this). "
        "Inference-only heuristic (AdaBlock-style idea); try 0.08–0.18.",
    )
    parser.add_argument(
        "--volatile-cfg-quantile",
        type=float,
        default=0.72,
        help="Quantile of recent latent deltas; above it counts as a spike (with --volatile-cfg-boost > 0).",
    )
    parser.add_argument(
        "--volatile-cfg-window",
        type=int,
        default=6,
        help="Rolling window length for volatile CFG heuristic (>=2).",
    )
    parser.add_argument(
        "--dual-stage-layout",
        action="store_true",
        help="Layout-first: denoise at lower latent res, upscale, then short high-res pass (KL VAE, no img2img/inpaint).",
    )
    parser.add_argument(
        "--dual-stage-div",
        type=int,
        default=2,
        help="Latent side divisor for layout stage (2 => half spatial resolution).",
    )
    parser.add_argument("--dual-layout-steps", type=int, default=24, help="Denoising steps for coarse layout stage.")
    parser.add_argument("--dual-detail-steps", type=int, default=20, help="Denoising steps after latent upscale.")
    parser.add_argument(
        "--dual-detail-strength",
        type=float,
        default=0.38,
        help="Noise level 0–1 when re-noising upscaled latent before detail stage.",
    )
    parser.add_argument(
        "--clip-guard-threshold",
        type=float,
        default=0.0,
        help="If >0: decode preview, CLIP cosine vs prompt; below threshold run short extra denoise (needs transformers). Try 0.20–0.28.",
    )
    parser.add_argument(
        "--clip-guard-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HF CLIP model id for --clip-guard-threshold.",
    )
    parser.add_argument(
        "--clip-guard-t-frac",
        type=float,
        default=0.22,
        help="Timestep fraction for CLIP-guard re-noising before refine loop.",
    )
    parser.add_argument("--clip-guard-steps", type=int, default=12, help="Steps for CLIP-guard extra sample_loop.")
    parser.add_argument(
        "--clip-monitor-every",
        type=int,
        default=0,
        help="If >0: decode x0_pred every N denoise steps, CLIP cosine vs prompt; below --clip-monitor-threshold "
        "multiply CFG by (1 + --clip-monitor-cfg-boost). Very slow (uses --clip-guard-model). 0=off.",
    )
    parser.add_argument(
        "--clip-monitor-threshold",
        type=float,
        default=0.22,
        help="CLIP cosine threshold for --clip-monitor-every (same scale as --clip-guard-threshold; try 0.18–0.28).",
    )
    parser.add_argument(
        "--clip-monitor-cfg-boost",
        type=float,
        default=0.12,
        help="CFG multiplicative boost when CLIP cosine drops below --clip-monitor-threshold (only with --clip-monitor-every > 0).",
    )
    parser.add_argument(
        "--clip-monitor-rewind",
        type=float,
        default=0.0,
        help="If >0 and --clip-monitor-every >0: soft-rewind latent when CLIP cosine drops below threshold. "
        "Applies x = (1-s)*x + s*x_prev (0–1). Try 0.15–0.4. Costs no extra forwards.",
    )
    parser.add_argument(
        "--speculative-draft-cfg-scale",
        type=float,
        default=0.0,
        help="Experimental: two CFG forwards (draft at this scale, then full). 0=off. Needs classifier-free uncond kwargs.",
    )
    parser.add_argument(
        "--speculative-close-thresh",
        type=float,
        default=0.0,
        help="If >0: when mean |full_pred - draft_pred| is below this, blend toward draft (see --speculative-blend).",
    )
    parser.add_argument(
        "--speculative-blend",
        type=float,
        default=0.35,
        help="Blend weight toward draft when close (0–1). Only used when --speculative-close-thresh > 0.",
    )
    parser.add_argument(
        "--flow-matching-sample",
        action="store_true",
        help="Rectified-flow Euler/Heun sampler (matches --flow-matching-training). Auto-on if checkpoint was flow-trained.",
    )
    parser.add_argument(
        "--force-vp-sample",
        action="store_true",
        help="Use VP DDIM sampler even when checkpoint has flow_matching_training (debug / wrong ckpt).",
    )
    parser.add_argument(
        "--flow-solver",
        type=str,
        default="euler",
        metavar="NAME",
        help="Flow ODE solver: euler/heun canonical; aliases rk2/edm_heun/etc. "
        "(see diffusion.gaussian_diffusion.canonicalize_flow_solver).",
    )
    parser.add_argument(
        "--domain-prior-latent",
        type=float,
        default=0.0,
        help="Latent high-frequency emphasis before decode (0=off; try 0.03–0.08).",
    )
    parser.add_argument(
        "--spectral-coherence-latent",
        type=float,
        default=0.0,
        help="FFT low-frequency blend on final latent before decode (0=off; try 0.05–0.2). See inference_research_hooks.spectral_latent_lowfreq_blend.",
    )
    parser.add_argument(
        "--spectral-coherence-cutoff",
        type=float,
        default=0.15,
        help="Normalized radial cutoff for --spectral-coherence-latent (smaller = tighter low-pass).",
    )
    # Test-time scaling: generate N candidates (--num) and keep the best (see IMPROVEMENTS.md)
    parser.add_argument(
        "--pick-best",
        type=str,
        default="none",
        choices=[
            "auto",
            "none",
            "clip",
            "edge",
            "ocr",
            "vit",
            "aesthetic",
            "combo",
            "combo_vit",
            "combo_vit_hq",
            "combo_vit_realism",
            "combo_count_vit",
            "combo_exposure",
            "combo_structural",
            "combo_hq",
            "combo_count",
            "combo_realism",
            "aesthetic_realism",
            "superior_composite",
        ],
        help="With --num > 1, score candidates; see IMPROVEMENTS.md. Includes aesthetic, aesthetic_realism, combo_vit_*, superior_composite.",
    )
    parser.add_argument(
        "--local-rag-jsonl",
        type=str,
        default="",
        help="JSONL corpus for local TF-IDF RAG (utils/superior/retrieval.py); merges top facts into prompt before encode.",
    )
    parser.add_argument(
        "--local-rag-top-k",
        type=int,
        default=8,
        help="Max facts retrieved from --local-rag-jsonl.",
    )
    parser.add_argument(
        "--superior-self-correct",
        action="store_true",
        help="After sampling, CLIP-gate a short refine pass when alignment score is low (see utils/superior/self_correct.py).",
    )
    parser.add_argument(
        "--expand-prompt",
        action="store_true",
        help="Heuristic prompt expansion before encode (utils/superior/prompt_expand.py).",
    )
    parser.add_argument(
        "--fdg-cfg-strength",
        type=float,
        default=0.0,
        help="Frequency-decoupled CFG blend (0=standard CFG, 1=full FDG; see utils/superior/frequency_cfg.py).",
    )
    parser.add_argument(
        "--fdg-cutoff-frac",
        type=float,
        default=0.15,
        help="Radial FFT cutoff for --fdg-cfg-strength (low vs high freq split).",
    )
    parser.add_argument(
        "--feature-cache-delta",
        type=float,
        default=0.0,
        help="Reuse DiT prediction when latent mean delta < threshold (SpeCa-lite; 0=off).",
    )
    parser.add_argument(
        "--feature-cache-max-reuse",
        type=int,
        default=2,
        help="Max consecutive feature-cache reuses per sample.",
    )
    parser.add_argument(
        "--block-cache-thresh",
        type=float,
        default=0.0,
        help="Block-wise DiT cache (BWCache-lite; 0=off, 0.15–0.25 typical).",
    )
    parser.add_argument(
        "--block-cache-recompute-every",
        type=int,
        default=4,
        help="Force full DiT block recompute every N denoise steps when block cache is on.",
    )
    parser.add_argument(
        "--taylor-cache",
        action="store_true",
        help="Use TaylorSeer forecast for block cache (ICCV 2025; needs --block-cache-thresh).",
    )
    parser.add_argument(
        "--taylor-cache-order",
        type=int,
        default=1,
        help="Taylor expansion order for --taylor-cache (0=reuse, 1=linear forecast).",
    )
    parser.add_argument(
        "--rcfgpp-tangent",
        type=float,
        default=0.0,
        help="Rectified-CFG++ tangent norm cap on flow/CFG delta (0=off, 0.85 typical).",
    )
    parser.add_argument(
        "--apg-parallel-eta",
        type=float,
        default=-1.0,
        help="Adaptive Projected Guidance: parallel component weight (0=remove oversaturation, 1=CFG). "
        "<0 disables. Mutually preferred over --rcfgpp-tangent when both set; FDG takes priority if --fdg-cfg-strength>0.",
    )
    parser.add_argument(
        "--zeresfdg-strength",
        type=float,
        default=0.0,
        help="ZeResFDG unified guidance (FDG+zero-projection+energy rescale). 0=off, 1=full (CADE 2.5).",
    )
    parser.add_argument(
        "--cfg-zero-star",
        action="store_true",
        help="CFG-Zero* for flow matching: optimized scale + zero-init early steps (arXiv:2503.18886).",
    )
    parser.add_argument(
        "--cfg-zero-init-frac",
        type=float,
        default=0.04,
        help="Fraction of ODE steps to zero when --cfg-zero-star (default 4%%).",
    )
    parser.add_argument(
        "--qsilk-micrograin",
        type=float,
        default=0.0,
        help="QSilk micrograin latent stabilizer strength at end of sampling (0=off, 0.12 typical).",
    )
    parser.add_argument(
        "--dynamic-dit-width",
        action="store_true",
        help="DyDiT-style timestep dynamic width: scale early-step predictions (training-free).",
    )
    parser.add_argument(
        "--dynamic-dit-early",
        type=float,
        default=0.88,
        help="Early-step width multiplier when --dynamic-dit-width (default 0.88).",
    )
    parser.add_argument(
        "--dynamic-sdt",
        action="store_true",
        help="Spatial dynamic tokens: attenuate updates on low-importance latent regions.",
    )
    parser.add_argument(
        "--apg-momentum-beta",
        type=float,
        default=0.0,
        help="APG reverse momentum across steps (0=off, 0.2 typical; needs --apg-parallel-eta>=0).",
    )
    parser.add_argument(
        "--cfg-pp-lambda",
        type=float,
        default=0.0,
        help="CFG++ manifold guidance strength in [0,1] (0=off, 0.55 typical; ICLR 2025).",
    )
    parser.add_argument(
        "--cfg-skip-early-frac",
        type=float,
        default=0.0,
        help="Skip CFG for first fraction of denoise steps (e.g. 0.15).",
    )
    parser.add_argument(
        "--cfg-skip-late-frac",
        type=float,
        default=0.0,
        help="Skip CFG for last fraction of denoise steps (e.g. 0.1).",
    )
    parser.add_argument(
        "--linear-attn-fraction",
        type=float,
        default=0.0,
        help="Blend linear attention into DiT blocks (0=off, 0.25 experimental SLA scaffold).",
    )
    parser.add_argument(
        "--tcfg-damping",
        type=float,
        default=0.0,
        help="TCFG tangential damping on uncond branch (0=off, 1=full; CVPR 2025).",
    )
    parser.add_argument(
        "--slg-scale",
        type=float,
        default=0.0,
        help="Skip Layer Guidance scale (0=off, 2.8 typical; extra cond forward).",
    )
    parser.add_argument(
        "--slg-skip-blocks",
        type=str,
        default="auto",
        help="Block indices to skip for SLG (comma list or auto).",
    )
    parser.add_argument(
        "--cfg-rejection-rerank",
        action="store_true",
        help="Rerank multi-sample batch by early CFG gap before decode (--num>1).",
    )
    parser.add_argument(
        "--dbc-separate-cfg",
        action="store_true",
        help="Cache-DiT style: fingerprint block cache on cond half of CFG batch only.",
    )
    parser.add_argument(
        "--lcm-ckpt",
        type=str,
        default="",
        help="Consistency-distilled student checkpoint for few-step flow sampling.",
    )
    parser.add_argument(
        "--lcm-steps",
        type=int,
        default=4,
        help="Inference steps when --lcm-ckpt is set (overrides --steps unless --steps explicitly high).",
    )
    parser.add_argument(
        "--pick-save-all", action="store_true", help="Also save each candidate as stem_cand{i} when using --pick-best"
    )
    parser.add_argument(
        "--pick-clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HF model id for --pick-best clip/combo",
    )
    parser.add_argument(
        "--pick-vit-ckpt",
        type=str,
        default="",
        help="Optional vq checkpoint (best.pt) for vit / combo_vit / combo_vit_* / combo_count_vit.",
    )
    parser.add_argument(
        "--pick-vit-use-adherence",
        action="store_true",
        help="When using --pick-best vit/combo_vit: blend in the adherence head (quality*0.65 + adherence*0.35).",
    )
    parser.add_argument(
        "--pick-vit-ar-blocks",
        type=int,
        default=-1,
        help=(
            "If 0/2/4: ViT quality scorer uses matching DiT block-AR regime (see utils/architecture/ar_block_conditioning.py). "
            "-1 = ViT unknown one-hot (checkpoint default)."
        ),
    )
    parser.add_argument(
        "--pick-auto-no-clip",
        action="store_true",
        help="With --pick-best auto: avoid CLIP in the default/photo branches (aesthetic, aesthetic_realism, ocr; combo_count still uses CLIP).",
    )
    parser.add_argument(
        "--pick-report-json",
        type=str,
        default="",
        help="Optional path to write a JSON sidecar with pick/beam scores + chosen indices (useful for debugging and preference mining).",
    )
    # Beam-style partial denoise search: run a few steps for N candidates, score previews, continue only the best.
    parser.add_argument(
        "--beam-width",
        type=int,
        default=0,
        help="If >0 (and --num=1): run a partial denoise for this many candidates, score previews, continue from the best. "
        "This is like diffusion beam search; compute-heavy but high leverage.",
    )
    parser.add_argument(
        "--beam-steps",
        type=int,
        default=0,
        help="How many early denoise steps to run in the beam stage (try 6–14). Only used when --beam-width > 0.",
    )
    parser.add_argument(
        "--beam-metric",
        type=str,
        default="",
        help="Metric for beam previews (defaults to --pick-best, else combo_vit_hq if --pick-vit-ckpt, else combo_vit).",
    )
    parser.add_argument(
        "--beam2-width",
        type=int,
        default=0,
        help="Optional second-stage micro-beam (after some denoise): branch from current latent into N variants and re-pick.",
    )
    parser.add_argument(
        "--beam2-steps",
        type=int,
        default=0,
        help="How many steps to run in the second-stage micro-beam (try 4–10). Only used when --beam2-width > 0.",
    )
    parser.add_argument(
        "--beam2-at-frac",
        type=float,
        default=0.65,
        help="When to run micro-beam, as a fraction of total steps (0–1). Example 0.65 means after ~65%% of steps.",
    )
    parser.add_argument(
        "--beam2-noise",
        type=float,
        default=0.03,
        help="Stddev of Gaussian noise added to the mid-latent to create micro-beam branches (try 0.01–0.06).",
    )
    parser.add_argument(
        "--beam2-metric",
        type=str,
        default="",
        help="Metric used for micro-beam pick (defaults to --beam-metric if set, else combo_vit).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Target people count for --pick-best combo_count (0=auto-infer from prompt).",
    )
    parser.add_argument(
        "--expected-count-target",
        type=str,
        default="auto",
        choices=["auto", "people", "objects"],
        help="Count verifier target for --pick-best combo_count.",
    )
    parser.add_argument(
        "--expected-count-object",
        type=str,
        default="",
        help="Optional object hint for combo_count object mode (e.g. coin, candle, window).",
    )
    parser.add_argument(
        "--auto-expected-text",
        action="store_true",
        default=True,
        help="If --expected-text is empty, infer quoted text from prompt for OCR/pick-best text scoring.",
    )
    parser.add_argument(
        "--no-auto-expected-text",
        action="store_false",
        dest="auto_expected_text",
        help="Disable prompt-based expected-text inference.",
    )
    parser.add_argument(
        "--auto-constraint-boost",
        action="store_true",
        default=True,
        help="If text/count constraints are detected, auto-raise --num to improve adherence.",
    )
    parser.add_argument(
        "--no-auto-constraint-boost",
        action="store_false",
        dest="auto_constraint_boost",
        help="Disable automatic candidate-count boost for constrained prompts.",
    )
    parser.add_argument(
        "--vae-tiling", action="store_true", help="Enable VAE tiling for decode (lower VRAM for large output)"
    )
    parser.add_argument(
        "--compile-inference",
        action="store_true",
        help="torch.compile DiT after load for faster sampling (warm-up compile; same numerics)",
    )
    parser.add_argument(
        "--grid", action="store_true", help="When --num > 1, also save a single N-up grid image (e.g. 2x2 for 4)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Reproducible decode: cudnn deterministic + benchmark off (same seed -> same image when supported)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable T5 encoding cache (use when prompt/negative change every run)"
    )
    try:
        from diffusion import list_timestep_schedules as _lts

        _ts_list = tuple(sorted(_lts()))
    except Exception:
        _ts_list = ()
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        metavar="NAME",
        help="Timestep index schedule; registered: "
        + ", ".join(_ts_list)
        + ". Or indices:HIGH,...,LOW (see diffusion.inference_timesteps). Composes with --steps and --solver.",
    )
    parser.add_argument(
        "--timestep-schedule",
        type=str,
        default=None,
        metavar="NAME",
        help="If set, overrides --scheduler (registered names plus indices:...).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="ddim",
        metavar="NAME",
        help="VP update solver (canonical ddim/heun; aliases euler/rk2/edm_heun/… — see list_inference_solver_aliases).",
    )
    parser.add_argument(
        "--guidance-schedule",
        "--cfg-schedule",
        dest="guidance_schedule",
        type=str,
        default=None,
        metavar="MODE",
        help="CFG vs denoise progress: linear | cosine | piecewise | snr (incompatible with --holy-grail).",
    )
    parser.add_argument(
        "--guidance-schedule-linear-start",
        dest="guidance_schedule_linear_start",
        type=float,
        default=0.7,
        help="Multiplier at first VP/flow step for --guidance-schedule linear (default 0.7).",
    )
    parser.add_argument(
        "--guidance-schedule-linear-end",
        dest="guidance_schedule_linear_end",
        type=float,
        default=1.0,
        help="Multiplier at last VP/flow step for --guidance-schedule linear (default 1.0).",
    )
    parser.add_argument(
        "--guidance-schedule-cosine-min",
        dest="guidance_schedule_cosine_min",
        type=float,
        default=0.65,
        help="Min cosine multiplier when --guidance-schedule cosine.",
    )
    parser.add_argument(
        "--guidance-schedule-cosine-max",
        dest="guidance_schedule_cosine_max",
        type=float,
        default=1.0,
        help="Max cosine multiplier when --guidance-schedule cosine.",
    )
    parser.add_argument(
        "--karras-rho",
        type=float,
        default=7.0,
        help="Exponent ρ for karras_rho schedule only (larger → more emphasis in very noisy σ region).",
    )
    parser.add_argument(
        "--no-neg-filter",
        action="store_true",
        help="Disable positive/negative conflict filter (default: remove from neg any token that appears in pos)",
    )
    parser.add_argument(
        "--text-in-image",
        action="store_true",
        help="Use text-friendly default negative (legible text, signs, lettering) so desired text is not suppressed",
    )
    parser.add_argument(
        "--expected-text", type=str, default="", help="Expected OCR text for --ocr-fix (comma-separated or JSON list)."
    )
    parser.add_argument(
        "--ocr-fix", action="store_true", help="Enable OCR validation and iterative inpainting to fix misrendered text."
    )
    parser.add_argument("--ocr-threshold", type=float, default=0.65, help="Stop when OCR accuracy_score >= this value.")
    parser.add_argument("--ocr-iters", type=int, default=2, help="Max OCR repair iterations.")
    parser.add_argument("--ocr-mask-dilate", type=int, default=0, help="Dilate OCR mask before inpainting (pixels).")
    parser.add_argument(
        "--ocr-inpaint-strength", type=float, default=0.55, help="MDM inpaint strength when repairing text via OCR."
    )
    parser.add_argument("--ocr-repair-iter", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--boost-quality",
        action="store_true",
        help="Prepend 'masterpiece, best quality' to the prompt for stronger adherence (complexamples/challenging prompts)",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Write a .txt sidecar next to output with prompt, negative, seed, steps (reproducibility)",
    )
    parser.add_argument(
        "--subject-first",
        action="store_true",
        help="Reorder comma-separated prompt so subject tags (1girl, 1boy, etc.) come first",
    )
    parser.add_argument(
        "--prompt-file", type=str, default="", help="Read prompt from file (overrides --prompt when set)"
    )
    parser.add_argument(
        "--agentic-facts-json",
        type=str,
        default="",
        help="Optional JSON/JSONL with retrieved facts (e.g. Gen-Searcher output) merged into prompt before encoding.",
    )
    parser.add_argument(
        "--agentic-facts-format",
        type=str,
        default="auto",
        choices=["auto", "gen_searcher", "jsonl_text"],
        help="Fact loader mode for --agentic-facts-json.",
    )
    parser.add_argument(
        "--agentic-max-facts",
        type=int,
        default=16,
        help="Max number of retrieved facts to merge into prompt.",
    )
    parser.add_argument(
        "--agentic-facts-max-chars",
        type=int,
        default=2400,
        help="Max total character budget for merged retrieved-facts context block.",
    )
    # Creative RAG: multimodal prompt enrichment using moondream2 + Qwen2.5 + semantic analysis
    parser.add_argument(
        "--creative-rag",
        action="store_true",
        help=(
            "Enable Creative RAG prompt enrichment: semantically decomposes the prompt, "
            "resolves cross-category contradictions, classifies intent, and enriches with "
            "novel context-aware additions. Uses moondream2 (pretrained/moondream2) for "
            "reference image understanding and Qwen2.5 (pretrained/Qwen2.5-14B-Instruct) "
            "for creative synthesis when available; falls back to lightweight semantic "
            "enrichment otherwise."
        ),
    )
    parser.add_argument(
        "--creative-rag-level",
        type=float,
        default=0.7,
        help=(
            "Creative RAG novelty level 0-1 (default 0.7). "
            "0.3=subtle quality improvements, 0.6=balanced, 0.9=push for genuinely novel directions."
        ),
    )
    parser.add_argument(
        "--creative-rag-image",
        type=str,
        default="",
        help=(
            "Reference image path for Creative RAG. moondream2 will describe this image "
            "in correlation with your prompt intent, grounding the creative synthesis."
        ),
    )
    parser.add_argument(
        "--creative-rag-images",
        type=str,
        default="",
        help=(
            "Comma-separated reference image paths for Creative RAG (max 16), similar to multi-image "
            "API workflows: dissection-derived facts use all paths; Moondream captions the first "
            "8 existent files to control latency."
        ),
    )
    parser.add_argument(
        "--composition-brief",
        type=str,
        choices=("off", "auto", "on"),
        default="off",
        help=(
            "Append concise composition and text-legibility cues. "
            "'auto' enables only for UI/posters/quoted-string style prompts."
        ),
    )
    _vdd_choices = None  # filled lazily to avoid importing visual_design during --help snapshots
    try:
        from utils.visual_design.compose import visual_design_cli_domain_choices

        _vdd_choices = visual_design_cli_domain_choices()
    except ImportError:
        _vdd_choices = (
            "none",
            "auto",
            "ui_ux",
            "architecture",
            "stem",
            "textbook",
            "brand",
            "infographic",
            "packaging",
            "wayfinding",
            "general_product",
            "editorial_layout",
            "presentation_slide",
            "technical_blueprint",
            "fashion_flat",
        )
    parser.add_argument(
        "--visual-design-domain",
        type=str,
        default="none",
        choices=_vdd_choices,
        help=(
            "Append utils.visual_design domain craft cues (UI, STEM, textbook, brand, packaging, …). "
            "'auto' picks from the prompt heuristic; pairs well with --composition-brief for UI/typography."
        ),
    )
    parser.add_argument(
        "--visual-design-intensity",
        type=str,
        default="standard",
        choices=("lite", "standard", "strong"),
        help="Tier for --visual-design-domain positives and optional negatives.",
    )
    parser.add_argument(
        "--visual-design-negative-pack",
        action="store_true",
        help="Merge domain negatives into the effective negative prompt (requires --visual-design-domain / auto-hit).",
    )
    try:
        from utils.visual_design.presets import preset_ids as _visual_design_preset_ids

        _vdp_help_extra = f"Known ids: {', '.join(_visual_design_preset_ids())}."
    except ImportError:

        def _visual_design_preset_ids():  # type: ignore[misc]
            return ()

        _vdp_help_extra = ""
    parser.add_argument(
        "--visual-design-preset",
        type=str,
        default="",
        help=(
            "Shortcut: set domain + intensity (and optional prompt prefix) via utils.visual_design.presets. "
            "Overrides --visual-design-domain / --visual-design-intensity when non-empty. "
            f"{_vdp_help_extra}"
        ),
    )
    parser.add_argument(
        "--multi-instance-preset",
        type=str,
        choices=(
            "none",
            "distinct_objects",
            "stacked_media",
            "turnaround_sheet",
            "panel_strip",
            "group_portrait",
        ),
        default="none",
        help=(
            "Bias toward several *different* instances in one frame (posters, books, panels, group shots, "
            "turnarounds): structure + anti-clone negatives + higher --num floor. See utils.prompt.multi_instance_scene."
        ),
    )
    parser.add_argument(
        "--multi-instance-count",
        type=int,
        default=0,
        help=("With --multi-instance-preset: set --expected-count for test-time scoring (e.g. 5 posters, 4 people)."),
    )
    parser.add_argument(
        "--multi-instance-auto",
        action="store_true",
        help=(
            "With --multi-instance-preset: if --composition-brief was off, switch to auto; "
            "if --pick-best is none, use combo_count; if --multi-instance-count>0 and "
            "--expected-count-target is auto, set people vs objects heuristic. Prints a workflow checklist."
        ),
    )
    parser.add_argument(
        "--detailed-scene-boost",
        type=str,
        choices=("off", "auto", "on"),
        default="off",
        help=(
            "Add separation, per-entity noun consistency, pose/weight/contact, optional anatomy + "
            "physics/creature heuristics (see utils.prompt.detailed_scene_entities). "
            "Use auto for group/count/long multi-clause prompts; on always appends."
        ),
    )
    parser.add_argument(
        "--detailed-scene-strength",
        type=str,
        choices=("lite", "strong"),
        default="lite",
        help="With --detailed-scene-boost: lite vs stronger anatomy/pose/shadow cues.",
    )
    parser.add_argument(
        "--prompt-breakdown",
        type=str,
        choices=("off", "auto", "on"),
        default="off",
        help=(
            "Heuristically split comma/semicolon clauses into layout-aligned buckets and reorder for encoders. "
            "Auto triggers on long or clause-heavy prompts. Skipped when --prompt-layout is used."
        ),
    )
    parser.add_argument(
        "--prompt-breakdown-format",
        type=str,
        choices=("ordered", "labeled"),
        default="ordered",
        help=(
            "With --prompt-breakdown: ordered = single reordered comma line (CLIP+T5); "
            "labeled = QUALITY:/SUBJECTS:/… blocks for T5 only (flat line still used for cache/CLIP consistency)."
        ),
    )
    parser.add_argument(
        "--prompt-breakdown-order",
        type=str,
        choices=("subject_first", "quality_first", "scene_first"),
        default="subject_first",
        help="Section priority when merging buckets (same presets as prompt_layout JSON).",
    )
    parser.add_argument(
        "--creative-rag-resolve-conflicts",
        action="store_true",
        default=True,
        help="Resolve semantic contradictions in the prompt before generation (e.g. 'photorealistic, anime'). Default: on.",
    )
    parser.add_argument(
        "--no-creative-rag-resolve-conflicts",
        action="store_false",
        dest="creative_rag_resolve_conflicts",
        help="Disable automatic semantic conflict resolution.",
    )
    parser.add_argument(
        "--hard-style",
        type=str,
        default=None,
        choices=["3d", "realistic", "3d_realistic", "style_mix"],
        help="Prepend recommended tags for hard styles (3d, realistic, 3d_realistic, style_mix); see config/defaults/prompt_domains.py for negatives",
    )
    parser.add_argument(
        "--naturalize",
        action="store_true",
        help="Reduce AI look: add anti-plastic/oversmooth negative, optional natural-look prompt prefix, and subtle film grain + micro-contrast post-process",
    )
    parser.add_argument(
        "--naturalize-grain",
        type=float,
        default=0.015,
        help="Film grain amount when --naturalize (0=off, 0.01-0.03 typical)",
    )
    parser.add_argument(
        "--naturalize-deep",
        action="store_true",
        help="With --naturalize: stronger anti-AI negatives + richer natural-photo prefix (more de-CGI)",
    )
    parser.add_argument(
        "--less-ai",
        action="store_true",
        help="Shorthand: --anti-ai-pack lite + --human-media photographic (when those are still none)",
    )
    parser.add_argument(
        "--human-made",
        type=str,
        default="none",
        choices=["none", "lite", "standard", "strong"],
        help="Human-made polish: anti-AI prompts + speckle/plastic/halo cleanup post-process.",
    )
    parser.add_argument(
        "--human-made-strength",
        type=float,
        default=-1.0,
        help="Override human-made post strength 0-1 (default: preset default).",
    )
    parser.add_argument(
        "--anti-ai-pack",
        type=str,
        default="none",
        choices=["none", "lite", "strong"],
        help="Reduce plastic/CGI/oversmooth look via prompt packs (pairs well with --naturalize post-process).",
    )
    parser.add_argument(
        "--human-media",
        dest="human_media_mode",
        type=str,
        default="none",
        choices=["none", "photographic", "dslr", "film"],
        help="Bias toward real camera / film capture instead of CG render.",
    )
    parser.add_argument(
        "--photo-realism-pack",
        type=str,
        default="none",
        choices=[
            "none",
            "documentary",
            "cinematic",
            "studio_portrait",
            "film_analog",
            "night_noir",
            "product_catalog",
            "fashion_editorial",
        ],
        help="Photography realism pack for prompt+negative guidance.",
    )
    parser.add_argument(
        "--photo-color-grade",
        type=str,
        default="none",
        choices=["none", "natural", "teal_orange", "kodak_portra", "cinestill_800t", "noir_bw", "fujifilm_eterna"],
        help="Photography color-grade direction.",
    )
    parser.add_argument(
        "--photo-lighting-technique",
        type=str,
        default="none",
        choices=[
            "none",
            "three_point",
            "golden_hour",
            "overcast_soft",
            "motivated_practical",
            "rim_backlight",
            "butterfly",
            "rembrandt",
        ],
        help="Photography lighting-technique cues.",
    )
    parser.add_argument(
        "--photo-filter",
        type=str,
        default="none",
        choices=["none", "pro_mist", "polarizer", "nd_long_exposure", "vintage_diffusion", "clean_digital"],
        help="Photographic filter-style cues.",
    )
    parser.add_argument(
        "--photo-grain-style",
        type=str,
        default="none",
        choices=["none", "fine_35mm", "medium_35mm", "heavy_16mm", "clean_digital"],
        help="Photography grain-style cue.",
    )
    parser.add_argument(
        "--photo-realism-strength",
        type=float,
        default=1.0,
        help="Prompt weighting strength for photo-realism cues (0.25-2.0).",
    )
    parser.add_argument(
        "--photo-postprocess",
        dest="photo_postprocess",
        action="store_true",
        help="Apply photography-focused post process (grade/filter/grain) based on selected photo controls (default: on).",
    )
    parser.add_argument(
        "--no-photo-postprocess",
        dest="photo_postprocess",
        action="store_false",
        help="Disable photography post process.",
    )
    parser.set_defaults(photo_postprocess=True)
    parser.add_argument(
        "--photo-post-strength",
        type=float,
        default=0.6,
        help="Strength for photography post process grade/filter (0-1).",
    )
    parser.add_argument(
        "--auto-photo-realism",
        dest="auto_photo_realism",
        action="store_true",
        help="Auto-infer photo-realism controls from prompt keywords (default: on).",
    )
    parser.add_argument(
        "--no-auto-photo-realism",
        dest="auto_photo_realism",
        action="store_false",
        help="Disable auto photo-realism inference.",
    )
    parser.set_defaults(auto_photo_realism=True)
    parser.add_argument(
        "--realism-autopilot",
        dest="realism_autopilot",
        action="store_true",
        help="Auto-tune photo post strength, grain, and auto pick-best metric for photographic prompts (default: on).",
    )
    parser.add_argument(
        "--no-realism-autopilot",
        dest="realism_autopilot",
        action="store_false",
        help="Disable realism autopilot.",
    )
    parser.set_defaults(realism_autopilot=True)
    parser.add_argument(
        "--lora-scaffold",
        type=str,
        default="none",
        choices=["none", "blend", "character_first", "style_first"],
        help="Prompt scaffolding when using --lora (fusion / character vs style priority).",
    )
    parser.add_argument(
        "--lora-scaffold-auto",
        action="store_true",
        help="If any --lora is set and --lora-scaffold is none, use blend scaffolding.",
    )
    parser.add_argument(
        "--anti-bleed",
        action="store_true",
        help="Reduce concept/color bleeding: add distinct-colors positive and color-bleed negative",
    )
    parser.add_argument(
        "--shortcomings-mitigation",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Append prompt/negative hints: photoreal, digital painting/concept/pixel/vector/game art, 3D render (docs/COMMON_SHORTCOMINGS_AI_IMAGES.md); auto=keyword match, all=full base pack",
    )
    parser.add_argument(
        "--shortcomings-2d",
        action="store_true",
        help="With --shortcomings-mitigation auto|all: include stylized 2D packs (anime/manga/cel/etc.)",
    )
    parser.add_argument(
        "--art-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Artist-first medium packs (traditional + digital + photo): auto=keyword match, all=full pack",
    )
    parser.add_argument(
        "--no-art-guidance-photography",
        action="store_true",
        help="With --art-guidance-mode auto|all: skip photography-specific packs",
    )
    parser.add_argument(
        "--anatomy-guidance",
        type=str,
        default="none",
        choices=["none", "lite", "strong"],
        help="Extra anatomy/proportion constraints: lite (only if people detected), strong (always)",
    )
    parser.add_argument(
        "--style-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Style-domain guidance (anime/comic/editorial/concept/game/photo language)",
    )
    parser.set_defaults(style_guidance_artists=True)
    parser.add_argument(
        "--no-style-guidance-artists",
        action="store_false",
        dest="style_guidance_artists",
        help="Disable artist/game-name reference stabilization cues in style guidance",
    )
    parser.add_argument(
        "--diversity",
        action="store_true",
        help="Reduce same-face/repetitive face: add diversity positive and repetitive-face negative",
    )
    parser.add_argument(
        "--anti-artifacts",
        action="store_true",
        help="Add artifact negative (white dots, speckles, spiky, pixel stretch)",
    )
    parser.add_argument(
        "--strong-watermark", action="store_true", help="Stronger watermark/logo negative (for stubborn baked-in logos)"
    )
    parser.add_argument(
        "--pose-mode",
        type=str,
        default="none",
        choices=["none", "complex", "action", "acrobatics"],
        help="Add pose scaffolding tokens for difficult body compositions.",
    )
    parser.add_argument(
        "--view-angle",
        type=str,
        default="none",
        choices=[
            "none",
            "eye_level",
            "low_angle",
            "high_angle",
            "bird_eye",
            "worm_eye",
            "dutch",
            "over_shoulder",
            "first_person",
            "third_person",
        ],
        help="Camera/viewpoint conditioning for hard perspective shots.",
    )
    parser.add_argument(
        "--subject-sex",
        type=str,
        default="none",
        choices=["none", "female", "male", "mixed", "nonbinary"],
        help="Anatomy consistency hint for subject sexamples/presentation.",
    )
    parser.add_argument(
        "--scene-domain",
        type=str,
        default="none",
        choices=["none", "objects", "vehicles", "buildings", "architecture", "mixed"],
        help="Grounding hints for objects/vehicles/buildings-heavy scenes.",
    )
    parser.add_argument(
        "--clothing-mode",
        type=str,
        default="none",
        choices=["none", "casual", "formal", "streetwear", "fantasy_armor", "swimwear", "lingerie", "nude"],
        help="Clothing/garment control pack.",
    )
    parser.add_argument(
        "--background-mode",
        type=str,
        default="none",
        choices=["none", "studio", "indoor", "outdoor", "urban", "nature", "minimal"],
        help="Background/environment stabilization pack.",
    )
    parser.add_argument(
        "--people-layout",
        type=str,
        default="none",
        choices=["none", "solo", "duo", "group_small", "group_large"],
        help="Multi-person layout control.",
    )
    parser.add_argument(
        "--relationship-mode",
        type=str,
        default="none",
        choices=["none", "neutral", "romantic", "combat", "teamwork"],
        help="Interaction mode for multiple people.",
    )
    parser.add_argument(
        "--object-layout",
        type=str,
        default="none",
        choices=["none", "foreground_anchor", "rule_of_thirds", "symmetrical", "asymmetrical"],
        help="Object placement strategy hints.",
    )
    parser.add_argument(
        "--hand-mode",
        type=str,
        default="none",
        choices=["none", "stable", "detailed", "grip"],
        help="Hand-quality control pack for hard hand generations.",
    )
    parser.add_argument(
        "--pose-naturalness",
        type=str,
        default="none",
        choices=["none", "natural", "dynamic_natural", "intimate_natural"],
        help="Natural pose/body mechanics pack (works for sfw/nsfw prompts).",
    )
    parser.add_argument(
        "--typography-mode",
        type=str,
        default="none",
        choices=["none", "clean", "poster", "ui"],
        help="Typography/text rendering control pack.",
    )
    parser.add_argument(
        "--quality-pack",
        type=str,
        default="none",
        choices=[
            "none",
            "top",
            "one_shot",
            "ultra_clean",
            "cinematic",
            "illustrative",
            "editorial",
            "micro_detail",
        ],
        help="High-quality artifact-control pack. 'top' = score ladder; 'one_shot' = ladder + composition/anatomy first-try tags; 'micro_detail' = texture/material fidelity.",
    )
    parser.add_argument(
        "--adherence-pack",
        type=str,
        default="none",
        choices=["none", "standard", "strict"],
        help="Prompt adherence scaffolding: literal scene interpretation, fewer missing/wrong props (use with long prompts).",
    )
    parser.add_argument(
        "--lighting-mode",
        type=str,
        default="none",
        choices=["none", "natural_daylight", "studio_softbox", "dramatic_rim", "low_key", "high_key"],
        help="Lighting stability/style pack.",
    )
    parser.add_argument(
        "--skin-detail-mode",
        type=str,
        default="none",
        choices=["none", "natural_texture", "clean_beauty", "stylized_skin"],
        help="Skin texture/detail behavior pack.",
    )
    parser.add_argument(
        "--body-proportion",
        type=str,
        default="none",
        choices=["none", "realistic", "exaggerated", "hyper"],
        help="Body proportion style - use 'hyper' for extreme sizes.",
    )
    parser.add_argument(
        "--interaction-intensity",
        type=str,
        default="none",
        choices=["none", "gentle", "passionate", "intense", "extreme"],
        help="Intensity of sexual interaction.",
    )
    parser.add_argument(
        "--style-mode",
        type=str,
        default="none",
        choices=["none", "3d", "photoreal", "semi_real", "anime", "painterly", "3d_photoreal"],
        help="Style adherence control pack for hard styles.",
    )
    parser.add_argument(
        "--style-lock",
        action="store_true",
        help="Push consistent single-style rendering and suppress style drift.",
    )
    parser.add_argument(
        "--anti-style-bleed",
        action="store_true",
        help="Add negatives to reduce mixed/bleeding styles.",
    )
    parser.add_argument(
        "--composition-mode",
        type=str,
        default="none",
        choices=["none", "single_subject", "group", "multi_character", "scene", "cinematic"],
        help="Composition stabilizer: use multi_character for 2+ distinct outfits/poses (stronger than group).",
    )
    parser.add_argument(
        "--artist-composition",
        type=str,
        default="none",
        choices=["none", "lite", "standard", "perspective", "classical", "full"],
        help="Classical art composition tags: rule of thirds / golden ratio / perspective / notan / S-curve (stacks with --composition-mode).",
    )
    parser.add_argument(
        "--anti-duplicate-subjects",
        action="store_true",
        help="Add negatives to reduce cloned faces/extra heads/duplicate subjects.",
    )
    parser.add_argument(
        "--anti-perspective-drift",
        action="store_true",
        help="Add perspective/scale stability cues to reduce warped geometry.",
    )
    parser.add_argument(
        "--cleanup-conflicting-tags",
        action="store_true",
        help="Remove obvious contradictory prompt tags (keeps earlier tag).",
    )
    parser.add_argument(
        "--auto-content-fix",
        dest="auto_content_fix",
        action="store_true",
        help="Auto-infer domain, view, pose, composition (1girl solo), hands, lighting from keywords (default: on).",
    )
    parser.add_argument(
        "--no-auto-content-fix",
        dest="auto_content_fix",
        action="store_false",
        help="Disable automatic keyword inference for content controls.",
    )
    parser.set_defaults(auto_content_fix=True)
    parser.add_argument(
        "--one-shot-boost",
        dest="one_shot_boost",
        action="store_true",
        help="Add one-shot composition/anatomy/scaffolding to pos+neg (default: on).",
    )
    parser.add_argument(
        "--no-one-shot-boost",
        dest="one_shot_boost",
        action="store_false",
        help="Disable extra one-shot scaffolding tokens.",
    )
    parser.set_defaults(one_shot_boost=True)
    parser.add_argument(
        "--prompt-clauses",
        type=str,
        default="",
        help=(
            "Comma-separated intent clauses from utils.prompt.stack (e.g. "
            "uncensored.fidelity,hands.stable,quality.micro). Applied after content controls."
        ),
    )
    parser.add_argument(
        "--no-prompt-stack-intelligence",
        dest="prompt_stack_intelligence",
        action="store_false",
        help="Disable PromptStack prompt analysis (complexity, auto quality hints).",
    )
    parser.set_defaults(prompt_stack_intelligence=True)
    parser.add_argument(
        "--no-prompt-stack-auto-quality",
        dest="prompt_stack_auto_quality",
        action="store_false",
        help="Disable light quality-tag injection for short prompts.",
    )
    parser.set_defaults(prompt_stack_auto_quality=True)
    parser.add_argument(
        "--prompt-special-helpers",
        type=str,
        default="auto",
        help=(
            "Route surreal/horror/narrative/technical/NSFW-precision helpers "
            "(utils.prompt.special_prompt_helpers). Use 'off' to disable."
        ),
    )
    parser.add_argument(
        "--invent-styles",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Invent N novel style genomes (utils.prompt.style_inventor) and apply "
            "genome #--style-genome-index via PromptStack before generation."
        ),
    )
    parser.add_argument(
        "--style-genome-file",
        type=str,
        default="",
        help="JSON file with one StyleGenome object or an array (use with --style-genome-index).",
    )
    parser.add_argument(
        "--style-genome-index",
        type=int,
        default=0,
        help="When --style-genome-file or --invent-styles returns multiple genomes, pick this index.",
    )
    parser.add_argument(
        "--style-inventor-creativity",
        type=float,
        default=0.75,
        help="0–1 novelty for --invent-styles (higher = more original axes).",
    )
    parser.add_argument(
        "--no-style-inventor-qwen",
        action="store_true",
        help="Use deterministic style genome fallback only (no Qwen2.5).",
    )
    parser.add_argument(
        "--explore-styles",
        action="store_true",
        help=(
            "Shorthand: --invent-styles 3, --num 3, --pick-best combo if unset. "
            "For full genome×mutation manifests use scripts.tools.explore_styles."
        ),
    )
    parser.add_argument(
        "--explore-styles-insane",
        action="store_true",
        help=(
            "Nuclear explore: --style-inventor-mode apocalypse --style-chaos-level 0.95 "
            "--invent-styles 4 --prompt-clauses style.chaos,style.surreal"
        ),
    )
    parser.add_argument(
        "--style-inventor-mode",
        type=str,
        default="normal",
        choices=("normal", "insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk"),
        help="Style genome invention mode (see utils.prompt.style_genome_chaos).",
    )
    parser.add_argument(
        "--style-chaos-level",
        type=float,
        default=0.0,
        help="0–1 extra chaos spice on top of invented genome (hyper-fragments, wild axes).",
    )
    parser.add_argument(
        "--style-genome-preset",
        type=str,
        default="",
        help=(
            "Force a named insane preset (glitch_cathedral, biolume_abyss, eldritch_taxonomy, …). "
            "Run: python -m scripts.tools explore_styles --list-presets"
        ),
    )
    parser.add_argument(
        "--style-genome-hypermutate",
        action="store_true",
        help="Hypermutate each invented genome before apply (sibling strain).",
    )
    parser.add_argument(
        "--style-genome-fusion",
        action="store_true",
        help="When inventing 2+ genomes, also build chimera fusion in explore manifest (explore_styles tool).",
    )
    parser.add_argument(
        "--gender-swap",
        action="store_true",
        help="Heuristic gender swap: girl<->boy, woman<->man, she<->he in the prompt",
    )
    parser.add_argument(
        "--anatomy-scale", type=str, default="", help="Comma-separated: longer,bigger,wider (anatomy proportions)"
    )
    parser.add_argument(
        "--object-scale", type=str, default="", help="Comma-separated: longer,bigger,wider (bigger/longer/wider props)"
    )
    parser.add_argument(
        "--scene-scale",
        type=str,
        default="",
        help="Comma-separated: longer,bigger,wider (wider/longer/bigger scene framing)",
    )
    parser.add_argument(
        "--character-sheet",
        type=str,
        default="",
        help="Path(s) to character sheet JSON (comma-separated for multi-character) to inject identity tokens.",
    )
    parser.add_argument(
        "--label-multi-character-sheets",
        action="store_true",
        help="With 2+ --character-sheet paths, wrap each sheet as (character N: ...) for clearer T5 separation.",
    )
    parser.add_argument(
        "--character-prompt-extra", type=str, default="", help="Extra character tokens appended to prompt"
    )
    parser.add_argument(
        "--character-negative-extra",
        type=str,
        default="",
        help="Extra negative tokens to append for the character (applied after defaults)",
    )
    parser.add_argument(
        "--scene-blueprint",
        type=str,
        default="",
        help="Path to JSON scene blueprint for deep structured scene customization.",
    )
    parser.add_argument(
        "--scene-blueprint-strength",
        type=float,
        default=1.0,
        help="Blueprint emphasis strength (0.5-2.0).",
    )
    parser.add_argument(
        "--character-strength",
        type=float,
        default=1.0,
        help="Character identity strength (0.5-2.0): higher reinforces profile traits.",
    )
    parser.add_argument(
        "--auto-original-character",
        dest="auto_original_character",
        action="store_true",
        help="Auto-synthesize an original character profile when prompt asks for OC/character design (default: on).",
    )
    parser.add_argument(
        "--no-auto-original-character",
        dest="auto_original_character",
        action="store_false",
        help="Disable automatic OC synthesis from prompt intent.",
    )
    parser.set_defaults(auto_original_character=True)
    parser.add_argument(
        "--auto-oc-seed-offset",
        type=int,
        default=0,
        help="Extra deterministic seed offset for auto-OC synthesis.",
    )
    parser.add_argument(
        "--uncensored-mode",
        action="store_true",
        default=True,
        help="Disable character-sheet safety sanitization and avoid anti-explicit negative injections. (enabled by default)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["sdxl", "flux", "anime", "zit"],
        help="Apply a sampler preset (soft defaults) from config.defaults.model_presets",
    )
    parser.add_argument(
        "--op-mode",
        type=str,
        default=None,
        choices=["portrait", "fullbody", "anime_char"],
        help="High-level OP mode (applied after preset)",
    )
    try:
        from diffusion.sampling import list_holy_grail_presets as _lhgp

        _hg_preset_choices = ["auto"] + _lhgp()
    except Exception:
        _hg_preset_choices = ["auto"]
    parser.add_argument(
        "--holy-grail-preset",
        type=str,
        default=None,
        choices=_hg_preset_choices,
        help="Apply a holy-grail preset bundle (auto|balanced|photoreal|anime|illustration|aggressive).",
    )
    parser.add_argument(
        "--box-layout",
        type=str,
        default="",
        help="JSON file: Ideogram-style boxes + per-region prompts (optional sketch/draw per box). "
        "See examples/box_layout.example.json and examples/box_layout_sketch.example.json",
    )
    parser.add_argument(
        "--box-layout-mode",
        type=str,
        default="regional_cfg",
        choices=["regional_cfg", "text_only"],
        help="With --box-layout: regional_cfg blends per-box prompts during denoising; "
        "text_only only merges layout into the global T5 prompt.",
    )
    parser.add_argument(
        "--prompt-layout",
        type=str,
        default="",
        help="JSON file: layered prompt (intent/subjects/scene/camera/…). See utils/prompt/prompt_layout.py and examples/prompt_layout.example.json",
    )
    parser.add_argument(
        "--t5-layout-encode",
        type=str,
        default="auto",
        choices=["auto", "flat", "blocks", "segmented"],
        help="With --prompt-layout: how T5 reads the positive (frozen encoder; clearer section boundaries). "
        "auto=blocks when layout is used else flat. segmented=concat tokenized sections, one forward. "
        "Triple mode + layout: CLIP-L and CLIP-bigG use a labeled compact caption (same string for both); "
        "T5 uses blocks/segmented/flat per this flag. Use flat if you rely on (word)/[word] emphasis.",
    )
    return parser


if __name__ == "__main__" and any(_h in sys.argv[1:] for _h in ("-h", "--help")):
    build_sample_parser().parse_args()


import numpy as np
import torch
from config.defaults.model_presets import apply_op_mode_to_args, apply_preset_to_args
from diffusion import create_diffusion
from diffusion.sampling import (
    apply_holy_grail_preset_to_args,
    recommend_holy_grail_preset,
    sanitize_holy_grail_kwargs,
)
from models.controlnet import control_type_to_id, infer_control_type_from_path
from PIL import Image
from utils.prompt.prompt_emphasis import parse_prompt_emphasis, token_weights_from_cleaned_segments
from utils.runtime.jsonutil import loads as json_loads
from utils.terminal import configure_stdio_for_console

configure_stdio_for_console()


def _maybe_rae_to_dit(z: torch.Tensor, ae_type: str, rae_bridge) -> torch.Tensor:
    """Map RAE latent (B,C,h,w) to DiT 4-channel space when checkpoint includes RAELatentBridge."""
    if z is None or ae_type != "rae" or rae_bridge is None:
        return z
    if z.shape[1] == 4:
        return z
    return rae_bridge.rae_to_dit(z)


def load_model_from_ckpt(ckpt_path, device="cuda"):
    from utils.checkpoint.checkpoint_loading import load_sampler_checkpoint

    return load_sampler_checkpoint(ckpt_path, device=device, reject_enhanced=True, verbose=True)


# T5 encoding cache (IMPROVEMENTS 3.2): key = (prompt, negative, style), value = (cond, uncond, style_emb or None)
_t5_cache = {}
_T5_CACHE_MAX = 32  # limit entries to avoid unbounded memory


def _parse_scale_csv(value: str) -> list:
    """Parse comma-separated scale modifiers (longer,bigger,wider) into a stable list."""
    allowed = {"longer", "bigger", "wider"}
    if not value:
        return []
    parts = [p.strip().lower() for p in value.split(",")]
    out = []
    for p in parts:
        if p in allowed and p not in out:
            out.append(p)
    return out


def _parse_lora_role_budgets(raw: str) -> dict:
    """Parse 'character=1.8,style=1.0,detail=0.8' into dict."""
    out = {}
    if not raw:
        return out
    for part in str(raw).split(","):
        p = part.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        try:
            out[str(k).strip().lower()] = float(v.strip())
        except Exception:
            continue
    return out


def _parse_lora_role_stage_weights(raw: str) -> dict:
    """
    Parse per-role stage multipliers:
    "character=1.15/1.0/0.85,style=0.9/1.0/1.1"
    where values are early/mid/late.
    """
    out = {}
    if not raw:
        return out
    for part in str(raw).split(","):
        p = part.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        nums = [x.strip() for x in v.split("/") if x.strip()]
        if len(nums) != 3:
            continue
        try:
            out[str(k).strip().lower()] = (float(nums[0]), float(nums[1]), float(nums[2]))
        except Exception:
            continue
    return out


def _parse_lora_spec(spec: str, *, default_role: str = "style") -> Tuple[str, float, str]:
    """
    Parse LoRA/DoRA/LyCORIS spec.
    Supported:
      - path
      - path:scale
      - path:scale:role
    """
    s = str(spec or "").strip()
    if not s:
        return "", 0.8, str(default_role or "style").lower()
    parts = s.split(":")
    role = str(default_role or "style").strip().lower()
    if len(parts) >= 3:
        maybe_role = parts[-1].strip().lower()
        try:
            scale = float(parts[-2].strip())
            path = ":".join(parts[:-2]).strip()
            if path:
                return path, scale, maybe_role or role
        except Exception:
            pass
    if len(parts) >= 2:
        try:
            scale = float(parts[-1].strip())
            path = ":".join(parts[:-1]).strip()
            if path:
                return path, scale, role
        except Exception:
            pass
    return s, 0.8, role


def _parse_weighted_style_mix(raw: str) -> list:
    """
    Parse weighted multi-style prompt.
    Supported forms (segments separated by '|'):
      - "anime::0.6 | watercolor::0.4"
      - "anime:0.6 | watercolor:0.4"
      - "anime | watercolor" (equal weights)
    Returns list[(text, normalized_weight)].
    """
    s = str(raw or "").strip()
    if not s:
        return []
    segs = [p.strip() for p in s.split("|") if p.strip()]
    out = []
    for seg in segs:
        txt = seg
        w = 1.0
        if "::" in seg:
            a, b = seg.rsplit("::", 1)
            try:
                w = float(b.strip())
                txt = a.strip()
            except Exception:
                pass
        elif ":" in seg:
            a, b = seg.rsplit(":", 1)
            try:
                w = float(b.strip())
                txt = a.strip()
            except Exception:
                pass
        if txt:
            out.append((txt, max(0.0, float(w))))
    if not out:
        return []
    sw = sum(w for _, w in out)
    if sw <= 1e-8:
        n = float(len(out))
        return [(t, 1.0 / n) for t, _ in out]
    return [(t, w / sw) for t, w in out]


def _parse_control_spec(
    spec: str,
    *,
    default_type: str = "auto",
    default_scale: float = 0.85,
) -> Tuple[str, str, float]:
    """
    Parse ControlNet spec with Windows-path-safe rules.
    Supported:
      - path
      - path:scale
      - path:type
      - path:type:scale
      - path:scale:type
    """
    s = str(spec or "").strip()
    if not s:
        return "", str(default_type or "auto").lower(), float(default_scale)
    parts = s.split(":")
    ctype = str(default_type or "auto").strip().lower()
    cscale = float(default_scale)
    if len(parts) >= 3:
        a = parts[-2].strip()
        b = parts[-1].strip().lower()
        # Try path:scale:type
        try:
            sc = float(a)
            path = ":".join(parts[:-2]).strip()
            if path:
                return path, (b or ctype), sc
        except Exception:
            pass
        # Try path:type:scale
        try:
            sc = float(parts[-1].strip())
            path = ":".join(parts[:-2]).strip()
            t = parts[-2].strip().lower()
            if path:
                return path, (t or ctype), sc
        except Exception:
            pass
    if len(parts) >= 2:
        tail = parts[-1].strip()
        path = ":".join(parts[:-1]).strip()
        if path:
            try:
                return path, ctype, float(tail)
            except Exception:
                return path, tail.lower() or ctype, cscale
    return s, ctype, cscale


def _resize_control_tensor(ctrl: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize control tensor preserving 4D (B,C,H,W) or 5D (B,K,C,H,W) layout."""
    if ctrl.ndim == 4:
        return torch.nn.functional.interpolate(
            ctrl,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    if ctrl.ndim == 5:
        b, k, c, h, w = ctrl.shape
        flat = ctrl.reshape(b * k, c, h, w)
        out = torch.nn.functional.interpolate(
            flat,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return out.view(b, k, c, target_h, target_w)
    raise ValueError(f"Unsupported control tensor shape: {tuple(ctrl.shape)}")


def _apply_gender_swap(prompt: str) -> str:
    """
    Swap common gendered tags/phrases in Danbooru-ish prompts.
    Note: this is heuristic text replacement; it does not guarantee semantic correctness.
    """
    if not prompt:
        return prompt
    p = prompt

    # Danbooru counts (use placeholders to avoid double-swaps)
    p = re.sub(r"\b(\d+)girls\b", r"\1__TMP_BOYS__", p, flags=re.IGNORECASE)
    p = re.sub(r"\b(\d+)boys\b", r"\1__TMP_GIRLS__", p, flags=re.IGNORECASE)
    p = re.sub(r"\b(\d+)__TMP_BOYS__\b", r"\1boys", p, flags=re.IGNORECASE)
    p = re.sub(r"\b(\d+)__TMP_GIRLS__\b", r"\1girls", p, flags=re.IGNORECASE)

    # Single tags
    p = re.sub(r"\bgirl\b", "__TMP_GIRL__", p, flags=re.IGNORECASE)
    p = re.sub(r"\bboy\b", "girl", p, flags=re.IGNORECASE)
    p = re.sub(r"\b__TMP_GIRL__\b", "boy", p, flags=re.IGNORECASE)

    p = re.sub(r"\bwoman\b", "__TMP_WOMAN__", p, flags=re.IGNORECASE)
    p = re.sub(r"\bman\b", "woman", p, flags=re.IGNORECASE)
    p = re.sub(r"\b__TMP_WOMAN__\b", "man", p, flags=re.IGNORECASE)

    # Adjectives
    p = re.sub(r"\bfemale\b", "__TMP_FEMALE__", p, flags=re.IGNORECASE)
    p = re.sub(r"\bmale\b", "female", p, flags=re.IGNORECASE)
    p = re.sub(r"\b__TMP_FEMALE__\b", "male", p, flags=re.IGNORECASE)

    # Pronouns (simple placeholders)
    p = re.sub(r"\bshe\b", "__TMP_SHE__", p, flags=re.IGNORECASE)
    p = re.sub(r"\bhe\b", "she", p, flags=re.IGNORECASE)
    p = re.sub(r"\b__TMP_SHE__\b", "he", p, flags=re.IGNORECASE)

    p = re.sub(r"\bher\b", "__TMP_HER__", p, flags=re.IGNORECASE)
    p = re.sub(r"\bhis\b", "her", p, flags=re.IGNORECASE)
    p = re.sub(r"\b__TMP_HER__\b", "his", p, flags=re.IGNORECASE)

    return p


def _build_size_tokens(anatomy_scales: list, object_scales: list, scene_scales: list) -> str:
    """Return comma-separated prompt tokens for requested size modifiers."""
    anatomy_map = {
        "longer": "longer limbs, longer legs, longer arms",
        "bigger": "larger body, bigger frame, broader build",
        "wider": "wider shoulders, broader chest, wider hips",
    }
    object_map = {
        "longer": "elongated props, longer objects",
        "bigger": "oversized props, larger accessories",
        "wider": "wider objects, broad props",
    }
    scene_map = {
        "longer": "extended composition, longer scene layout",
        "bigger": "large-scale scene, big environment",
        "wider": "wide view, wider perspective",
    }
    tokens: list[str] = []
    for s in anatomy_scales:
        tokens.append(anatomy_map[s])
    for s in object_scales:
        tokens.append(object_map[s])
    for s in scene_scales:
        tokens.append(scene_map[s])
    return ", ".join([t for t in tokens if t])


SCALE_DISTORTION_NEGATIVE = (
    # Keep scaling requests “shape-like” without drifting into warped outputs.
    "deformed, warped anatomy, stretched anatomy, bad proportions, misproportioned, wrong scale, "
    "extra limbs, fused limbs, melted, distorted"
)


def _parse_expected_texts(raw: str) -> list:
    """
    Parse expected text for OCR validation.
    Accepts: comma-separated string or a JSON list string.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        if raw.startswith("["):
            data = json_loads(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _infer_expected_texts_from_prompt(prompt: str) -> list:
    """
    Infer likely intended on-image text from quoted fragments in prompt.
    """
    p = str(prompt or "")
    if not p.strip():
        return []
    out = []
    for m in re.finditer(r'"([^"\n]{1,80})"', p):
        t = (m.group(1) or "").strip()
        if not t:
            continue
        if not re.search(r"[A-Za-z0-9]", t):
            continue
        out.append(t)
    # Keep stable order + de-duplicate.
    dedup = []
    seen = set()
    for t in out:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(t)
    return dedup[:4]


def _maybe_append_text_says(prompt: str, expected_texts: list) -> str:
    """Ensure prompt contains 'text that says "<t>"' for expected OCR text."""
    p = prompt or ""
    if not expected_texts:
        return p
    # Use first expected string as the "anchor" for exact OCR.
    t = expected_texts[0]
    if not t:
        return p
    quoted = f'"{t}"'
    if quoted.lower() in p.lower() or t.lower() in p.lower():
        return p
    # Append in a way our prompt-negative logic understands (TEXT_IN_IMAGE_PHRASES).
    # "text that says" is also used in config defaults.
    return f"{p.strip()}, text that says {quoted}"


def _refine_gate_score(
    *,
    image_rgb_u8: np.ndarray,
    expected_texts: list,
) -> tuple[float, dict]:
    """
    Return (score in [0,1], details) where higher means "already good enough".
    """
    try:
        from utils.quality import test_time_pick as _ttp
    except Exception:
        return 0.0, {"reason": "metrics_unavailable"}
    edge = float(_ttp.score_edge_sharpness(image_rgb_u8))
    exp = float(_ttp.score_exposure_balance(image_rgb_u8))
    edge_n = float(np.clip(edge / 400.0, 0.0, 1.0))
    parts = [0.45 * edge_n, 0.45 * exp]
    details = {"edge_sharpness": edge, "edge_norm": edge_n, "exposure_balance": exp}
    if expected_texts:
        try:
            ocr = float(_ttp.score_ocr_match(image_rgb_u8, str(expected_texts[0])))
        except Exception:
            ocr = 0.5
        details["ocr_match"] = ocr
        parts.append(0.10 * ocr)
    score = float(np.clip(sum(parts), 0.0, 1.0))
    details["score"] = score
    return score, details


SHEET_FUTA_REPLACEMENT = "androgynous presentation"
SHEET_SAFE_WARN_PREFIX = "Character sheet safety sanitizer:"


def _normalize_list_or_str(v) -> list:
    """Accept either a string or list[str] and return list[str]."""
    if v is None:
        return []
    if isinstance(v, str):
        if not v.strip():
            return []
        return [v.strip()]
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _sanitize_character_prompt_tokens(
    tokens: list, negative_tokens: list, *, uncensored_mode: bool = False
) -> Tuple[list, list]:
    """
    Prevent explicitly sexual tokens from being injected.
    If user includes "futa" or similar, we replace with androgynous presentation.
    """
    if uncensored_mode:
        return tokens, negative_tokens
    banned_direct = ["futa", "trap"]
    lowered = [t.lower() for t in tokens]
    swapped = False
    for i, t in enumerate(tokens):
        tl = lowered[i]
        if any(b in tl for b in banned_direct):
            tokens[i] = SHEET_FUTA_REPLACEMENT
            swapped = True
    if swapped:
        # Add a mild negative to reduce explicit outcomes.
        negative_tokens.extend(["explicit genital content"])
        # Keep warning concise; don't spam if this is called repeatedly.
        print(
            f"{SHEET_SAFE_WARN_PREFIX} Replaced explicit gender term with '{SHEET_FUTA_REPLACEMENT}'.", file=sys.stderr
        )
    return tokens, negative_tokens


def _load_character_sheet(
    sheet_path: str, *, uncensored_mode: bool = False, character_strength: float = 1.0
) -> Tuple[str, str]:
    """
    Load a character sheet JSON file and return (positive_additions, negative_additions).
    Supported keys (all optional):
      - prompt / positive / appearance / style_tags / clothing / accessories
      - negative / negative_prompt
      - gender_presentation: androgynous|male|female
      - subject_label / character_slot: short name for multi-sheet labeling (e.g. left girl)
      - spatial_anchor / screen_position: e.g. left side, right foreground, background center
    Values can be strings or lists of strings.
    """
    p = Path(sheet_path)
    if not p.exists():
        raise ValueError(f"character-sheet not found: {p}")

    data = json_loads(p.read_text(encoding="utf-8", errors="ignore"))

    from utils.consistency.character_customization import build_character_prompt_additions

    pos, neg = build_character_prompt_additions(
        data,
        uncensored_mode=uncensored_mode,
        character_strength=character_strength,
    )
    return pos, neg


def _apply_character_gender_presentation(tokens: list, gender_presentation: str) -> list:
    gp = (gender_presentation or "").strip().lower()
    if gp in {"", "auto"}:
        return tokens
    if gp == "androgynous":
        tokens.append("androgynous presentation")
    elif gp == "male":
        tokens.append("male-presenting")
    elif gp == "female":
        tokens.append("female-presenting")
    return tokens


@torch.inference_mode()
def encode_text(
    captions,
    tokenizer,
    text_encoder,
    device,
    max_length=300,
    dtype=torch.float32,
    text_bundle=None,
    clip_captions=None,
    long_clip_captions=None,
    segment_texts=None,
):
    if text_bundle is not None:
        return text_bundle.encode(
            captions,
            device,
            max_length=max_length,
            dtype=dtype,
            train_fusion=False,
            clip_captions=clip_captions,
            long_clip_captions=long_clip_captions,
            segment_texts=segment_texts,
        )
    if segment_texts is not None:
        from utils.modeling.t5_segmented_encode import encode_t5_segment_concat

        return encode_t5_segment_concat(
            segment_texts, tokenizer, text_encoder, device, max_length=max_length, dtype=dtype
        )
    nbc = device.type == "cuda"
    tok = tokenizer(captions, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = tok.input_ids.to(device, non_blocking=nbc)
    attention_mask = tok.attention_mask.to(device, non_blocking=nbc)
    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state.to(dtype)


def main():  # pyright: ignore[reportGeneralTypeIssues] — body exceeds analyzer complexity limits
    parser = build_sample_parser()
    args = parser.parse_args()
    pick_report: dict = {
        "prompt": str(getattr(args, "prompt", "") or ""),
        "negative_prompt": str(getattr(args, "negative_prompt", "") or ""),
        "seed": int(getattr(args, "seed", 0) or 0),
        "steps": int(getattr(args, "steps", 0) or 0),
        "num": int(getattr(args, "num", 0) or 0),
    }

    # Apply preset and OP mode as soft defaults (only for unset args)
    if getattr(args, "preset", None):
        apply_preset_to_args(args, args.preset)
    if getattr(args, "op_mode", None):
        apply_op_mode_to_args(args, args.op_mode)
    if getattr(args, "holy_grail_preset", None):
        hg_name = str(getattr(args, "holy_grail_preset")).strip().lower()
        if hg_name == "auto":
            hg_name = recommend_holy_grail_preset(
                prompt=str(getattr(args, "prompt", "") or ""),
                style=str(getattr(args, "style", "") or ""),
                has_control=bool(getattr(args, "control_image", "") or getattr(args, "control", [])),
                has_lora=bool(getattr(args, "lora", [])),
            )
        apply_holy_grail_preset_to_args(args, hg_name)

    if str(getattr(args, "human_made", "none") or "none").lower() not in ("none", "off", "0", ""):
        from utils.quality.human_made import apply_human_made_prompt_flags

        apply_human_made_prompt_flags(args)

    _lin_attn = float(getattr(args, "linear_attn_fraction", 0.0) or 0.0)
    if _lin_attn > 0.0:
        from models.attention import set_linear_attention_fraction

        set_linear_attention_fraction(_lin_attn)

    has_tags = bool(getattr(args, "tags", "").strip() or getattr(args, "tags_file", "").strip())
    has_prompt_file = bool(getattr(args, "prompt_file", "").strip())
    has_prompt_layout = bool(getattr(args, "prompt_layout", "").strip())
    has_box_layout = bool(getattr(args, "box_layout", "").strip())
    if not (args.prompt or has_tags or has_prompt_file or has_prompt_layout or has_box_layout):
        parser.error(
            "Provide at least one of --prompt, --prompt-file, --tags, --tags-file, --prompt-layout, or --box-layout"
        )

    # Build effective prompt from --prompt-file, --tags / --tags-file, and optional --lora-trigger
    if has_prompt_file:
        pf_path = Path(args.prompt_file)
        prompt_for_encoding = (
            pf_path.read_text(encoding="utf-8", errors="ignore").strip()
            if pf_path.exists()
            else (args.prompt or "").strip()
        )
        if not prompt_for_encoding and pf_path.exists():
            print(f"Warning: --prompt-file is empty: {pf_path}", file=sys.stderr)
    else:
        prompt_for_encoding = (args.prompt or "").strip()
    tags_list = []
    if getattr(args, "tags", "").strip():
        tags_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    if getattr(args, "tags_file", "").strip():
        tags_path = Path(args.tags_file)
        if tags_path.exists():
            raw = tags_path.read_text(encoding="utf-8", errors="ignore")
            for line in raw.strip().split("\n"):
                for t in line.split(","):
                    if t.strip():
                        tags_list.append(t.strip())
        else:
            print(f"Warning: --tags-file not found: {tags_path}", file=sys.stderr)
    if tags_list:
        from data.caption_utils import prompt_from_tags

        tag_str = prompt_from_tags(tags_list)
        prompt_for_encoding = (tag_str + ", " + prompt_for_encoding) if prompt_for_encoding else tag_str
    if getattr(args, "lora_trigger", "").strip() and args.lora:
        trigger = args.lora_trigger.strip()
        prompt_for_encoding = f"{trigger}, {prompt_for_encoding}" if prompt_for_encoding else trigger

    args._prompt_layout_negative = ""
    args._used_prompt_layout = False
    args._layout_compiled = None
    args._box_layout_spec = None
    args._regional_cfg_plan = None
    if has_box_layout:
        try:
            from utils.generation.regional_box_prompting import layout_text_from_regions, load_box_layout_file

            box_spec = load_box_layout_file(str(args.box_layout).strip())
            args._box_layout_spec = box_spec
            layout_line = layout_text_from_regions(box_spec)
            if box_spec.global_prompt:
                prompt_for_encoding = (
                    f"{box_spec.global_prompt}, {prompt_for_encoding}" if prompt_for_encoding else box_spec.global_prompt
                )
            if not prompt_for_encoding:
                prompt_for_encoding = layout_line
            elif str(getattr(args, "box_layout_mode", "regional_cfg") or "regional_cfg").lower() == "text_only":
                prompt_for_encoding = layout_line if not prompt_for_encoding else f"{prompt_for_encoding}. {layout_line}"
            if box_spec.global_negative and not getattr(args, "negative_prompt", "").strip():
                args.negative_prompt = box_spec.global_negative
            print(
                f"Box layout: {len(box_spec.regions)} region(s) "
                f"({', '.join(r.name for r in box_spec.regions)})",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Warning: --box-layout failed: {e}", file=sys.stderr)
    if has_prompt_layout:
        try:
            from utils.prompt.prompt_layout import load_prompt_layout_file, merge_prompt_with_layout

            compiled = load_prompt_layout_file(str(args.prompt_layout).strip())
            args._prompt_layout_negative = compiled.negative or ""
            args._used_prompt_layout = True
            args._layout_compiled = compiled
            prompt_for_encoding = merge_prompt_with_layout(compiled.positive, prompt_for_encoding, layout_first=True)
        except Exception as e:
            print(f"Warning: --prompt-layout failed: {e}", file=sys.stderr)

    if (
        getattr(args, "subject_first", False)
        and prompt_for_encoding
        and not getattr(args, "_used_prompt_layout", False)
    ):
        from data.caption_utils import prompt_from_tags

        parts = [p.strip() for p in prompt_for_encoding.split(",") if p.strip()]
        if len(parts) > 1:
            prompt_for_encoding = prompt_from_tags(parts)
    if prompt_for_encoding:
        args.prompt = prompt_for_encoding

    anatomy_scales = _parse_scale_csv(getattr(args, "anatomy_scale", ""))
    object_scales = _parse_scale_csv(getattr(args, "object_scale", ""))
    scene_scales = _parse_scale_csv(getattr(args, "scene_scale", ""))

    if getattr(args, "gender_swap", False) and getattr(args, "prompt", ""):
        args.prompt = _apply_gender_swap(args.prompt)

    size_tokens = _build_size_tokens(anatomy_scales, object_scales, scene_scales)
    if size_tokens and getattr(args, "prompt", ""):
        args.prompt = f"{args.prompt}, {size_tokens}"
    elif size_tokens:
        args.prompt = size_tokens

    auto_oc_negative_additions = ""
    if getattr(args, "auto_original_character", True) and getattr(args, "prompt", ""):
        try:
            from utils.prompt.auto_oc import infer_auto_original_character

            _auto_profile = infer_auto_original_character(
                str(getattr(args, "prompt", "") or ""),
                seed=int(getattr(args, "seed", 0)) + int(getattr(args, "auto_oc_seed_offset", 0)),
                style_context=", ".join(
                    [
                        str(getattr(args, "style", "") or ""),
                        str(getattr(args, "hard_style", "") or ""),
                        str(getattr(args, "preset", "") or ""),
                        str(getattr(args, "op_mode", "") or ""),
                    ]
                ),
            )
            if _auto_profile is not None and not str(getattr(args, "character_sheet", "") or "").strip():
                _auto_block = _auto_profile.to_prompt_block()
                if _auto_block:
                    args.prompt = f"{args.prompt}, {_auto_block}".strip(", ")
                auto_oc_negative_additions = str(getattr(_auto_profile, "negative_block", "") or "").strip()
        except Exception:
            pass

    # Character sheet injection: adds user-defined character appearance tokens to prompt
    # and character-specific negative tokens to discourage drift.
    character_positive_additions = ""
    character_negative_additions = ""
    if getattr(args, "character_sheet", "").strip():
        pos_all = []
        neg_all = []
        sheet_paths = [s.strip() for s in str(args.character_sheet).split(",") if s.strip()]
        for sp in sheet_paths:
            try:
                pos_i, neg_i = _load_character_sheet(
                    sp,
                    uncensored_mode=bool(getattr(args, "uncensored_mode", False)),
                    character_strength=float(getattr(args, "character_strength", 1.0)),
                )
                if pos_i:
                    pos_all.append(pos_i)
                if neg_i:
                    neg_all.append(neg_i)
            except Exception as e:
                print(f"Warning: failed to load --character-sheet '{sp}': {e}", file=sys.stderr)
        use_labels = bool(getattr(args, "label_multi_character_sheets", False)) or (
            len(sheet_paths) >= 2 and str(getattr(args, "composition_mode", "none") or "none") == "multi_character"
        )
        if use_labels and len(pos_all) >= 2:
            from utils.prompt.multi_subject import (
                merge_character_sheet_negatives,
                merge_character_sheet_positives,
                multi_sheet_extra_negatives_csv,
            )

            character_positive_additions = merge_character_sheet_positives(pos_all)
            character_negative_additions = merge_character_sheet_negatives(neg_all)
            extra_neg = multi_sheet_extra_negatives_csv()
            if extra_neg:
                character_negative_additions = (
                    f"{character_negative_additions}, {extra_neg}".strip(", ")
                    if character_negative_additions
                    else extra_neg
                )
        else:
            character_positive_additions = ", ".join([x for x in pos_all if x]).strip(", ")
            character_negative_additions = ", ".join([x for x in neg_all if x]).strip(", ")
    if getattr(args, "character_negative_extra", "").strip():
        character_negative_additions = f"{character_negative_additions}, {args.character_negative_extra}".strip(", ")
    if auto_oc_negative_additions:
        character_negative_additions = f"{character_negative_additions}, {auto_oc_negative_additions}".strip(", ")
    if getattr(args, "character_prompt_extra", "").strip():
        character_positive_additions = f"{character_positive_additions}, {args.character_prompt_extra}".strip(", ")

    if character_positive_additions and getattr(args, "prompt", ""):
        args.prompt = f"{args.prompt}, {character_positive_additions}"
    elif character_positive_additions:
        args.prompt = character_positive_additions

    # Structured scene blueprint injection (actors, relations, camera, composition, constraints).
    scene_positive_additions = ""
    scene_negative_additions = ""
    if getattr(args, "scene_blueprint", "").strip():
        try:
            from utils.prompt.scene_blueprint import load_scene_blueprint

            scene_positive_additions, scene_negative_additions = load_scene_blueprint(
                args.scene_blueprint,
                strength=float(getattr(args, "scene_blueprint_strength", 1.0)),
            )
        except Exception as e:
            print(f"Warning: failed to load --scene-blueprint: {e}", file=sys.stderr)
            scene_positive_additions, scene_negative_additions = "", ""
    if scene_positive_additions and getattr(args, "prompt", ""):
        args.prompt = f"{args.prompt}, {scene_positive_additions}"
    elif scene_positive_additions:
        args.prompt = scene_positive_additions

    if getattr(args, "hard_style", None):
        try:
            from config.defaults.prompt_domains import HARD_STYLE_RECOMMENDED_PROMPTS

            prefix = HARD_STYLE_RECOMMENDED_PROMPTS.get(args.hard_style, [None])[0]
            if prefix and args.prompt:
                args.prompt = f"{prefix}, {args.prompt}"
            elif prefix:
                args.prompt = prefix
        except ImportError:
            pass

    if getattr(args, "naturalize", False) and args.prompt:
        try:
            from config.defaults.prompt_domains import NATURAL_LOOK_POSITIVE, NATURAL_LOOK_POSITIVE_DEEP

            _nat_pre = NATURAL_LOOK_POSITIVE_DEEP if getattr(args, "naturalize_deep", False) else NATURAL_LOOK_POSITIVE
            args.prompt = f"{_nat_pre}, {args.prompt}"
        except ImportError:
            try:
                from config.defaults.prompt_domains import NATURAL_LOOK_POSITIVE

                args.prompt = f"{NATURAL_LOOK_POSITIVE}, {args.prompt}"
            except ImportError:
                pass
    if getattr(args, "anti_bleed", False) and args.prompt:
        try:
            from config.defaults.prompt_domains import CONCEPT_BLEEDING_POSITIVE

            args.prompt = f"{CONCEPT_BLEEDING_POSITIVE}, {args.prompt}"
        except ImportError:
            pass
    if getattr(args, "diversity", False) and args.prompt:
        try:
            from config.defaults.prompt_domains import DIVERSITY_POSITIVE

            args.prompt = f"{DIVERSITY_POSITIVE}, {args.prompt}"
        except ImportError:
            pass

    # Originality / novelty: inject composition tokens (shared with train.py originality augment).
    if getattr(args, "originality", 0.0) and args.prompt:
        try:
            from utils.prompt.originality_augment import inject_originality_tokens

            strength = max(0.0, min(1.0, float(args.originality)))
            rng = np.random.default_rng(int(args.seed) + int(1000 * strength))
            args.prompt = inject_originality_tokens(args.prompt, strength, rng)
            if getattr(args, "creativity", None) is None:
                args.creativity = strength
        except Exception:
            pass

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if getattr(args, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if device.type == "cuda":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    elif device.type == "cuda":
        from utils.training.device_perf import configure_inference_cuda

        configure_inference_cuda(cudnn_benchmark=True, enable_tf32=True)

    print("Loading checkpoint and encoders...")
    model, cfg, rae_bridge, fusion_sd = load_model_from_ckpt(args.ckpt, device)

    lcm_ckpt = str(getattr(args, "lcm_ckpt", "") or "").strip()
    if lcm_ckpt:
        from utils.checkpoint.checkpoint_loading import load_dit_text_checkpoint

        lcm_model, _lcm_cfg, _lcm_bridge, _, _ = load_dit_text_checkpoint(
            lcm_ckpt, device=str(device), reject_enhanced=True
        )
        model.load_state_dict(lcm_model.state_dict(), strict=False)
        lcm_steps = int(getattr(args, "lcm_steps", 4) or 4)
        if int(args.steps) >= 16:
            args.steps = max(2, lcm_steps)
        if not getattr(args, "flow_matching_training", False):
            setattr(args, "flow_matching_sample", True)
        print(f"LCM few-step mode: loaded {lcm_ckpt}, steps={args.steps}, flow sampler.", file=sys.stderr)

    if getattr(args, "compile_inference", False) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Inference: torch.compile enabled (mode=reduce-overhead).")
        except Exception as comp_ex:
            print(f"Inference: torch.compile skipped ({comp_ex}).", file=sys.stderr)

    # Apply LoRAs
    if args.lora:
        from models.lora import apply_loras

        lora_specs = []
        role_counts = {}
        default_role = str(getattr(args, "lora_default_role", "style") or "style").strip().lower()
        for spec in args.lora:
            path, scale, role = _parse_lora_spec(spec, default_role=default_role)
            role = (role or default_role).strip().lower()
            lora_specs.append((path.strip(), float(scale), role))
            role_counts[role] = int(role_counts.get(role, 0)) + 1
        role_budgets = _parse_lora_role_budgets(getattr(args, "lora_role_budgets", ""))
        role_stage_weights = _parse_lora_role_stage_weights(getattr(args, "lora_role_stage_weights", ""))
        stage_policy = str(getattr(args, "lora_stage_policy", "auto") or "auto")
        _, num_keys = apply_loras(
            model,
            lora_specs,
            normalize_scales=not bool(getattr(args, "no_lora_normalize_scales", False)),
            max_total_scale=float(getattr(args, "lora_max_total_scale", 1.5)),
            role_budgets=role_budgets,
            stage_policy=stage_policy,
            layer_group=str(getattr(args, "lora_layers", "all") or "all"),
            role_stage_weights=role_stage_weights,
        )
        role_mix = ", ".join(f"{k}:{v}" for k, v in sorted(role_counts.items()))
        print(
            f"Applied {len(lora_specs)} adapter(s) (LoRA/DoRA/LyCORIS), {num_keys} layer(s), "
            f"normalize={not bool(getattr(args, 'no_lora_normalize_scales', False))}, "
            f"max_total={float(getattr(args, 'lora_max_total_scale', 1.5)):.2f}, "
            f"stage_policy={stage_policy}, roles=[{role_mix}]"
        )

    from diffusers import AutoencoderKL, AutoencoderRAE
    from transformers import AutoTokenizer, T5EncoderModel
    from utils.modeling.text_encoder_bundle import attach_fusion_weights, load_text_encoder_bundle

    text_bundle = None
    _enc_mode = str(getattr(cfg, "text_encoder_mode", "t5") or "t5").lower()
    if _enc_mode in ("triple", "penta"):
        text_bundle = load_text_encoder_bundle(cfg, device)
        if text_bundle is None:
            raise RuntimeError(f"Checkpoint config requests {_enc_mode} text encoders but bundle failed to load.")
        if fusion_sd is not None:
            attach_fusion_weights(text_bundle, fusion_sd)
        tokenizer = text_bundle.tokenizer
        text_encoder = text_bundle.text_encoder
        if _enc_mode == "penta":
            print("Penta text encoder (T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L) loaded.")
        else:
            print("Triple text encoder (T5 + CLIP-L + CLIP-bigG) loaded.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder)
        text_encoder = T5EncoderModel.from_pretrained(cfg.text_encoder).to(device).eval()
    # Warn if prompt is very long (T5 truncates at max_length; important content may be lost)
    try:
        tok_out = tokenizer(args.prompt, return_tensors="pt", truncation=False)
        n_tok = tok_out.input_ids.shape[1]
        if n_tok > 250:
            print(
                f"Note: prompt has {n_tok} tokens; T5 truncates at 300. Put key elements first for best adherence.",
                file=sys.stderr,
            )
    except Exception:
        pass
    ae_type = getattr(cfg, "autoencoder_type", "kl")
    if ae_type == "rae":
        vae = AutoencoderRAE.from_pretrained(cfg.vae_model).to(device).eval()
        latent_scale = 1.0  # RAE checkpoints handle latent normalization internally
    else:
        vae = AutoencoderKL.from_pretrained(cfg.vae_model).to(device).eval()
        latent_scale = getattr(cfg, "latent_scale", 0.18215)
    image_size = getattr(cfg, "image_size", 256)

    # Compatibility guard for RAE: this repo's DiT expects SD-style 4-channel latents.
    if ae_type == "rae":
        latent_hw_expected = int(image_size) // 8
        latent_channels_expected = 4
        ae_cfg = getattr(vae, "config", None)
        latent_channels_rae = getattr(ae_cfg, "encoder_hidden_size", None) if ae_cfg is not None else None
        encoder_input_size = getattr(ae_cfg, "encoder_input_size", None) if ae_cfg is not None else None
        encoder_patch_size = getattr(ae_cfg, "encoder_patch_size", None) if ae_cfg is not None else None
        latent_hw_rae = None
        if encoder_input_size is not None and encoder_patch_size is not None:
            try:
                latent_hw_rae = int(encoder_input_size) // int(encoder_patch_size)
            except Exception:
                latent_hw_rae = None

        if latent_channels_rae is not None and int(latent_channels_rae) != latent_channels_expected:
            if rae_bridge is None:
                raise ValueError(
                    "AutoencoderRAE latent channels != 4 but checkpoint has no rae_latent_bridge. "
                    f"encoder_hidden_size={latent_channels_rae}. Train with "
                    "`train.py --autoencoder-type rae` (bridge is created automatically) and sample this checkpoint."
                )
        if latent_hw_rae is not None and int(latent_hw_rae) != latent_hw_expected:
            raise ValueError(
                "AutoencoderRAE selected, but the RAE latent spatial size doesn't match this repo's assumptions. "
                f"RAE latent_hw={latent_hw_rae}, expected={latent_hw_expected} (image_size//8). "
                "To use RAE properly, update the DiT/diffusion latent spatial assumptions."
            )
    out_w = args.width if getattr(args, "width", 0) > 0 else image_size
    out_h = args.height if getattr(args, "height", 0) > 0 else image_size
    if getattr(args, "vae_tiling", False) or (out_w * out_h > 512 * 512):
        try:
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
                print("Autoencoder tiling enabled (decode in tiles for lower VRAM).")
        except Exception:
            pass
    latent_size = image_size // 8

    diffusion = create_diffusion(
        timestep_respacing=getattr(cfg, "timestep_respacing", ""),
        num_timesteps=getattr(cfg, "num_timesteps", 1000),
        beta_schedule=getattr(cfg, "beta_schedule", "linear"),
        prediction_type=getattr(cfg, "prediction_type", "epsilon"),
    )

    # Encode prompts (cond = positive, uncond = negative; default negative when empty)
    if getattr(args, "explore_styles_insane", False):
        args.explore_styles = True
        args.style_inventor_mode = "apocalypse"
        args.style_chaos_level = max(float(getattr(args, "style_chaos_level", 0) or 0), 0.95)
        args.invent_styles = max(int(getattr(args, "invent_styles", 0) or 0), 4)
        _cl = str(getattr(args, "prompt_clauses", "") or "").strip()
        _insane_clauses = "style.chaos,style.surreal,style.apocalypse"
        args.prompt_clauses = f"{_cl},{_insane_clauses}".strip(",") if _cl else _insane_clauses
        print("Explore-styles-insane: apocalypse mode engaged.", file=sys.stderr)

    if getattr(args, "explore_styles", False):
        if int(getattr(args, "invent_styles", 0) or 0) < 1:
            args.invent_styles = 3
        _pb_explore = str(getattr(args, "pick_best", "none") or "none").strip().lower()
        if _pb_explore in ("none", ""):
            args.pick_best = "combo"
        print(
            f"Explore-styles: inventing {args.invent_styles} genomes "
            f"(mode={getattr(args, 'style_inventor_mode', 'normal')}, "
            f"chaos={float(getattr(args, 'style_chaos_level', 0) or 0):.2f}); "
            f"use --style-genome-index 0..{args.invent_styles - 1} per run or scripts.tools.explore_styles",
            file=sys.stderr,
        )

    num_gen = max(1, getattr(args, "num", 1))
    if getattr(args, "explore_styles", False) and num_gen < int(getattr(args, "invent_styles", 3) or 3):
        num_gen = int(args.invent_styles)
        args.num = num_gen
    if num_gen < 2 and str(getattr(args, "pick_best", "none")).lower() not in ("none", ""):
        print("Note: --pick-best only applies with --num >= 2; ignoring.", file=sys.stderr)
    # Resolve emphasis (word)/[word] first so we have prompt_to_encode for conflict filter
    if "(" in args.prompt or "[" in args.prompt:
        prompt_to_encode, _emphasis_segments = parse_prompt_emphasis(args.prompt)
    else:
        prompt_to_encode, _emphasis_segments = args.prompt, []
    if getattr(args, "expand_prompt", False) and prompt_to_encode.strip():
        from utils.superior.prompt_expand import expand_prompt_heuristic

        prompt_to_encode = expand_prompt_heuristic(prompt_to_encode)
        args.prompt = prompt_to_encode
        print("Superior: expanded prompt heuristically.", file=sys.stderr)
    _hm = str(getattr(args, "human_made", "none") or "none").lower().strip()
    if _hm not in ("none", "off", "0", ""):
        from utils.quality.human_made import append_human_made_prompt_fragments

        prompt_to_encode, args.negative_prompt = append_human_made_prompt_fragments(
            prompt_to_encode,
            str(getattr(args, "negative_prompt", "") or ""),
            _hm,
        )
        args.prompt = prompt_to_encode
    local_rag = str(getattr(args, "local_rag_jsonl", "") or "").strip()
    if local_rag:
        try:
            from utils.prompt.rag_prompt import merge_facts_into_prompt
            from utils.superior.auto_stack import SuperiorPromptStack

            stack = SuperiorPromptStack(rag_jsonl=local_rag, rag_top_k=int(getattr(args, "local_rag_top_k", 8) or 8))
            prompt_to_encode = stack.enrich(prompt_to_encode)
            args.prompt = prompt_to_encode
            print(f"Local RAG: enriched prompt from {local_rag}", file=sys.stderr)
        except Exception as e:
            print(f"Local RAG skipped: {e}", file=sys.stderr)
    # Optional agentic-search grounding (e.g. Gen-Searcher JSON/JSONL output).
    facts_path = str(getattr(args, "agentic_facts_json", "") or "").strip()
    if facts_path:
        try:
            from utils.prompt.rag_prompt import (
                load_facts_from_gen_searcher_json,
                load_facts_from_jsonl,
                merge_facts_into_prompt,
            )

            facts_mode = str(getattr(args, "agentic_facts_format", "auto") or "auto").lower().strip()
            max_facts = max(1, int(getattr(args, "agentic_max_facts", 16) or 16))
            max_chars = max(256, int(getattr(args, "agentic_facts_max_chars", 2400) or 2400))
            facts: list = []
            if facts_mode in ("auto", "gen_searcher"):
                facts = load_facts_from_gen_searcher_json(
                    facts_path,
                    max_entries=max_facts,
                )
            if not facts and facts_mode in ("auto", "jsonl_text"):
                facts = load_facts_from_jsonl(
                    facts_path,
                    max_entries=max_facts,
                )
            if facts:
                prompt_to_encode = merge_facts_into_prompt(
                    prompt_to_encode,
                    facts,
                    max_chars=max_chars,
                )
                args.prompt = prompt_to_encode
                print(
                    f"Agentic grounding merged {len(facts)} facts from {facts_path}",
                    file=sys.stderr,
                )
            else:
                print(f"Agentic grounding: no facts found in {facts_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: agentic fact grounding failed: {e}", file=sys.stderr)

    _comp_brief = str(getattr(args, "composition_brief", "off") or "off")
    if _comp_brief != "off" and prompt_to_encode.strip():
        try:
            from utils.prompt.composition_brief import apply_composition_brief

            prompt_to_encode = apply_composition_brief(
                prompt_to_encode,
                _comp_brief,  # type: ignore[arg-type]
            )
            args.prompt = prompt_to_encode
        except Exception as _cb_e:
            print(f"Composition brief skipped: {_cb_e}", file=sys.stderr)

    setattr(args, "_visual_design_negative", "")
    _vd_preset = str(getattr(args, "visual_design_preset", "") or "").strip().lower()
    if _vd_preset and prompt_to_encode.strip():
        try:
            from utils.visual_design.presets import apply_visual_design_preset_to_prompt

            prompt_to_encode, _vd_pre_dom, _vd_pre_int = apply_visual_design_preset_to_prompt(
                prompt_to_encode, _vd_preset
            )
            args.prompt = prompt_to_encode
            setattr(args, "visual_design_domain", _vd_pre_dom)
            setattr(args, "visual_design_intensity", _vd_pre_int)
        except Exception as _vp_e:
            print(f"Visual design preset skipped: {_vp_e}", file=sys.stderr)

    _vd_dom = str(getattr(args, "visual_design_domain", "none") or "none").lower().strip()
    if _vd_dom != "none" and prompt_to_encode.strip():
        try:
            from utils.visual_design.sampling import apply_visual_design_stage

            vd_out = apply_visual_design_stage(
                prompt_to_encode,
                cli_domain=_vd_dom,
                intensity=str(getattr(args, "visual_design_intensity", "standard") or "standard"),
                use_negative_pack=bool(getattr(args, "visual_design_negative_pack", False)),
                emit=lambda m: print(m, file=sys.stderr),
            )
            prompt_to_encode = vd_out.prompt
            args.prompt = prompt_to_encode
            if vd_out.negative_addon:
                setattr(args, "_visual_design_negative", vd_out.negative_addon)
        except Exception as _vd_e:
            print(f"Visual design pack skipped: {_vd_e}", file=sys.stderr)

    # Creative RAG: semantic conflict resolution + intent-aware enrichment + optional moondream2/Qwen2.5
    if getattr(args, "creative_rag", False) and prompt_to_encode.strip():
        try:
            from utils.prompt.advanced_prompting import (
                CreativeRAGOptimizer,
                classify_prompt_intent,
                resolve_semantic_conflicts,
            )

            _rag_level = float(getattr(args, "creative_rag_level", 0.7) or 0.7)
            _rag_image = str(getattr(args, "creative_rag_image", "") or "").strip()
            _rag_multi = str(getattr(args, "creative_rag_images", "") or "").strip()
            _rag_resolve = bool(getattr(args, "creative_rag_resolve_conflicts", True))

            _rag_ref_list: list[str] = []
            if _rag_image:
                _rag_ref_list.append(_rag_image)
            if _rag_multi:
                for _p in _rag_multi.split(","):
                    _s = _p.strip()
                    if _s and _s not in _rag_ref_list:
                        _rag_ref_list.append(_s)
            _rag_ref_list = _rag_ref_list[:16]

            # Always run semantic conflict resolution (fast, no heavy models)
            if _rag_resolve:
                _resolved, _removed = resolve_semantic_conflicts(prompt_to_encode)
                if _removed:
                    print(
                        f"Creative RAG: resolved {len(_removed)} semantic contradiction(s): "
                        + ", ".join(f"'{r}'" for r in _removed[:4]),
                        file=sys.stderr,
                    )
                    prompt_to_encode = _resolved
                    args.prompt = prompt_to_encode

            # Classify intent and report
            _intent = classify_prompt_intent(prompt_to_encode)
            print(f"Creative RAG: detected intent={_intent}, creativity_level={_rag_level:.2f}", file=sys.stderr)
            if _rag_ref_list:
                print(
                    f"Creative RAG: reference images={len(_rag_ref_list)} (facts/dissection up to 16; moondream up to 8)",
                    file=sys.stderr,
                )

            # Full enrichment (uses moondream2 + Qwen2.5 if available, else fast fallback)
            _rag_opt = CreativeRAGOptimizer(device=str(device))
            _rag_result = _rag_opt.optimize(
                prompt_to_encode,
                reference_image_paths=_rag_ref_list if _rag_ref_list else None,
                creativity_level=_rag_level,
                optimization_level="balanced",
                seed=int(getattr(args, "seed", 42)),
                use_rag=True,
            )

            if _rag_result.get("optimized_prompt", "").strip():
                _enriched = _rag_result["optimized_prompt"]
                _added_count = len(_enriched.split(",")) - len(prompt_to_encode.split(","))
                _fallback = getattr(_rag_result.get("rag_result"), "fallback_used", True)
                _mode = "semantic fallback" if _fallback else "Qwen2.5 synthesis"
                print(
                    f"Creative RAG ({_mode}): added {max(0, _added_count)} enrichment(s) [intent={_intent}]",
                    file=sys.stderr,
                )
                prompt_to_encode = _enriched
                args.prompt = prompt_to_encode

                # Merge RAG negative additions into the negative prompt
                _rag_neg = str(_rag_result.get("negative_additions", "") or "").strip()
                if _rag_neg and not getattr(args, "negative_prompt", "").strip():
                    # Only auto-inject if user didn't provide a custom negative
                    # (we don't want to override user intent)
                    pass  # negative additions are available via _rag_result for downstream use

        except Exception as e:
            print(f"Creative RAG failed (non-fatal): {e}", file=sys.stderr)
    if getattr(args, "boost_quality", False) and prompt_to_encode.strip():
        try:
            from data.caption_utils import QUALITY_PREFIX

            if not prompt_to_encode.strip().lower().startswith("masterpiece"):
                prompt_to_encode = f"{QUALITY_PREFIX}{prompt_to_encode}".strip()
        except ImportError:
            prompt_to_encode = f"masterpiece, best quality, {prompt_to_encode}".strip()

    setattr(args, "_multi_instance_negative", "")
    _mi_preset = str(getattr(args, "multi_instance_preset", "none") or "none").lower().strip()
    if _mi_preset not in ("none", "") and prompt_to_encode.strip():
        try:
            from utils.prompt.multi_instance_scene import apply_multi_instance_preset, describe_limitation_short

            _mi_user_n = max(0, int(getattr(args, "multi_instance_count", 0) or 0))
            prompt_to_encode, _mi_neg, _mi_min_n, _mi_ec, _mi_note = apply_multi_instance_preset(
                prompt_to_encode,
                _mi_preset,
                user_expected_count=_mi_user_n,
            )
            args.prompt = prompt_to_encode
            if _mi_neg:
                setattr(args, "_multi_instance_negative", _mi_neg)
            if _mi_min_n > num_gen:
                num_gen = max(num_gen, _mi_min_n)
                args.num = int(num_gen)
                print(
                    f"Multi-instance preset '{_mi_preset}': raised --num to {num_gen} for broader search "
                    f"({describe_limitation_short()})",
                    file=sys.stderr,
                )
            if _mi_user_n > 0:
                setattr(args, "expected_count", int(_mi_user_n))
            elif _mi_ec is not None and _mi_ec > 0 and int(getattr(args, "expected_count", 0) or 0) <= 0:
                setattr(args, "expected_count", int(_mi_ec))
            if _mi_note:
                print(f"Multi-instance: {_mi_note}", file=sys.stderr)
        except Exception as _mi_e:
            print(f"Multi-instance preset skipped: {_mi_e}", file=sys.stderr)

    if (
        bool(getattr(args, "multi_instance_auto", False))
        and _mi_preset not in ("none", "")
        and prompt_to_encode.strip()
    ):
        try:
            from utils.prompt.composition_brief import apply_composition_brief
            from utils.prompt.multi_instance_scene import (
                multi_instance_auto_settings,
                print_multi_instance_hints,
            )

            _st = multi_instance_auto_settings(_mi_preset)
            if _st.get("composition_brief_boost") and str(getattr(args, "composition_brief", "off")).lower() == "off":
                setattr(args, "composition_brief", "auto")
                prompt_to_encode = apply_composition_brief(prompt_to_encode, "auto")
                args.prompt = prompt_to_encode
                print("Multi-instance auto: composition-brief -> auto", file=sys.stderr)
            _pb0 = str(getattr(args, "pick_best", "none") or "none").strip().lower()
            if _pb0 in ("none", "") and _st.get("pick_best"):
                setattr(args, "pick_best", str(_st["pick_best"]))
                print(f"Multi-instance auto: pick-best -> {_st['pick_best']}", file=sys.stderr)
            _mic_auto = int(getattr(args, "multi_instance_count", 0) or 0)
            _ect0 = str(getattr(args, "expected_count_target", "auto") or "auto").lower().strip()
            if _mic_auto > 0 and _ect0 == "auto" and _st.get("expected_count_target"):
                setattr(args, "expected_count_target", str(_st["expected_count_target"]))
                print(
                    f"Multi-instance auto: expected-count-target -> {_st['expected_count_target']}",
                    file=sys.stderr,
                )
            if str(getattr(args, "pick_best", "none") or "").strip().lower() not in ("none", ""):
                if num_gen < 2:
                    num_gen = max(2, num_gen)
                    setattr(args, "num", int(num_gen))
                    print("Multi-instance auto: --num raised to >= 2 because pick-best is active", file=sys.stderr)
            print_multi_instance_hints(_mi_preset)
        except Exception as _mia:
            print(f"Multi-instance auto skipped: {_mia}", file=sys.stderr)

    setattr(args, "_detailed_scene_negative", "")
    _dsb = str(getattr(args, "detailed_scene_boost", "auto") or "auto")
    _dss = str(getattr(args, "detailed_scene_strength", "lite") or "lite")
    if _dsb != "off" and prompt_to_encode.strip():
        try:
            from utils.prompt.detailed_scene_entities import (
                apply_detailed_scene_boost,
                detailed_scene_warrants_boost,
                extract_key_segments,
            )

            if _dsb == "on" or (_dsb == "auto" and detailed_scene_warrants_boost(prompt_to_encode)):
                _ds_pt, _ds_neg = apply_detailed_scene_boost(
                    prompt_to_encode,
                    "on",
                    strength=_dss,  # type: ignore[arg-type]
                )
                if _ds_neg:
                    setattr(args, "_detailed_scene_negative", _ds_neg)
                if _ds_pt != prompt_to_encode:
                    prompt_to_encode = _ds_pt
                    args.prompt = prompt_to_encode
                _segs = extract_key_segments(prompt_to_encode)
                if _segs:
                    print(
                        f"Detailed-scene boost ({_dss}): {len(_segs)} key clauses (e.g. {_segs[:3]!r})",
                        file=sys.stderr,
                    )
        except Exception as _dse:
            print(f"Detailed-scene boost skipped: {_dse}", file=sys.stderr)

    expected_texts = _parse_expected_texts(getattr(args, "expected_text", ""))
    if not expected_texts and bool(getattr(args, "auto_expected_text", True)):
        expected_texts = _infer_expected_texts_from_prompt(prompt_to_encode)
        if expected_texts:
            print(f"Inferred expected text from prompt: {expected_texts}", file=sys.stderr)
    expected_count = int(getattr(args, "expected_count", 0) or 0)
    expected_count_target = str(getattr(args, "expected_count_target", "auto") or "auto").lower().strip()
    if expected_count <= 0:
        try:
            from utils.quality.test_time_pick import infer_expected_object_count, infer_expected_people_count
        except Exception:
            infer_expected_people_count = None
            infer_expected_object_count = None
        if expected_count_target == "objects":
            if infer_expected_object_count is not None:
                expected_count, _ = infer_expected_object_count(prompt_to_encode)
        else:
            if infer_expected_people_count is not None:
                expected_count = infer_expected_people_count(prompt_to_encode)
            if expected_count <= 0 and expected_count_target == "auto" and infer_expected_object_count is not None:
                expected_count, _ = infer_expected_object_count(prompt_to_encode)
    if bool(getattr(args, "auto_constraint_boost", True)):
        has_constraints = bool(expected_texts) or expected_count > 0
        if has_constraints and num_gen < 4:
            print(f"Auto constraint boost: num {num_gen} -> 4", file=sys.stderr)
            num_gen = 4
    if getattr(args, "ocr_fix", False) and expected_texts:
        # Encourage the model not to suppress text and bias toward exact content.
        args.text_in_image = True
        prompt_to_encode = _maybe_append_text_says(prompt_to_encode, expected_texts)
        args.prompt = prompt_to_encode
    _pr_grade_ef = str(getattr(args, "_pr_grade_ef", "none") or "none")
    _pr_filter_ef = str(getattr(args, "_pr_filter_ef", "none") or "none")
    _pr_grain_ef = str(getattr(args, "_pr_grain_ef", "none") or "none")
    _pr_pack_ef = str(getattr(args, "_pr_pack_ef", "none") or "none")
    _is_photo_prompt = bool(getattr(args, "_is_photo_prompt", False))
    _auto_pick_metric = str(getattr(args, "_auto_pick_metric", "") or "")

    from utils.prompt.stack import apply_sample_prompt_stack

    prompt_to_encode, negative_text = apply_sample_prompt_stack(
        args,
        prompt_to_encode,
        character_negative_additions=character_negative_additions,
        scene_negative_additions=scene_negative_additions,
        apply_scale_distortion=bool(anatomy_scales or object_scales or scene_scales),
    )
    args.prompt = prompt_to_encode
    # Style: explicit --style (supports weighted mix with '|') or auto-extract from prompt.
    effective_style = (args.style or "").strip()
    if getattr(cfg, "style_embed_dim", 0) and not effective_style and getattr(args, "auto_style_from_prompt", False):
        try:
            from config.defaults.style_artists import extract_style_from_text

            effective_style = extract_style_from_text(prompt_to_encode) or ""
            if effective_style:
                print(
                    f'Auto-style from prompt: "{effective_style[:50]}{"..." if len(effective_style) > 50 else ""}"',
                    file=sys.stderr,
                )
        except Exception:
            pass
    style_mix = _parse_weighted_style_mix(effective_style)
    if style_mix:
        style_key = "|".join(f"{t}::{w:.4f}" for t, w in style_mix)
    else:
        style_key = effective_style if getattr(cfg, "style_embed_dim", 0) else ""
    compiled_layout = getattr(args, "_layout_compiled", None)
    raw_t5_layout = str(getattr(args, "t5_layout_encode", "auto") or "auto").lower()
    if raw_t5_layout == "auto":
        t5_layout_enc = "blocks" if compiled_layout is not None else "flat"
    else:
        t5_layout_enc = raw_t5_layout
    _t5_hint = str(getattr(args, "_encode_t5_positive_hint", "") or "").strip()
    t5_positive_prompt = _t5_hint if _t5_hint else prompt_to_encode
    segment_texts = None
    clip_caps = None
    long_clip_caps = None
    if compiled_layout is not None and t5_layout_enc == "blocks":
        from utils.prompt.prompt_layout import substitute_compiled_layout_in_t5_prompt

        t5_positive_prompt = substitute_compiled_layout_in_t5_prompt(prompt_to_encode, compiled_layout)
    elif compiled_layout is not None and t5_layout_enc == "segmented":
        from utils.prompt.prompt_layout import t5_segment_texts_for_full_prompt

        segment_texts = t5_segment_texts_for_full_prompt(compiled_layout, prompt_to_encode)
    if text_bundle is not None and compiled_layout is not None:
        from utils.prompt.prompt_layout import multi_clip_caption

        clip_caps = [multi_clip_caption(compiled_layout, prompt_to_encode)]
    if text_bundle is not None and getattr(text_bundle, "mode", "") == "penta":
        long_clip_caps = [prompt_to_encode]

    layout_cache_tag = (
        (t5_layout_enc, str(getattr(args, "prompt_layout", "") or "")) if compiled_layout is not None else ("flat", "")
    )
    _t5_cache_extra = hashlib.sha256(t5_positive_prompt.encode("utf-8")).hexdigest()[:20]
    cache_key = (
        (prompt_to_encode, negative_text, style_key, layout_cache_tag, _t5_cache_extra)
        if not getattr(args, "no_cache", False)
        else None
    )
    if cache_key is not None and cache_key in _t5_cache:
        cond_emb, uncond_emb, style_emb_cached = _t5_cache[cache_key]
        cond_emb = cond_emb.to(device)
        uncond_emb = uncond_emb.to(device)
        if style_emb_cached is not None:
            style_emb_cached = style_emb_cached.to(device)
        print("T5 cache hit.")
    else:
        if segment_texts is not None:
            cond_emb = encode_text(
                [t5_positive_prompt],
                tokenizer,
                text_encoder,
                device,
                text_bundle=text_bundle,
                clip_captions=clip_caps,
                long_clip_captions=long_clip_caps,
                segment_texts=segment_texts,
            )
            uncond_emb = encode_text([negative_text], tokenizer, text_encoder, device, text_bundle=text_bundle)
        else:
            both_long = long_clip_caps
            if text_bundle is not None and getattr(text_bundle, "mode", "") == "penta":
                both_long = [prompt_to_encode, negative_text]
            both_emb = encode_text(
                [t5_positive_prompt, negative_text],
                tokenizer,
                text_encoder,
                device,
                text_bundle=text_bundle,
                clip_captions=clip_caps,
                long_clip_captions=both_long,
            )
            cond_emb, uncond_emb = both_emb[0:1], both_emb[1:2]
        style_emb_cached = None
        if effective_style and getattr(cfg, "style_embed_dim", 0):
            if style_mix and len(style_mix) > 1:
                style_texts = [t for t, _ in style_mix]
                style_weights = torch.tensor([w for _, w in style_mix], device=device, dtype=torch.float32)
                style_enc = encode_text(style_texts, tokenizer, text_encoder, device, text_bundle=text_bundle).mean(
                    dim=1
                )
                style_emb_cached = (style_enc * style_weights[:, None].to(style_enc.dtype)).sum(dim=0, keepdim=True)
            else:
                style_emb_cached = encode_text(
                    [effective_style], tokenizer, text_encoder, device, text_bundle=text_bundle
                ).mean(dim=1)
        if cache_key is not None:
            while len(_t5_cache) >= _T5_CACHE_MAX:
                _t5_cache.pop(next(iter(_t5_cache)))
            _t5_cache[cache_key] = (
                cond_emb.cpu(),
                uncond_emb.cpu(),
                style_emb_cached.cpu() if style_emb_cached is not None else None,
            )
    if num_gen > 1:
        cond_emb = cond_emb.expand(num_gen, -1, -1)
        uncond_emb = uncond_emb.expand(num_gen, -1, -1)
    model_kwargs_cond = {"encoder_hidden_states": cond_emb}
    model_kwargs_uncond = {"encoder_hidden_states": uncond_emb}
    if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
        lh = max(1, image_size // 8)
        lw = max(1, image_size // 8)
        bsz = cond_emb.shape[0]
        sz = torch.tensor([[float(lh), float(lw)]], device=device, dtype=torch.float32).expand(bsz, -1)
        model_kwargs_cond["size_embed"] = sz
        model_kwargs_uncond["size_embed"] = sz
    if style_emb_cached is not None:
        if num_gen > 1:
            style_emb_cached = style_emb_cached.expand(num_gen, -1)
        model_kwargs_cond["style_embedding"] = style_emb_cached
        model_kwargs_cond["style_strength"] = args.style_strength
    # IMPROVEMENTS 2.2: prompt emphasis (word) -> 1.2, [word] -> 0.8
    if _emphasis_segments:
        tw = token_weights_from_cleaned_segments(prompt_to_encode, _emphasis_segments, tokenizer, 300, device=device)
        if tw is not None:
            model_kwargs_cond["token_weights"] = tw

    # Creativity/diversity knob (only if model has creativity_embed_dim)
    if getattr(cfg, "creativity_embed_dim", 0) and args.creativity is not None:
        c0 = max(0.0, min(1.0, float(args.creativity)))
        jitter = max(0.0, float(getattr(args, "creativity_jitter", 0.0) or 0.0))
        if jitter > 0 and num_gen >= 1:
            g = torch.Generator(device=device)
            g.manual_seed(int(args.seed) + 90211)
            deltas = torch.randn(num_gen, generator=g, device=device, dtype=torch.float32) * jitter
            cre = (c0 + deltas).clamp(0.0, 1.0).to(dtype=cond_emb.dtype)
        else:
            cre = torch.full((num_gen,), c0, device=device, dtype=cond_emb.dtype)
        model_kwargs_cond["creativity"] = cre

    # Control image(s): supports single --control-image and stacked --control specs.
    control_specs: list[tuple[str, str, float]] = []
    if args.control_image:
        control_specs.append(
            (
                str(args.control_image),
                str(getattr(args, "control_type", "auto") or "auto"),
                float(getattr(args, "control_scale", 0.85)),
            )
        )
    for raw_spec in list(getattr(args, "control", []) or []):
        p, t, s = _parse_control_spec(
            raw_spec,
            default_type=str(getattr(args, "control_type", "auto") or "auto"),
            default_scale=float(getattr(args, "control_scale", 0.85)),
        )
        if p:
            control_specs.append((p, t, s))
    box_spec_ctrl = getattr(args, "_box_layout_spec", None)
    if box_spec_ctrl is not None and not control_specs:
        try:
            from utils.generation.regional_box_sketch import build_composite_sketch_tensor, spec_has_sketches

            if spec_has_sketches(box_spec_ctrl):
                sk = build_composite_sketch_tensor(
                    box_spec_ctrl,
                    image_size,
                    source_dir=getattr(box_spec_ctrl, "source_dir", None),
                    device=device,
                )
                model_kwargs_cond["control_image"] = sk.unsqueeze(0)
                model_kwargs_cond["control_scale"] = float(
                    getattr(args, "box_sketch_control_scale", 0.75) or 0.75
                )
                from models.controlnet import control_type_to_id

                model_kwargs_cond["control_type"] = torch.tensor(
                    [int(control_type_to_id("scribble"))], device=device, dtype=torch.long
                )
                print("Box-layout sketch control image enabled (scribble).", file=sys.stderr)
        except Exception as e:
            print(f"Warning: box sketch control image skipped: {e}", file=sys.stderr)
    if control_specs:
        ctrl_tensors = []
        ctrl_type_ids = []
        ctrl_scales = []
        for cpath, ctype_raw, cscale in control_specs:
            pil = Image.open(cpath).convert("RGB")
            w, h = pil.size
            if w != image_size or h != image_size:
                pil = pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
            arr = np.array(pil).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            ctrl_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            ct_name = str(ctype_raw or "auto").strip().lower()
            if ct_name in {"", "auto"}:
                ct_name = infer_control_type_from_path(str(cpath))
            ctrl_type_ids.append(int(control_type_to_id(ct_name)))
            ctrl_scales.append(float(max(0.0, cscale)))
        if len(ctrl_tensors) == 1:
            model_kwargs_cond["control_image"] = ctrl_tensors[0].unsqueeze(0).to(device)
            model_kwargs_cond["control_scale"] = float(ctrl_scales[0])
            model_kwargs_cond["control_type"] = torch.tensor([ctrl_type_ids[0]], device=device, dtype=torch.long)
        else:
            stack = torch.stack(ctrl_tensors, dim=0).unsqueeze(0).to(device)  # (1, K, C, H, W)
            model_kwargs_cond["control_image"] = stack
            model_kwargs_cond["control_scale"] = torch.tensor(ctrl_scales, device=device, dtype=torch.float32)
            model_kwargs_cond["control_type"] = torch.tensor(ctrl_type_ids, device=device, dtype=torch.long)

    ref_path = str(getattr(args, "reference_image", "") or "").strip()
    ref_strength = float(getattr(args, "reference_strength", 0.0) or 0.0)
    if ref_path and ref_strength > 0:
        try:
            from models.reference_token_projection import ReferenceTokenProjector
            from utils.generation.clip_reference_embed import encode_reference_image_pil

            pil_r = Image.open(ref_path).convert("RGB")
            clip_id = str(getattr(args, "reference_clip_model", "") or "openai/clip-vit-large-patch14")
            emb, clip_dim = encode_reference_image_pil(pil_r, device=device, model_id=clip_id, dtype=torch.float32)
            if num_gen > 1:
                emb = emb.expand(num_gen, -1)
            hs = int(getattr(cfg, "hidden_size", 1152))
            ntok = max(1, int(getattr(args, "reference_tokens", 4) or 4))
            proj = ReferenceTokenProjector(clip_dim, hs, ntok).to(device)
            apt = str(getattr(args, "reference_adapter_pt", "") or "").strip()
            if apt:
                sd = torch.load(apt, map_location="cpu", weights_only=False)
                if isinstance(sd, dict):
                    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                        sd = sd["state_dict"]
                    elif "projector" in sd and isinstance(sd["projector"], dict):
                        sd = sd["projector"]
                proj.load_state_dict(sd, strict=False)
            proj.eval()
            with torch.no_grad():
                rt = proj(emb)
            model_kwargs_cond["reference_tokens"] = rt
            model_kwargs_cond["reference_scale"] = ref_strength
        except Exception as e:
            print(f"Reference image conditioning skipped: {e}", file=sys.stderr)

    # Optional: prompt-driven dissection -> composite init image + inpaint mask.
    # Only runs when explicitly enabled and when user hasn't already provided init/mask.
    if (
        bool(getattr(args, "auto_init_from_dissection", False))
        and not str(getattr(args, "init_image", "") or "").strip()
        and not str(getattr(args, "mask", "") or "").strip()
    ):
        raw_refs = str(getattr(args, "dissect_refs", "") or "").strip()
        if raw_refs:
            try:
                ref_paths = [p.strip() for p in raw_refs.split(",") if p.strip()]
                if ref_paths:
                    from utils.generation.image_dissection import dissect_images_to_parts
                    from utils.generation.part_compositing import build_init_and_inpaint_mask

                    out_dir = Path("runs") / "dissection_composites"
                    reqs, parts, _facts = dissect_images_to_parts(
                        str(getattr(args, "prompt", "") or ""),
                        ref_paths,
                        output_dir=out_dir,
                        enable_heavy_models=True,
                    )
                    spec = build_init_and_inpaint_mask(
                        reference_images=ref_paths,
                        parts=parts,
                        output_dir=out_dir,
                        target_size=(image_size, image_size),
                        lock_background_if_requested=bool(getattr(args, "dissection_lock_background", False)),
                    )
                    if spec.preserved_parts > 0:
                        args.init_image = spec.init_image_path
                        args.mask = spec.mask_path
                        # MDM inpaint behaves better for preserving known regions.
                        if str(getattr(args, "inpaint_mode", "legacy")) == "legacy":
                            args.inpaint_mode = "mdm"
                        print(
                            f"Auto dissection init enabled: parts={spec.preserved_parts}, "
                            f"background_locked={spec.background_locked} -> init={args.init_image}, mask={args.mask}",
                            file=sys.stderr,
                        )
            except Exception as _e:
                pass

    # Img2img / from-z / inpainting
    num_timesteps = getattr(cfg, "num_timesteps", 1000)
    use_flow_sample = bool(getattr(args, "flow_matching_sample", False)) or (
        bool(getattr(cfg, "flow_matching_training", False)) and not bool(getattr(args, "force_vp_sample", False))
    )
    if use_flow_sample and args.mask:
        print(
            "Flow sampling disabled: structured inpaint uses VP q_sample; use --force-vp-sample or drop --mask.",
            file=sys.stderr,
        )
        use_flow_sample = False
    x_init = None
    start_timestep = None
    inpaint_mask_latent = None
    inpaint_x0 = None
    inpaint_noise = None
    if args.init_latent:
        z = torch.load(args.init_latent, map_location=device, weights_only=True)
        if z.dim() == 3:
            z = z.unsqueeze(0)
        x_init = z.to(device=device, dtype=torch.float32)
        x_init = _maybe_rae_to_dit(x_init, ae_type, rae_bridge)
        start_timestep = int(args.strength * num_timesteps)
        start_timestep = min(max(1, start_timestep), num_timesteps - 1)
        if use_flow_sample:
            den_fm = max(num_timesteps - 1, 1)
            s0 = float(start_timestep) / float(den_fm)
            z_fm = torch.randn_like(x_init, device=device, dtype=x_init.dtype)
            x_init = (1.0 - s0) * x_init + s0 * z_fm
        else:
            x_init = diffusion.q_sample(x_init, torch.tensor([start_timestep], device=device).expand(x_init.shape[0]))
        print(f"From-z: strength={args.strength} -> t_start={start_timestep}")
    elif args.init_image:
        pil_init = Image.open(args.init_image).convert("RGB")
        if pil_init.size != (image_size, image_size):
            pil_init = pil_init.resize((image_size, image_size), Image.Resampling.LANCZOS)
        arr = np.array(pil_init).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32)
        with torch.no_grad():
            enc = vae.encode(img_t)
            if hasattr(enc, "latent_dist"):
                z0 = enc.latent_dist.sample() * latent_scale
            else:
                z0 = enc.latent
            z0 = _maybe_rae_to_dit(z0, ae_type, rae_bridge)
        start_timestep = int(args.strength * num_timesteps)
        start_timestep = min(max(1, start_timestep), num_timesteps - 1)
        if args.mask:
            pil_mask = Image.open(args.mask).convert("L")
            if pil_mask.size != (image_size, image_size):
                pil_mask = pil_mask.resize((image_size, image_size), Image.Resampling.LANCZOS)
            mask_np = np.array(pil_mask).astype(np.float32) / 255.0
            mask_np = (mask_np >= 0.5).astype(np.float32)
            mask_latent = torch.from_numpy(mask_np).to(device).view(1, 1, image_size, image_size)
            mask_latent = torch.nn.functional.interpolate(mask_latent, size=(latent_size, latent_size), mode="nearest")
            t_start = torch.tensor([start_timestep], device=device).expand(z0.shape[0])
            if args.inpaint_mode == "mdm":
                # Approximate MDM-style "fill the blanks" at inference by:
                # - giving masked/unmasked regions different forward-noise eps at t_start
                # - freezing the unmasked regions to q_sample(x0_known, t) after every denoise step
                noise_known = torch.randn_like(z0, device=device, dtype=z0.dtype)
                noise_masked = torch.randn_like(z0, device=device, dtype=z0.dtype)
                x_t_known = diffusion.q_sample(z0, t_start, noise=noise_known)
                x_t_masked = diffusion.q_sample(z0, t_start, noise=noise_masked)
                x_init = mask_latent * x_t_masked + (1 - mask_latent) * x_t_known
                inpaint_mask_latent = mask_latent
                inpaint_x0 = z0
                inpaint_noise = noise_known
            else:
                # Legacy behavior kept for backward compatibility.
                noise = torch.randn_like(z0, device=device, dtype=z0.dtype)
                x_t_full = diffusion.q_sample(z0, t_start, noise=noise)
                x_init = mask_latent * noise + (1 - mask_latent) * x_t_full
            print(f"Inpainting: strength={args.strength} -> t_start={start_timestep}")
        else:
            if use_flow_sample:
                den_fm = max(num_timesteps - 1, 1)
                s0 = float(start_timestep) / float(den_fm)
                z_fm = torch.randn_like(z0, device=device, dtype=z0.dtype)
                x_init = (1.0 - s0) * z0 + s0 * z_fm
            else:
                x_init = diffusion.q_sample(z0, torch.tensor([start_timestep], device=device).expand(z0.shape[0]))
            print(f"Img2img: strength={args.strength} -> t_start={start_timestep}")

    shape = (num_gen, 4, latent_size, latent_size)
    regional_cfg_plan = None
    box_spec = getattr(args, "_box_layout_spec", None)
    box_mode = str(getattr(args, "box_layout_mode", "regional_cfg") or "regional_cfg").lower()
    if box_spec is not None and box_mode == "regional_cfg":

        def _encode_regions(captions: list) -> torch.Tensor:
            return encode_text(captions, tokenizer, text_encoder, device, text_bundle=text_bundle)

        try:
            from utils.generation.regional_box_prompting import encode_regional_plan

            regional_cfg_plan = encode_regional_plan(
                box_spec,
                encode_fn=_encode_regions,
                device=device,
                latent_h=latent_size,
                latent_w=latent_size,
                pixel_size=image_size,
                base_negative=str(getattr(args, "negative_prompt", "") or ""),
            )
            args._regional_cfg_plan = regional_cfg_plan
            sketch_n = sum(1 for r in box_spec.regions if r.sketch_path or r.strokes)
            msg = f"Regional CFG enabled for {len(box_spec.regions)} box(es)."
            if sketch_n:
                msg += f" {sketch_n} with draw+describe sketches."
            print(msg, file=sys.stderr)
        except Exception as e:
            print(f"Warning: regional box CFG disabled: {e}", file=sys.stderr)
    if x_init is not None and x_init.shape[0] != num_gen:
        x_init = x_init.expand(num_gen, -1, -1, -1)
        if inpaint_x0 is not None:
            inpaint_x0 = inpaint_x0.expand(num_gen, -1, -1, -1)
        if inpaint_noise is not None:
            inpaint_noise = inpaint_noise.expand(num_gen, -1, -1, -1)
        if inpaint_mask_latent is not None:
            inpaint_mask_latent = inpaint_mask_latent.expand(num_gen, -1, -1, -1)
    cfg_scale = getattr(args, "cfg_scale", 7.5)
    cfg_rescale = getattr(args, "cfg_rescale", 0.0)
    dyn_thresh_p = getattr(args, "dynamic_threshold_percentile", 0.0)

    if getattr(args, "originality", 0.0):
        # Encourage variation: lower CFG a bit so the model can explore beyond the most literal token match.
        try:
            strength = max(0.0, min(1.0, float(getattr(args, "originality", 0.0))))
            cfg_scale = max(1.0, cfg_scale - strength * 2.0)
            if cfg_rescale == 0.0 and cfg_scale > 6.0:
                cfg_rescale = 0.6
        except Exception:
            pass
    # IMPROVEMENTS 2.3: when CFG is high and user didn't set rescale/threshold, auto-enable to avoid oversaturation
    if cfg_scale > 10 and cfg_rescale == 0.0 and dyn_thresh_p == 0.0:
        cfg_rescale = 0.7
        dyn_thresh_p = 99.5
        print(
            f"High CFG ({cfg_scale}): auto-enabled cfg_rescale=0.7 and dynamic_threshold_percentile=99.5 to reduce oversaturation.",
            file=sys.stderr,
        )
    _vol_kw = dict(
        volatile_cfg_boost=float(getattr(args, "volatile_cfg_boost", 0.0) or 0.0),
        volatile_cfg_quantile=float(getattr(args, "volatile_cfg_quantile", 0.72) or 0.72),
        volatile_cfg_window=int(getattr(args, "volatile_cfg_window", 6) or 6),
    )
    _spec_kw = dict(
        speculative_draft_cfg_scale=float(getattr(args, "speculative_draft_cfg_scale", 0.0) or 0.0),
        speculative_close_thresh=float(getattr(args, "speculative_close_thresh", 0.0) or 0.0),
        speculative_blend=float(getattr(args, "speculative_blend", 0.35) or 0.35),
    )
    _flow_kw = dict(
        flow_matching_sample=bool(use_flow_sample),
        flow_solver=str(getattr(args, "flow_solver", "euler") or "euler"),
    )
    _control_kw = dict(
        control_guidance_start=float(getattr(args, "control_guidance_start", 0.0) or 0.0),
        control_guidance_end=float(getattr(args, "control_guidance_end", 1.0) or 1.0),
        control_guidance_decay=float(getattr(args, "control_guidance_decay", 1.0) or 1.0),
    )
    _holy_kw = dict(
        holy_grail_enable=bool(getattr(args, "holy_grail", False)),
        holy_grail_cfg_early_ratio=float(getattr(args, "holy_grail_cfg_early_ratio", 0.72) or 0.72),
        holy_grail_cfg_late_ratio=float(getattr(args, "holy_grail_cfg_late_ratio", 1.0) or 1.0),
        holy_grail_control_mult=float(getattr(args, "holy_grail_control_mult", 1.0) or 1.0),
        holy_grail_adapter_mult=float(getattr(args, "holy_grail_adapter_mult", 1.0) or 1.0),
        holy_grail_frontload_control=not bool(getattr(args, "holy_grail_no_frontload_control", False)),
        holy_grail_late_adapter_boost=float(getattr(args, "holy_grail_late_adapter_boost", 1.15) or 1.15),
        holy_grail_cads_strength=float(getattr(args, "holy_grail_cads_strength", 0.0) or 0.0),
        holy_grail_cads_min_strength=float(getattr(args, "holy_grail_cads_min_strength", 0.0) or 0.0),
        holy_grail_cads_power=float(getattr(args, "holy_grail_cads_power", 1.0) or 1.0),
        holy_grail_unsharp_sigma=float(getattr(args, "holy_grail_unsharp_sigma", 0.0) or 0.0),
        holy_grail_unsharp_amount=float(getattr(args, "holy_grail_unsharp_amount", 0.0) or 0.0),
        holy_grail_clamp_quantile=float(getattr(args, "holy_grail_clamp_quantile", 0.0) or 0.0),
        holy_grail_clamp_floor=float(getattr(args, "holy_grail_clamp_floor", 1.0) or 1.0),
    )
    _holy_kw = sanitize_holy_grail_kwargs(_holy_kw)
    if _holy_kw["holy_grail_enable"]:
        print("Holy-grail diffusion policy enabled.", file=sys.stderr)
    _gs_raw = getattr(args, "guidance_schedule", None)
    _gs = _gs_raw.strip() if isinstance(_gs_raw, str) and _gs_raw.strip() else None
    if _holy_kw["holy_grail_enable"] and _gs:
        print("--guidance-schedule ignored when holy-grail is enabled.", file=sys.stderr)
        _gs = None
    _guidance_kw = dict(
        cfg_guidance_schedule=_gs,
        cfg_guidance_linear_start_multiplier=float(getattr(args, "guidance_schedule_linear_start", 0.7)),
        cfg_guidance_linear_end_multiplier=float(getattr(args, "guidance_schedule_linear_end", 1.0)),
        cfg_guidance_cosine_min_multiplier=float(getattr(args, "guidance_schedule_cosine_min", 0.65)),
        cfg_guidance_cosine_max_multiplier=float(getattr(args, "guidance_schedule_cosine_max", 1.0)),
    )
    _regional_kw = dict(regional_cfg_plan=getattr(args, "_regional_cfg_plan", None))
    if use_flow_sample:
        print(
            f"Using flow-matching sampler (solver={_flow_kw['flow_solver']}); "
            "pair with checkpoints trained using --flow-matching-training.",
            file=sys.stderr,
        )
    _sag_kw = dict(
        sag_blur_sigma=float(getattr(args, "sag_blur_sigma", 0.0) or 0.0),
        sag_scale=float(getattr(args, "sag_scale", 0.0) or 0.0),
    )
    _ada_kw = dict(
        ada_early_exit_delta_threshold=(
            getattr(args, "ada_exit_delta_threshold", 0.0) if getattr(args, "ada_early_exit", False) else 0.0
        ),
        ada_early_exit_patience=(
            int(getattr(args, "ada_exit_patience", 0)) if getattr(args, "ada_early_exit", False) else 0
        ),
        ada_early_exit_min_steps=int(getattr(args, "ada_exit_min_steps", 0)),
    )
    _superior_kw = dict(
        fdg_cfg_strength=float(getattr(args, "fdg_cfg_strength", 0.0) or 0.0),
        fdg_cutoff_frac=float(getattr(args, "fdg_cutoff_frac", 0.15) or 0.15),
        feature_cache_delta_threshold=float(getattr(args, "feature_cache_delta", 0.0) or 0.0),
        feature_cache_max_reuse=int(getattr(args, "feature_cache_max_reuse", 2) or 2),
        block_cache_threshold=float(getattr(args, "block_cache_thresh", 0.0) or 0.0),
        block_cache_recompute_every=int(getattr(args, "block_cache_recompute_every", 4) or 4),
        taylor_cache=bool(getattr(args, "taylor_cache", False)),
        taylor_cache_order=int(getattr(args, "taylor_cache_order", 1) or 1),
        taylor_cache_interval=int(getattr(args, "block_cache_recompute_every", 4) or 4),
        rcfgpp_tangent=float(getattr(args, "rcfgpp_tangent", 0.0) or 0.0),
        apg_parallel_eta=float(getattr(args, "apg_parallel_eta", -1.0) or -1.0),
        zeresfdg_strength=float(getattr(args, "zeresfdg_strength", 0.0) or 0.0),
        cfg_zero_star=bool(getattr(args, "cfg_zero_star", False)),
        cfg_zero_init_frac=float(getattr(args, "cfg_zero_init_frac", 0.04) or 0.04),
        qsilk_micrograin=float(getattr(args, "qsilk_micrograin", 0.0) or 0.0),
        dynamic_dit_width=bool(getattr(args, "dynamic_dit_width", False)),
        dynamic_dit_early=float(getattr(args, "dynamic_dit_early", 0.88) or 0.88),
        dynamic_sdt=bool(getattr(args, "dynamic_sdt", False)),
        apg_momentum_beta=float(getattr(args, "apg_momentum_beta", 0.0) or 0.0),
        cfg_pp_lambda=float(getattr(args, "cfg_pp_lambda", 0.0) or 0.0),
        cfg_skip_early_frac=float(getattr(args, "cfg_skip_early_frac", 0.0) or 0.0),
        cfg_skip_late_frac=float(getattr(args, "cfg_skip_late_frac", 0.0) or 0.0),
        tcfg_damping=float(getattr(args, "tcfg_damping", 0.0) or 0.0),
        slg_scale=float(getattr(args, "slg_scale", 0.0) or 0.0),
        slg_skip_blocks=str(getattr(args, "slg_skip_blocks", "auto") or "auto"),
        cfg_rejection_rerank=bool(getattr(args, "cfg_rejection_rerank", False))
        and int(getattr(args, "num", 1) or 1) > 1,
        dbc_separate_cfg=bool(getattr(args, "dbc_separate_cfg", False))
        or float(getattr(args, "block_cache_thresh", 0.0) or 0.0) > 0.0,
    )

    _clip_mon_n = int(getattr(args, "clip_monitor_every", 0) or 0)
    if _clip_mon_n > 0:
        from utils.generation.clip_alignment import latent_x0_clip_cosine

        _cm_thr = float(getattr(args, "clip_monitor_threshold", 0.22) or 0.22)
        _cm_boost = float(getattr(args, "clip_monitor_cfg_boost", 0.12) or 0.12)
        _cm_rewind = float(getattr(args, "clip_monitor_rewind", 0.0) or 0.0)
        _cm_model = str(getattr(args, "clip_guard_model", "openai/clip-vit-base-patch32"))

        def _periodic_clip_fn(_step_i: int, x0p: torch.Tensor) -> float:
            return latent_x0_clip_cosine(
                x0p,
                prompt=prompt_to_encode,
                model_id=_cm_model,
                device=device,
                vae=vae,
                latent_scale=latent_scale,
                ae_type=ae_type,
                rae_bridge=rae_bridge,
            )

        _periodic_kw = dict(
            periodic_alignment_interval=_clip_mon_n,
            periodic_alignment_threshold=_cm_thr,
            periodic_alignment_cfg_boost=_cm_boost,
            periodic_alignment_fn=_periodic_clip_fn,
            periodic_alignment_rewind_strength=_cm_rewind,
        )
    else:
        _periodic_kw = dict(
            periodic_alignment_interval=0,
            periodic_alignment_threshold=0.0,
            periodic_alignment_cfg_boost=0.0,
            periodic_alignment_fn=None,
            periodic_alignment_rewind_strength=0.0,
        )

    dual_stage = bool(getattr(args, "dual_stage_layout", False))
    if use_flow_sample and dual_stage:
        print(
            "Dual-stage layout disabled: second stage uses VP re-noising (incompatible with flow sampler).",
            file=sys.stderr,
        )
        dual_stage = False
    if dual_stage:
        if ae_type != "kl" or args.mask or args.init_image or args.init_latent or x_init is not None:
            print("Dual-stage layout disabled (need KL VAE, no inpaint/img2img/from-z/init).", file=sys.stderr)
            dual_stage = False
    plan_ds = None
    if dual_stage:
        from utils.generation.inference_research_hooks import apply_size_embed_to_model_kwargs, plan_dual_stage_latents

        try:
            plan_ds = plan_dual_stage_latents(
                image_size,
                layout_scale_div=int(getattr(args, "dual_stage_div", 2) or 2),
                layout_steps=int(getattr(args, "dual_layout_steps", 24) or 24),
                detail_steps=int(getattr(args, "dual_detail_steps", 20) or 20),
            )
        except ValueError as e:
            print(f"Dual-stage skipped: {e}", file=sys.stderr)
            dual_stage = False

    if dual_stage and plan_ds is not None:
        ch, cw = plan_ds.layout_latent_hw
        fh, fw = plan_ds.target_latent_hw
        shape_c = (num_gen, 4, ch, cw)
        mk_c1, mk_u1 = apply_size_embed_to_model_kwargs(
            model_kwargs_cond,
            model_kwargs_uncond,
            cfg=cfg,
            cond_emb=cond_emb,
            latent_h=ch,
            latent_w=cw,
            device=device,
        )
        if "control_image" in model_kwargs_cond:
            mk_c1 = dict(mk_c1)
            mk_c1["control_image"] = _resize_control_tensor(model_kwargs_cond["control_image"], ch * 8, cw * 8)
            mk_c1["control_scale"] = model_kwargs_cond.get("control_scale", args.control_scale)
            if "control_type" in model_kwargs_cond:
                mk_c1["control_type"] = model_kwargs_cond["control_type"]
        print(
            f"Dual-stage: layout latent {ch}x{cw} ({plan_ds.layout_steps} steps) -> {fh}x{fw} ({plan_ds.detail_steps} steps)...",
            file=sys.stderr,
        )
        with torch.no_grad():
            x0 = diffusion.sample_loop(
                model,
                shape_c,
                model_kwargs_cond=mk_c1,
                model_kwargs_uncond=mk_u1,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                num_inference_steps=plan_ds.layout_steps,
                eta=0.0,
                device=device,
                dtype=torch.float32,
                x_init=None,
                start_timestep=None,
                dynamic_threshold_percentile=dyn_thresh_p,
                dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
                dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
                scheduler=getattr(args, "scheduler", "ddim"),
                timestep_schedule=getattr(args, "timestep_schedule", None),
                solver=getattr(args, "solver", "ddim"),
                karras_rho=float(getattr(args, "karras_rho", 7.0)),
                inpaint_mask=None,
                inpaint_x0=None,
                inpaint_noise=None,
                inpaint_freeze_known=False,
                ada_early_exit_delta_threshold=0.0,
                ada_early_exit_patience=0,
                ada_early_exit_min_steps=0,
                pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
                pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
                **_sag_kw,
                **_vol_kw,
                **_spec_kw,
                **_flow_kw,
                **_control_kw,
                **_holy_kw,
                **_periodic_kw,
                **_superior_kw,
                **_guidance_kw,
            **_regional_kw,
            )
            x0_up = torch.nn.functional.interpolate(x0.float(), size=(fh, fw), mode="bicubic", align_corners=False).to(
                dtype=x0.dtype
            )
            t_d = int(float(getattr(args, "dual_detail_strength", 0.38)) * (num_timesteps - 1))
            t_d = min(max(1, t_d), num_timesteps - 1)
            noise_d = torch.randn_like(x0_up, device=device, dtype=x0.dtype)
            t_batch = torch.tensor([t_d], device=device, dtype=torch.long).expand(x0_up.shape[0])
            x_hinit = diffusion.q_sample(x0_up, t_batch, noise=noise_d)
            mk_c2, mk_u2 = apply_size_embed_to_model_kwargs(
                model_kwargs_cond,
                model_kwargs_uncond,
                cfg=cfg,
                cond_emb=cond_emb,
                latent_h=fh,
                latent_w=fw,
                device=device,
            )
            if "control_image" in model_kwargs_cond:
                mk_c2 = dict(mk_c2)
                mk_c2["control_image"] = _resize_control_tensor(
                    model_kwargs_cond["control_image"], image_size, image_size
                )
                mk_c2["control_scale"] = model_kwargs_cond.get("control_scale", args.control_scale)
                if "control_type" in model_kwargs_cond:
                    mk_c2["control_type"] = model_kwargs_cond["control_type"]
            shape_f = (num_gen, 4, fh, fw)
            x0 = diffusion.sample_loop(
                model,
                shape_f,
                model_kwargs_cond=mk_c2,
                model_kwargs_uncond=mk_u2,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                num_inference_steps=plan_ds.detail_steps,
                eta=0.0,
                device=device,
                dtype=torch.float32,
                x_init=x_hinit,
                start_timestep=t_d,
                dynamic_threshold_percentile=dyn_thresh_p,
                dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
                dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
                scheduler="euler",
                timestep_schedule=None,
                solver=getattr(args, "solver", "ddim"),
                karras_rho=float(getattr(args, "karras_rho", 7.0)),
                inpaint_mask=None,
                inpaint_x0=None,
                inpaint_noise=None,
                inpaint_freeze_known=False,
                ada_early_exit_delta_threshold=0.0,
                ada_early_exit_patience=0,
                ada_early_exit_min_steps=0,
                pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
                pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
                sag_blur_sigma=0.0,
                sag_scale=0.0,
                **_vol_kw,
                **_spec_kw,
                **_flow_kw,
                **_control_kw,
                **_holy_kw,
                **_periodic_kw,
                **_superior_kw,
                **_guidance_kw,
            **_regional_kw,
            )
    else:
        print(f"Sampling (steps={args.steps}, num={num_gen}, cfg_scale={cfg_scale})...")
        beam_w = int(getattr(args, "beam_width", 0) or 0)
        beam_steps = int(getattr(args, "beam_steps", 0) or 0)
        do_beam = (
            beam_w > 1
            and beam_steps > 0
            and int(num_gen) == 1
            and not bool(use_flow_sample)
            and x_init is None
            and start_timestep is None
            and inpaint_mask_latent is None
            and inpaint_x0 is None
            and inpaint_noise is None
        )

        if do_beam:
            try:
                from utils.quality.test_time_pick import pick_best_indices

                pick_m0 = (str(getattr(args, "beam_metric", "") or "")).strip().lower()
                if not pick_m0 or pick_m0 in ("auto", "none"):
                    pick_m0 = (str(getattr(args, "pick_best", "") or "")).strip().lower()
                if not pick_m0 or pick_m0 in ("auto", "none"):
                    pick_m0 = "combo_vit_hq" if str(getattr(args, "pick_vit_ckpt", "") or "").strip() else "combo_vit"
                print(f"Beam search: width={beam_w} steps={beam_steps} metric={pick_m0}", file=sys.stderr)

                shape_b = (beam_w, shape[1], shape[2], shape[3])
                with torch.no_grad():
                    x_t, x0_pred, t_resume = diffusion.sample_loop(
                        model,
                        shape_b,
                        model_kwargs_cond=model_kwargs_cond,
                        model_kwargs_uncond=model_kwargs_uncond,
                        cfg_scale=cfg_scale,
                        cfg_rescale=cfg_rescale,
                        num_inference_steps=beam_steps,
                        eta=0.0,
                        device=device,
                        dtype=torch.float32,
                        x_init=None,
                        start_timestep=None,
                        dynamic_threshold_percentile=dyn_thresh_p,
                        dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
                        dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
                        scheduler=getattr(args, "scheduler", "ddim"),
                        timestep_schedule=getattr(args, "timestep_schedule", None),
                        solver=getattr(args, "solver", "ddim"),
                        karras_rho=float(getattr(args, "karras_rho", 7.0)),
                        inpaint_mask=None,
                        inpaint_x0=None,
                        inpaint_noise=None,
                        inpaint_freeze_known=False,
                        pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
                        pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
                        **_sag_kw,
                        **_vol_kw,
                        **_spec_kw,
                        **_flow_kw,
                        **_control_kw,
                        **_holy_kw,
                        **_ada_kw,
                        **_superior_kw,
                        **_periodic_kw,
                        **_guidance_kw,
            **_regional_kw,
                        return_intermediate_state=True,
                    )

                    # Decode cheap previews from x0_pred for scoring.
                    x0p = x0_pred
                    if ae_type == "kl":
                        x0p = x0p / latent_scale
                    elif ae_type == "rae" and rae_bridge is not None:
                        x0p = rae_bridge.dit_to_rae(x0p)
                    im = vae.decode(x0p).sample
                    im = (im * 0.5 + 0.5).clamp(0, 1)
                    rgb_list = []
                    for bi in range(im.shape[0]):
                        arr = im[bi].permute(1, 2, 0).detach().cpu().numpy()
                        rgb_list.append((arr * 255).round().astype("uint8"))

                    exp_ocr = ""
                    if isinstance(expected_texts, list) and expected_texts:
                        exp_ocr = str(expected_texts[0])
                    best_i, scores = pick_best_indices(
                        rgb_list,
                        prompt_to_encode,
                        pick_m0,
                        str(device),
                        exp_ocr,
                        getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32"),
                        int(getattr(args, "expected_count", 0) or 0),
                        str(getattr(args, "expected_count_target", "auto") or "auto"),
                        str(getattr(args, "expected_count_object", "") or ""),
                        str(getattr(args, "pick_vit_ckpt", "") or ""),
                        bool(getattr(args, "pick_vit_use_adherence", False)),
                        int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
                    )
                    print(f"beam scores={scores} -> keep {best_i} resume_t={t_resume}", file=sys.stderr)
                    pick_report["beam1"] = {
                        "metric": pick_m0,
                        "scores": [float(x) for x in scores],
                        "best_index": int(best_i),
                        "resume_t": int(t_resume),
                        "beam_width": int(beam_w),
                        "beam_steps": int(beam_steps),
                    }
                    x_init = x_t[best_i : best_i + 1].contiguous()
                    start_timestep = int(t_resume)
                    shape = (1, shape[1], shape[2], shape[3])
            except Exception as e:
                print(f"Beam search disabled (fallback): {e}", file=sys.stderr)

        steps_main = int(args.steps)
        if do_beam and beam_steps > 0 and x_init is not None and start_timestep is not None:
            steps_main = max(1, int(args.steps) - int(beam_steps))

        # Optional second-stage micro-beam: branch from a mid-latent and re-pick.
        beam2_w = int(getattr(args, "beam2_width", 0) or 0)
        beam2_steps = int(getattr(args, "beam2_steps", 0) or 0)
        beam2_at = float(getattr(args, "beam2_at_frac", 0.65) or 0.65)
        beam2_noise = float(getattr(args, "beam2_noise", 0.03) or 0.03)
        do_beam2 = (
            beam2_w > 1
            and beam2_steps > 0
            and int(num_gen) == 1
            and not bool(use_flow_sample)
            and inpaint_mask_latent is None
            and inpaint_x0 is None
            and inpaint_noise is None
            and (x_init is None) == (start_timestep is None)
        )

        if do_beam2 and steps_main >= (beam2_steps + 2):
            try:
                from utils.quality.test_time_pick import pick_best_indices

                pick_m2 = (str(getattr(args, "beam2_metric", "") or "")).strip().lower()
                if not pick_m2 or pick_m2 in ("auto", "none"):
                    pick_m2 = (str(getattr(args, "beam_metric", "") or "")).strip().lower()
                if not pick_m2 or pick_m2 in ("auto", "none"):
                    pick_m2 = (str(getattr(args, "pick_best", "") or "")).strip().lower()
                if not pick_m2 or pick_m2 in ("auto", "none"):
                    pick_m2 = "combo_vit_hq" if str(getattr(args, "pick_vit_ckpt", "") or "").strip() else "combo_vit"

                split = int(round(float(steps_main) * float(min(0.95, max(0.05, beam2_at)))))
                split = max(1, min(split, steps_main - (beam2_steps + 1)))
                with torch.no_grad():
                    # Run up to the split point and capture latent state.
                    x_mid, x0_mid, t_mid = diffusion.sample_loop(
                        model,
                        shape,
                        model_kwargs_cond=model_kwargs_cond,
                        model_kwargs_uncond=model_kwargs_uncond,
                        cfg_scale=cfg_scale,
                        cfg_rescale=cfg_rescale,
                        num_inference_steps=split,
                        eta=0.0,
                        device=device,
                        dtype=torch.float32,
                        x_init=x_init,
                        start_timestep=start_timestep,
                        dynamic_threshold_percentile=dyn_thresh_p,
                        dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
                        dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
                        scheduler=getattr(args, "scheduler", "ddim"),
                        timestep_schedule=getattr(args, "timestep_schedule", None),
                        solver=getattr(args, "solver", "ddim"),
                        karras_rho=float(getattr(args, "karras_rho", 7.0)),
                        inpaint_mask=None,
                        inpaint_x0=None,
                        inpaint_noise=None,
                        inpaint_freeze_known=False,
                        pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
                        pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
                        **_sag_kw,
                        **_vol_kw,
                        **_spec_kw,
                        **_flow_kw,
                        **_control_kw,
                        **_holy_kw,
                        **_ada_kw,
                        **_superior_kw,
                        **_periodic_kw,
                        **_guidance_kw,
            **_regional_kw,
                        return_intermediate_state=True,
                    )

                    # Branch into beam2 variants with small noise around x_mid.
                    x_seed = x_mid.expand(beam2_w, -1, -1, -1).contiguous()
                    if beam2_noise > 0.0:
                        x_seed = x_seed + beam2_noise * torch.randn_like(x_seed, device=device, dtype=x_seed.dtype)
                    shape2 = (beam2_w, shape[1], shape[2], shape[3])
                    x2, x0_2, t2 = diffusion.sample_loop(
                        model,
                        shape2,
                        model_kwargs_cond=model_kwargs_cond,
                        model_kwargs_uncond=model_kwargs_uncond,
                        cfg_scale=cfg_scale,
                        cfg_rescale=cfg_rescale,
                        num_inference_steps=beam2_steps,
                        eta=0.0,
                        device=device,
                        dtype=torch.float32,
                        x_init=x_seed,
                        start_timestep=int(t_mid),
                        dynamic_threshold_percentile=dyn_thresh_p,
                        dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
                        dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
                        scheduler=getattr(args, "scheduler", "ddim"),
                        timestep_schedule=getattr(args, "timestep_schedule", None),
                        solver=getattr(args, "solver", "ddim"),
                        karras_rho=float(getattr(args, "karras_rho", 7.0)),
                        inpaint_mask=None,
                        inpaint_x0=None,
                        inpaint_noise=None,
                        inpaint_freeze_known=False,
                        pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
                        pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
                        **_sag_kw,
                        **_vol_kw,
                        **_spec_kw,
                        **_flow_kw,
                        **_control_kw,
                        **_holy_kw,
                        **_ada_kw,
                        **_superior_kw,
                        **_periodic_kw,
                        **_guidance_kw,
            **_regional_kw,
                        return_intermediate_state=True,
                    )

                    # Decode previews from x0_2 and pick best.
                    x0p2 = x0_2
                    if ae_type == "kl":
                        x0p2 = x0p2 / latent_scale
                    elif ae_type == "rae" and rae_bridge is not None:
                        x0p2 = rae_bridge.dit_to_rae(x0p2)
                    im2 = vae.decode(x0p2).sample
                    im2 = (im2 * 0.5 + 0.5).clamp(0, 1)
                    rgb2 = []
                    for bi in range(im2.shape[0]):
                        arr = im2[bi].permute(1, 2, 0).detach().cpu().numpy()
                        rgb2.append((arr * 255).round().astype("uint8"))
                    exp_ocr2 = ""
                    if isinstance(expected_texts, list) and expected_texts:
                        exp_ocr2 = str(expected_texts[0])
                    best2, scores2 = pick_best_indices(
                        rgb2,
                        prompt_to_encode,
                        pick_m2,
                        str(device),
                        exp_ocr2,
                        getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32"),
                        int(getattr(args, "expected_count", 0) or 0),
                        str(getattr(args, "expected_count_target", "auto") or "auto"),
                        str(getattr(args, "expected_count_object", "") or ""),
                        str(getattr(args, "pick_vit_ckpt", "") or ""),
                        bool(getattr(args, "pick_vit_use_adherence", False)),
                        int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
                    )
                    print(
                        f"Micro-beam: split={split}/{steps_main} width={beam2_w} steps={beam2_steps} "
                        f"metric={pick_m2} scores={scores2} -> keep {best2} resume_t={t2}",
                        file=sys.stderr,
                    )
                    pick_report["beam2"] = {
                        "metric": pick_m2,
                        "scores": [float(x) for x in scores2],
                        "best_index": int(best2),
                        "resume_t": int(t2),
                        "beam2_width": int(beam2_w),
                        "beam2_steps": int(beam2_steps),
                        "beam2_at_frac": float(beam2_at),
                        "beam2_noise": float(beam2_noise),
                        "split_steps": int(split),
                    }
                    x_init = x2[best2 : best2 + 1].contiguous()
                    start_timestep = int(t2)
                    steps_main = max(1, int(steps_main) - int(split) - int(beam2_steps))
            except Exception as e:
                print(f"Micro-beam disabled (fallback): {e}", file=sys.stderr)

        x0 = diffusion.sample_loop(
            model,
            shape,
            model_kwargs_cond=model_kwargs_cond,
            model_kwargs_uncond=model_kwargs_uncond,
            cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            num_inference_steps=steps_main,
            eta=0.0,
            device=device,
            dtype=torch.float32,
            x_init=x_init,
            start_timestep=start_timestep,
            dynamic_threshold_percentile=dyn_thresh_p,
            dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
            dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
            scheduler=getattr(args, "scheduler", "ddim"),
            timestep_schedule=getattr(args, "timestep_schedule", None),
            solver=getattr(args, "solver", "ddim"),
            karras_rho=float(getattr(args, "karras_rho", 7.0)),
            inpaint_mask=inpaint_mask_latent,
            inpaint_x0=inpaint_x0,
            inpaint_noise=inpaint_noise,
            inpaint_freeze_known=(args.mask and args.inpaint_mode == "mdm"),
            pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
            pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
            **_sag_kw,
            **_vol_kw,
            **_spec_kw,
            **_flow_kw,
            **_control_kw,
            **_holy_kw,
            **_ada_kw,
            **_superior_kw,
            **_periodic_kw,
            **_guidance_kw,
            **_regional_kw,
        )

    if bool(getattr(args, "cfg_rejection_rerank", False)) and num_gen > 1:
        _probe = getattr(diffusion, "_last_guidance_probe", None)
        if _probe is not None:
            _order = _probe.rerank_indices()
            if _order and len(_order) == int(x0.shape[0]):
                x0 = x0[_order]
                print(f"CFG-rejection rerank order={_order}", file=sys.stderr)

    if float(getattr(args, "clip_guard_threshold", 0.0) or 0.0) > 0.0:
        try:
            from utils.generation.clip_alignment import maybe_clip_refine_latent

            def _clip_refiner(xc: torch.Tensor, t_s: int, n_st: int) -> torch.Tensor:
                if use_flow_sample:
                    den_r = max(num_timesteps - 1, 1)
                    s0 = float(t_s) / float(den_r)
                    noise2 = torch.randn_like(xc, device=device, dtype=xc.dtype)
                    xi2 = (1.0 - s0) * xc + s0 * noise2
                else:
                    noise2 = torch.randn_like(xc, device=device, dtype=xc.dtype)
                    tb2 = torch.tensor([t_s], device=device, dtype=torch.long).expand(xc.shape[0])
                    xi2 = diffusion.q_sample(xc, tb2, noise=noise2)
                return diffusion.sample_loop(
                    model,
                    xc.shape,
                    model_kwargs_cond=model_kwargs_cond,
                    model_kwargs_uncond=model_kwargs_uncond,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    num_inference_steps=max(4, int(n_st)),
                    eta=0.0,
                    device=device,
                    dtype=torch.float32,
                    x_init=xi2,
                    start_timestep=t_s,
                    dynamic_threshold_percentile=dyn_thresh_p,
                    dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
                    dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
                    scheduler="euler",
                    timestep_schedule=None,
                    solver=getattr(args, "solver", "ddim"),
                    karras_rho=float(getattr(args, "karras_rho", 7.0)),
                    inpaint_mask=None,
                    inpaint_x0=None,
                    inpaint_noise=None,
                    inpaint_freeze_known=False,
                    ada_early_exit_delta_threshold=0.0,
                    ada_early_exit_patience=0,
                    ada_early_exit_min_steps=0,
                    pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
                    pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
                    sag_blur_sigma=0.0,
                    sag_scale=0.0,
                    **_vol_kw,
                    **_spec_kw,
                    **_flow_kw,
                    **_control_kw,
                    **_holy_kw,
                    **_superior_kw,
                    **_guidance_kw,
            **_regional_kw,
                )

            x0 = maybe_clip_refine_latent(
                x0,
                prompt=prompt_to_encode,
                sim_threshold=float(getattr(args, "clip_guard_threshold", 0.0)),
                model_id=str(getattr(args, "clip_guard_model", "openai/clip-vit-base-patch32")),
                device=device,
                vae=vae,
                latent_scale=latent_scale,
                ae_type=ae_type,
                rae_bridge=rae_bridge,
                num_timesteps=num_timesteps,
                t_frac=float(getattr(args, "clip_guard_t_frac", 0.22)),
                refine_steps=int(getattr(args, "clip_guard_steps", 12)),
                refiner=_clip_refiner,
            )
        except Exception as e:
            print(f"CLIP guard refine skipped: {e}", file=sys.stderr)

    if (
        getattr(args, "superior_self_correct", False)
        and float(getattr(args, "clip_guard_threshold", 0.0) or 0.0) <= 0.0
    ):
        try:
            from utils.superior.self_correct import SelfCorrectConfig, SelfCorrectPolicy

            _sc_pol = SelfCorrectPolicy(
                SelfCorrectConfig(clip_model_id=str(getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32")))
            )
            _sc_score = _sc_pol.score(
                x0,
                prompt_to_encode,
                vae=vae,
                latent_scale=latent_scale,
                ae_type=ae_type,
                rae_bridge=rae_bridge,
                device=device,
            )
            if _sc_pol.needs_correction(_sc_score):
                t_s, n_st = _sc_pol.refine_plan(num_timesteps)

                def _sc_refiner(xc: torch.Tensor, t_start: int, n_steps: int) -> torch.Tensor:
                    noise2 = torch.randn_like(xc, device=device, dtype=xc.dtype)
                    tb2 = torch.tensor([t_start], device=device, dtype=torch.long).expand(xc.shape[0])
                    xi2 = diffusion.q_sample(xc, tb2, noise=noise2)
                    return diffusion.sample_loop(
                        model,
                        xc.shape,
                        model_kwargs_cond=model_kwargs_cond,
                        model_kwargs_uncond=model_kwargs_uncond,
                        cfg_scale=cfg_scale,
                        cfg_rescale=cfg_rescale,
                        num_inference_steps=max(4, int(n_steps)),
                        eta=0.0,
                        device=device,
                        dtype=torch.float32,
                        x_init=xi2,
                        start_timestep=t_start,
                        dynamic_threshold_percentile=dyn_thresh_p,
                        scheduler="euler",
                        solver=getattr(args, "solver", "ddim"),
                        inpaint_mask=None,
                        inpaint_x0=None,
                        inpaint_noise=None,
                        inpaint_freeze_known=False,
                        ada_early_exit_delta_threshold=0.0,
                        sag_blur_sigma=0.0,
                        sag_scale=0.0,
                        **_vol_kw,
                        **_spec_kw,
                        **_flow_kw,
                        **_control_kw,
                        **_holy_kw,
                        **_superior_kw,
                        **_guidance_kw,
            **_regional_kw,
                    )

                x0 = _sc_refiner(x0, t_s, n_st)
                _sc_pol.note_refinement_done()
                print(f"Superior self-correct: refined (score={_sc_score:.3f}, t={t_s})", file=sys.stderr)
        except Exception as e:
            print(f"Superior self-correct skipped: {e}", file=sys.stderr)

    if float(getattr(args, "domain_prior_latent", 0.0) or 0.0) > 0.0:
        try:
            from utils.generation.inference_research_hooks import highfreq_layout_prior

            x0 = highfreq_layout_prior(x0, strength=float(getattr(args, "domain_prior_latent", 0.0)))
        except Exception:
            pass

    if float(getattr(args, "spectral_coherence_latent", 0.0) or 0.0) > 0.0:
        try:
            from utils.generation.inference_research_hooks import spectral_latent_lowfreq_blend

            x0 = spectral_latent_lowfreq_blend(
                x0,
                strength=float(getattr(args, "spectral_coherence_latent", 0.0)),
                cutoff_frac=float(getattr(args, "spectral_coherence_cutoff", 0.15) or 0.15),
            )
        except Exception:
            pass

    # Latent-space hires fix: upscale clean latent, re-noise, short CFG denoise (works best with flexible DiT + KL VAE).
    if getattr(args, "hires_fix", False):
        if use_flow_sample:
            print("Hires-fix skipped: incompatible with flow-matching sampler (VP re-noising).", file=sys.stderr)
        elif ae_type != "kl":
            print("Hires-fix skipped: only supported for KL VAE checkpoints.", file=sys.stderr)
        elif args.mask or args.init_image or args.init_latent:
            print("Hires-fix skipped: not combined with inpaint / img2img / from-z.", file=sys.stderr)
        else:
            hs_scale = float(getattr(args, "hires_scale", 1.5) or 1.5)
            if int(getattr(args, "width", 0) or 0) > 0 or int(getattr(args, "height", 0) or 0) > 0:
                tw_px = int(out_w)
                th_px = int(out_h)
            else:
                tw_px = max(int(out_w), int(round(image_size * max(hs_scale, 1.001))))
                th_px = max(int(out_h), int(round(image_size * max(hs_scale, 1.001))))
            tlw = max(1, (tw_px + 7) // 8)
            tlh = max(1, (th_px + 7) // 8)
            if tlw <= latent_size and tlh <= latent_size:
                print(
                    f"Hires-fix skipped: latent already {latent_size}x{latent_size} (target {tlh}x{tlw}).",
                    file=sys.stderr,
                )
            else:
                with torch.no_grad():
                    x0_up = torch.nn.functional.interpolate(
                        x0.float(), size=(tlh, tlw), mode="bicubic", align_corners=False
                    ).to(dtype=x0.dtype)
                    t_hires = int(float(getattr(args, "hires_strength", 0.35)) * (num_timesteps - 1))
                    t_hires = min(max(1, t_hires), num_timesteps - 1)
                    noise_h = torch.randn_like(x0_up, device=device, dtype=x0.dtype)
                    t_batch = torch.tensor([t_hires], device=device, dtype=torch.long).expand(x0_up.shape[0])
                    x_hinit = diffusion.q_sample(x0_up, t_batch, noise=noise_h)
                    shape_h = (num_gen, 4, tlh, tlw)
                    mk_c = dict(model_kwargs_cond)
                    mk_u = dict(model_kwargs_uncond)
                    if int(getattr(cfg, "size_embed_dim", 0) or 0) > 0:
                        bsz = cond_emb.shape[0]
                        sz_h = torch.tensor([[float(tlh), float(tlw)]], device=device, dtype=torch.float32).expand(
                            bsz, -1
                        )
                        mk_c["size_embed"] = sz_h
                        mk_u["size_embed"] = sz_h
                    if "control_image" in model_kwargs_cond:
                        mk_c["control_image"] = _resize_control_tensor(model_kwargs_cond["control_image"], th_px, tw_px)
                        mk_c["control_scale"] = model_kwargs_cond.get("control_scale", args.control_scale)
                        if "control_type" in model_kwargs_cond:
                            mk_c["control_type"] = model_kwargs_cond["control_type"]
                    hires_steps = max(1, int(getattr(args, "hires_steps", 15)))
                    cfg_h = float(getattr(args, "hires_cfg_scale", -1.0))
                    if cfg_h < 0:
                        cfg_h = float(cfg_scale)
                    x0 = diffusion.sample_loop(
                        model,
                        shape_h,
                        model_kwargs_cond=mk_c,
                        model_kwargs_uncond=mk_u,
                        cfg_scale=cfg_h,
                        cfg_rescale=cfg_rescale,
                        num_inference_steps=hires_steps,
                        eta=0.0,
                        device=device,
                        dtype=torch.float32,
                        x_init=x_hinit,
                        start_timestep=t_hires,
                        dynamic_threshold_percentile=dyn_thresh_p,
                        dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
                        dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
                        scheduler="euler",
                        timestep_schedule=None,
                        solver=getattr(args, "solver", "ddim"),
                        karras_rho=float(getattr(args, "karras_rho", 7.0)),
                        inpaint_mask=None,
                        inpaint_x0=None,
                        inpaint_noise=None,
                        inpaint_freeze_known=False,
                        ada_early_exit_delta_threshold=0.0,
                        ada_early_exit_patience=0,
                        ada_early_exit_min_steps=0,
                        pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
                        pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
                        sag_blur_sigma=0.0,
                        sag_scale=0.0,
                        volatile_cfg_boost=float(getattr(args, "volatile_cfg_boost", 0.0) or 0.0),
                        volatile_cfg_quantile=float(getattr(args, "volatile_cfg_quantile", 0.72) or 0.72),
                        volatile_cfg_window=int(getattr(args, "volatile_cfg_window", 6) or 6),
                        **_spec_kw,
                        **_flow_kw,
                        **_control_kw,
                        **_holy_kw,
                        **_periodic_kw,
                        **_superior_kw,
                        **_guidance_kw,
            **_regional_kw,
                    )
                print(
                    f"Hires-fix: refined latent {tlh}x{tlw} ({hires_steps} steps, t_start={t_hires}, cfg={cfg_h}).",
                    file=sys.stderr,
                )

    # Optional refinement pass: add a little noise to the final latent and denoise once more.
    # This tends to reduce small artifacts while keeping composition.
    if not getattr(args, "no_refine", False) and not use_flow_sample:
        run_refine = True
        if str(getattr(args, "refine_gate", "off") or "off").lower() == "auto":
            try:
                with torch.no_grad():
                    _x0_preview = x0
                    if ae_type == "kl":
                        _x0_preview = _x0_preview / latent_scale
                    elif ae_type == "rae" and rae_bridge is not None:
                        _x0_preview = rae_bridge.dit_to_rae(_x0_preview)
                    _im_preview = vae.decode(_x0_preview).sample
                    _im_preview = (_im_preview * 0.5 + 0.5).clamp(0, 1)
                    _rgb = (_im_preview[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
                    gate_score, gate_details = _refine_gate_score(
                        image_rgb_u8=_rgb,
                        expected_texts=expected_texts if isinstance(expected_texts, list) else [],
                    )
                    thr = float(getattr(args, "refine_gate_threshold", 0.62) or 0.62)
                    run_refine = gate_score < thr
                    print(
                        f"refine-gate: score={gate_score:.3f} thr={thr:.3f} run_refine={run_refine} details={gate_details}",
                        file=sys.stderr,
                    )
            except Exception as e:
                # Fallback to existing behavior if gate path fails.
                print(f"refine-gate fallback: {e}", file=sys.stderr)
                run_refine = True
        if run_refine:
            with torch.no_grad():
                t_refine = int(getattr(args, "refine_t", 50))
                t_refine = min(max(0, t_refine), getattr(cfg, "num_timesteps", 1000) - 1)
                t = torch.full((x0.shape[0],), t_refine, device=device, dtype=torch.long)
                noise = torch.randn_like(x0, device=device)
                x_t = diffusion.q_sample(x0, t, noise=noise)
                x0_refined, _ = diffusion.p_step(model, x_t, t, model_kwargs=model_kwargs_cond)
                x0 = x0_refined

    if args.save_attn:
        with torch.no_grad():
            t0 = torch.zeros(1, device=device, dtype=torch.long)
            out = model(x0[:1], t0, return_attn=True, **model_kwargs_cond)
        if isinstance(out, tuple):
            _, attn_weights = out
            torch.save({"attn": attn_weights.cpu(), "prompt": args.prompt}, args.save_attn)
            print(f"Saved attention: {args.save_attn}")
        else:
            print("Model does not support --save-attn (e.g. DiT-P); skipping.", file=sys.stderr)

    # Decode with VAE / RAE
    if ae_type == "kl":
        x0 = x0 / latent_scale
    elif ae_type == "rae" and rae_bridge is not None:
        x0 = rae_bridge.dit_to_rae(x0)
    image = vae.decode(x0).sample
    image = (image * 0.5 + 0.5).clamp(0, 1)
    _, _, dec_h, dec_w = image.shape
    out_h, out_w = int(dec_h), int(dec_w)
    if args.width > 0 or args.height > 0:
        out_w = int(args.width or dec_w)
        out_h = int(args.height or dec_h)
        if (out_w != dec_w or out_h != dec_h) and str(
            getattr(args, "resize_mode", "stretch") or "stretch"
        ) == "stretch":
            image = torch.nn.functional.interpolate(image, size=(out_h, out_w), mode="bilinear", align_corners=False)
    # Non-native resolution often causes blur/artifacts
    if out_w != image_size or out_h != image_size:
        if max(out_w, out_h) > image_size * 1.5 or min(out_w, out_h) < image_size * 0.5:
            print(
                f"Note: output {out_w}x{out_h} differs from model native {image_size}x{image_size}; for best quality use native or enable --vae-tiling for large decode.",
                file=sys.stderr,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem, ext = out_path.stem, out_path.suffix or ".png"
    saved_imgs = []

    processed = []
    for i in range(num_gen):
        img_np = image[i].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).round().astype("uint8")
        if (out_w != dec_w or out_h != dec_h) and str(
            getattr(args, "resize_mode", "stretch") or "stretch"
        ) != "stretch":
            try:
                from utils.image_resize import fit_image_to_size

                img_np = fit_image_to_size(
                    img_np,
                    out_h,
                    out_w,
                    mode=str(getattr(args, "resize_mode", "stretch") or "stretch"),
                    saliency_face_bias=float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0),
                )
            except Exception as e:
                print(f"Resize mode fallback to stretch: {e}", file=sys.stderr)
                img_np = np.array(
                    Image.fromarray(img_np, mode="RGB").resize((int(out_w), int(out_h)), Image.Resampling.BILINEAR)
                )
        _sat = float(getattr(args, "saturation", 1.0))
        _preset = str(getattr(args, "finishing_preset", "none") or "none").lower()
        _need_finishing = (
            args.contrast != 1.0
            or args.sharpen > 0
            or abs(_sat - 1.0) > 1e-6
            or float(getattr(args, "clarity", 0.0)) > 0
            or float(getattr(args, "tone_punch", 0.0)) > 0
            or float(getattr(args, "chroma_smooth", 0.0)) > 0
            or float(getattr(args, "polish", 0.0)) > 0
            or _preset != "none"
        )
        if _need_finishing:
            try:
                from utils.quality import (
                    FINISHING_PRESET_BASELINES,
                    chroma_smooth_light,
                    contrast,
                    gentle_s_curve_luminance,
                    luminance_clarity,
                    polish_pass,
                    saturation_rgb,
                    sharpen,
                )

                _bc, _bt, _bch = FINISHING_PRESET_BASELINES.get(_preset, (0.0, 0.0, 0.0))
                _eff_clarity = min(0.55, float(getattr(args, "clarity", 0.0)) + _bc)
                _eff_tone = min(0.42, float(getattr(args, "tone_punch", 0.0)) + _bt)
                _eff_chroma = min(0.65, float(getattr(args, "chroma_smooth", 0.0)) + _bch)
                _polish = float(getattr(args, "polish", 0.0))
                if args.contrast != 1.0:
                    img_np = contrast(img_np.astype(np.float32), factor=args.contrast)
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                if abs(_sat - 1.0) > 1e-6:
                    img_np = saturation_rgb(img_np, factor=_sat)
                if _eff_tone > 0:
                    img_np = gentle_s_curve_luminance(img_np, strength=_eff_tone)
                if _eff_chroma > 0:
                    img_np = chroma_smooth_light(img_np, amount=_eff_chroma)
                if _eff_clarity > 0:
                    img_np = luminance_clarity(img_np, amount=_eff_clarity)
                if _polish > 0:
                    img_np = polish_pass(img_np, amount=_polish, seed=int(args.seed) + i)
                if args.sharpen > 0:
                    img_np = sharpen(img_np, amount=args.sharpen)
            except Exception:
                pass
        _run_photo_post = bool(getattr(args, "photo_postprocess", True)) and (
            str(_pr_grade_ef).lower() != "none"
            or str(_pr_filter_ef).lower() != "none"
            or str(_pr_grain_ef).lower() != "none"
        )
        if _run_photo_post:
            try:
                from utils.quality import add_film_grain, apply_photo_color_grade, apply_photo_filter

                _pps = float(getattr(args, "photo_post_strength", 0.6) or 0.6)
                img_np = apply_photo_color_grade(img_np, preset=str(_pr_grade_ef or "none"), strength=_pps)
                img_np = apply_photo_filter(
                    img_np,
                    filter_name=str(_pr_filter_ef or "none"),
                    strength=_pps,
                    seed=int(args.seed) + i,
                )
                _grain_map = {
                    "none": 0.0,
                    "fine_35mm": 0.008,
                    "medium_35mm": 0.015,
                    "heavy_16mm": 0.026,
                    "clean_digital": 0.002,
                }
                _g = _grain_map.get(str(_pr_grain_ef or "none").lower(), 0.0)
                if _g > 0:
                    img_np = add_film_grain(img_np, amount=float(_g), seed=int(args.seed) + i)
            except Exception:
                pass
        if getattr(args, "naturalize", False):
            try:
                from utils.quality import naturalize

                grain = max(0.0, getattr(args, "naturalize_grain", 0.015))
                img_np = naturalize(img_np, grain_amount=grain, micro_contrast=1.02, seed=args.seed + i)
            except Exception:
                pass
        # Artistic post-processing: compositional director, value structure, asymmetry, SSS, etc.
        _artistic_needed = any(
            [
                str(getattr(args, "composition_guide", "none") or "none") != "none",
                float(getattr(args, "value_shadow_lift", 0.0) or 0.0) > 0,
                float(getattr(args, "value_highlight_roll", 0.0) or 0.0) > 0,
                float(getattr(args, "value_midtone_contrast", 0.0) or 0.0) > 0,
                bool(getattr(args, "value_structure", False)),
                float(getattr(args, "asymmetry", 0.0) or 0.0) > 0,
                float(getattr(args, "lost_found_edges", 0.0) or 0.0) > 0,
                float(getattr(args, "sss", 0.0) or 0.0) > 0,
                float(getattr(args, "chromatic_aberration", 0.0) or 0.0) > 0,
                float(getattr(args, "vignette", 0.0) or 0.0) > 0,
                float(getattr(args, "micro_detail", 0.0) or 0.0) > 0,
            ]
        )
        if _artistic_needed:
            try:
                from utils.quality.artistic_post_process import ArtisticPostConfig, apply_artistic_pipeline

                _art_cfg = ArtisticPostConfig(
                    composition_mode=str(getattr(args, "composition_guide", "none") or "none"),
                    composition_strength=float(getattr(args, "composition_guide_strength", 0.15) or 0.15),
                    value_structure=bool(getattr(args, "value_structure", False)),
                    value_shadow_lift=float(getattr(args, "value_shadow_lift", 0.0) or 0.0),
                    value_highlight_roll=float(getattr(args, "value_highlight_roll", 0.0) or 0.0),
                    value_midtone_contrast=float(getattr(args, "value_midtone_contrast", 0.0) or 0.0),
                    asymmetry_strength=float(getattr(args, "asymmetry", 0.0) or 0.0),
                    asymmetry_seed=int(args.seed) + i,
                    lost_found_strength=float(getattr(args, "lost_found_edges", 0.0) or 0.0),
                    lost_found_seed=int(args.seed) + i + 1000,
                    sss_strength=float(getattr(args, "sss", 0.0) or 0.0),
                    sss_radius=float(getattr(args, "sss_radius", 3.0) or 3.0),
                    chromatic_aberration=float(getattr(args, "chromatic_aberration", 0.0) or 0.0),
                    vignette_strength=float(getattr(args, "vignette", 0.0) or 0.0),
                    micro_detail=float(getattr(args, "micro_detail", 0.0) or 0.0),
                )
                img_np = apply_artistic_pipeline(img_np, _art_cfg)
            except Exception as e:
                print(f"Artistic post-process failed (non-fatal): {e}", file=sys.stderr)
        _hm_cfg = None
        try:
            from utils.quality.human_made import apply_human_made_pipeline, config_from_args

            _hm_cfg = config_from_args(args)
            if _hm_cfg is not None:
                _hm_cfg.seed = int(args.seed) + i
                img_np = apply_human_made_pipeline(img_np, _hm_cfg)
        except Exception as e:
            print(f"Human-made post-process failed (non-fatal): {e}", file=sys.stderr)
        if getattr(args, "face_enhance", False):
            try:
                from utils.quality.face_region_enhance import enhance_faces_in_rgb

                img_np = enhance_faces_in_rgb(
                    img_np,
                    padding=float(getattr(args, "face_enhance_padding", 0.25)),
                    sharpen_amount=float(getattr(args, "face_enhance_sharpen", 0.35)),
                    micro_contrast=float(getattr(args, "face_enhance_contrast", 1.04)),
                    max_faces=int(getattr(args, "face_enhance_max", 4)),
                )
            except Exception:
                pass
        pr = str(getattr(args, "post_reference_image", "") or "").strip()
        pa = float(getattr(args, "post_reference_alpha", 0.0) or 0.0)
        if pr and pa > 0:
            try:
                from utils.quality.face_region_enhance import blend_reference_rgb

                ref_np = np.array(Image.open(pr).convert("RGB"))
                img_np = blend_reference_rgb(img_np, ref_np, alpha=pa)
            except Exception:
                pass
        processed.append(img_np)
    saved_imgs = processed

    pick_m = (getattr(args, "pick_best", None) or "none").lower()
    if pick_m == "auto":
        if bool(getattr(args, "realism_autopilot", True)) and str(_auto_pick_metric).strip():
            pick_m = str(_auto_pick_metric).strip().lower()
        elif str(getattr(args, "pick_vit_ckpt", "") or "").strip():
            if isinstance(expected_texts, list) and expected_texts:
                pick_m = "combo_vit_realism"
            elif re.search(
                r"\b(exactly\s+\d+|\d+\s+(people|persons|person|characters?|girls?|boys?|coins?|candles?|windows?))\b",
                str(prompt_to_encode).lower(),
            ):
                pick_m = "combo_count_vit"
            elif _is_photo_prompt or str(_pr_pack_ef).lower() != "none":
                pick_m = "combo_vit_realism"
            else:
                pick_m = "combo_vit_hq"
        elif bool(getattr(args, "pick_auto_no_clip", False)):
            if isinstance(expected_texts, list) and expected_texts:
                pick_m = "ocr"
            elif re.search(
                r"\b(exactly\s+\d+|\d+\s+(people|persons|person|characters?|girls?|boys?|coins?|candles?|windows?))\b",
                str(prompt_to_encode).lower(),
            ):
                pick_m = "combo_count"
            elif _is_photo_prompt or str(_pr_pack_ef).lower() != "none":
                pick_m = "aesthetic_realism"
            else:
                pick_m = "aesthetic"
        elif isinstance(expected_texts, list) and expected_texts:
            pick_m = "combo"
        elif re.search(
            r"\b(exactly\s+\d+|\d+\s+(people|persons|person|characters?|girls?|boys?|coins?|candles?|windows?))\b",
            str(prompt_to_encode).lower(),
        ):
            pick_m = "combo_count"
        elif _is_photo_prompt or str(_pr_pack_ef).lower() != "none":
            pick_m = "combo_realism"
        else:
            pick_m = "combo_hq"
        print(f"pick-best auto -> {pick_m}", file=sys.stderr)
    best_idx = 0
    if num_gen > 1 and pick_m != "none":
        exp_ocr = ""
        if isinstance(expected_texts, list) and expected_texts:
            exp_ocr = str(expected_texts[0])
        if pick_m == "superior_composite":
            from utils.superior.composite_ranker import CompositeRanker

            ranker = CompositeRanker()
            best_idx, scores = ranker.pick_best_index(
                processed,
                prompt=prompt_to_encode,
                device=str(device),
                vit_ckpt=str(getattr(args, "pick_vit_ckpt", "") or ""),
            )
        else:
            from utils.quality.test_time_pick import pick_best_indices

            best_idx, scores = pick_best_indices(
                processed,
                prompt_to_encode,
                pick_m,
                str(device),
                exp_ocr,
                getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32"),
                int(getattr(args, "expected_count", 0) or 0),
                str(getattr(args, "expected_count_target", "auto") or "auto"),
                str(getattr(args, "expected_count_object", "") or ""),
                str(getattr(args, "pick_vit_ckpt", "") or ""),
                bool(getattr(args, "pick_vit_use_adherence", False)),
                int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
            )
        print(f"pick-best ({pick_m}): scores={scores} -> best index {best_idx}")
        pick_report["pick_best"] = {
            "metric": str(pick_m),
            "scores": [float(x) for x in scores],
            "best_index": int(best_idx),
            "expected_text": str(exp_ocr or ""),
            "expected_count": int(getattr(args, "expected_count", 0) or 0),
            "expected_count_target": str(getattr(args, "expected_count_target", "auto") or "auto"),
            "expected_count_object": str(getattr(args, "expected_count_object", "") or ""),
            "pick_clip_model": str(getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32") or ""),
            "pick_vit_ckpt": str(getattr(args, "pick_vit_ckpt", "") or ""),
            "pick_vit_use_adherence": bool(getattr(args, "pick_vit_use_adherence", False)),
            "pick_vit_ar_blocks": int(getattr(args, "pick_vit_ar_blocks", -1) or -1),
            "pick_auto_no_clip": bool(getattr(args, "pick_auto_no_clip", False)),
        }

    if getattr(args, "pick_save_all", False) and num_gen > 1:
        for i in range(num_gen):
            cand_path = out_path.parent / f"{stem}_cand{i}{ext}"
            Image.fromarray(processed[i]).save(cand_path)
            print(f"Saved candidate: {cand_path}")

    if num_gen == 1:
        Image.fromarray(processed[0]).save(out_path)
        print(f"Saved: {out_path}")
    elif pick_m != "none":
        Image.fromarray(processed[best_idx]).save(out_path)
        print(f"Saved best ({pick_m}): {out_path}")
    else:
        for i in range(num_gen):
            save_path = out_path.parent / f"{stem}_{i}{ext}"
            Image.fromarray(processed[i]).save(save_path)
            print(f"Saved: {save_path}")

    # Optional: write prompt/seed/steps sidecar for reproducibility
    if getattr(args, "save_prompt", False):
        params_lines = [
            f"prompt: {args.prompt}",
            f"negative: {(args.negative_prompt or '').strip() or '(default)'}",
            f"seed: {args.seed}",
            f"steps: {args.steps}",
            f"cfg_scale: {getattr(args, 'cfg_scale', 7.5)}",
            f"scheduler: {getattr(args, 'timestep_schedule', None) or getattr(args, 'scheduler', 'ddim')}, "
            f"solver: {getattr(args, 'solver', 'ddim')}",
            f"hires_fix: {getattr(args, 'hires_fix', False)}, hires_scale: {getattr(args, 'hires_scale', 1.5)}, "
            f"hires_steps: {getattr(args, 'hires_steps', 15)}, saturation: {getattr(args, 'saturation', 1.0)}",
            f"finishing_preset: {getattr(args, 'finishing_preset', 'none')}, clarity: {getattr(args, 'clarity', 0.0)}, "
            f"tone_punch: {getattr(args, 'tone_punch', 0.0)}, chroma_smooth: {getattr(args, 'chroma_smooth', 0.0)}, "
            f"polish: {getattr(args, 'polish', 0.0)}",
        ]
        prompt_txt = out_path.parent / f"{stem}.txt"
        prompt_txt.write_text("\n".join(params_lines), encoding="utf-8")
        print(f"Saved params: {prompt_txt}")

    if str(getattr(args, "pick_report_json", "") or "").strip():
        try:
            import json as _json

            rp = Path(str(getattr(args, "pick_report_json", "") or "")).expanduser()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(_json.dumps(pick_report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved pick report: {rp}")
        except Exception as e:
            print(f"pick-report-json: failed to write ({e})", file=sys.stderr)

    # Optional: OCR validate + iterative inpainting to fix text rendering.
    if (
        getattr(args, "ocr_fix", False)
        and isinstance(expected_texts, list)
        and expected_texts
        and num_gen == 1
        and int(getattr(args, "ocr_repair_iter", 0)) < int(getattr(args, "ocr_iters", 0))
    ):
        try:
            import cv2 as _cv2
            import numpy as _np
            from utils.generation.text_rendering import create_text_rendering_pipeline

            pipe = create_text_rendering_pipeline()
            text_engine = pipe["engine"]
            ocr_engine = pipe["inpainting"]

            pil_img = Image.open(out_path).convert("RGB")
            val = text_engine.validate_text_rendering(pil_img, expected_texts)
            acc = float(val.get("accuracy_score", 0.0))

            if acc < float(getattr(args, "ocr_threshold", 0.65)):
                mask_pil = ocr_engine.create_text_edit_mask(
                    pil_img,
                    target_text=expected_texts[0] if len(expected_texts) == 1 else None,
                )
                if mask_pil is None:
                    print("OCR fix: could not build mask; stopping.", file=sys.stderr)
                else:
                    if getattr(args, "ocr_mask_dilate", 0) > 0:
                        dil_px = int(getattr(args, "ocr_mask_dilate", 0))
                        arr = _np.array(mask_pil.convert("L"))
                        kernel = _np.ones((dil_px * 2 + 1, dil_px * 2 + 1), dtype=_np.uint8)
                        dil = _cv2.dilate(arr, kernel, iterations=1)
                        mask_pil = Image.fromarray(dil, mode="L")

                    ocr_masks_dir = out_path.parent / "ocr_masks"
                    ocr_masks_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = ocr_masks_dir / f"{stem}_ocrmask_iter{args.ocr_repair_iter}.png"
                    mask_pil.save(mask_path)

                    # Bias prompt toward exact text.
                    repair_prompt = _maybe_append_text_says(args.prompt, expected_texts)

                    # Re-run sample.py inpainting mode, freezing known regions (mdm).
                    repair_cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--ckpt",
                        args.ckpt,
                        "--prompt",
                        repair_prompt,
                        "--negative-prompt",
                        args.negative_prompt or "",
                        "--out",
                        str(out_path),
                        "--num",
                        "1",
                        "--steps",
                        str(args.steps),
                        "--seed",
                        str(args.seed),
                        "--device",
                        args.device,
                        "--scheduler",
                        getattr(args, "timestep_schedule", None) or getattr(args, "scheduler", "ddim"),
                        "--solver",
                        getattr(args, "solver", "ddim"),
                        "--karras-rho",
                        str(getattr(args, "karras_rho", 7.0)),
                        "--cfg-scale",
                        str(getattr(args, "cfg_scale", 7.5)),
                        "--cfg-rescale",
                        str(getattr(args, "cfg_rescale", 0.0)),
                        "--strength",
                        str(getattr(args, "ocr_inpaint_strength", 0.55)),
                        "--init-image",
                        str(out_path),
                        "--mask",
                        str(mask_path),
                        "--inpaint-mode",
                        "mdm",
                        "--expected-text",
                        ",".join(expected_texts),
                        "--ocr-fix",
                        "--ocr-threshold",
                        str(getattr(args, "ocr_threshold", 0.65)),
                        "--ocr-iters",
                        str(getattr(args, "ocr_iters", 2)),
                        "--ocr-mask-dilate",
                        str(getattr(args, "ocr_mask_dilate", 0)),
                        "--ocr-inpaint-strength",
                        str(getattr(args, "ocr_inpaint_strength", 0.55)),
                        "--ocr-repair-iter",
                        str(int(getattr(args, "ocr_repair_iter", 0)) + 1),
                    ]
                    from utils.generation.sample_cli_passthrough import append_sample_repair_passthrough

                    append_sample_repair_passthrough(repair_cmd, args)

                    print(
                        f"OCR fix: acc={acc:.3f} < {args.ocr_threshold}; repairing with mdm inpainting...",
                        file=sys.stderr,
                    )
                    subprocess.run(repair_cmd, check=True)
                    # Stop this process: the repair subprocess overwrote args.out.
                    sys.exit(0)
            else:
                print(f"OCR fix: acc={acc:.3f} >= {args.ocr_threshold}; done.", file=sys.stderr)
        except Exception as e:
            print(f"OCR fix failed (non-fatal): {e}", file=sys.stderr)

    frs = str(getattr(args, "face_restore_shell", "") or "").strip()
    if frs:
        try:
            import shlex

            # Security: split into a list so the OS never interprets shell metacharacters.
            # {src} and {dst} are replaced with the literal path strings before splitting,
            # so the final argv list contains the path as a single token — no injection risk.
            cmd_str = frs.replace("{src}", str(out_path)).replace("{dst}", str(out_path))
            cmd_list = shlex.split(cmd_str)
            subprocess.run(cmd_list, shell=False, check=False)
        except Exception as e:
            print(f"face-restore-shell failed: {e}", file=sys.stderr)

    # IMPROVEMENTS 9: optional grid image when --num > 1
    if num_gen > 1 and getattr(args, "grid", False) and saved_imgs:
        try:
            ncols = int(np.ceil(np.sqrt(num_gen)))
            nrows = int(np.ceil(num_gen / ncols))
            h, w = saved_imgs[0].shape[0], saved_imgs[0].shape[1]
            grid_img = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
            for i in range(num_gen):
                r, c = i // ncols, i % ncols
                grid_img[r * h : (r + 1) * h, c * w : (c + 1) * w] = saved_imgs[i]
            grid_path = out_path.parent / f"{stem}_grid{ext}"
            Image.fromarray(grid_img).save(grid_path)
            print(f"Saved grid: {grid_path}")
        except Exception as e:
            print(f"Grid save failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    from utils.runtime.profiling import consume_profile_args, run_with_cprofile

    _argv, _pcfg = consume_profile_args(sys.argv)
    sys.argv = _argv
    if _pcfg is not None:
        run_with_cprofile(main, _pcfg)
    else:
        main()
