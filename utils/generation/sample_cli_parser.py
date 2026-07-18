"""Sample CLI parser builder (extracted from sample.py).

Import-light: argparse only, so ``python sample.py --help`` stays dependency-light.
"""

from __future__ import annotations

import argparse


def build_sample_parser() -> argparse.ArgumentParser:
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
        "--lora-bank",
        action="store_true",
        help="Resolve @artist / @style:name mentions to LoRA adapters from lora_bank/index.json (default when index exists).",
    )
    parser.add_argument(
        "--no-lora-bank",
        action="store_true",
        help="Disable automatic LoRA loading from the LoRA bank index.",
    )
    parser.add_argument(
        "--lora-bank-index",
        type=str,
        default="",
        help="Path to lora_bank/index.json (default: $SDX_DATA/lora_bank/index.json).",
    )
    parser.add_argument(
        "--style-lora-strength",
        type=float,
        default=1.0,
        help="Multiplier for @style:anime / @lora:style adapter scales from the LoRA bank.",
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
        "--style-ref",
        type=str,
        default="",
        help="Multi style references: path:strength,path2:strength (Krea-style weighted style transfer).",
    )
    parser.add_argument(
        "--style-references-json",
        type=str,
        default="",
        help="JSON file with references list (see examples/style_references.example.json).",
    )
    parser.add_argument(
        "--moodboard-json",
        type=str,
        default="",
        help="JSON moodboard image list (see examples/moodboard.example.json).",
    )
    parser.add_argument(
        "--moodboard-images",
        type=str,
        default="",
        help="Comma-separated moodboard image paths (pooled into one style embedding).",
    )
    parser.add_argument(
        "--moodboard-strength",
        type=float,
        default=1.0,
        help="Per-image weight for moodboard paths before pooling.",
    )
    parser.add_argument(
        "--creativity-mode",
        type=str,
        choices=("raw", "low", "medium", "high"),
        default="",
        help="Krea-style prompt expansion: raw=literal, high=more aesthetic fill-in for short prompts.",
    )
    parser.add_argument(
        "--slider-intensity",
        type=float,
        default=0.0,
        help="Generative slider −100…100: muted (−) vs bold stylization (+).",
    )
    parser.add_argument(
        "--slider-complexity",
        type=float,
        default=0.0,
        help="Generative slider −100…100: minimal (−) vs dense detail (+).",
    )
    parser.add_argument(
        "--slider-movement",
        type=float,
        default=0.0,
        help="Generative slider −100…100: static (−) vs dynamic motion (+).",
    )
    parser.add_argument(
        "--krea-turbo-preset",
        action="store_true",
        help="Few-step turbo profile (~8 steps, CFG≈1) inspired by Krea 2 Turbo.",
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
        help="Rectified-flow sampler (matches --flow-matching-training). Auto-on if checkpoint was flow-trained.",
    )
    parser.add_argument(
        "--force-vp-sample",
        action="store_true",
        help="Use VP sampler even when checkpoint has flow_matching_training (debug / wrong ckpt).",
    )
    parser.add_argument(
        "--flow-solver",
        type=str,
        default="dpmpp_2m",
        metavar="NAME",
        help="Flow ODE solver: euler | heun | midpoint | dpmpp_2m (aliases: rk2, dpm++_2m, …).",
    )
    parser.add_argument(
        "--flow-schedule",
        type=str,
        default="ays",
        metavar="NAME",
        help="Continuous flow time grid: linear | karras | ays | ays_dit (default: ays).",
    )
    parser.add_argument(
        "--flow-karras-rho",
        type=float,
        default=7.0,
        help="Rho for --flow-schedule karras (default 7).",
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
        default="ays_dit",
        metavar="NAME",
        help="Timestep index schedule; registered: "
        + ", ".join(_ts_list)
        + ". Or indices:HIGH,...,LOW (see diffusion.inference_timesteps). "
        "Recommended: ays_dit (DiT) or ays. Composes with --steps and --solver.",
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
        default="dpmpp_2m",
        metavar="NAME",
        help="VP ODE solver: ddim | heun | dpmpp_2m | dpmpp_3m | unipc "
        "(aliases: dpm++_2m, uni_pc, …). Recommended: dpmpp_2m.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity η (0=deterministic). Multistep DPM++/UniPC stay deterministic.",
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
    # Creative RAG: multimodal prompt enrichment using moondream + Qwen3 + semantic analysis
    parser.add_argument(
        "--creative-rag",
        action="store_true",
        help=(
            "Enable Creative RAG prompt enrichment: semantically decomposes the prompt, "
            "resolves cross-category contradictions, classifies intent, and enriches with "
            "novel context-aware additions. Uses moondream (pretrained/moondream3-preview, "
            "or legacy moondream2) for "
            "reference image understanding and Qwen3 (pretrained/Qwen3-14B, or legacy Qwen2.5-14B-Instruct) "
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
            "Reference image path for Creative RAG. moondream will describe this image "
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
        "--no-artist-style",
        dest="artist_style",
        action="store_false",
        help="Disable @artist expansion (e.g. '@Kantoku' -> the trained artist tag).",
    )
    parser.set_defaults(artist_style=True)
    parser.add_argument(
        "--artist-strength",
        type=float,
        default=1.0,
        help="Style emphasis for @artist tags; >1.0 wraps as (tag:strength) to push adherence (try 1.1-1.4).",
    )
    parser.add_argument(
        "--artist-index",
        type=str,
        default="",
        help="Path to artist_index.json from scraped data (default: $SDX_ARTIST_INDEX or data/artist_index.json).",
    )
    parser.set_defaults(prompt_compose=True)
    parser.add_argument(
        "--no-prompt-compose",
        dest="prompt_compose",
        action="store_false",
        help="Disable +category / @artist prompt composer (use raw prompt only).",
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
        "--frontier", action="store_true", help="Enable frontier research hooks (serendipity, witness POV)."
    )
    parser.add_argument(
        "--frontier-serendipity", type=float, default=0.25, help="Serendipity dial 0–1 when --frontier."
    )
    parser.add_argument(
        "--frontier-auto-resolve",
        action="store_true",
        help="With --frontier: auto-rewrite contradictory prompt phrases.",
    )
    parser.add_argument(
        "--frontier-subject",
        action="store_true",
        help="Subject-aware frontier: anatomy, creatures, mediums, realism, mature quality.",
    )
    parser.add_argument(
        "--frontier-perfect",
        action="store_true",
        help="Full perfect frontier: deep + subject + composition/lighting/materials + safety steering.",
    )
    parser.add_argument(
        "--safety-tier",
        type=str,
        choices=("off", "moderate", "strict"),
        default="moderate",
        help="Content policy tier when --frontier-perfect (moderate=steer/refuse high-risk prompts).",
    )
    parser.add_argument(
        "--frontier-creative",
        action="store_true",
        help="Creative frontier: surreal, cinema, mood physics, mutations — not duplicate art-medium tags.",
    )
    parser.add_argument(
        "--creative-mutate",
        type=int,
        default=0,
        help="With --frontier-creative: generate N prompt variants for explore/auto-refine.",
    )
    parser.add_argument(
        "--creative-random-constraint",
        action="store_true",
        help="With --frontier-creative: apply one random art-school constraint (monochrome, silhouette, etc.).",
    )
    parser.add_argument(
        "--character-session",
        type=str,
        default="",
        help="JSON file: locked character prompt additions, refs, and negative.",
    )
    parser.add_argument(
        "--save-character-session",
        type=str,
        default="",
        help="After generation, write character session JSON to this path.",
    )
    parser.add_argument(
        "--box-attn-layout",
        action="store_true",
        help="With --box-layout: Dense Diffusion–style cross-attn layout plan (early steps).",
    )
    parser.add_argument(
        "--box-attn-inject-frac", type=float, default=0.4, help="Fraction of steps for box-attn layout."
    )
    parser.add_argument("--box-attn-strength", type=float, default=0.85, help="Box-attn bias strength.")
    parser.add_argument(
        "--per-region-cads",
        action="store_true",
        help="With --box-layout: per-region condition annealing (boosts holy-grail CADS).",
    )
    parser.add_argument(
        "--fix-region",
        type=str,
        default="",
        help="With --init-image and --box-layout: inpaint only this region name (MDM).",
    )
    parser.add_argument(
        "--explain-adherence",
        type=str,
        default="",
        help="Save prompt-adherence heatmap PNG (runs extra attn forward).",
    )
    parser.add_argument(
        "--export-comfy-workflow",
        type=str,
        default="",
        help="Write ComfyUI-style workflow JSON from current args and exit (no sampling).",
    )
    parser.add_argument(
        "--auto-refine",
        type=int,
        default=0,
        metavar="N",
        help="Generate N seed variants, keep best by heuristic score (N>1).",
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
