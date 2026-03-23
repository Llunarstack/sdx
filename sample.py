"""
Generate an image from a text prompt using a trained checkpoint.
Supports: prompt, negative prompt, steps, width, height, CFG, scheduler (ddim/euler).
Optional: style, control-image, lora, img2img, inpainting, sharpen, contrast, emphasis (word)/[word].

Presets and OP modes:
- --preset sdxl|flux|anime|zit: apply a model-style preset from config.model_presets.
- --op-mode portrait|fullbody|anime_char: apply a high-level OP bundle on top.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.model_presets import apply_op_mode_to_args, apply_preset_to_args
from diffusion import create_diffusion
from utils.checkpoint_loading import load_dit_text_checkpoint


def _maybe_rae_to_dit(z: torch.Tensor, ae_type: str, rae_bridge) -> torch.Tensor:
    """Map RAE latent (B,C,h,w) to DiT 4-channel space when checkpoint includes RAELatentBridge."""
    if z is None or ae_type != "rae" or rae_bridge is None:
        return z
    if z.shape[1] == 4:
        return z
    return rae_bridge.rae_to_dit(z)


def load_model_from_ckpt(ckpt_path, device="cuda"):
    model, cfg, rae_bridge, model_name, fusion_sd = load_dit_text_checkpoint(
        ckpt_path,
        device=device,
        reject_enhanced=True,
    )
    try:
        from config.pixai_reference import get_pixai_style_label

        print(f"Model: {model_name} — {get_pixai_style_label(model_name)}")
    except ImportError:
        pass
    if rae_bridge is not None:
        rae_c = int(rae_bridge.to_dit.weight.shape[1])
        print(f"Loaded RAELatentBridge: rae_channels={rae_c} -> 4 (DiT latent space)")
    return model, cfg, rae_bridge, fusion_sd


# T5 encoding cache (IMPROVEMENTS 3.2): key = (prompt, negative, style), value = (cond, uncond, style_emb or None)
_t5_cache = {}
_T5_CACHE_MAX = 32  # limit entries to avoid unbounded memory


def _parse_prompt_emphasis(prompt: str):
    """Parse (word) -> weight 1.2, [word] -> 0.8. Returns (cleaned_prompt, segments) with segments = [(start, end, weight), ...] in cleaned."""
    cleaned = ""
    segments = []
    parts = re.split(r"(\([^)]*\)|\[[^\]]*\))", prompt)
    for p in parts:
        if p.startswith("(") and p.endswith(")"):
            content = p[1:-1]
            start = len(cleaned)
            cleaned += content
            segments.append((start, len(cleaned), 1.2))
        elif p.startswith("[") and p.endswith("]"):
            content = p[1:-1]
            start = len(cleaned)
            cleaned += content
            segments.append((start, len(cleaned), 0.8))
        else:
            start = len(cleaned)
            cleaned += p
            if p:
                segments.append((start, len(cleaned), 1.0))
    return cleaned.strip(), segments


def _positive_token_set(text: str) -> set:
    """Normalize prompt to a set of tokens (comma/space split, lowercased) for conflict detection."""
    if not (text or text.strip()):
        return set()
    tokens = []
    for part in text.split(","):
        tokens.extend(part.split())
    return {t.strip().lower() for t in tokens if t.strip()}


def _filter_negative_by_positive(positive: str, negative: str) -> str:
    """
    Remove from the negative prompt any token that also appears in the positive,
    so CFG does not push away from what the user asked for (pos/neg conflict resolution).
    Splits on comma and space; comparison is case-insensitive.
    """
    pos_set = _positive_token_set(positive)
    if not pos_set:
        return negative
    kept = []
    for part in negative.split(","):
        words = part.split()
        filtered_words = [w for w in words if w.strip().lower() not in pos_set]
        if filtered_words:
            kept.append(" ".join(filtered_words))
    result = ", ".join(kept).strip()
    return result if result else " "


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
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


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


def _sanitize_character_prompt_tokens(tokens: list, negative_tokens: list) -> Tuple[list, list]:
    """
    Prevent explicitly sexual tokens from being injected.
    If user includes "futa" or similar, we replace with androgynous presentation.
    """
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


def _load_character_sheet(sheet_path: str) -> Tuple[str, str]:
    """
    Load a character sheet JSON file and return (positive_additions, negative_additions).
    Supported keys (all optional):
      - prompt / positive / appearance / style_tags / clothing / accessories
      - negative / negative_prompt
      - gender_presentation: androgynous|male|female
    Values can be strings or lists of strings.
    """
    import json

    p = Path(sheet_path)
    if not p.exists():
        raise ValueError(f"character-sheet not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))

    positive_tokens: list = []
    negative_tokens: list = []

    positive_tokens.extend(_normalize_list_or_str(data.get("prompt")))
    positive_tokens.extend(_normalize_list_or_str(data.get("positive")))
    positive_tokens.extend(_normalize_list_or_str(data.get("appearance")))
    positive_tokens.extend(_normalize_list_or_str(data.get("style_tags")))
    positive_tokens.extend(_normalize_list_or_str(data.get("clothing")))
    positive_tokens.extend(_normalize_list_or_str(data.get("accessories")))

    negative_tokens.extend(_normalize_list_or_str(data.get("negative")))
    negative_tokens.extend(_normalize_list_or_str(data.get("negative_prompt")))

    gender_pres = str(data.get("gender_presentation", "") or "").strip().lower()
    if gender_pres in {"androgynous", "androgynous presentation", "gender-ambiguous", "gender ambiguous"}:
        positive_tokens.append("androgynous presentation")
    elif gender_pres in {"male", "male-presenting", "man-presenting"}:
        positive_tokens.append("male-presenting")
    elif gender_pres in {"female", "female-presenting", "woman-presenting"}:
        positive_tokens.append("female-presenting")

    positive_tokens, negative_tokens = _sanitize_character_prompt_tokens(positive_tokens, negative_tokens)

    pos = ", ".join([t for t in positive_tokens if t])
    neg = ", ".join([t for t in negative_tokens if t])
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


def _token_weights_from_segments(cleaned: str, segments: list, tokenizer, max_length: int, device):
    """Return (L,) tensor of per-token weights from segments. Uses offset_mapping if available."""
    try:
        enc = tokenizer(
            [cleaned],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
    except TypeError:
        return None
    offset_mapping = enc.get("offset_mapping")
    if offset_mapping is None:
        return None
    offset_mapping = offset_mapping[0]
    weights = []
    for s, e in offset_mapping:
        if s == 0 and e == 0:  # padding
            weights.append(1.0)
            continue
        w = 1.0
        for seg_start, seg_end, seg_w in segments:
            if s < seg_end and e > seg_start:
                w = seg_w
                break
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def encode_text(
    captions,
    tokenizer,
    text_encoder,
    device,
    max_length=300,
    dtype=torch.float32,
    text_bundle=None,
):
    if text_bundle is not None:
        return text_bundle.encode(
            captions,
            device,
            max_length=max_length,
            dtype=dtype,
            train_fusion=False,
        )
    tok = tokenizer(captions, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)
    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state.to(dtype)


def main():
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
    parser.add_argument("--control-scale", type=float, default=0.85, help="ControlNet strength")
    parser.add_argument(
        "--lora",
        type=str,
        nargs="*",
        default=[],
        help="LoRA paths (.pt or .safetensors) with optional scale, e.g. path.safetensors path2.pt:0.6",
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
        help="Comma-separated tags; prepended to prompt with subject-first order (PixAI/Danbooru-style)",
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
        "--creativity",
        type=float,
        default=None,
        help="Creativity/diversity 0-1 (only if model was trained with --creativity-embed-dim)",
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
    # Test-time scaling: generate N candidates (--num) and keep the best (§11.3 IMPROVEMENTS.md)
    parser.add_argument(
        "--pick-best",
        type=str,
        default="none",
        choices=["none", "clip", "edge", "ocr", "combo", "combo_exposure"],
        help="With --num > 1, score candidates and save the best to --out (clip|edge|ocr|combo|combo_exposure)",
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
        "--vae-tiling", action="store_true", help="Enable VAE tiling for decode (lower VRAM for large output)"
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
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "euler"],
        help="Sampling scheduler: ddim (default) or euler",
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
        help="Prepend 'masterpiece, best quality' to the prompt for stronger adherence (complex/challenging prompts)",
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
        "--hard-style",
        type=str,
        default=None,
        choices=["3d", "realistic", "3d_realistic", "style_mix"],
        help="Prepend recommended tags for hard styles (3d, realistic, 3d_realistic, style_mix); see config/prompt_domains.py for negatives",
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
        "--anti-bleed",
        action="store_true",
        help="Reduce concept/color bleeding: add distinct-colors positive and color-bleed negative",
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
        help="Path to character sheet JSON to inject appearance tokens into prompt",
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
        "--preset",
        type=str,
        default=None,
        choices=["sdxl", "flux", "anime", "zit"],
        help="Apply a sampler preset (soft defaults) from config.model_presets",
    )
    parser.add_argument(
        "--op-mode",
        type=str,
        default=None,
        choices=["portrait", "fullbody", "anime_char"],
        help="High-level OP mode (applied after preset)",
    )
    args = parser.parse_args()

    # Apply preset and OP mode as soft defaults (only for unset args)
    if getattr(args, "preset", None):
        apply_preset_to_args(args, args.preset)
    if getattr(args, "op_mode", None):
        apply_op_mode_to_args(args, args.op_mode)

    has_tags = bool(getattr(args, "tags", "").strip() or getattr(args, "tags_file", "").strip())
    has_prompt_file = bool(getattr(args, "prompt_file", "").strip())
    if not (args.prompt or has_tags or has_prompt_file):
        parser.error("Provide at least one of --prompt, --prompt-file, --tags, or --tags-file")

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
    if getattr(args, "subject_first", False) and prompt_for_encoding:
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

    # Character sheet injection: adds user-defined character appearance tokens to prompt
    # and character-specific negative tokens to discourage drift.
    character_positive_additions = ""
    character_negative_additions = ""
    if getattr(args, "character_sheet", "").strip():
        try:
            character_positive_additions, character_negative_additions = _load_character_sheet(args.character_sheet)
        except Exception as e:
            print(f"Warning: failed to load --character-sheet: {e}", file=sys.stderr)
            character_positive_additions, character_negative_additions = "", ""
    if getattr(args, "character_negative_extra", "").strip():
        character_negative_additions = f"{character_negative_additions}, {args.character_negative_extra}".strip(", ")
    if getattr(args, "character_prompt_extra", "").strip():
        character_positive_additions = f"{character_positive_additions}, {args.character_prompt_extra}".strip(", ")

    if character_positive_additions and getattr(args, "prompt", ""):
        args.prompt = f"{args.prompt}, {character_positive_additions}"
    elif character_positive_additions:
        args.prompt = character_positive_additions

    if getattr(args, "hard_style", None):
        try:
            from config.prompt_domains import HARD_STYLE_RECOMMENDED_PROMPTS

            prefix = HARD_STYLE_RECOMMENDED_PROMPTS.get(args.hard_style, [None])[0]
            if prefix and args.prompt:
                args.prompt = f"{prefix}, {args.prompt}"
            elif prefix:
                args.prompt = prefix
        except ImportError:
            pass

    if getattr(args, "naturalize", False) and args.prompt:
        try:
            from config.prompt_domains import NATURAL_LOOK_POSITIVE

            args.prompt = f"{NATURAL_LOOK_POSITIVE}, {args.prompt}"
        except ImportError:
            pass
    if getattr(args, "anti_bleed", False) and args.prompt:
        try:
            from config.prompt_domains import CONCEPT_BLEEDING_POSITIVE

            args.prompt = f"{CONCEPT_BLEEDING_POSITIVE}, {args.prompt}"
        except ImportError:
            pass
    if getattr(args, "diversity", False) and args.prompt:
        try:
            from config.prompt_domains import DIVERSITY_POSITIVE

            args.prompt = f"{DIVERSITY_POSITIVE}, {args.prompt}"
        except ImportError:
            pass

    # Originality / novelty: inject a few deterministic "unique composition" tokens near the
    # start of the prompt (after subject/people descriptor tags) and, if supported, auto-set
    # the model's creativity embedding.
    if getattr(args, "originality", 0.0) and args.prompt:
        try:
            from config.prompt_domains import ORIGINALITY_POSITIVE_TOKENS
            from data.caption_utils import (
                AGE_TAGS,
                ANATOMY_FRAMING_TAGS,
                BODY_PART_TAGS,
                BUILD_BODY_TAGS,
                HEIGHT_TAGS,
                SUBJECT_PREFIXES,
            )

            strength = float(args.originality)
            strength = max(0.0, min(1.0, strength))

            tokens = ORIGINALITY_POSITIVE_TOKENS
            if tokens:
                k = 1 + int(round(strength * 3))
                k = max(1, min(k, len(tokens)))
                rng = np.random.default_rng(args.seed)
                chosen = list(rng.choice(tokens, size=k, replace=False))

                parts = [p.strip() for p in args.prompt.split(",") if p.strip()]
                person_terms = (
                    list(SUBJECT_PREFIXES)
                    + list(AGE_TAGS)
                    + list(HEIGHT_TAGS)
                    + list(BUILD_BODY_TAGS)
                    + list(ANATOMY_FRAMING_TAGS)
                    + list(BODY_PART_TAGS)
                )

                def _norm(x: str) -> str:
                    return x.lower().strip().replace("_", " ")

                person_norm = [_norm(t) for t in person_terms]

                def _is_person_term(tag: str) -> bool:
                    t = _norm(tag)
                    return any(t == pt or t.startswith(pt + " ") for pt in person_norm)

                insert_at = 0
                while insert_at < len(parts) and _is_person_term(parts[insert_at]):
                    insert_at += 1

                parts[insert_at:insert_at] = chosen
                args.prompt = ", ".join(parts)

            # Auto-set creativity embedding when user didn't set it.
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

    print("Loading checkpoint and encoders...")
    model, cfg, rae_bridge, fusion_sd = load_model_from_ckpt(args.ckpt, device)

    # Apply LoRAs
    if args.lora:
        from models.lora import apply_loras

        lora_specs = []
        for spec in args.lora:
            if ":" in spec:
                path, scale = spec.rsplit(":", 1)
                lora_specs.append((path.strip(), float(scale)))
            else:
                lora_specs.append((spec.strip(), 0.8))
        _, num_keys = apply_loras(model, lora_specs)
        print(f"Applied {len(lora_specs)} LoRA(s), {num_keys} layer(s)")

    from diffusers import AutoencoderKL, AutoencoderRAE
    from transformers import AutoTokenizer, T5EncoderModel
    from utils.text_encoder_bundle import attach_fusion_weights, load_text_encoder_bundle

    text_bundle = None
    if str(getattr(cfg, "text_encoder_mode", "t5") or "t5").lower() == "triple":
        text_bundle = load_text_encoder_bundle(cfg, device)
        if text_bundle is None:
            raise RuntimeError("Checkpoint config requests triple text encoders but bundle failed to load.")
        if fusion_sd is not None:
            attach_fusion_weights(text_bundle, fusion_sd)
        tokenizer = text_bundle.tokenizer
        text_encoder = text_bundle.text_encoder
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
    num_gen = max(1, getattr(args, "num", 1))
    if num_gen < 2 and str(getattr(args, "pick_best", "none")).lower() not in ("none", ""):
        print("Note: --pick-best only applies with --num >= 2; ignoring.", file=sys.stderr)
    # Resolve emphasis (word)/[word] first so we have prompt_to_encode for conflict filter
    if "(" in args.prompt or "[" in args.prompt:
        prompt_to_encode, _emphasis_segments = _parse_prompt_emphasis(args.prompt)
    else:
        prompt_to_encode, _emphasis_segments = args.prompt, []
    if getattr(args, "boost_quality", False) and prompt_to_encode.strip():
        try:
            from data.caption_utils import QUALITY_PREFIX

            if not prompt_to_encode.strip().lower().startswith("masterpiece"):
                prompt_to_encode = f"{QUALITY_PREFIX}{prompt_to_encode}".strip()
        except ImportError:
            prompt_to_encode = f"masterpiece, best quality, {prompt_to_encode}".strip()

    expected_texts = _parse_expected_texts(getattr(args, "expected_text", ""))
    if getattr(args, "ocr_fix", False) and expected_texts:
        # Encourage the model not to suppress text and bias toward exact content.
        args.text_in_image = True
        prompt_to_encode = _maybe_append_text_says(prompt_to_encode, expected_texts)
        args.prompt = prompt_to_encode
    try:
        from config.prompt_domains import ANTI_AI_LOOK_NEGATIVE

        from config import DEFAULT_NEGATIVE_PROMPT, TEXT_IN_IMAGE_NEGATIVE, TEXT_IN_IMAGE_PHRASES
    except ImportError:
        DEFAULT_NEGATIVE_PROMPT = " "
        TEXT_IN_IMAGE_NEGATIVE = (
            "garbled text, misspelled, wrong spelling, illegible, watermark, signature, low quality, blurry"
        )
        TEXT_IN_IMAGE_PHRASES = ("sign that says", "text that says", "lettering", "written", 'reads "', 'says "')
        ANTI_AI_LOOK_NEGATIVE = (
            "oversaturated, plastic skin, smooth skin, airbrushed, waxy, doll-like, synthetic, artificial, CGI, uncanny"
        )
    # When user wants text in the image (sign, lettering, etc.), use a negative that avoids bad text but doesn't suppress desired text
    user_neg = (args.negative_prompt or "").strip()
    if user_neg:
        negative_text_raw = user_neg
    elif getattr(args, "text_in_image", False):
        negative_text_raw = TEXT_IN_IMAGE_NEGATIVE
    else:
        prompt_lower = prompt_to_encode.lower()
        if any(phrase in prompt_lower for phrase in TEXT_IN_IMAGE_PHRASES):
            negative_text_raw = TEXT_IN_IMAGE_NEGATIVE
            print(
                "Text-in-image detected: using text-friendly negative (legible text not suppressed).", file=sys.stderr
            )
        else:
            negative_text_raw = DEFAULT_NEGATIVE_PROMPT
    if getattr(args, "naturalize", False):
        negative_text_raw = f"{negative_text_raw}, {ANTI_AI_LOOK_NEGATIVE}".strip()
    if getattr(args, "anti_bleed", False):
        try:
            from config.prompt_domains import CONCEPT_BLEEDING_NEGATIVE

            negative_text_raw = f"{negative_text_raw}, {CONCEPT_BLEEDING_NEGATIVE}".strip()
        except ImportError:
            pass
    if getattr(args, "diversity", False):
        try:
            from config.prompt_domains import FLUX_FACE_DIVERSITY_NEGATIVE

            negative_text_raw = f"{negative_text_raw}, {FLUX_FACE_DIVERSITY_NEGATIVE}".strip()
        except ImportError:
            pass
    if getattr(args, "anti_artifacts", False):
        try:
            from config.prompt_domains import ARTIFACT_NEGATIVES

            negative_text_raw = f"{negative_text_raw}, {ARTIFACT_NEGATIVES}".strip()
        except ImportError:
            pass
    if getattr(args, "strong_watermark", False):
        try:
            from config.prompt_domains import WATERMARK_NEGATIVE_STRONG

            negative_text_raw = f"{negative_text_raw}, {WATERMARK_NEGATIVE_STRONG}".strip()
        except ImportError:
            pass
    if anatomy_scales or object_scales or scene_scales:
        negative_text_raw = f"{negative_text_raw}, {SCALE_DISTORTION_NEGATIVE}".strip()
    if character_negative_additions:
        negative_text_raw = f"{negative_text_raw}, {character_negative_additions}".strip()
    # Pos/neg conflict: remove from negative any token that appears in positive so CFG doesn't fight the user's intent
    if getattr(args, "no_neg_filter", False):
        negative_text = negative_text_raw
    else:
        negative_text = _filter_negative_by_positive(prompt_to_encode, negative_text_raw)
        if not negative_text.strip():
            negative_text = " "
        if negative_text != negative_text_raw:
            print(
                f'Negative prompt filtered (conflict resolution): "{negative_text_raw[:60]}{"..." if len(negative_text_raw) > 60 else ""}" -> "{negative_text[:60]}{"..." if len(negative_text) > 60 else ""}"',
                file=sys.stderr,
            )
    # Style: explicit --style or auto-extract from prompt (artist/style tags from PixAI, Danbooru, etc.)
    effective_style = (args.style or "").strip()
    if getattr(cfg, "style_embed_dim", 0) and not effective_style and getattr(args, "auto_style_from_prompt", False):
        try:
            from config.style_artists import extract_style_from_text

            effective_style = extract_style_from_text(prompt_to_encode) or ""
            if effective_style:
                print(
                    f'Auto-style from prompt: "{effective_style[:50]}{"..." if len(effective_style) > 50 else ""}"',
                    file=sys.stderr,
                )
        except Exception:
            pass
    style_key = effective_style if getattr(cfg, "style_embed_dim", 0) else ""
    cache_key = (prompt_to_encode, negative_text, style_key) if not getattr(args, "no_cache", False) else None
    if cache_key is not None and cache_key in _t5_cache:
        cond_emb, uncond_emb, style_emb_cached = _t5_cache[cache_key]
        cond_emb = cond_emb.to(device)
        uncond_emb = uncond_emb.to(device)
        if style_emb_cached is not None:
            style_emb_cached = style_emb_cached.to(device)
        print("T5 cache hit.")
    else:
        cond_emb = encode_text([prompt_to_encode], tokenizer, text_encoder, device, text_bundle=text_bundle)
        uncond_emb = encode_text([negative_text], tokenizer, text_encoder, device, text_bundle=text_bundle)
        style_emb_cached = None
        if effective_style and getattr(cfg, "style_embed_dim", 0):
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
        tw = _token_weights_from_segments(prompt_to_encode, _emphasis_segments, tokenizer, 300, device)
        if tw is not None:
            model_kwargs_cond["token_weights"] = tw

    # Creativity/diversity knob (only if model has creativity_embed_dim)
    if getattr(cfg, "creativity_embed_dim", 0) and args.creativity is not None:
        c = max(0.0, min(1.0, float(args.creativity)))
        model_kwargs_cond["creativity"] = torch.tensor([c], device=device, dtype=cond_emb.dtype)

    # Control image
    if args.control_image:
        pil = Image.open(args.control_image).convert("RGB")
        w, h = pil.size
        if w != image_size or h != image_size:
            pil = pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
        arr = np.array(pil).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        ctrl = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        model_kwargs_cond["control_image"] = ctrl
        model_kwargs_cond["control_scale"] = args.control_scale

    # Img2img / from-z / inpainting
    num_timesteps = getattr(cfg, "num_timesteps", 1000)
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
            x_init = diffusion.q_sample(z0, torch.tensor([start_timestep], device=device).expand(z0.shape[0]))
            print(f"Img2img: strength={args.strength} -> t_start={start_timestep}")

    shape = (num_gen, 4, latent_size, latent_size)
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
    print(f"Sampling (steps={args.steps}, num={num_gen}, cfg_scale={cfg_scale})...")
    x0 = diffusion.sample_loop(
        model,
        shape,
        model_kwargs_cond=model_kwargs_cond,
        model_kwargs_uncond=model_kwargs_uncond,
        cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        num_inference_steps=args.steps,
        eta=0.0,
        device=device,
        dtype=torch.float32,
        x_init=x_init,
        start_timestep=start_timestep,
        dynamic_threshold_percentile=dyn_thresh_p,
        dynamic_threshold_type=getattr(args, "dynamic_threshold_type", "percentile"),
        dynamic_threshold_value=getattr(args, "dynamic_threshold_value", 0.0),
        scheduler=getattr(args, "scheduler", "ddim"),
        inpaint_mask=inpaint_mask_latent,
        inpaint_x0=inpaint_x0,
        inpaint_noise=inpaint_noise,
        inpaint_freeze_known=(args.mask and args.inpaint_mode == "mdm"),
        ada_early_exit_delta_threshold=(
            getattr(args, "ada_exit_delta_threshold", 0.0) if getattr(args, "ada_early_exit", False) else 0.0
        ),
        ada_early_exit_patience=(
            int(getattr(args, "ada_exit_patience", 0)) if getattr(args, "ada_early_exit", False) else 0
        ),
        ada_early_exit_min_steps=int(getattr(args, "ada_exit_min_steps", 0)),
        pbfm_edge_boost=float(getattr(args, "pbfm_edge_boost", 0.0)),
        pbfm_edge_kernel=int(getattr(args, "pbfm_edge_kernel", 3)),
    )

    # Optional refinement pass: add a little noise to the final latent and denoise once more.
    # This tends to reduce small artifacts while keeping composition.
    if not getattr(args, "no_refine", False):
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
    out_h, out_w = image_size, image_size
    if args.width > 0 or args.height > 0:
        out_w = args.width or image_size
        out_h = args.height or image_size
        image = torch.nn.functional.interpolate(image, size=(out_h, out_w), mode="bilinear", align_corners=False)
    # Civitai-style tip: non-native resolution often causes blur/artifacts
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
        if args.contrast != 1.0 or args.sharpen > 0:
            try:
                from utils.quality import contrast, sharpen

                if args.contrast != 1.0:
                    img_np = contrast(img_np.astype(np.float32), factor=args.contrast)
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                if args.sharpen > 0:
                    img_np = sharpen(img_np, amount=args.sharpen)
            except Exception:
                pass
        if getattr(args, "naturalize", False):
            try:
                from utils.quality import naturalize

                grain = max(0.0, getattr(args, "naturalize_grain", 0.015))
                img_np = naturalize(img_np, grain_amount=grain, micro_contrast=1.02, seed=args.seed + i)
            except Exception:
                pass
        processed.append(img_np)
    saved_imgs = processed

    pick_m = (getattr(args, "pick_best", None) or "none").lower()
    best_idx = 0
    if num_gen > 1 and pick_m != "none":
        from utils.test_time_pick import pick_best_indices

        exp_ocr = ""
        if isinstance(expected_texts, list) and expected_texts:
            exp_ocr = str(expected_texts[0])
        best_idx, scores = pick_best_indices(
            processed,
            prompt_to_encode,
            pick_m,
            str(device),
            exp_ocr,
            getattr(args, "pick_clip_model", "openai/clip-vit-base-patch32"),
        )
        print(f"pick-best ({pick_m}): scores={scores} -> best index {best_idx}")

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
            f"scheduler: {getattr(args, 'scheduler', 'ddim')}",
        ]
        prompt_txt = out_path.parent / f"{stem}.txt"
        prompt_txt.write_text("\n".join(params_lines), encoding="utf-8")
        print(f"Saved params: {prompt_txt}")

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
            from utils.text_rendering import create_text_rendering_pipeline

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
                        getattr(args, "scheduler", "ddim"),
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

                    # Forward common generation knobs that impact text quality.
                    if (
                        getattr(args, "dynamic_threshold_type", "percentile") != "percentile"
                        or getattr(args, "dynamic_threshold_value", 0.0) > 0.0
                        or getattr(args, "dynamic_threshold_percentile", 0.0) > 0.0
                    ):
                        repair_cmd += [
                            "--dynamic-threshold-percentile",
                            str(getattr(args, "dynamic_threshold_percentile", 0.0)),
                            "--dynamic-threshold-type",
                            getattr(args, "dynamic_threshold_type", "percentile"),
                            "--dynamic-threshold-value",
                            str(getattr(args, "dynamic_threshold_value", 0.0)),
                        ]
                    if getattr(args, "vae_tiling", False):
                        repair_cmd.append("--vae-tiling")
                    if getattr(args, "deterministic", False):
                        repair_cmd.append("--deterministic")
                    if getattr(args, "no_cache", False):
                        repair_cmd.append("--no-cache")
                    if getattr(args, "no_refine", False):
                        repair_cmd.append("--no-refine")
                    if getattr(args, "refine_t", None) is not None:
                        repair_cmd += ["--refine-t", str(getattr(args, "refine_t", 50))]
                    if getattr(args, "no_neg_filter", False):
                        repair_cmd.append("--no-neg-filter")
                    # Ensure text isn't suppressed.
                    repair_cmd.append("--text-in-image")

                    # Optional prompt controls.
                    if getattr(args, "gender_swap", False):
                        repair_cmd.append("--gender-swap")
                    if getattr(args, "anatomy_scale", ""):
                        repair_cmd += ["--anatomy-scale", str(getattr(args, "anatomy_scale"))]
                    if getattr(args, "object_scale", ""):
                        repair_cmd += ["--object-scale", str(getattr(args, "object_scale"))]
                    if getattr(args, "scene_scale", ""):
                        repair_cmd += ["--scene-scale", str(getattr(args, "scene_scale"))]
                    if getattr(args, "character_sheet", ""):
                        repair_cmd += ["--character-sheet", str(getattr(args, "character_sheet"))]
                    if getattr(args, "character_prompt_extra", ""):
                        repair_cmd += ["--character-prompt-extra", str(getattr(args, "character_prompt_extra"))]
                    if getattr(args, "character_negative_extra", ""):
                        repair_cmd += ["--character-negative-extra", str(getattr(args, "character_negative_extra"))]
                    if getattr(args, "preset", None):
                        repair_cmd += ["--preset", str(getattr(args, "preset"))]
                    if getattr(args, "op_mode", None):
                        repair_cmd += ["--op-mode", str(getattr(args, "op_mode"))]
                    if getattr(args, "hard_style", None):
                        repair_cmd += ["--hard-style", str(getattr(args, "hard_style"))]
                    if getattr(args, "boost_quality", False):
                        repair_cmd.append("--boost-quality")
                    if getattr(args, "style", ""):
                        repair_cmd += ["--style", str(getattr(args, "style"))]
                        repair_cmd += ["--style-strength", str(getattr(args, "style_strength", 0.7))]
                    if getattr(args, "control_image", ""):
                        repair_cmd += ["--control-image", str(getattr(args, "control_image"))]
                        repair_cmd += ["--control-scale", str(getattr(args, "control_scale", 0.85))]
                    if getattr(args, "lora", None):
                        repair_cmd += ["--lora"] + [str(x) for x in getattr(args, "lora", [])]
                        if getattr(args, "lora_trigger", ""):
                            repair_cmd += ["--lora-trigger", str(getattr(args, "lora_trigger"))]
                    if getattr(args, "naturalize", False):
                        repair_cmd.append("--naturalize")
                        repair_cmd += ["--naturalize-grain", str(getattr(args, "naturalize_grain", 0.015))]
                    if getattr(args, "anti_bleed", False):
                        repair_cmd.append("--anti-bleed")
                    if getattr(args, "diversity", False):
                        repair_cmd.append("--diversity")
                    if getattr(args, "anti_artifacts", False):
                        repair_cmd.append("--anti-artifacts")
                    if getattr(args, "strong_watermark", False):
                        repair_cmd.append("--strong-watermark")

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
    main()
