"""
Generate an image from a text prompt using a trained checkpoint.
Supports: prompt, negative prompt, steps, width, height, CFG, timestep schedules (ddim, euler, karras_rho, …) and solvers (ddim, heun).
Optional: style, control-image, lora, img2img, inpainting, dual-stage layout (coarse latent then detail pass),
hires-fix (latent upscale + refine), volatile CFG (spike-aware guidance), CLIP-guard extra denoise,
CLIP monitor (mid-loop CFG boost on low cosine), spectral-coherence latent (FFT lowfreq blend),
domain latent prior, sharpen, contrast, saturation, clarity / tone-punch / chroma-smooth / polish /
finishing-preset (cross-style post), emphasis (word)/[word].

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
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.model_presets import apply_op_mode_to_args, apply_preset_to_args
from diffusion import INFERENCE_SOLVERS, create_diffusion, list_timestep_schedules
from diffusion.holy_grail import (
    apply_holy_grail_preset_to_args,
    list_holy_grail_presets,
    recommend_holy_grail_preset,
    sanitize_holy_grail_kwargs,
)
from models.controlnet import control_type_to_id, infer_control_type_from_path
from utils.checkpoint.checkpoint_loading import load_dit_text_checkpoint
from utils.prompt.neg_filter import filter_negative_by_positive
from utils.prompt.prompt_emphasis import parse_prompt_emphasis, token_weights_from_cleaned_segments


def _configure_stdio_for_console() -> None:
    """
    Avoid UnicodeEncodeError on Windows terminals using legacy encodings (e.g. cp1252).
    This keeps --help and logging robust even when text includes Unicode symbols.
    """
    if os.name != "nt":
        return
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass


_configure_stdio_for_console()


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
            data = json.loads(raw)
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


def _sanitize_character_prompt_tokens(tokens: list, negative_tokens: list, *, uncensored_mode: bool = False) -> Tuple[list, list]:
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
    import json

    p = Path(sheet_path)
    if not p.exists():
        raise ValueError(f"character-sheet not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))

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


@torch.no_grad()
def encode_text(
    captions,
    tokenizer,
    text_encoder,
    device,
    max_length=300,
    dtype=torch.float32,
    text_bundle=None,
    clip_captions=None,
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
            segment_texts=segment_texts,
        )
    if segment_texts is not None:
        from utils.modeling.t5_segmented_encode import encode_t5_segment_concat

        return encode_t5_segment_concat(
            segment_texts, tokenizer, text_encoder, device, max_length=max_length, dtype=dtype
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
        help=(
            "Override per-role early/mid/late multipliers, e.g. "
            "'character=1.15/1.0/0.85,style=0.9/1.0/1.1'."
        ),
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
        choices=["euler", "heun"],
        help="ODE solver when using flow-matching sampling.",
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
    # Test-time scaling: generate N candidates (--num) and keep the best (§11.3 IMPROVEMENTS.md)
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
            "combo",
            "combo_exposure",
            "combo_structural",
            "combo_hq",
            "combo_count",
        ],
        help="With --num > 1, score candidates and save the best to --out (clip|edge|ocr|combo|combo_exposure|combo_structural|combo_hq|combo_count)",
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
    _ts_choices = tuple(sorted(list_timestep_schedules()))
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=_ts_choices,
        metavar="NAME",
        help=f"Timestep index schedule (noise→clean path): {', '.join(_ts_choices)}. Composes with --steps and --solver.",
    )
    parser.add_argument(
        "--timestep-schedule",
        type=str,
        default=None,
        choices=_ts_choices,
        metavar="NAME",
        help="If set, overrides --scheduler (same names).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="ddim",
        choices=tuple(INFERENCE_SOLVERS),
        help="Update rule: ddim (1 model eval per step) or heun (2 evals/step, often sharper).",
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
        "--safety-mode",
        type=str,
        default="nsfw",
        choices=["none", "sfw", "nsfw"],
        help="Content intent mode: sfw/nsfw scaffolding for better adherence. (default: nsfw for uncensored model)",
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
        help="Anatomy consistency hint for subject sex/presentation.",
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
        "--nsfw-pack",
        type=str,
        default="none",
        choices=["none", "soft", "explicit_detail", "romantic", "extreme"],
        help="Adult-content pose/anatomy stability pack.",
    )
    parser.add_argument(
        "--nsfw-civitai-pack",
        type=str,
        default="none",
        choices=[
            "none",
            "hits",
            "hits_lite",
            "snippets",
            "snippets_lite",
            "action",
            "complex",
            "easy",
            "clothing",
            "objects",
            "style",
        ],
        help="Extra NSFW pack: hits*=freq CSV tags; snippets*=short name/trigger fragments (deduped vs hits). Stacks with --nsfw-pack when --safety-mode nsfw.",
    )
    parser.add_argument(
        "--civitai-trigger-bank",
        type=str,
        default="none",
        choices=[
            "none",
            "light",
            "medium",
            "heavy",
            "frequency_light",
            "frequency_medium",
            "frequency_heavy",
        ],
        help="Append triggers: light/medium/heavy = CSV row order; frequency_* = top_triggers_by_frequency.txt (requires --safety-mode nsfw).",
    )
    parser.add_argument(
        "--civitai-model-bank-csv",
        type=str,
        default="",
        help="Override CSV path for row-order trigger bank (default: data/civitai/nsfw_illustrious_noobai_models.csv).",
    )
    parser.add_argument(
        "--civitai-frequency-txt",
        type=str,
        default="",
        help="Override path for frequency_* trigger bank (default: data/civitai/top_triggers_by_frequency.txt).",
    )
    parser.add_argument(
        "--sex-position",
        type=str,
        default="none",
        choices=["none", "standing_missionary", "doggy", "cowgirl", "missionary", "spooning", "standing"],
        help="Specific sex position for extreme scene control.",
    )
    parser.add_argument(
        "--penetration-detail",
        type=str,
        default="none",
        choices=["none", "normal", "deep", "extreme"],
        help="Level of penetration detail and anatomical exaggeration.",
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
        "--sfw-mood",
        type=str,
        default="none",
        choices=["none", "wholesome", "heroic", "serene", "adventurous", "joyful", "cozy"],
        help="SFW mood control for wholesome and artistic scenes.",
    )
    parser.add_argument(
        "--sfw-pose",
        type=str,
        default="none",
        choices=["none", "elegant", "heroic", "contemplative", "playful", "dynamic", "relaxed"],
        help="SFW pose control for elegant and wholesome scenes.",
    )
    parser.add_argument(
        "--sfw-clothing",
        type=str,
        default="none",
        choices=["none", "casual", "elegant", "cozy", "adventurer", "fantasy", "historical", "sporty"],
        help="SFW clothing control for wholesome outfits.",
    )
    parser.add_argument(
        "--sfw-environment",
        type=str,
        default="none",
        choices=["none", "forest", "cabin", "meadow", "library", "garden", "night"],
        help="SFW environment control for peaceful scenes.",
    )
    parser.add_argument(
        "--sfw-expression",
        type=str,
        default="none",
        choices=["none", "gentle", "joyful", "curious", "serene", "determined"],
        help="SFW facial expression control.",
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
        help="Apply a sampler preset (soft defaults) from config.model_presets",
    )
    parser.add_argument(
        "--op-mode",
        type=str,
        default=None,
        choices=["portrait", "fullbody", "anime_char"],
        help="High-level OP mode (applied after preset)",
    )
    _hg_preset_choices = ["auto"] + list_holy_grail_presets()
    parser.add_argument(
        "--holy-grail-preset",
        type=str,
        default=None,
        choices=_hg_preset_choices,
        help="Apply a holy-grail preset bundle (auto|balanced|photoreal|anime|illustration|aggressive).",
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
    args = parser.parse_args()

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

    has_tags = bool(getattr(args, "tags", "").strip() or getattr(args, "tags_file", "").strip())
    has_prompt_file = bool(getattr(args, "prompt_file", "").strip())
    has_prompt_layout = bool(getattr(args, "prompt_layout", "").strip())
    if not (args.prompt or has_tags or has_prompt_file or has_prompt_layout):
        parser.error(
            "Provide at least one of --prompt, --prompt-file, --tags, --tags-file, or --prompt-layout"
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
    if has_prompt_layout:
        try:
            from utils.prompt.prompt_layout import load_prompt_layout_file, merge_prompt_with_layout

            compiled = load_prompt_layout_file(str(args.prompt_layout).strip())
            args._prompt_layout_negative = compiled.negative or ""
            args._used_prompt_layout = True
            args._layout_compiled = compiled
            prompt_for_encoding = merge_prompt_with_layout(
                compiled.positive, prompt_for_encoding, layout_first=True
            )
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
            from config.prompt_domains import NATURAL_LOOK_POSITIVE, NATURAL_LOOK_POSITIVE_DEEP

            _nat_pre = NATURAL_LOOK_POSITIVE_DEEP if getattr(args, "naturalize_deep", False) else NATURAL_LOOK_POSITIVE
            args.prompt = f"{_nat_pre}, {args.prompt}"
        except ImportError:
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
        # Fixed latent shapes across steps: pick faster conv algorithms (not bit-reproducible).
        torch.backends.cudnn.benchmark = True

    print("Loading checkpoint and encoders...")
    model, cfg, rae_bridge, fusion_sd = load_model_from_ckpt(args.ckpt, device)

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
        prompt_to_encode, _emphasis_segments = parse_prompt_emphasis(args.prompt)
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
    _sm_mode = str(getattr(args, "shortcomings_mitigation", "none") or "none").lower()
    _ag_mode = str(getattr(args, "art_guidance_mode", "none") or "none").lower()
    _ag_anat = str(getattr(args, "anatomy_guidance", "none") or "none").lower()
    _sg_mode = str(getattr(args, "style_guidance_mode", "none") or "none").lower()
    _ag_pos = ""
    _ag_neg = ""
    _sg_pos = ""
    _sg_neg = ""
    if _sm_mode in ("auto", "all") and prompt_to_encode.strip():
        try:
            from config.ai_image_shortcomings import mitigation_fragments

            _pos_sm, _ = mitigation_fragments(
                prompt_to_encode,
                _sm_mode,  # type: ignore[arg-type]
                include_2d_pack=bool(getattr(args, "shortcomings_2d", False)),
            )
            if _pos_sm:
                prompt_to_encode = f"{prompt_to_encode}, {_pos_sm}".strip().strip(",")
                args.prompt = prompt_to_encode
        except Exception:
            pass
    if (_ag_mode in ("auto", "all") or _ag_anat in ("lite", "strong")) and prompt_to_encode.strip():
        try:
            from config.art_mediums import guidance_fragments

            _ag_pos, _ag_neg = guidance_fragments(
                prompt_to_encode,
                _ag_mode,  # type: ignore[arg-type]
                include_photography=not bool(getattr(args, "no_art_guidance_photography", False)),
                anatomy_mode=_ag_anat,  # type: ignore[arg-type]
            )
            if _ag_pos:
                prompt_to_encode = f"{prompt_to_encode}, {_ag_pos}".strip().strip(",")
                args.prompt = prompt_to_encode
        except Exception:
            pass
    if _sg_mode in ("auto", "all") and prompt_to_encode.strip():
        try:
            from config.style_guidance import style_guidance_fragments

            _sg_pos, _sg_neg = style_guidance_fragments(
                prompt_to_encode,
                _sg_mode,  # type: ignore[arg-type]
                include_artist_refs=bool(getattr(args, "style_guidance_artists", True)),
            )
            if _sg_pos:
                prompt_to_encode = f"{prompt_to_encode}, {_sg_pos}".strip().strip(",")
                args.prompt = prompt_to_encode
        except Exception:
            pass
    try:
        from config.prompt_domains import ANTI_AI_LOOK_NEGATIVE, ANTI_AI_LOOK_NEGATIVE_STRONG

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
        ANTI_AI_LOOK_NEGATIVE_STRONG = ANTI_AI_LOOK_NEGATIVE
    # When user wants text in the image (sign, lettering, etc.), use a negative that avoids bad text but doesn't suppress desired text
    user_neg = (args.negative_prompt or "").strip()
    layout_neg = (getattr(args, "_prompt_layout_negative", None) or "").strip()
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
    if layout_neg:
        negative_text_raw = (
            f"{negative_text_raw}, {layout_neg}".strip().strip(",") if negative_text_raw else layout_neg
        )
    if getattr(args, "naturalize", False):
        _anti_nat = ANTI_AI_LOOK_NEGATIVE_STRONG if getattr(args, "naturalize_deep", False) else ANTI_AI_LOOK_NEGATIVE
        negative_text_raw = f"{negative_text_raw}, {_anti_nat}".strip()
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
    if _sm_mode in ("auto", "all"):
        try:
            from config.ai_image_shortcomings import mitigation_fragments

            _, _neg_sm = mitigation_fragments(
                prompt_to_encode,
                _sm_mode,  # type: ignore[arg-type]
                include_2d_pack=bool(getattr(args, "shortcomings_2d", False)),
            )
            if _neg_sm:
                negative_text_raw = f"{negative_text_raw}, {_neg_sm}".strip()
        except Exception:
            pass
    if _ag_neg:
        negative_text_raw = f"{negative_text_raw}, {_ag_neg}".strip()
    if _sg_neg:
        negative_text_raw = f"{negative_text_raw}, {_sg_neg}".strip()
    if getattr(args, "less_ai", False):
        if str(getattr(args, "anti_ai_pack", "none") or "none") == "none":
            args.anti_ai_pack = "lite"
        if str(getattr(args, "human_media_mode", "none") or "none") == "none":
            args.human_media_mode = "photographic"
    if len(getattr(args, "lora", []) or []) > 1:
        try:
            from config.prompt_domains import LORA_STACK_NEGATIVE

            negative_text_raw = f"{negative_text_raw}, {LORA_STACK_NEGATIVE}".strip()
        except ImportError:
            pass
    # Optional prompt controls for hard cases: sfw/nsfw, pose, viewpoint, domain grounding.
    # Also applies stronger anti-text/logo suppression when text is not desired.
    try:
        from utils.prompt.content_controls import apply_content_controls, infer_content_controls_from_prompt

        scene_domain = str(getattr(args, "scene_domain", "none") or "none")
        view_angle = str(getattr(args, "view_angle", "none") or "none")
        pose_mode = str(getattr(args, "pose_mode", "none") or "none")
        style_mode = str(getattr(args, "style_mode", "none") or "none")
        safety_mode = str(getattr(args, "safety_mode", "none") or "none")
        clothing_mode = str(getattr(args, "clothing_mode", "none") or "none")
        composition_mode = str(getattr(args, "composition_mode", "none") or "none")
        people_layout = str(getattr(args, "people_layout", "none") or "none")
        hand_mode = str(getattr(args, "hand_mode", "none") or "none")
        lighting_mode = str(getattr(args, "lighting_mode", "none") or "none")
        nsfw_pack = str(getattr(args, "nsfw_pack", "none") or "none")
        sex_position = str(getattr(args, "sex_position", "none") or "none")
        human_media_mode = str(getattr(args, "human_media_mode", "none") or "none")
        anti_ai_pack = str(getattr(args, "anti_ai_pack", "none") or "none")
        lora_scaffold_ef = str(getattr(args, "lora_scaffold", "none") or "none")
        adherence_pack = str(getattr(args, "adherence_pack", "none") or "none")
        if getattr(args, "lora_scaffold_auto", False) and getattr(args, "lora", None) and lora_scaffold_ef == "none":
            lora_scaffold_ef = "blend"
        sfw_mood = str(getattr(args, "sfw_mood", "none") or "none")
        sfw_pose = str(getattr(args, "sfw_pose", "none") or "none")
        sfw_clothing = str(getattr(args, "sfw_clothing", "none") or "none")
        sfw_environment = str(getattr(args, "sfw_environment", "none") or "none")
        sfw_expression = str(getattr(args, "sfw_expression", "none") or "none")
        if getattr(args, "auto_content_fix", False):
            inferred = infer_content_controls_from_prompt(prompt_to_encode)
            if scene_domain == "none":
                scene_domain = inferred.get("scene_domain", scene_domain)
            if view_angle == "none":
                view_angle = inferred.get("view_angle", view_angle)
            if pose_mode == "none":
                pose_mode = inferred.get("pose_mode", pose_mode)
            if style_mode == "none":
                style_mode = inferred.get("style_mode", style_mode)
            if safety_mode == "none" and inferred.get("safety_mode"):
                safety_mode = inferred["safety_mode"]
            if composition_mode == "none" and inferred.get("composition_mode"):
                composition_mode = inferred["composition_mode"]
            if people_layout == "none" and inferred.get("people_layout"):
                people_layout = inferred["people_layout"]
            if hand_mode == "none" and inferred.get("hand_mode"):
                hand_mode = inferred["hand_mode"]
            if lighting_mode == "none" and inferred.get("lighting_mode"):
                lighting_mode = inferred["lighting_mode"]
            if clothing_mode == "none" and inferred.get("clothing_mode"):
                clothing_mode = inferred["clothing_mode"]
            if nsfw_pack == "none" and inferred.get("nsfw_pack"):
                nsfw_pack = inferred["nsfw_pack"]
            if sex_position == "none" and inferred.get("sex_position"):
                sex_position = inferred["sex_position"]
            if human_media_mode == "none" and inferred.get("human_media_mode"):
                human_media_mode = inferred["human_media_mode"]
            if adherence_pack == "none" and inferred.get("adherence_pack"):
                adherence_pack = str(inferred["adherence_pack"])

        prompt_to_encode, negative_text_raw = apply_content_controls(
            prompt_to_encode,
            negative_text_raw,
            safety_mode=safety_mode,
            pose_mode=pose_mode,
            view_angle=view_angle,
            subject_sex=str(getattr(args, "subject_sex", "none") or "none"),
            scene_domain=scene_domain,
            clothing_mode=clothing_mode,
            background_mode=str(getattr(args, "background_mode", "none") or "none"),
            people_layout=people_layout,
            relationship_mode=str(getattr(args, "relationship_mode", "none") or "none"),
            object_layout=str(getattr(args, "object_layout", "none") or "none"),
            hand_mode=hand_mode,
            pose_naturalness=str(getattr(args, "pose_naturalness", "none") or "none"),
            typography_mode=str(getattr(args, "typography_mode", "none") or "none"),
            quality_pack=str(getattr(args, "quality_pack", "none") or "none"),
            lighting_mode=lighting_mode,
            skin_detail_mode=str(getattr(args, "skin_detail_mode", "none") or "none"),
            nsfw_pack=nsfw_pack,
            sex_position=sex_position,
            penetration_detail=str(getattr(args, "penetration_detail", "none") or "none"),
            body_proportion=str(getattr(args, "body_proportion", "none") or "none"),
            interaction_intensity=str(getattr(args, "interaction_intensity", "none") or "none"),
            sfw_mood=sfw_mood,
            sfw_pose=sfw_pose,
            sfw_clothing=sfw_clothing,
            sfw_environment=sfw_environment,
            sfw_expression=sfw_expression,
            style_mode=style_mode,
            style_lock=bool(getattr(args, "style_lock", False)),
            anti_style_bleed=bool(getattr(args, "anti_style_bleed", False)),
            composition_mode=composition_mode,
            anti_duplicate_subjects=bool(getattr(args, "anti_duplicate_subjects", False)),
            anti_perspective_drift=bool(getattr(args, "anti_perspective_drift", False)),
            cleanup_conflicting_tags=bool(getattr(args, "cleanup_conflicting_tags", False)),
            allow_text_in_image=bool(getattr(args, "text_in_image", False)),
            nsfw_civitai_pack=str(getattr(args, "nsfw_civitai_pack", "none") or "none"),
            civitai_trigger_bank=str(getattr(args, "civitai_trigger_bank", "none") or "none"),
            civitai_model_bank_csv=(str(getattr(args, "civitai_model_bank_csv", "") or "").strip() or None),
            civitai_frequency_txt=(str(getattr(args, "civitai_frequency_txt", "") or "").strip() or None),
            one_shot_boost=bool(getattr(args, "one_shot_boost", True)),
            anti_ai_pack=anti_ai_pack,
            human_media_mode=human_media_mode,
            lora_scaffold=lora_scaffold_ef,
            adherence_pack=adherence_pack,
        )
    except Exception as e:
        print(f"Warning: apply_content_controls failed: {e}", file=sys.stderr)
        if os.environ.get("SDX_DEBUG", "").strip():
            traceback.print_exc()
    if anatomy_scales or object_scales or scene_scales:
        negative_text_raw = f"{negative_text_raw}, {SCALE_DISTORTION_NEGATIVE}".strip()
    if character_negative_additions:
        negative_text_raw = f"{negative_text_raw}, {character_negative_additions}".strip()
    if scene_negative_additions:
        negative_text_raw = f"{negative_text_raw}, {scene_negative_additions}".strip()
    # Pos/neg conflict: remove from negative any token that appears in positive so CFG doesn't fight the user's intent
    if getattr(args, "no_neg_filter", False):
        negative_text = negative_text_raw
    else:
        negative_text = filter_negative_by_positive(prompt_to_encode, negative_text_raw)
        if not negative_text.strip():
            negative_text = " "
        if negative_text != negative_text_raw:
            print(
                f'Negative prompt filtered (conflict resolution): "{negative_text_raw[:60]}{"..." if len(negative_text_raw) > 60 else ""}" -> "{negative_text[:60]}{"..." if len(negative_text) > 60 else ""}"',
                file=sys.stderr,
            )
    # Style: explicit --style (supports weighted mix with '|') or auto-extract from prompt.
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
    t5_positive_prompt = prompt_to_encode
    segment_texts = None
    clip_caps = None
    if compiled_layout is not None and t5_layout_enc == "blocks":
        from utils.prompt.prompt_layout import substitute_compiled_layout_in_t5_prompt

        t5_positive_prompt = substitute_compiled_layout_in_t5_prompt(prompt_to_encode, compiled_layout)
    elif compiled_layout is not None and t5_layout_enc == "segmented":
        from utils.prompt.prompt_layout import t5_segment_texts_for_full_prompt

        segment_texts = t5_segment_texts_for_full_prompt(compiled_layout, prompt_to_encode)
    if text_bundle is not None and compiled_layout is not None:
        from utils.prompt.prompt_layout import triple_clip_caption

        clip_caps = [triple_clip_caption(compiled_layout, prompt_to_encode)]

    layout_cache_tag = (
        (t5_layout_enc, str(getattr(args, "prompt_layout", "") or "")) if compiled_layout is not None else ("flat", "")
    )
    cache_key = (
        (prompt_to_encode, negative_text, style_key, layout_cache_tag) if not getattr(args, "no_cache", False) else None
    )
    if cache_key is not None and cache_key in _t5_cache:
        cond_emb, uncond_emb, style_emb_cached = _t5_cache[cache_key]
        cond_emb = cond_emb.to(device)
        uncond_emb = uncond_emb.to(device)
        if style_emb_cached is not None:
            style_emb_cached = style_emb_cached.to(device)
        print("T5 cache hit.")
    else:
        cond_emb = encode_text(
            [t5_positive_prompt],
            tokenizer,
            text_encoder,
            device,
            text_bundle=text_bundle,
            clip_captions=clip_caps,
            segment_texts=segment_texts,
        )
        uncond_emb = encode_text([negative_text], tokenizer, text_encoder, device, text_bundle=text_bundle)
        style_emb_cached = None
        if effective_style and getattr(cfg, "style_embed_dim", 0):
            if style_mix and len(style_mix) > 1:
                style_texts = [t for t, _ in style_mix]
                style_weights = torch.tensor([w for _, w in style_mix], device=device, dtype=torch.float32)
                style_enc = encode_text(style_texts, tokenizer, text_encoder, device, text_bundle=text_bundle).mean(dim=1)
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
        tw = token_weights_from_cleaned_segments(
            prompt_to_encode, _emphasis_segments, tokenizer, 300, device=device
        )
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
            emb, clip_dim = encode_reference_image_pil(
                pil_r, device=device, model_id=clip_id, dtype=torch.float32
            )
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

    _clip_mon_n = int(getattr(args, "clip_monitor_every", 0) or 0)
    if _clip_mon_n > 0:
        from utils.generation.clip_alignment import latent_x0_clip_cosine

        _cm_thr = float(getattr(args, "clip_monitor_threshold", 0.22) or 0.22)
        _cm_boost = float(getattr(args, "clip_monitor_cfg_boost", 0.12) or 0.12)
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
        )
    else:
        _periodic_kw = dict(
            periodic_alignment_interval=0,
            periodic_alignment_threshold=0.0,
            periodic_alignment_cfg_boost=0.0,
            periodic_alignment_fn=None,
        )

    dual_stage = bool(getattr(args, "dual_stage_layout", False))
    if use_flow_sample and dual_stage:
        print("Dual-stage layout disabled: second stage uses VP re-noising (incompatible with flow sampler).", file=sys.stderr)
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
            )
            x0_up = torch.nn.functional.interpolate(
                x0.float(), size=(fh, fw), mode="bicubic", align_corners=False
            ).to(dtype=x0.dtype)
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
                mk_c2["control_image"] = _resize_control_tensor(model_kwargs_cond["control_image"], image_size, image_size)
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
            )
    else:
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
            **_periodic_kw,
        )

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
        if (out_w != dec_w or out_h != dec_h) and str(getattr(args, "resize_mode", "stretch") or "stretch") == "stretch":
            image = torch.nn.functional.interpolate(
                image, size=(out_h, out_w), mode="bilinear", align_corners=False
            )
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
        if (out_w != dec_w or out_h != dec_h) and str(getattr(args, "resize_mode", "stretch") or "stretch") != "stretch":
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
        if getattr(args, "naturalize", False):
            try:
                from utils.quality import naturalize

                grain = max(0.0, getattr(args, "naturalize_grain", 0.015))
                img_np = naturalize(img_np, grain_amount=grain, micro_contrast=1.02, seed=args.seed + i)
            except Exception:
                pass
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
        if isinstance(expected_texts, list) and expected_texts:
            pick_m = "combo"
        elif re.search(r"\b(exactly\s+\d+|\d+\s+(people|persons|person|characters?|girls?|boys?|coins?|candles?|windows?))\b", str(prompt_to_encode).lower()):
            pick_m = "combo_count"
        else:
            pick_m = "combo_hq"
        print(f"pick-best auto -> {pick_m}", file=sys.stderr)
    best_idx = 0
    if num_gen > 1 and pick_m != "none":
        from utils.quality.test_time_pick import pick_best_indices

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
            int(getattr(args, "expected_count", 0) or 0),
            str(getattr(args, "expected_count_target", "auto") or "auto"),
            str(getattr(args, "expected_count_object", "") or ""),
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
                    if int(getattr(args, "width", 0) or 0) > 0:
                        repair_cmd += ["--width", str(int(getattr(args, "width", 0) or 0))]
                    if int(getattr(args, "height", 0) or 0) > 0:
                        repair_cmd += ["--height", str(int(getattr(args, "height", 0) or 0))]
                    _rm = str(getattr(args, "resize_mode", "stretch") or "stretch").lower()
                    if _rm in ("stretch", "center_crop", "saliency_crop"):
                        repair_cmd += ["--resize-mode", _rm]
                    if float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0) > 0:
                        repair_cmd += [
                            "--resize-saliency-face-bias",
                            str(float(getattr(args, "resize_saliency_face_bias", 0.0) or 0.0)),
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
                    if str(getattr(args, "refine_gate", "off") or "off").lower() in ("off", "auto"):
                        repair_cmd += ["--refine-gate", str(getattr(args, "refine_gate", "off"))]
                    if getattr(args, "refine_gate_threshold", None) is not None:
                        repair_cmd += ["--refine-gate-threshold", str(getattr(args, "refine_gate_threshold", 0.62))]
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
                    if getattr(args, "label_multi_character_sheets", False):
                        repair_cmd.append("--label-multi-character-sheets")
                    if getattr(args, "character_prompt_extra", ""):
                        repair_cmd += ["--character-prompt-extra", str(getattr(args, "character_prompt_extra"))]
                    if getattr(args, "character_negative_extra", ""):
                        repair_cmd += ["--character-negative-extra", str(getattr(args, "character_negative_extra"))]
                    if getattr(args, "prompt_layout", ""):
                        repair_cmd += ["--prompt-layout", str(getattr(args, "prompt_layout"))]
                    if str(getattr(args, "t5_layout_encode", "auto") or "auto").lower() != "auto":
                        repair_cmd += ["--t5-layout-encode", str(getattr(args, "t5_layout_encode"))]
                    if getattr(args, "scene_blueprint", ""):
                        repair_cmd += ["--scene-blueprint", str(getattr(args, "scene_blueprint"))]
                    if float(getattr(args, "scene_blueprint_strength", 1.0)) != 1.0:
                        repair_cmd += ["--scene-blueprint-strength", str(getattr(args, "scene_blueprint_strength"))]
                    if float(getattr(args, "character_strength", 1.0)) != 1.0:
                        repair_cmd += ["--character-strength", str(getattr(args, "character_strength"))]
                    if bool(getattr(args, "uncensored_mode", False)):
                        repair_cmd += ["--uncensored-mode"]
                    if getattr(args, "clothing_mode", "none") != "none":
                        repair_cmd += ["--clothing-mode", str(getattr(args, "clothing_mode"))]
                    if getattr(args, "background_mode", "none") != "none":
                        repair_cmd += ["--background-mode", str(getattr(args, "background_mode"))]
                    if getattr(args, "people_layout", "none") != "none":
                        repair_cmd += ["--people-layout", str(getattr(args, "people_layout"))]
                    if getattr(args, "relationship_mode", "none") != "none":
                        repair_cmd += ["--relationship-mode", str(getattr(args, "relationship_mode"))]
                    if getattr(args, "object_layout", "none") != "none":
                        repair_cmd += ["--object-layout", str(getattr(args, "object_layout"))]
                    if getattr(args, "hand_mode", "none") != "none":
                        repair_cmd += ["--hand-mode", str(getattr(args, "hand_mode"))]
                    if getattr(args, "pose_naturalness", "none") != "none":
                        repair_cmd += ["--pose-naturalness", str(getattr(args, "pose_naturalness"))]
                    if getattr(args, "typography_mode", "none") != "none":
                        repair_cmd += ["--typography-mode", str(getattr(args, "typography_mode"))]
                    if getattr(args, "quality_pack", "none") != "none":
                        repair_cmd += ["--quality-pack", str(getattr(args, "quality_pack"))]
                    if getattr(args, "adherence_pack", "none") != "none":
                        repair_cmd += ["--adherence-pack", str(getattr(args, "adherence_pack"))]
                    if getattr(args, "lighting_mode", "none") != "none":
                        repair_cmd += ["--lighting-mode", str(getattr(args, "lighting_mode"))]
                    if getattr(args, "skin_detail_mode", "none") != "none":
                        repair_cmd += ["--skin-detail-mode", str(getattr(args, "skin_detail_mode"))]
                    if getattr(args, "nsfw_pack", "none") != "none":
                        repair_cmd += ["--nsfw-pack", str(getattr(args, "nsfw_pack"))]
                    if getattr(args, "nsfw_civitai_pack", "none") != "none":
                        repair_cmd += ["--nsfw-civitai-pack", str(getattr(args, "nsfw_civitai_pack"))]
                    if getattr(args, "civitai_trigger_bank", "none") != "none":
                        repair_cmd += ["--civitai-trigger-bank", str(getattr(args, "civitai_trigger_bank"))]
                    if str(getattr(args, "civitai_model_bank_csv", "") or "").strip():
                        repair_cmd += ["--civitai-model-bank-csv", str(getattr(args, "civitai_model_bank_csv"))]
                    if str(getattr(args, "civitai_frequency_txt", "") or "").strip():
                        repair_cmd += ["--civitai-frequency-txt", str(getattr(args, "civitai_frequency_txt"))]
                    if getattr(args, "style_mode", "none") != "none":
                        repair_cmd += ["--style-mode", str(getattr(args, "style_mode"))]
                    if bool(getattr(args, "style_lock", False)):
                        repair_cmd += ["--style-lock"]
                    if bool(getattr(args, "anti_style_bleed", False)):
                        repair_cmd += ["--anti-style-bleed"]
                    if getattr(args, "preset", None):
                        repair_cmd += ["--preset", str(getattr(args, "preset"))]
                    if getattr(args, "op_mode", None):
                        repair_cmd += ["--op-mode", str(getattr(args, "op_mode"))]
                    if getattr(args, "holy_grail_preset", None):
                        repair_cmd += ["--holy-grail-preset", str(getattr(args, "holy_grail_preset"))]
                    if getattr(args, "hard_style", None):
                        repair_cmd += ["--hard-style", str(getattr(args, "hard_style"))]
                    if getattr(args, "boost_quality", False):
                        repair_cmd.append("--boost-quality")
                    if getattr(args, "style", ""):
                        repair_cmd += ["--style", str(getattr(args, "style"))]
                        repair_cmd += ["--style-strength", str(getattr(args, "style_strength", 0.7))]
                    if getattr(args, "control_image", ""):
                        repair_cmd += ["--control-image", str(getattr(args, "control_image"))]
                        repair_cmd += ["--control-type", str(getattr(args, "control_type", "auto"))]
                        repair_cmd += ["--control-scale", str(getattr(args, "control_scale", 0.85))]
                        repair_cmd += ["--control-guidance-start", str(getattr(args, "control_guidance_start", 0.0))]
                        repair_cmd += ["--control-guidance-end", str(getattr(args, "control_guidance_end", 1.0))]
                        repair_cmd += ["--control-guidance-decay", str(getattr(args, "control_guidance_decay", 1.0))]
                    if bool(getattr(args, "holy_grail", False)):
                        repair_cmd.append("--holy-grail")
                        repair_cmd += ["--holy-grail-cfg-early-ratio", str(getattr(args, "holy_grail_cfg_early_ratio", 0.72))]
                        repair_cmd += ["--holy-grail-cfg-late-ratio", str(getattr(args, "holy_grail_cfg_late_ratio", 1.0))]
                        repair_cmd += ["--holy-grail-control-mult", str(getattr(args, "holy_grail_control_mult", 1.0))]
                        repair_cmd += ["--holy-grail-adapter-mult", str(getattr(args, "holy_grail_adapter_mult", 1.0))]
                        if bool(getattr(args, "holy_grail_no_frontload_control", False)):
                            repair_cmd.append("--holy-grail-no-frontload-control")
                        repair_cmd += [
                            "--holy-grail-late-adapter-boost",
                            str(getattr(args, "holy_grail_late_adapter_boost", 1.15)),
                        ]
                        repair_cmd += ["--holy-grail-cads-strength", str(getattr(args, "holy_grail_cads_strength", 0.0))]
                        repair_cmd += [
                            "--holy-grail-cads-min-strength",
                            str(getattr(args, "holy_grail_cads_min_strength", 0.0)),
                        ]
                        repair_cmd += ["--holy-grail-cads-power", str(getattr(args, "holy_grail_cads_power", 1.0))]
                        repair_cmd += ["--holy-grail-unsharp-sigma", str(getattr(args, "holy_grail_unsharp_sigma", 0.0))]
                        repair_cmd += ["--holy-grail-unsharp-amount", str(getattr(args, "holy_grail_unsharp_amount", 0.0))]
                        repair_cmd += [
                            "--holy-grail-clamp-quantile",
                            str(getattr(args, "holy_grail_clamp_quantile", 0.0)),
                        ]
                        repair_cmd += ["--holy-grail-clamp-floor", str(getattr(args, "holy_grail_clamp_floor", 1.0))]
                    if getattr(args, "control", None):
                        repair_cmd += ["--control"] + [str(x) for x in getattr(args, "control", [])]
                    if getattr(args, "lora", None):
                        repair_cmd += ["--lora"] + [str(x) for x in getattr(args, "lora", [])]
                        if getattr(args, "no_lora_normalize_scales", False):
                            repair_cmd.append("--no-lora-normalize-scales")
                        repair_cmd += ["--lora-max-total-scale", str(getattr(args, "lora_max_total_scale", 1.5))]
                        if str(getattr(args, "lora_default_role", "style") or "style").strip().lower() != "style":
                            repair_cmd += ["--lora-default-role", str(getattr(args, "lora_default_role"))]
                        lrb = str(getattr(args, "lora_role_budgets", "") or "").strip()
                        if lrb:
                            repair_cmd += ["--lora-role-budgets", lrb]
                        lsp = str(getattr(args, "lora_stage_policy", "auto") or "auto").strip().lower()
                        if lsp and lsp != "auto":
                            repair_cmd += ["--lora-stage-policy", lsp]
                        lrsw = str(getattr(args, "lora_role_stage_weights", "") or "").strip()
                        if lrsw:
                            repair_cmd += ["--lora-role-stage-weights", lrsw]
                        if getattr(args, "lora_trigger", ""):
                            repair_cmd += ["--lora-trigger", str(getattr(args, "lora_trigger"))]
                    if getattr(args, "naturalize", False):
                        repair_cmd.append("--naturalize")
                        repair_cmd += ["--naturalize-grain", str(getattr(args, "naturalize_grain", 0.015))]
                    if getattr(args, "naturalize_deep", False):
                        repair_cmd.append("--naturalize-deep")
                    if getattr(args, "face_enhance", False):
                        repair_cmd.append("--face-enhance")
                        repair_cmd += ["--face-enhance-sharpen", str(getattr(args, "face_enhance_sharpen", 0.35))]
                        repair_cmd += ["--face-enhance-contrast", str(getattr(args, "face_enhance_contrast", 1.04))]
                        repair_cmd += ["--face-enhance-padding", str(getattr(args, "face_enhance_padding", 0.25))]
                        repair_cmd += ["--face-enhance-max", str(getattr(args, "face_enhance_max", 4))]
                    pri = str(getattr(args, "post_reference_image", "") or "").strip()
                    if pri:
                        repair_cmd += ["--post-reference-image", pri]
                        repair_cmd += ["--post-reference-alpha", str(float(getattr(args, "post_reference_alpha", 0.0) or 0.0))]
                    frsh = str(getattr(args, "face_restore_shell", "") or "").strip()
                    if frsh:
                        repair_cmd += ["--face-restore-shell", frsh]
                    rif = str(getattr(args, "reference_image", "") or "").strip()
                    if rif:
                        repair_cmd += ["--reference-image", rif]
                        repair_cmd += ["--reference-strength", str(float(getattr(args, "reference_strength", 1.0) or 0.0))]
                        repair_cmd += ["--reference-tokens", str(int(getattr(args, "reference_tokens", 4) or 4))]
                        repair_cmd += [
                            "--reference-clip-model",
                            str(getattr(args, "reference_clip_model", "openai/clip-vit-large-patch14")),
                        ]
                    rap = str(getattr(args, "reference_adapter_pt", "") or "").strip()
                    if rap:
                        repair_cmd += ["--reference-adapter-pt", rap]
                    if float(getattr(args, "sag_blur_sigma", 0.0) or 0.0) > 0 and float(
                        getattr(args, "sag_scale", 0.0) or 0.0
                    ) > 0:
                        repair_cmd += ["--sag-blur-sigma", str(getattr(args, "sag_blur_sigma", 0.0))]
                        repair_cmd += ["--sag-scale", str(getattr(args, "sag_scale", 0.0))]
                    if getattr(args, "less_ai", False):
                        repair_cmd.append("--less-ai")
                    if str(getattr(args, "anti_ai_pack", "none") or "none") != "none":
                        repair_cmd += ["--anti-ai-pack", str(getattr(args, "anti_ai_pack"))]
                    if str(getattr(args, "human_media_mode", "none") or "none") != "none":
                        repair_cmd += ["--human-media", str(getattr(args, "human_media_mode"))]
                    if str(getattr(args, "lora_scaffold", "none") or "none") != "none":
                        repair_cmd += ["--lora-scaffold", str(getattr(args, "lora_scaffold"))]
                    if getattr(args, "lora_scaffold_auto", False):
                        repair_cmd.append("--lora-scaffold-auto")
                    if getattr(args, "anti_bleed", False):
                        repair_cmd.append("--anti-bleed")
                    if getattr(args, "diversity", False):
                        repair_cmd.append("--diversity")
                    if getattr(args, "anti_artifacts", False):
                        repair_cmd.append("--anti-artifacts")
                    if getattr(args, "strong_watermark", False):
                        repair_cmd.append("--strong-watermark")
                    _rsm = str(getattr(args, "shortcomings_mitigation", "none") or "none").lower()
                    if _rsm in ("auto", "all"):
                        repair_cmd += ["--shortcomings-mitigation", _rsm]
                    if getattr(args, "shortcomings_2d", False):
                        repair_cmd.append("--shortcomings-2d")
                    _rag = str(getattr(args, "art_guidance_mode", "none") or "none").lower()
                    if _rag in ("auto", "all"):
                        repair_cmd += ["--art-guidance-mode", _rag]
                    if getattr(args, "no_art_guidance_photography", False):
                        repair_cmd.append("--no-art-guidance-photography")
                    _anat = str(getattr(args, "anatomy_guidance", "none") or "none").lower()
                    if _anat in ("lite", "strong"):
                        repair_cmd += ["--anatomy-guidance", _anat]
                    _sg = str(getattr(args, "style_guidance_mode", "none") or "none").lower()
                    if _sg in ("auto", "all"):
                        repair_cmd += ["--style-guidance-mode", _sg]
                    if not bool(getattr(args, "style_guidance_artists", True)):
                        repair_cmd.append("--no-style-guidance-artists")
                    if not getattr(args, "auto_content_fix", True):
                        repair_cmd.append("--no-auto-content-fix")
                    if not getattr(args, "one_shot_boost", True):
                        repair_cmd.append("--no-one-shot-boost")

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
            cmd = frs.replace("{src}", str(out_path)).replace("{dst}", str(out_path))
            subprocess.run(cmd, shell=True, check=False)
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
    main()
