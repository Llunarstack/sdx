"""Helpers for sample.py generation path (extracted from sample.py)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.runtime.jsonutil import loads as json_loads


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


def _parse_scale_csv(value: str) -> list[Any]:
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


def _parse_lora_role_budgets(raw: str) -> dict[str, Any]:
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


def _parse_lora_role_stage_weights(raw: str) -> dict[str, Any]:
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


def _parse_lora_spec(spec: str, *, default_role: str = "style") -> tuple[str, float, str]:
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


def _parse_weighted_style_mix(raw: str) -> list[Any]:
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
) -> tuple[str, str, float]:
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


def _build_size_tokens(anatomy_scales: list[Any], object_scales: list[Any], scene_scales: list[Any]) -> str:
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
    tokens: list[Any] = []
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


def _parse_expected_texts(raw: str) -> list[Any]:
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


def _infer_expected_texts_from_prompt(prompt: str) -> list[Any]:
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


def _maybe_append_text_says(prompt: str, expected_texts: list[Any]) -> str:
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
    expected_texts: list[Any],
) -> tuple[float, dict[str, Any]]:
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


def _normalize_list_or_str(v) -> list[Any]:
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
    tokens: list[Any], negative_tokens: list[Any], *, uncensored_mode: bool = False
) -> tuple[list[Any], list[Any]]:
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
) -> tuple[str, str]:
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


def _apply_character_gender_presentation(tokens: list[Any], gender_presentation: str) -> list[Any]:
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
