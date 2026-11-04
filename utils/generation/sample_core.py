"""``sample.py`` models and conditioning: spec parsing, checkpoint / encoder loading, Krea controls, the quality policy, text embeddings and init latents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusion import create_diffusion
from diffusion.sampling import dit_rgb_stride_px, sampling_canvas_hw
from diffusion.schedules import sd3_resolution_shift, time_shift, vp_shifted_timestep
from PIL import Image

from utils.modeling.encoders import encode_text
from utils.runtime import json_loads
from utils.safety.minor_guard import is_sexual, require_allowed, require_image_allowed

# Child runs (auto-refine candidates, OCR repair) go through the entry script, which sets up sys.path.
SAMPLE_PY = Path(__file__).resolve().parents[2] / "sample.py"


def _maybe_rae_to_dit(z: torch.Tensor, ae_type: str, rae_bridge) -> torch.Tensor:
    """Map RAE latent (B,C,h,w) to DiT 4-channel space when checkpoint includes RAELatentBridge."""
    if z is None or ae_type != "rae" or rae_bridge is None:
        return z
    if z.shape[1] == 4:
        return z
    return rae_bridge.rae_to_dit(z)


def load_model_from_ckpt(ckpt_path, device="cuda"):
    from utils.modeling.checkpoints import load_sampler_checkpoint

    return load_sampler_checkpoint(ckpt_path, device=device, reject_enhanced=True, verbose=True)


# Per-process text-embedding cache: (prompt, negative, style, layout, t5 hash) -> (cond, uncond, style_emb).
_t5_cache = {}
_T5_CACHE_MAX = 32
# T5 sequence length every prompt is padded/truncated to (CLIP pooled captions stop at 77).
TEXT_MAX_TOKENS = 300


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


def renoise_for_grid(
    diffusion,
    x0: torch.Tensor,
    latent_h: int,
    latent_w: int,
    strength: float,
    *,
    native_latent_hw: tuple[int, int],
    flow: bool,
) -> tuple[torch.Tensor, int]:
    """Bicubic-upscale a clean latent to ``latent_h x latent_w`` and forward-noise it; returns ``(x_t, t_start)``.

    ``strength`` (0-1) is the noise level at the native grid. A larger grid starts from its SD3-shifted
    equivalent (Esser et al. 2024: ``alpha = sqrt(tokens / native tokens)``; flow ``time_shift``, VP
    ``vp_shifted_timestep``), because the same noise level destroys less structure on more pixels.
    """
    x_up = torch.nn.functional.interpolate(x0.float(), size=(latent_h, latent_w), mode="bicubic", align_corners=False)
    x_up = x_up.to(dtype=x0.dtype)
    a = sd3_resolution_shift(latent_h * latent_w, native_latent_hw[0] * native_latent_hw[1])
    t_last = int(diffusion.num_timesteps) - 1
    noise = torch.randn_like(x_up)
    if flow:
        t0 = min(max(1, round(float(time_shift(float(strength), a)) * t_last)), t_last)
        return (1.0 - t0 / t_last) * x_up + (t0 / t_last) * noise, t0
    ac = diffusion.alpha_cumprod.detach().cpu().numpy()
    t0 = min(max(1, vp_shifted_timestep(ac, round(float(strength) * t_last), a)), t_last)
    t_batch = torch.full((x_up.shape[0],), t0, device=x_up.device, dtype=torch.long)
    return diffusion.q_sample(x_up, t_batch, noise=noise), t0


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
    if not dedup:  # curly / single quotes after a text cue, unquoted capitals ("a sign that says OPEN")
        from utils.prompt.realism import extract_expected_texts

        dedup = extract_expected_texts(p)
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

    from utils.consistency import build_character_prompt_additions

    return build_character_prompt_additions(
        data,
        uncensored_mode=uncensored_mode,
        character_strength=character_strength,
    )


class CreativityMode(str, Enum):
    RAW = "raw"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class StyleReference:
    path: str
    strength: float = 1.0


@dataclass(slots=True)
class GenerativeSliderPlan:
    intensity: float = 0.0
    complexity: float = 0.0
    movement: float = 0.0
    creativity_mode: CreativityMode = CreativityMode.MEDIUM
    prompt_additions: str = ""
    negative_additions: str = ""
    cfg_multiplier: float = 1.0
    reference_strength_multiplier: float = 1.0
    serendipity_offset: float = 0.0
    trace: list[str] = field(default_factory=list)


def _clamp_slider(v: float) -> float:
    return max(-100.0, min(100.0, float(v)))


def _norm_weights(strengths: Sequence[float]) -> list[float]:
    ws = [max(0.0, float(s)) for s in strengths]
    total = sum(ws)
    if total <= 0.0:
        n = max(1, len(ws))
        return [1.0 / n] * n
    return [w / total for w in ws]


def parse_style_references(
    *,
    json_path: str = "",
    csv_spec: str = "",
) -> list[StyleReference]:
    """
    Parse style refs from JSON (``{"references": [{"path", "strength"}]}``)
    or CSV ``path:weight,path2:weight`` (weight defaults to 1.0).
    """
    out: list[StyleReference] = []
    jp = (json_path or "").strip()
    if jp:
        data = json.loads(Path(jp).read_text(encoding="utf-8"))
        rows = data.get("references", data.get("style_references", data.get("images", [])))
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, str):
                    out.append(StyleReference(path=row.strip(), strength=1.0))
                elif isinstance(row, Mapping):
                    p = str(row.get("path", row.get("image", row.get("url", ""))) or "").strip()
                    if p and not p.startswith("http"):
                        out.append(
                            StyleReference(
                                path=p,
                                strength=float(row.get("strength", row.get("weight", 1.0)) or 1.0),
                            )
                        )
    spec = (csv_spec or "").strip()
    if spec:
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                path, wt = chunk.rsplit(":", 1)
                out.append(StyleReference(path=path.strip(), strength=float(wt.strip() or 1.0)))
            else:
                out.append(StyleReference(path=chunk, strength=1.0))
    return out


def load_moodboard_paths(
    *,
    json_path: str = "",
    csv_paths: str = "",
) -> list[str]:
    """Load moodboard image paths from JSON or comma-separated list."""
    paths: list[str] = []
    jp = (json_path or "").strip()
    if jp:
        data = json.loads(Path(jp).read_text(encoding="utf-8"))
        rows = data.get("images", data.get("paths", data.get("moodboard", [])))
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, str):
                    paths.append(row.strip())
                elif isinstance(row, Mapping):
                    p = str(row.get("path", row.get("image", "")) or "").strip()
                    if p:
                        paths.append(p)
    spec = (csv_paths or "").strip()
    if spec:
        paths.extend(p.strip() for p in spec.split(",") if p.strip())
    return paths


@lru_cache(maxsize=1)
def _clip_vision(model_id: str, device: Any) -> tuple[Any, Any]:
    import transformers as tr

    proc = tr.CLIPImageProcessor.from_pretrained(model_id)
    model = tr.CLIPVisionModelWithProjection.from_pretrained(model_id).eval().requires_grad_(False)
    return model.to(device), proc


def encode_reference_image_pil(pil_image: Any, *, device: Any, model_id: str, dtype: Any) -> tuple[Any, int]:
    """CLIP vision ``image_embeds`` ``(1, D)`` for a reference image, and ``D``."""
    import torch

    model, proc = _clip_vision(model_id, device)
    inputs = {k: v.to(device) for k, v in proc(images=pil_image, return_tensors="pt").items()}
    with torch.no_grad():
        emb = model(**inputs).image_embeds
    return emb.to(dtype=dtype), int(emb.shape[-1])


def aggregate_style_embeddings(
    refs: Sequence[StyleReference],
    *,
    device: Any,
    model_id: str,
    dtype: Any,
) -> tuple[Any, int]:
    """Weighted-mean CLIP image embedding from multiple style references."""
    import torch
    from PIL import Image

    if not refs:
        raise ValueError("aggregate_style_embeddings requires at least one reference")

    embeds: list[Any] = []
    weights: list[float] = []
    for ref in refs:
        p = Path(ref.path)
        if not p.is_file():
            raise FileNotFoundError(f"style reference not found: {p}")
        emb, dim = encode_reference_image_pil(
            Image.open(p).convert("RGB"),
            device=device,
            model_id=model_id,
            dtype=dtype,
        )
        embeds.append(emb)
        weights.append(ref.strength)

    ws = _norm_weights(weights)
    stacked = torch.stack([e.squeeze(0) for e in embeds], dim=0)
    w = torch.tensor(ws, device=stacked.device, dtype=stacked.dtype).view(-1, 1)
    blended = (stacked * w).sum(dim=0, keepdim=True)
    return blended, int(stacked.shape[-1])


def build_generative_slider_plan(
    *,
    intensity: float = 0.0,
    complexity: float = 0.0,
    movement: float = 0.0,
    creativity_mode: str = "medium",
) -> GenerativeSliderPlan:
    """Map Krea-style sliders (−100…100) to prompt fragments and sampling knobs."""
    plan = GenerativeSliderPlan(
        intensity=_clamp_slider(intensity),
        complexity=_clamp_slider(complexity),
        movement=_clamp_slider(movement),
        creativity_mode=CreativityMode(str(creativity_mode or "medium").lower()),
    )
    pos: list[str] = []
    neg: list[str] = []

    if plan.intensity < -25:
        pos.append("muted palette, soft contrast, understated styling")
        plan.cfg_multiplier *= 0.92
        plan.trace.append("slider:intensity_low")
    elif plan.intensity > 25:
        pos.append("bold stylization, strong contrast, vivid color grading")
        plan.cfg_multiplier *= 1.06
        plan.reference_strength_multiplier *= 1.08
        plan.trace.append("slider:intensity_high")

    if plan.complexity < -25:
        pos.append("minimal composition, clean negative space, simple forms")
        neg.append("cluttered background, busy detail overload")
        plan.trace.append("slider:complexity_low")
    elif plan.complexity > 25:
        pos.append("rich environmental detail, layered textures, dense visual interest")
        plan.trace.append("slider:complexity_high")

    if plan.movement < -25:
        pos.append("static pose, stable camera, calm composition")
        neg.append("motion blur, dynamic action blur")
        plan.trace.append("slider:movement_low")
    elif plan.movement > 25:
        pos.append("dynamic pose, energetic motion, dramatic camera angle, kinetic composition")
        plan.serendipity_offset += 0.05
        plan.trace.append("slider:movement_high")

    plan.prompt_additions = ", ".join(pos)
    plan.negative_additions = ", ".join(neg)
    return plan


def apply_generative_sliders(args: Any) -> GenerativeSliderPlan:
    """Read slider args from ``args`` namespace; mutate prompt/negative/cfg in place."""
    mode = str(getattr(args, "creativity_mode", "") or "medium").lower()
    plan = build_generative_slider_plan(
        intensity=float(getattr(args, "slider_intensity", 0) or 0),
        complexity=float(getattr(args, "slider_complexity", 0) or 0),
        movement=float(getattr(args, "slider_movement", 0) or 0),
        creativity_mode=mode,
    )
    if plan.prompt_additions:
        base = (getattr(args, "prompt", "") or "").strip()
        args.prompt = f"{base}, {plan.prompt_additions}" if base else plan.prompt_additions
    if plan.negative_additions:
        base_neg = (getattr(args, "negative_prompt", "") or "").strip()
        args.negative_prompt = f"{base_neg}, {plan.negative_additions}" if base_neg else plan.negative_additions
    if plan.cfg_multiplier != 1.0 and getattr(args, "cfg_scale", None):
        args.cfg_scale = float(args.cfg_scale) * plan.cfg_multiplier
    rs = float(getattr(args, "reference_strength", 1.0) or 1.0)
    if plan.reference_strength_multiplier != 1.0 and rs > 0:
        args.reference_strength = rs * plan.reference_strength_multiplier
    if plan.serendipity_offset and getattr(args, "frontier_serendipity", None) is not None:
        args.frontier_serendipity = min(1.0, float(args.frontier_serendipity) + plan.serendipity_offset)
    args._generative_slider_plan = plan
    return plan


def apply_creativity_mode_to_prompt(args: Any) -> CreativityMode:
    """Expand or preserve prompt based on Krea-style creativity mode."""
    raw = str(getattr(args, "creativity_mode", "") or "medium").lower().strip()
    try:
        mode = CreativityMode(raw)
    except ValueError:
        mode = CreativityMode.MEDIUM

    prompt = (getattr(args, "prompt", "") or "").strip()
    if not prompt:
        args._creativity_mode = mode
        return mode

    word_count = len(re.findall(r"\w+", prompt))
    short = word_count < 10

    if mode == CreativityMode.RAW:
        args._skip_prompt_expansion = True
    elif mode == CreativityMode.LOW and short:
        low = prompt.lower()
        if "lighting" not in low:
            args.prompt = f"{prompt}, soft natural lighting"
    elif mode == CreativityMode.MEDIUM and short:
        from utils.agentic.quality_loop import expand_prompt_heuristic

        args.prompt = expand_prompt_heuristic(prompt)
    elif mode == CreativityMode.HIGH:
        from utils.agentic.quality_loop import expand_prompt_heuristic

        expanded = expand_prompt_heuristic(prompt)
        if mode == CreativityMode.HIGH and word_count < 16:
            expanded = f"{expanded}, expressive art direction, cinematic mood, rich color palette"
        args.prompt = expanded
        if getattr(args, "frontier_serendipity", None) is not None:
            args.frontier_serendipity = min(1.0, float(args.frontier_serendipity) + 0.08)
        elif getattr(args, "novelty", None) is not None:
            args.novelty = min(1.0, float(args.novelty) + 0.1)

    args._creativity_mode = mode
    return mode


def apply_turbo_preset(args: Any) -> None:
    """Few-step sampling profile inspired by Krea 2 Turbo (8 steps, low CFG)."""
    if int(getattr(args, "steps", 0) or 0) in (0, 50, 28):
        args.steps = 8
    args.cfg_scale = float(getattr(args, "cfg_scale", 4.5) or 4.5)
    if args.cfg_scale > 2.0:
        args.cfg_scale = 1.0
    args._krea_turbo_preset = True


def apply_krea_controls_to_args(args: Any) -> dict[str, Any]:
    """
    Prompt-time Krea controls: turbo preset, creativity mode, generative sliders.

    Returns metadata dict (empty if nothing applied).
    """
    meta: dict[str, Any] = {}
    if getattr(args, "krea_turbo_preset", False):
        apply_turbo_preset(args)
        meta["turbo_preset"] = True
    if str(getattr(args, "creativity_mode", "") or "").strip():
        meta["creativity_mode"] = apply_creativity_mode_to_prompt(args).value
    if any(
        abs(float(getattr(args, k, 0) or 0)) > 1e-6
        for k in ("slider_intensity", "slider_complexity", "slider_movement")
    ):
        meta["generative_sliders"] = apply_generative_sliders(args)
    return meta


def _collect_style_refs(args: Any) -> list[StyleReference]:
    refs = parse_style_references(
        json_path=str(getattr(args, "style_references_json", "") or ""),
        csv_spec=str(getattr(args, "style_ref", "") or ""),
    )
    mood_paths = load_moodboard_paths(
        json_path=str(getattr(args, "moodboard_json", "") or ""),
        csv_paths=str(getattr(args, "moodboard_images", "") or ""),
    )
    if mood_paths:
        mb_strength = float(getattr(args, "moodboard_strength", 1.0) or 1.0)
        for p in mood_paths:
            refs.append(StyleReference(path=p, strength=mb_strength))
    single = str(getattr(args, "reference_image", "") or "").strip()
    if single and not refs:
        refs.append(StyleReference(path=single, strength=float(getattr(args, "reference_strength", 1.0) or 1.0)))
    return refs


def inject_reference_conditioning(
    model_kwargs_cond: dict[str, Any],
    *,
    args: Any,
    cfg: Any,
    device: Any,
    num_gen: int,
) -> None:
    """Blend style/moodboard refs → ``reference_tokens`` on ``model_kwargs_cond``."""
    import torch
    from PIL import Image

    refs = _collect_style_refs(args)
    if not refs:
        return

    global_scale = float(getattr(args, "reference_strength", 1.0) or 1.0)
    if len(refs) == 1:
        ref_strength = refs[0].strength * global_scale
    else:
        ref_strength = global_scale

    if ref_strength <= 0:
        return

    from models.conditioning import ReferenceTokenProjector

    clip_id = str(getattr(args, "reference_clip_model", "") or "openai/clip-vit-large-patch14")

    if len(refs) == 1:
        pil_r = Image.open(refs[0].path).convert("RGB")
        emb, clip_dim = encode_reference_image_pil(pil_r, device=device, model_id=clip_id, dtype=torch.float32)
    else:
        emb, clip_dim = aggregate_style_embeddings(refs, device=device, model_id=clip_id, dtype=torch.float32)

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


_COLOUR_WORD = re.compile(
    r"\b(?:red|orange|yellow|green|blue|purple|violet|pink|teal|gold|silver|black|white|brown|gr[ae]y)\b", re.I
)


def apply_best_model_policy(args: Any, *, prompt: str | None = None) -> dict[str, Any]:
    """
    Soft-fill sampling args from compete-mode ``QualityPolicy``.

    Always-on unless ``args.no_quality_defaults`` or ``POLICY.enabled`` is False.
    Explicit CLI values always win (only unset fields are filled). The holy-grail preset it picks is
    expanded by the caller; it never picks ``--preset`` / ``--op-mode``, whose prompt packs are medium-blind.
    """
    from config.presets import POLICY, recipe_for_prompt, soft_set

    meta: dict[str, Any] = {"profile": None, "applied": [], "aggression": POLICY.aggression}
    if not POLICY.enabled or bool(getattr(args, "no_quality_defaults", False)):
        meta["skipped"] = True
        return meta

    text = prompt if prompt is not None else str(getattr(args, "prompt", "") or "")
    style = str(getattr(args, "style", "") or "")
    recipe = recipe_for_prompt(text, style=style)
    meta["profile"] = recipe.name

    def _note(key: str, ok: bool) -> None:
        if ok:
            meta["applied"].append(key)

    if POLICY.enable_holy_grail and recipe.holy_grail:
        _note("holy_grail", soft_set(args, "holy_grail", True, unset=(False,)))
        _note(
            "holy_grail_preset",
            soft_set(args, "holy_grail_preset", recipe.holy_grail_preset, unset=(None, "")),
        )

    if recipe.cfg_scale is not None:
        _note("cfg_scale", soft_set(args, "cfg_scale", recipe.cfg_scale, unset=(0.0, 7.5, None)))
    _note("cfg_rescale", soft_set(args, "cfg_rescale", recipe.cfg_rescale, unset=(0.0, None)))
    if recipe.apg_parallel_eta is not None:
        _note("apg", soft_set(args, "apg_parallel_eta", recipe.apg_parallel_eta, unset=(-1.0, None)))
    if recipe.apg_momentum_beta is not None:
        _note(
            "apg_mom",
            soft_set(args, "apg_momentum_beta", recipe.apg_momentum_beta, unset=(0.0, None)),
        )
    if recipe.fdg_low is not None:
        _note("fdg_low", soft_set(args, "fdg_low", recipe.fdg_low, unset=(1.0, None)))
    if recipe.fdg_high is not None:
        _note("fdg_high", soft_set(args, "fdg_high", recipe.fdg_high, unset=(1.0, None)))
    if recipe.qsilk_micrograin is not None:
        _note(
            "qsilk",
            soft_set(args, "qsilk_micrograin", recipe.qsilk_micrograin, unset=(0.0, None)),
        )
    if recipe.steps is not None:
        _note("steps", soft_set(args, "steps", recipe.steps, unset=(0, 50, None)))

    # Photo-capture packs (skin pores, DSLR, "authentic photograph") only where a photograph is possible:
    # a named painting / anime / 3-D medium keeps its own look.
    from utils.prompt.art_direction import plan_art_direction

    ad = getattr(args, "_art_direction", None) or plan_art_direction(text)
    photo_ok = ad.medium_family not in ("2d", "traditional", "3d")
    if photo_ok:
        _note(
            "human_made",
            soft_set(args, "human_made", recipe.human_made, unset=("none", "off", "0", "", None)),
        )
    if recipe.less_ai and photo_ok:
        _note("less_ai", soft_set(args, "less_ai", True, unset=(False,)))
    if recipe.naturalize and photo_ok:
        _note("naturalize", soft_set(args, "naturalize", True, unset=(False,)))
    if recipe.naturalize_deep and photo_ok:
        _note("naturalize_deep", soft_set(args, "naturalize_deep", True, unset=(False,)))
    if photo_ok:
        _note(
            "anti_ai_pack",
            soft_set(args, "anti_ai_pack", recipe.anti_ai_pack, unset=("none", "", None)),
        )
    if recipe.human_media and photo_ok:
        _note(
            "human_media",
            soft_set(args, "human_media_mode", recipe.human_media, unset=("none", "", None)),
        )
    _note(
        "shortcomings",
        soft_set(
            args,
            "shortcomings_mitigation",
            recipe.shortcomings_mitigation,
            unset=("none", "", None),
        ),
    )

    anat = recipe.anatomy_guidance
    if anat == "auto":
        pl = text.lower()
        anat = (
            "strong"
            if any(k in pl for k in ("hand", "hands", "finger", "full body", "nude", "anatomy", "person", "people"))
            else "lite"
        )
    _note(
        "anatomy_guidance",
        soft_set(args, "anatomy_guidance", anat, unset=("none", "auto", "", None)),
    )
    if recipe.hand_mode and ad.has_people:
        _note(
            "hand_mode",
            soft_set(args, "hand_mode", recipe.hand_mode, unset=("none", "off", "", None)),
        )
    if recipe.pose_naturalness and ad.has_people:
        _note(
            "pose",
            soft_set(args, "pose_naturalness", recipe.pose_naturalness, unset=("none", "", None)),
        )
    if recipe.typography_mode:
        _note(
            "typography",
            soft_set(args, "typography_mode", recipe.typography_mode, unset=("none", "", None)),
        )

    if recipe.expand_prompt:
        _note("expand_prompt", soft_set(args, "expand_prompt", True, unset=(False,)))

    if recipe.photo_realism_prefer and ad.photographic:
        _note(
            "photo_pack",
            soft_set(
                args,
                "photo_realism_pack",
                recipe.photo_realism_prefer,
                unset=("none", "", None),
            ),
        )

    # Text-in-image + OCR auto
    needs_text = recipe.text_in_image or (
        POLICY.enable_ocr_auto and ('"' in text or "“" in text or "[text:" in text.lower() or "says" in text.lower())
    )
    if needs_text:
        _note("text_in_image", soft_set(args, "text_in_image", True, unset=(False,)))
        if recipe.ocr_fix or POLICY.enable_ocr_auto:
            _note("ocr_fix", soft_set(args, "ocr_fix", True, unset=(False,)))

    # Multi-candidate pick (compete mode)
    if POLICY.enable_pick_best and recipe.pick_best and POLICY.aggression == "max":
        if soft_set(args, "pick_best", recipe.pick_best, unset=("none", "", None)):
            _note("pick_best", True)
        if recipe.num is not None and soft_set(args, "num", int(recipe.num), unset=(1, None)):
            _note("num", True)

    # The subject planner pre-empts --frontier / --frontier-perfect / --frontier-creative, so it only
    # fills in when the user asked for none of them.
    frontier_asked = any(bool(getattr(args, f, False)) for f in ("frontier", "frontier_perfect", "frontier_creative"))
    if POLICY.enable_frontier_subject and recipe.frontier_subject and not frontier_asked:
        _note("frontier_subject", soft_set(args, "frontier_subject", True, unset=(False,)))
    if recipe.superior_self_correct:
        _note(
            "self_correct",
            soft_set(args, "superior_self_correct", True, unset=(False,)),
        )
    # The diversity pack is all faces, and the colour-separation pack only helps when two named colours
    # can bleed into each other; elsewhere they take the token budget from the scene.
    if recipe.diversity and ad.has_people:
        _note("diversity", soft_set(args, "diversity", True, unset=(False,)))

    if recipe.anti_slop and photo_ok:
        from frontier.planning import AntiSlopScanner

        # Only prompts that already ask for photorealism get the photo-tell packs.
        plan = AntiSlopScanner().plan(text)
        if plan.positive:
            cur = str(getattr(args, "prompt", "") or "")
            if plan.positive.lower() not in cur.lower():
                args.prompt = f"{cur}, {plan.positive}" if cur.strip() else plan.positive
                meta["applied"].append("anti_slop_pos")
        if plan.negative:
            base_neg = str(getattr(args, "negative_prompt", "") or "")
            if plan.negative.lower() not in base_neg.lower():
                args.negative_prompt = f"{base_neg}, {plan.negative}".strip(", ").strip()
                meta["applied"].append("anti_slop_neg")

    # Guidance interval + hard-token emphasis (compete mode)
    if POLICY.enable_cfg_interval and recipe.cfg_interval:
        _note("cfg_interval", soft_set(args, "cfg_interval", recipe.cfg_interval, unset=("", None)))
    if POLICY.enable_token_emphasis and recipe.token_emphasis:
        if not bool(getattr(args, "frontier_perfect", False)):
            from utils.generation.guidance import apply_token_emphasis_to_args

            te = apply_token_emphasis_to_args(args)
            if "weights" in te:
                meta["applied"].append("token_emphasis")

    bleed_risk = len({m.group(0).lower() for m in _COLOUR_WORD.finditer(text)}) >= 2
    for flag in ("anti_bleed", "anti_artifacts", "strong_watermark"):
        want = getattr(recipe, flag, True) and (flag != "anti_bleed" or bleed_risk)
        if want and soft_set(args, flag, True, unset=(False,)):
            meta["applied"].append(flag)

    if POLICY.enable_box_attn:
        for flag in ("box_attn_layout", "per_region_cads"):
            if not hasattr(args, flag):
                setattr(args, flag, False)
            _note(flag, soft_set(args, flag, True, unset=(False,)))
    return meta


def _as_offset_pairs(raw: Any, seq_len: int) -> list[tuple[int, int]] | None:
    """Normalize tokenizer ``offset_mapping`` to ``[(start, end), ...]`` (unbatched)."""
    if raw is None:
        return None
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().tolist()
    elif hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not raw:
        return None
    first = raw[0]
    # Batched: [[(s, e), ...]] or [[[s, e], ...]]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        raw = first
    pairs: list[tuple[int, int]] = []
    for i, item in enumerate(raw):
        if i >= seq_len:
            break
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        a, b = item[0], item[1]
        if a is None or b is None:
            continue
        pairs.append((int(a), int(b)))
    return pairs or None


def _char_proportion_span(char_start: int, char_end: int, seq_len: int, prompt_len: int) -> slice:
    denom = max(int(prompt_len), 1)
    start = int(char_start * seq_len / denom)
    end = int(char_end * seq_len / denom)
    start = max(0, min(seq_len, start))
    end = max(0, min(seq_len, end))
    if end <= start:
        end = min(seq_len, start + 1)
    if start >= seq_len:
        start = max(0, seq_len - 1)
        end = seq_len
    return slice(start, end)


def _token_spans_for_words(prompt: str, seq_len: int, tokenizer: Any = None) -> list[slice]:
    """Map whitespace words to embedding-token slices (tokenizer offsets, else char proportion)."""
    seq_len = int(seq_len)
    if seq_len <= 0:
        return []
    matches = list(re.finditer(r"\S+", prompt or ""))
    if not matches:
        return []
    prompt_len = max(len(prompt or ""), 1)

    def _fallback(cs: int, ce: int) -> slice:
        return _char_proportion_span(cs, ce, seq_len, prompt_len)

    offsets: list[tuple[int, int]] | None = None
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                prompt,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=seq_len,
            )
            raw = None
            if encoded is not None:
                if hasattr(encoded, "get"):
                    raw = encoded.get("offset_mapping")
                if raw is None:
                    raw = getattr(encoded, "offset_mapping", None)
            offsets = _as_offset_pairs(raw, seq_len)
        except Exception:
            offsets = None

    if not offsets:
        return [_fallback(m.start(), m.end()) for m in matches]

    spans: list[slice] = []
    for m in matches:
        ws, we = m.start(), m.end()
        hit = [i for i, (ts, te) in enumerate(offsets) if te > ts and ts < we and te > ws]
        if not hit:
            spans.append(_fallback(ws, we))
            continue
        start = max(0, min(hit))
        end = min(seq_len, max(hit) + 1)
        if end <= start:
            spans.append(_fallback(ws, we))
        else:
            spans.append(slice(start, end))
    return spans


def modulate_text_for_adherence(
    enc: torch.Tensor,
    prompt: str,
    *,
    negation_scale: float = 0.35,
    binding_boost: float = 0.08,
    tokenizer: Any = None,
) -> torch.Tensor:
    """Training-free text-embedding modulation from ``PromptParser``."""
    if enc is None or enc.ndim != 3 or not (prompt or "").strip():
        return enc
    from models.conditioning import PromptParser

    parsed = PromptParser().parse(prompt)
    out = enc.clone()
    _b, length, _d = out.shape
    spans = _token_spans_for_words(prompt, length, tokenizer)
    n_words = len(spans)
    if n_words == 0:
        return out

    damp = float(max(0.0, min(1.0, negation_scale)))
    if damp > 0.0:
        scale = 1.0 - damp
        for wi in getattr(parsed, "negation_indices", []) or []:
            for j in range(int(wi) + 1, min(int(wi) + 4, n_words)):
                sl = spans[j]
                if sl.start < sl.stop:
                    out[:, sl, :] = out[:, sl, :] * scale

    boost = float(max(0.0, min(0.5, binding_boost)))
    if boost > 0.0:
        gain = 1.0 + boost
        for triple in getattr(parsed, "triples", []) or []:
            for idx in getattr(triple, "token_indices", []) or []:
                ii = int(idx)
                if 0 <= ii < n_words:
                    sl = spans[ii]
                    if sl.start < sl.stop:
                        out[:, sl, :] = out[:, sl, :] * gain
        spatial = {
            "left",
            "right",
            "behind",
            "between",
            "beside",
            "holding",
            "under",
            "reflection",
            "exactly",
        }
        for i, match in enumerate(re.finditer(r"\S+", prompt or "")):
            if i >= n_words:
                break
            token = match.group(0).strip(".,;:").lower()
            if token in spatial:
                sl = spans[i]
                if sl.start < sl.stop:
                    out[:, sl, :] = out[:, sl, :] * (1.0 + 0.5 * boost)
    return out


@dataclass
class Models:
    """Everything loaded for one run, plus the latent grid it samples on."""

    device: torch.device
    model: Any
    cfg: Any
    rae_bridge: Any
    tokenizer: Any
    text_encoder: Any
    text_bundle: Any
    vae: Any
    ae_type: str
    latent_scale: float
    image_size: int
    req_w: int
    req_h: int
    canvas_h: int
    canvas_w: int
    diffusion: Any

    @property
    def latent_channels(self) -> int:
        """Channels the DiT denoises: 4 for SD-style VAEs, 16 for FLUX / SD3 / REPA-E ones."""
        return int(getattr(self.model, "in_channels", 4))

    @property
    def latent_size(self) -> int:
        return self.image_size // 8

    @property
    def lat_h(self) -> int:
        return self.canvas_h // 8

    @property
    def lat_w(self) -> int:
        return self.canvas_w // 8

    def to_vae_space(self, x0: torch.Tensor) -> torch.Tensor:
        """DiT latent -> autoencoder latent (KL scale, RAE bridge)."""
        if self.ae_type == "kl":
            return x0 / self.latent_scale
        if self.ae_type == "rae" and self.rae_bridge is not None:
            return self.rae_bridge.dit_to_rae(x0)
        return x0

    def decode_unit(self, x0: torch.Tensor) -> torch.Tensor:
        """Decode DiT latents to RGB in ``[0, 1]``."""
        return (self.vae.decode(self.to_vae_space(x0)).sample * 0.5 + 0.5).clamp(0, 1)


def lcm_should_enable_flow_sample(dit_cfg, lcm_cfg) -> bool:
    """An LCM overlay needs the flow sampler only if the base DiT or the overlay was flow-trained."""
    return any(bool(getattr(c, "flow_matching_training", False)) for c in (dit_cfg, lcm_cfg))


def _seed_and_backends(args: Any) -> torch.device:
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
        from utils.training.perf import configure_inference_cuda

        configure_inference_cuda(cudnn_benchmark=True, enable_tf32=True)
    return device


def _apply_lcm(args: Any, model: Any, cfg: Any, device: torch.device) -> None:
    """``--lcm-ckpt``: overlay few-step weights and drop to ``--lcm-steps``."""
    lcm_ckpt = str(getattr(args, "lcm_ckpt", "") or "").strip()
    if not lcm_ckpt:
        return
    from utils.modeling.checkpoints import load_dit_text_checkpoint

    lcm_model, lcm_cfg, _, _, _ = load_dit_text_checkpoint(lcm_ckpt, device=str(device), reject_enhanced=True)
    model.load_state_dict(lcm_model.state_dict(), strict=False)
    if int(args.steps) >= 16:
        args.steps = max(2, int(getattr(args, "lcm_steps", 4) or 4))
    if lcm_should_enable_flow_sample(cfg, lcm_cfg) and not getattr(args, "force_vp_sample", False):
        args.flow_matching_sample = True
    sampler = "flow" if getattr(args, "flow_matching_sample", False) else "VP"
    print(f"LCM few-step mode: loaded {lcm_ckpt}, steps={args.steps}, {sampler} sampler.", file=sys.stderr)


def _apply_loras(args: Any, model: Any) -> None:
    """LoRA bank auto-adapters, then every ``--lora path[:scale[:role]]`` with role budgets."""
    if not getattr(args, "no_lora_bank", False):
        try:
            from utils.modeling.checkpoints import augment_sample_lora_args, default_bank_index_path

            if getattr(args, "lora_bank", False) or default_bank_index_path().is_file():
                resolved = augment_sample_lora_args(args)
                if resolved:
                    print(f"LoRA bank: auto adapters for {', '.join(resolved)}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: LoRA bank resolve failed: {e}", file=sys.stderr)
    if not args.lora:
        return
    from models.lora import apply_loras

    default_role = str(getattr(args, "lora_default_role", "style") or "style").strip().lower()
    specs, role_counts = [], {}
    for spec in args.lora:
        path, scale, role = _parse_lora_spec(spec, default_role=default_role)
        role = (role or default_role).strip().lower()
        specs.append((path.strip(), float(scale), role))
        role_counts[role] = role_counts.get(role, 0) + 1
    stage_policy = str(getattr(args, "lora_stage_policy", "auto") or "auto")
    normalize = not bool(getattr(args, "no_lora_normalize_scales", False))
    max_total = float(getattr(args, "lora_max_total_scale", 1.5))
    _, num_keys = apply_loras(
        model,
        specs,
        normalize_scales=normalize,
        max_total_scale=max_total,
        role_budgets=_parse_lora_role_budgets(getattr(args, "lora_role_budgets", "")),
        stage_policy=stage_policy,
        layer_group=str(getattr(args, "lora_layers", "all") or "all"),
        role_stage_weights=_parse_lora_role_stage_weights(getattr(args, "lora_role_stage_weights", "")),
    )
    role_mix = ", ".join(f"{k}:{v}" for k, v in sorted(role_counts.items()))
    print(
        f"Applied {len(specs)} adapter(s) (LoRA/DoRA/LyCORIS), {num_keys} layer(s), normalize={normalize}, "
        f"max_total={max_total:.2f}, stage_policy={stage_policy}, roles=[{role_mix}]"
    )


def _load_text_encoders(cfg: Any, fusion_sd: Any, device: torch.device) -> tuple[Any, Any, Any]:
    """(tokenizer, text_encoder, text_bundle); the bundle is set for triple / penta encoder checkpoints."""
    mode = str(getattr(cfg, "text_encoder_mode", "t5") or "t5").lower()
    if mode in ("triple", "penta"):
        from utils.modeling.encoders import attach_fusion_weights, load_text_encoder_bundle

        bundle = load_text_encoder_bundle(cfg, device)
        if bundle is None:
            raise RuntimeError(f"Checkpoint config requests {mode} text encoders but bundle failed to load.")
        if fusion_sd is not None:
            attach_fusion_weights(bundle, fusion_sd)
        names = "T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L" if mode == "penta" else "T5 + CLIP-L + CLIP-bigG"
        print(f"{mode.capitalize()} text encoder ({names}) loaded.")
        return bundle.tokenizer, bundle.text_encoder, bundle
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder)
    return tokenizer, T5EncoderModel.from_pretrained(cfg.text_encoder).to(device).eval(), None


def _check_rae(cfg: Any, vae: Any, rae_bridge: Any, image_size: int) -> None:
    """The DiT samples ``cfg.latent_channels``-channel latents at ``image_size // 8``; an RAE must map onto that."""
    ae_cfg = getattr(vae, "config", None)
    channels = getattr(ae_cfg, "encoder_hidden_size", None)
    in_size = getattr(ae_cfg, "encoder_input_size", None)
    patch = getattr(ae_cfg, "encoder_patch_size", None)
    if channels is not None and int(channels) != int(getattr(cfg, "latent_channels", 4) or 4) and rae_bridge is None:
        raise ValueError(
            "AutoencoderRAE latent channels differ from the DiT's but checkpoint has no rae_latent_bridge. "
            f"encoder_hidden_size={channels}. Train with "
            "`train.py --autoencoder-type rae` (bridge is created automatically) and sample this checkpoint."
        )
    if in_size is not None and patch is not None and int(in_size) // int(patch) != image_size // 8:
        raise ValueError(
            "AutoencoderRAE selected, but the RAE latent spatial size doesn't match this repo's assumptions. "
            f"RAE latent_hw={int(in_size) // int(patch)}, expected={image_size // 8} (image_size//8). "
            "To use RAE properly, update the DiT/diffusion latent spatial assumptions."
        )


def load_models(args: Any) -> Models:
    """Seed, load checkpoint + adapters + encoders + autoencoder, and pick the sampling canvas."""
    device = _seed_and_backends(args)
    print("Loading checkpoint and encoders...")
    model, cfg, rae_bridge, fusion_sd = load_model_from_ckpt(args.ckpt, device)
    _apply_lcm(args, model, cfg, device)
    if getattr(args, "compile_inference", False) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Inference: torch.compile enabled (mode=reduce-overhead).")
        except Exception as e:
            print(f"Inference: torch.compile skipped ({e}).", file=sys.stderr)
    _apply_loras(args, model)

    tokenizer, text_encoder, text_bundle = _load_text_encoders(cfg, fusion_sd, device)
    try:
        n_tok = tokenizer(args.prompt, return_tensors="pt", truncation=False).input_ids.shape[1]
        if n_tok > TEXT_MAX_TOKENS - 50:
            print(
                f"Note: prompt has {n_tok} tokens; T5 truncates at {TEXT_MAX_TOKENS}. "
                "Put key elements first for best adherence.",
                file=sys.stderr,
            )
    except Exception:
        pass

    from utils.modeling.encoders import load_autoencoder

    ae_type = getattr(cfg, "autoencoder_type", "kl")
    vae = load_autoencoder(
        ae_type, cfg.vae_model, subfolder=str(getattr(cfg, "vae_subfolder", "") or ""), device=device
    )
    latent_scale = 1.0 if ae_type == "rae" else getattr(cfg, "latent_scale", 0.18215)
    image_size = getattr(cfg, "image_size", 256)
    if ae_type == "rae":
        _check_rae(cfg, vae, rae_bridge, int(image_size))

    req_w, req_h = int(getattr(args, "width", 0) or 0), int(getattr(args, "height", 0) or 0)
    if not (req_w or req_h) and str(getattr(args, "init_image", "") or "").strip():
        try:
            req_w, req_h = Image.open(args.init_image).size  # img2img keeps the init image's aspect ratio
        except OSError:
            pass
    req_w, req_h = req_w or image_size, req_h or image_size
    if (getattr(args, "vae_tiling", False) or req_w * req_h > 512 * 512) and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
            print("Autoencoder tiling enabled (decode in tiles for lower VRAM).")
        except Exception:
            pass
    if ae_type == "rae":
        canvas_h = canvas_w = image_size  # the RAE latent grid is fixed
    else:
        # Hires-fix composes at the native pixel area and reaches the requested size in its own pass.
        canvas_h, canvas_w = sampling_canvas_hw(
            req_h,
            req_w,
            native_px=image_size,
            stride_px=dit_rgb_stride_px(model),
            max_area_ratio=1.0 if getattr(args, "hires_fix", False) else 2.0,
        )
    if (canvas_w, canvas_h) != (image_size, image_size):
        print(f"Sampling canvas {canvas_w}x{canvas_h} (latent {canvas_h // 8}x{canvas_w // 8}).", file=sys.stderr)
    diffusion = create_diffusion(
        timestep_respacing=getattr(cfg, "timestep_respacing", ""),
        num_timesteps=getattr(cfg, "num_timesteps", 1000),
        beta_schedule=getattr(cfg, "beta_schedule", "linear"),
        prediction_type=getattr(cfg, "prediction_type", "epsilon"),
    )
    return Models(
        device=device,
        model=model,
        cfg=cfg,
        rae_bridge=rae_bridge,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_bundle=text_bundle,
        vae=vae,
        ae_type=ae_type,
        latent_scale=latent_scale,
        image_size=int(image_size),
        req_w=int(req_w),
        req_h=int(req_h),
        canvas_h=int(canvas_h),
        canvas_w=int(canvas_w),
        diffusion=diffusion,
    )


@dataclass
class Conditioning:
    cond: dict
    uncond: dict
    sexual: bool


@dataclass
class InitLatents:
    """Starting point of the trajectory: pure noise (all None), img2img / from-z, or structured inpaint."""

    flow: bool
    num_timesteps: int
    x_init: torch.Tensor | None = None
    start_timestep: int | None = None
    inpaint_mask: torch.Tensor | None = None
    inpaint_x0: torch.Tensor | None = None
    inpaint_noise: torch.Tensor | None = None


def conditioning_image_paths(args) -> list[str]:
    """Every user-supplied image that conditions generation, for the minor guard's image layer."""

    keys = (
        "init_image",
        "edit_image",
        "control_image",
        "reference_image",
        "post_reference_image",
        "creative_rag_image",
    )
    paths = [getattr(args, k, "") for k in keys]
    for k in ("creative_rag_images", "dissect_refs"):
        paths += str(getattr(args, k, "") or "").split(",")
    paths += [_parse_control_spec(s)[0] for s in getattr(args, "control", None) or []]
    paths += [r.path for r in _collect_style_refs(args)]
    return list(dict.fromkeys(p for p in (str(x or "").strip() for x in paths) if p))


def _load_rgb_tensor(path: str, width: int, height: int) -> torch.Tensor:
    """``(3, H, W)`` in ``[-1, 1]``, resized to the canvas."""
    pil = Image.open(path).convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def _style_text(args: Any, cfg: Any, prompt: str) -> str:
    """``--style`` (``a::0.6 | b::0.4`` mixes) or, with ``--auto-style-from-prompt``, a style read off the prompt."""
    style = (args.style or "").strip()
    if not (getattr(cfg, "style_embed_dim", 0) and not style and getattr(args, "auto_style_from_prompt", False)):
        return style
    try:
        from config.styles import extract_style_from_text

        style = extract_style_from_text(prompt) or ""
        if style:
            print(f'Auto-style from prompt: "{style[:50]}{"..." if len(style) > 50 else ""}"', file=sys.stderr)
    except Exception:
        pass
    return style


def encode_conditioning(args: Any, m: Any, plan: Any) -> Conditioning:
    """Screen every positive channel, encode (cached), and build the cond / uncond model kwargs."""
    prompt, num_gen, device = plan.prompt, plan.num_gen, m.device
    style = _style_text(args, m.cfg, prompt)
    style_mix = _parse_weighted_style_mix(style)
    if style_mix:
        style_key = "|".join(f"{t}::{w:.4f}" for t, w in style_mix)
    else:
        style_key = style if getattr(m.cfg, "style_embed_dim", 0) else ""

    layout = getattr(args, "_layout_compiled", None)
    layout_enc = str(getattr(args, "t5_layout_encode", "auto") or "auto").lower()
    if layout_enc == "auto":
        layout_enc = "blocks" if layout is not None else "flat"
    t5_prompt = str(getattr(args, "_encode_t5_positive_hint", "") or "").strip() or prompt
    segment_texts = clip_caps = long_clip_caps = None
    if layout is not None and layout_enc == "blocks":
        from utils.prompt.text import substitute_compiled_layout_in_t5_prompt

        t5_prompt = substitute_compiled_layout_in_t5_prompt(prompt, layout)
    elif layout is not None and layout_enc == "segmented":
        from utils.prompt.text import t5_segment_texts_for_full_prompt

        segment_texts = t5_segment_texts_for_full_prompt(layout, prompt)
    if m.text_bundle is not None and layout is not None:
        from utils.prompt.text import multi_clip_caption

        clip_caps = [multi_clip_caption(layout, prompt)]
    if m.text_bundle is not None and getattr(m.text_bundle, "mode", "") == "penta":
        long_clip_caps = [prompt]

    negative = plan.negative
    box_regions = getattr(getattr(args, "_box_layout_spec", None), "regions", None) or []
    adapters = [_parse_lora_spec(s)[:2] for s in getattr(args, "lora", None) or []]
    positive_union = ", ".join(
        [
            prompt,
            t5_prompt,
            style,
            *(segment_texts or []),
            *(clip_caps or []),
            *(r.prompt for r in box_regions),
            *(Path(p).stem for p, scale in adapters if scale >= 0),
        ]
    )
    negative = require_allowed(positive_union, negative)
    # A negative-scale adapter subtracts its concept the way a negative prompt does.
    if negative_adapters := ", ".join(Path(p).stem for p, scale in adapters if scale < 0):
        require_allowed(positive_union, negative_adapters)
    sexual = is_sexual(positive_union)
    if float(getattr(args, "cfg_scale", 7.5) or 0.0) < 1.0:
        # Below CFG 1 the uncond branch gets positive weight, so the negative is part of the prompt.
        require_allowed(f"{prompt}, {negative}")
        sexual = sexual or is_sexual(negative)
    if sexual:
        require_image_allowed(conditioning_image_paths(args), sexual=True, conditioning=True)

    def enc(captions, **kw):
        return encode_text(captions, m.tokenizer, m.text_encoder, device, text_bundle=m.text_bundle, **kw)

    layout_tag = (layout_enc, str(getattr(args, "prompt_layout", "") or "")) if layout is not None else ("flat", "")
    t5_hash = hashlib.sha256(t5_prompt.encode("utf-8")).hexdigest()[:20]
    cache_key = None if getattr(args, "no_cache", False) else (prompt, negative, style_key, layout_tag, t5_hash)
    if cache_key is not None and cache_key in _t5_cache:
        cond_emb, uncond_emb, style_emb = (t.to(device) if t is not None else None for t in _t5_cache[cache_key])
        print("T5 cache hit.")
    else:
        if segment_texts is not None:
            cond_emb = enc(
                [t5_prompt], clip_captions=clip_caps, long_clip_captions=long_clip_caps, segment_texts=segment_texts
            )
            uncond_emb = enc([negative])
        else:
            both_long = long_clip_caps
            if m.text_bundle is not None and getattr(m.text_bundle, "mode", "") == "penta":
                both_long = [prompt, negative]
            both = enc([t5_prompt, negative], clip_captions=clip_caps, long_clip_captions=both_long)
            cond_emb, uncond_emb = both[0:1], both[1:2]
        style_emb = None
        if style and getattr(m.cfg, "style_embed_dim", 0):
            if len(style_mix) > 1:
                weights = torch.tensor([w for _, w in style_mix], device=device, dtype=torch.float32)
                style_enc = enc([t for t, _ in style_mix]).mean(dim=1)
                style_emb = (style_enc * weights[:, None].to(style_enc.dtype)).sum(dim=0, keepdim=True)
            else:
                style_emb = enc([style]).mean(dim=1)
        if cache_key is not None:
            while len(_t5_cache) >= _T5_CACHE_MAX:
                _t5_cache.pop(next(iter(_t5_cache)))
            _t5_cache[cache_key] = tuple(t.cpu() if t is not None else None for t in (cond_emb, uncond_emb, style_emb))

    cond = {"encoder_hidden_states": cond_emb.expand(num_gen, -1, -1)}
    uncond = {"encoder_hidden_states": uncond_emb.expand(num_gen, -1, -1)}
    if style_emb is not None:
        cond["style_embedding"] = style_emb.expand(num_gen, -1)
        cond["style_strength"] = args.style_strength
    if plan.emphasis_segments:
        from utils.prompt.text import token_weights_from_cleaned_segments

        tw = token_weights_from_cleaned_segments(
            prompt, plan.emphasis_segments, m.tokenizer, TEXT_MAX_TOKENS, device=device
        )
        if tw is not None and tw.shape[0] == cond_emb.shape[1]:
            # (word) -> 1.2, [word] -> 0.8 on the cond tokens; batch-shaped so batched CFG and regional
            # forwards can stack it, and explicit ones for the negative branch.
            tw = tw.view(1, -1, 1).expand(num_gen, -1, -1)
            cond["token_weights"] = tw
            uncond["token_weights"] = torch.ones_like(tw)
    if getattr(m.cfg, "creativity_embed_dim", 0) and args.creativity is not None:
        c0 = max(0.0, min(1.0, float(args.creativity)))
        jitter = max(0.0, float(getattr(args, "creativity_jitter", 0.0) or 0.0))
        if jitter > 0:
            g = torch.Generator(device=device).manual_seed(int(args.seed) + 90211)
            deltas = torch.randn(num_gen, generator=g, device=device, dtype=torch.float32) * jitter
            cond["creativity"] = (c0 + deltas).clamp(0.0, 1.0).to(dtype=cond_emb.dtype)
        else:
            cond["creativity"] = torch.full((num_gen,), c0, device=device, dtype=cond_emb.dtype)
    for flag in ("internal_guidance_block", "genesis_ops"):
        if bool(getattr(args, flag, False)):
            cond[flag] = uncond[flag] = True
    _control_kwargs(args, m, cond, num_gen)
    _reference_kwargs(args, m, cond, num_gen)
    _edit_kwargs(args, m, cond, uncond, num_gen)
    args._regional_cfg_plan = _regional_plan(args, m, enc)
    return Conditioning(cond=cond, uncond=uncond, sexual=sexual)


def _edit_kwargs(args: Any, m: Any, cond: dict, uncond: dict, num_gen: int) -> None:
    """``--edit-image``: source-image latents for an instruction-edit checkpoint.

    They go into both CFG branches, so guidance amplifies the instruction rather than pushing away from
    the source image (the negative branch is "this image, no instruction").
    """
    path = str(getattr(args, "edit_image", "") or "").strip()
    if not path:
        return
    if getattr(m.model, "edit_type_embed", None) is None:
        raise SystemExit("--edit-image needs an instruction-edit checkpoint (train.py --edit-conditioning).")
    if m.ae_type != "kl":
        raise SystemExit("--edit-image needs a KL-VAE checkpoint.")
    vae_dtype = next(m.vae.parameters()).dtype
    x = _load_rgb_tensor(path, m.canvas_w, m.canvas_h).unsqueeze(0).to(m.device, dtype=vae_dtype)
    with torch.no_grad():
        z = m.vae.encode(x).latent_dist.mode() * m.latent_scale
    cond["edit_latents"] = uncond["edit_latents"] = z.float().expand(num_gen, -1, -1, -1)


def _control_kwargs(args: Any, m: Any, cond: dict, num_gen: int) -> None:
    """ControlNet inputs: ``--control-image`` / stacked ``--control`` specs, else box-layout sketches.

    Tensors are batch-shaped like the text embeddings, so batched CFG can stack the branches.
    """
    from models.conditioning import control_type_to_id, infer_control_type_from_path

    default_type = str(getattr(args, "control_type", "auto") or "auto")
    default_scale = float(getattr(args, "control_scale", 0.85))
    specs: list[tuple[str, str, float]] = []
    if args.control_image:
        specs.append((str(args.control_image), default_type, default_scale))
    for raw in list(getattr(args, "control", []) or []):
        path, ctype, scale = _parse_control_spec(raw, default_type=default_type, default_scale=default_scale)
        if path:
            specs.append((path, ctype, scale))
    box_spec = getattr(args, "_box_layout_spec", None)
    if box_spec is not None and not specs:
        try:
            from utils.generation.regional import build_composite_sketch_tensor, spec_has_sketches

            if spec_has_sketches(box_spec):
                sketch = build_composite_sketch_tensor(
                    box_spec, m.image_size, source_dir=getattr(box_spec, "source_dir", None), device=m.device
                )
                cond["control_image"] = sketch.unsqueeze(0).expand(num_gen, -1, -1, -1)
                cond["control_scale"] = float(getattr(args, "box_sketch_control_scale", 0.75) or 0.75)
                cond["control_type"] = torch.full(
                    (num_gen,), int(control_type_to_id("scribble")), device=m.device, dtype=torch.long
                )
                print("Box-layout sketch control image enabled (scribble).", file=sys.stderr)
        except Exception as e:
            print(f"Warning: box sketch control image skipped: {e}", file=sys.stderr)
    if not specs:
        return
    images, type_ids, scales = [], [], []
    for path, ctype, scale in specs:
        images.append(_load_rgb_tensor(path, m.canvas_w, m.canvas_h))
        name = str(ctype or "auto").strip().lower()
        if name in ("", "auto"):
            name = infer_control_type_from_path(str(path))
        type_ids.append(int(control_type_to_id(name)))
        scales.append(float(max(0.0, scale)))
    types = torch.tensor(type_ids, device=m.device, dtype=torch.long)
    if len(images) == 1:
        cond["control_image"] = images[0].to(m.device).expand(num_gen, -1, -1, -1)
        cond["control_scale"] = scales[0]
        cond["control_type"] = types.expand(num_gen)
    else:
        stack = torch.stack(images, dim=0).to(m.device)  # (K, C, H, W)
        cond["control_image"] = stack.expand(num_gen, *stack.shape)
        cond["control_scale"] = torch.tensor(scales, device=m.device, dtype=torch.float32).expand(num_gen, -1)
        cond["control_type"] = types.expand(num_gen, -1)


def _reference_kwargs(args: Any, m: Any, cond: dict, num_gen: int) -> None:
    """``--reference-image`` / ``--style-ref`` / moodboards -> reference tokens."""
    ref_path = str(getattr(args, "reference_image", "") or "").strip()
    multi = any(
        str(getattr(args, k, "") or "").strip()
        for k in ("style_ref", "style_references_json", "moodboard_json", "moodboard_images")
    )
    if not (multi or (ref_path and float(getattr(args, "reference_strength", 0.0) or 0.0) > 0)):
        return
    try:
        inject_reference_conditioning(cond, args=args, cfg=m.cfg, device=m.device, num_gen=num_gen)
    except Exception as e:
        print(f"Reference image conditioning skipped: {e}", file=sys.stderr)


def _regional_plan(args: Any, m: Any, enc: Any) -> Any:
    """Per-box embeddings and masks for regional CFG (``--box-layout`` in ``regional_cfg`` mode)."""
    spec = getattr(args, "_box_layout_spec", None)
    if (
        spec is None
        or str(getattr(args, "box_layout_mode", "regional_cfg") or "regional_cfg").lower() != "regional_cfg"
    ):
        return None
    try:
        from utils.generation.regional import encode_regional_plan

        plan = encode_regional_plan(
            spec,
            encode_fn=enc,
            device=m.device,
            latent_h=m.lat_h,
            latent_w=m.lat_w,
            pixel_size=max(m.canvas_h, m.canvas_w),
            base_negative=str(getattr(args, "negative_prompt", "") or ""),
        )
    except Exception as e:
        print(f"Warning: regional box CFG disabled: {e}", file=sys.stderr)
        return None
    sketches = sum(1 for r in spec.regions if r.sketch_path or r.strokes)
    msg = f"Regional CFG enabled for {len(spec.regions)} box(es)."
    print(msg + (f" {sketches} with draw+describe sketches." if sketches else ""), file=sys.stderr)
    return plan


def _auto_init_from_dissection(args: Any, m: Any) -> None:
    """``--auto-init-from-dissection``: composite reference parts into an init image + inpaint mask."""
    refs = [p.strip() for p in str(getattr(args, "dissect_refs", "") or "").split(",") if p.strip()]
    if not (bool(getattr(args, "auto_init_from_dissection", False)) and refs and not args.init_image and not args.mask):
        return
    try:
        from utils.generation.editing import build_init_and_inpaint_mask, dissect_images_to_parts

        out_dir = Path("runs") / "dissection_composites"
        _reqs, parts, _facts = dissect_images_to_parts(
            str(getattr(args, "prompt", "") or ""), refs, output_dir=out_dir, enable_heavy_models=True
        )
        spec = build_init_and_inpaint_mask(
            reference_images=refs,
            parts=parts,
            output_dir=out_dir,
            target_size=(m.canvas_w, m.canvas_h),
            lock_background_if_requested=bool(getattr(args, "dissection_lock_background", False)),
        )
    except Exception as e:
        print(f"Auto dissection init skipped: {e}", file=sys.stderr)
        return
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


def _noise_to(diffusion: Any, z0: torch.Tensor, t_start: int, num_timesteps: int, flow: bool) -> torch.Tensor:
    """Forward-noise a clean latent to ``t_start`` (VP ``q_sample`` or the flow interpolation)."""
    if flow:
        s0 = float(t_start) / float(max(num_timesteps - 1, 1))
        return (1.0 - s0) * z0 + s0 * torch.randn_like(z0)
    return diffusion.q_sample(z0, torch.tensor([t_start], device=z0.device).expand(z0.shape[0]))


def prepare_init_latents(args: Any, m: Any, num_gen: int) -> InitLatents:
    """``--init-latent`` (from-z), ``--init-image`` (img2img) and ``--mask`` (inpaint) starting latents."""
    _auto_init_from_dissection(args, m)
    num_timesteps = int(getattr(m.cfg, "num_timesteps", 1000))
    flow = bool(getattr(args, "flow_matching_sample", False)) or (
        bool(getattr(m.cfg, "flow_matching_training", False)) and not bool(getattr(args, "force_vp_sample", False))
    )
    if flow and args.mask:
        print(
            "Flow sampling disabled: structured inpaint uses VP q_sample; use --force-vp-sample or drop --mask.",
            file=sys.stderr,
        )
        flow = False
    init = InitLatents(flow=flow, num_timesteps=num_timesteps)
    if not (args.init_latent or args.init_image):
        return init
    t_start = min(max(1, int(args.strength * num_timesteps)), num_timesteps - 1)
    init.start_timestep = t_start
    if args.init_latent:
        z = torch.load(args.init_latent, map_location=m.device, weights_only=True)
        z = _maybe_rae_to_dit(
            (z.unsqueeze(0) if z.dim() == 3 else z).to(device=m.device, dtype=torch.float32), m.ae_type, m.rae_bridge
        )
        init.x_init = _noise_to(m.diffusion, z, t_start, num_timesteps, flow)
        print(f"From-z: strength={args.strength} -> t_start={t_start}")
    else:
        img = _load_rgb_tensor(args.init_image, m.canvas_w, m.canvas_h).unsqueeze(0).to(m.device, torch.float32)
        with torch.no_grad():
            enc = m.vae.encode(img)
            z0 = enc.latent_dist.sample() * m.latent_scale if hasattr(enc, "latent_dist") else enc.latent
            z0 = _maybe_rae_to_dit(z0, m.ae_type, m.rae_bridge)
        if args.mask:
            pil_mask = Image.open(args.mask).convert("L")
            if pil_mask.size != (m.canvas_w, m.canvas_h):
                pil_mask = pil_mask.resize((m.canvas_w, m.canvas_h), Image.Resampling.LANCZOS)
            mask = torch.from_numpy((np.asarray(pil_mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32))
            mask = torch.nn.functional.interpolate(
                mask.to(m.device).view(1, 1, m.canvas_h, m.canvas_w), size=(m.lat_h, m.lat_w), mode="nearest"
            )
            t_b = torch.tensor([t_start], device=m.device).expand(z0.shape[0])
            noise_known = torch.randn_like(z0)
            if args.inpaint_mode == "mdm":
                # MDM-style fill-in: known and masked regions get different forward noise at t_start, and the
                # sampler re-imposes q_sample(x0_known, t) on the known region after every step.
                x_masked = m.diffusion.q_sample(z0, t_b, noise=torch.randn_like(z0))
                init.x_init = mask * x_masked + (1 - mask) * m.diffusion.q_sample(z0, t_b, noise=noise_known)
                init.inpaint_mask, init.inpaint_x0, init.inpaint_noise = mask, z0, noise_known
            else:
                init.x_init = mask * noise_known + (1 - mask) * m.diffusion.q_sample(z0, t_b, noise=noise_known)
            print(f"Inpainting: strength={args.strength} -> t_start={t_start}")
        else:
            init.x_init = _noise_to(m.diffusion, z0, t_start, num_timesteps, flow)
            print(f"Img2img: strength={args.strength} -> t_start={t_start}")
    if init.x_init.shape[0] != num_gen:
        for name in ("x_init", "inpaint_x0", "inpaint_noise", "inpaint_mask"):
            t = getattr(init, name)
            if t is not None:
                setattr(init, name, t.expand(num_gen, -1, -1, -1))
    return init
