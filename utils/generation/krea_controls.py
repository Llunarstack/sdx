"""
Krea 2–inspired generation controls (no hosted API required).

- Multi-reference **style** conditioning (weighted CLIP blend → reference tokens)
- **Moodboards** (many images → one pooled style embedding)
- **Generative sliders** (intensity / complexity / movement)
- **Creativity modes** (raw → high prompt expansion)
- **Turbo preset** (few-step, low CFG — mirrors Krea 2 Turbo defaults)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "CreativityMode",
    "GenerativeSliderPlan",
    "StyleReference",
    "aggregate_style_embeddings",
    "apply_creativity_mode_to_prompt",
    "apply_generative_sliders",
    "apply_krea_controls_to_args",
    "apply_turbo_preset",
    "build_generative_slider_plan",
    "inject_reference_conditioning",
    "load_moodboard_paths",
    "parse_style_references",
]


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
    trace: List[str] = field(default_factory=list)


def _clamp_slider(v: float) -> float:
    return max(-100.0, min(100.0, float(v)))


def _norm_weights(strengths: Sequence[float]) -> List[float]:
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
) -> List[StyleReference]:
    """
    Parse style refs from JSON (``{"references": [{"path", "strength"}]}``)
    or CSV ``path:weight,path2:weight`` (weight defaults to 1.0).
    """
    out: List[StyleReference] = []
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
) -> List[str]:
    """Load moodboard image paths from JSON or comma-separated list."""
    paths: List[str] = []
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


def aggregate_style_embeddings(
    refs: Sequence[StyleReference],
    *,
    device: Any,
    model_id: str,
    dtype: Any,
) -> Tuple[Any, int]:
    """Weighted-mean CLIP image embedding from multiple style references."""
    import torch
    from PIL import Image

    from utils.generation.clip_reference_embed import encode_reference_image_pil

    if not refs:
        raise ValueError("aggregate_style_embeddings requires at least one reference")

    embeds: List[Any] = []
    weights: List[float] = []
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
    pos: List[str] = []
    neg: List[str] = []

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
    setattr(args, "_generative_slider_plan", plan)
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
        setattr(args, "_creativity_mode", mode)
        return mode

    word_count = len(re.findall(r"\w+", prompt))
    short = word_count < 10

    if mode == CreativityMode.RAW:
        setattr(args, "_skip_prompt_expansion", True)
    elif mode == CreativityMode.LOW and short:
        low = prompt.lower()
        if "lighting" not in low:
            args.prompt = f"{prompt}, soft natural lighting"
    elif mode == CreativityMode.MEDIUM and short:
        from utils.superior.prompt_expand import expand_prompt_heuristic

        args.prompt = expand_prompt_heuristic(prompt)
    elif mode == CreativityMode.HIGH:
        from utils.superior.prompt_expand import expand_prompt_heuristic

        expanded = expand_prompt_heuristic(prompt)
        if mode == CreativityMode.HIGH and word_count < 16:
            expanded = f"{expanded}, expressive art direction, cinematic mood, rich color palette"
        args.prompt = expanded
        if getattr(args, "frontier_serendipity", None) is not None:
            args.frontier_serendipity = min(1.0, float(args.frontier_serendipity) + 0.08)
        elif getattr(args, "novelty", None) is not None:
            args.novelty = min(1.0, float(args.novelty) + 0.1)

    setattr(args, "_creativity_mode", mode)
    return mode


def apply_turbo_preset(args: Any) -> None:
    """Few-step sampling profile inspired by Krea 2 Turbo (8 steps, low CFG)."""
    if int(getattr(args, "steps", 0) or 0) in (0, 50, 28):
        args.steps = 8
    args.cfg_scale = float(getattr(args, "cfg_scale", 4.5) or 4.5)
    if args.cfg_scale > 2.0:
        args.cfg_scale = 1.0
    setattr(args, "_krea_turbo_preset", True)


def apply_krea_controls_to_args(args: Any) -> Dict[str, Any]:
    """
    Prompt-time Krea controls: turbo preset, creativity mode, generative sliders.

    Returns metadata dict (empty if nothing applied).
    """
    meta: Dict[str, Any] = {}
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


def _collect_style_refs(args: Any) -> List[StyleReference]:
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
    model_kwargs_cond: Dict[str, Any],
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

    from models.reference_token_projection import ReferenceTokenProjector

    from utils.generation.clip_reference_embed import encode_reference_image_pil

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
