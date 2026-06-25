"""
Non-UI sample.py feature helpers: frontier, auto-refine, adherence maps, sessions.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "apply_sample_feature_args",
    "run_auto_refine_candidates",
    "run_creative_refine",
    "score_image_heuristic",
    "build_region_fix_mask",
    "export_adherence_heatmap",
    "load_character_session",
    "save_character_session",
    "append_benchmark_history",
]


def apply_sample_feature_args(args: Any, *, steps: int = 50) -> Dict[str, Any]:
    """Apply --frontier, --character-session, etc. Mutates ``args`` in place."""
    extra_kw: Dict[str, Any] = {}
    extra_kw.update(_apply_character_session(args))
    extra_kw.update(_apply_frontier_perfect(args, steps=steps))
    extra_kw.update(_apply_frontier_creative(args, steps=steps))
    extra_kw.update(_apply_frontier(args, steps=steps))
    extra_kw.update(_apply_attention_layout(args, steps=steps))
    extra_kw.update(_apply_per_region_cads(args))
    return extra_kw


def apply_sample_feature_prompt_phase(args: Any, *, steps: int = 50) -> None:
    """Prompt-time mutations (before text encoding)."""
    _apply_character_session(args)
    _apply_krea_controls(args)
    _apply_frontier_perfect(args, steps=steps)
    _apply_frontier_creative(args, steps=steps)
    _apply_frontier(args, steps=steps, diffusion_only=False)


def _apply_krea_controls(args: Any) -> None:
    if not any(
        (
            getattr(args, "krea_turbo_preset", False),
            str(getattr(args, "creativity_mode", "") or "").strip(),
            str(getattr(args, "style_ref", "") or "").strip(),
            str(getattr(args, "style_references_json", "") or "").strip(),
            str(getattr(args, "moodboard_json", "") or "").strip(),
            str(getattr(args, "moodboard_images", "") or "").strip(),
            abs(float(getattr(args, "slider_intensity", 0) or 0)) > 1e-6,
            abs(float(getattr(args, "slider_complexity", 0) or 0)) > 1e-6,
            abs(float(getattr(args, "slider_movement", 0) or 0)) > 1e-6,
        )
    ):
        return
    from utils.generation.krea_controls import apply_krea_controls_to_args

    apply_krea_controls_to_args(args)


def apply_sample_feature_diffusion_phase(args: Any, *, steps: int = 50) -> Dict[str, Any]:
    """Diffusion-time kwargs (after layout / regional plan)."""
    extra: Dict[str, Any] = {}
    extra.update(_apply_frontier_perfect(args, steps=steps, diffusion_only=True))
    extra.update(_apply_frontier_creative(args, steps=steps, diffusion_only=True))
    extra.update(_apply_frontier(args, steps=steps, diffusion_only=True))
    extra.update(_apply_attention_layout(args, steps=steps))
    extra.update(_apply_per_region_cads(args))
    return extra


def _apply_frontier_perfect(args: Any, *, steps: int, diffusion_only: bool = False) -> Dict[str, Any]:
    perfect = getattr(args, "frontier_perfect", False)
    subject_only = getattr(args, "frontier_subject", False)
    if not perfect and not subject_only:
        return {}
    if diffusion_only:
        return {}

    if perfect:
        from frontier.perfect import analyze_perfect, perfect_sample_kwargs
        from frontier.safety import PolicyTier

        tier_name = str(getattr(args, "safety_tier", "moderate") or "moderate").lower()
        tier = PolicyTier(tier_name)
        plan = analyze_perfect(
            args.prompt or "",
            num_steps=steps,
            serendipity_dial=float(getattr(args, "frontier_serendipity", 0.25) or 0.25),
            layout_regions=len(getattr(getattr(args, "_box_layout_spec", None), "regions", []) or []),
            auto_resolve_contradictions=bool(getattr(args, "frontier_auto_resolve", False)),
            safety_tier=tier,
        )
        args.perfect_frontier_plan = plan
        if plan.refused:
            raise ValueError(
                "Prompt refused by content policy (--safety-tier). "
                f"Reasons: {', '.join(plan.refuse_reasons) or 'prohibited combination'}"
            )
        kw = perfect_sample_kwargs(plan, base_negative=getattr(args, "negative_prompt", "") or "")
        args.prompt = kw.get("prompt", args.prompt)
        args.negative_prompt = kw.get("negative_prompt", getattr(args, "negative_prompt", ""))
        if kw.get("cfg_scale_multiplier") and getattr(args, "cfg_scale", None):
            args.cfg_scale = float(args.cfg_scale) * float(kw["cfg_scale_multiplier"])
        return {k: v for k, v in kw.items() if k not in ("prompt", "negative_prompt")}

    from frontier.subject import analyze_subject, subject_sample_kwargs

    sub = analyze_subject(args.prompt or "")
    args.subject_frontier_plan = sub
    kw = subject_sample_kwargs(sub, base_negative=getattr(args, "negative_prompt", "") or "")
    args.prompt = kw.get("prompt", args.prompt)
    args.negative_prompt = kw.get("negative_prompt", getattr(args, "negative_prompt", ""))
    if kw.get("cfg_scale_multiplier") and getattr(args, "cfg_scale", None):
        args.cfg_scale = float(args.cfg_scale) * float(kw["cfg_scale_multiplier"])
    return {k: v for k, v in kw.items() if k not in ("prompt", "negative_prompt")}


def _apply_frontier_creative(args: Any, *, steps: int, diffusion_only: bool = False) -> Dict[str, Any]:
    if not getattr(args, "frontier_creative", False):
        return {}

    from frontier.imagination import analyze_imagination, imagination_sample_kwargs

    seed = int(getattr(args, "seed", 0) or 0)
    rnd_seed = seed if getattr(args, "creative_random_constraint", False) else None
    plan = analyze_imagination(
        args.prompt or "",
        num_steps=steps,
        base_serendipity=float(getattr(args, "frontier_serendipity", 0.25) or 0.25),
        mutate_count=int(getattr(args, "creative_mutate", 0) or 0),
        mutate_seed=seed,
        random_constraint_seed=rnd_seed,
    )
    args.imagination_plan = plan

    if diffusion_only:
        extra: Dict[str, Any] = {}
        if plan.step_emphasis:
            extra["step_emphasis"] = list(plan.step_emphasis)
        from frontier.chaos.serendipity import SerendipityInjector

        inj = SerendipityInjector(num_steps=steps)
        curve = inj.curve(plan.serendipity_dial)
        extra["step_noise_scales"] = list(curve.scales)
        return extra

    kw = imagination_sample_kwargs(plan, base_negative=getattr(args, "negative_prompt", "") or "")
    args.prompt = kw.get("prompt", args.prompt)
    args.negative_prompt = kw.get("negative_prompt", getattr(args, "negative_prompt", ""))
    if plan.suppress_contradiction_resolve:
        args.frontier_auto_resolve = False
    if kw.get("cfg_scale_multiplier") and getattr(args, "cfg_scale", None):
        args.cfg_scale = float(args.cfg_scale) * float(kw["cfg_scale_multiplier"])
    if plan.mutations:
        args.creative_prompt_mutations = plan.mutations
    args.frontier_serendipity = plan.serendipity_dial
    return {k: v for k, v in kw.items() if k not in ("prompt", "negative_prompt")}


def write_fix_region_mask(
    box_spec: Any,
    region_name: str,
    *,
    image_size: int,
    out_path: str | Path,
) -> Tuple[Path, str]:
    """Rasterize a box region to an inpaint mask PNG; returns (path, regional prompt)."""
    import torch
    from PIL import Image

    latent_h = latent_w = max(8, image_size // 8)
    mask_latent, reg_prompt = build_region_fix_mask(
        box_spec, region_name, latent_h=latent_h, latent_w=latent_w, device=torch.device("cpu")
    )
    mask_up = torch.nn.functional.interpolate(mask_latent, size=(image_size, image_size), mode="nearest")
    arr = (mask_up[0, 0].numpy() * 255).astype("uint8")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(p)
    return p, reg_prompt


def _apply_frontier(args: Any, *, steps: int, diffusion_only: bool = False) -> Dict[str, Any]:
    if getattr(args, "frontier_perfect", False) or getattr(args, "frontier_subject", False):
        return {}
    if getattr(args, "frontier_creative", False):
        return {}
    if not getattr(args, "frontier", False):
        return {}
    from frontier.hooks import apply_frontier_to_args, frontier_diffusion_hooks

    dial = float(getattr(args, "frontier_serendipity", 0.25) or 0.25)
    if diffusion_only and getattr(args, "frontier_plan", None) is not None:
        plan = args.frontier_plan
    else:
        from frontier.engine import FrontierEngine

        eng = FrontierEngine(num_steps=steps, serendipity_dial=dial)
        plan = apply_frontier_to_args(args, engine=eng)
        if getattr(args, "frontier_auto_resolve", False):
            from frontier.logic.contradiction import ContradictionScanner

            args.prompt = ContradictionScanner().suggest_rewrite(args.prompt or "")
    hooks = frontier_diffusion_hooks(plan)
    extra: Dict[str, Any] = {}
    if hooks.get("serendipity_scales"):
        extra["step_noise_scales"] = hooks["serendipity_scales"]
    if hooks.get("entropy_per_step"):
        extra["step_noise_scales"] = hooks.get("entropy_per_step")
    if not diffusion_only:
        witness_cfg = getattr(args, "cfg_scale", None)
        if plan.witness and plan.witness.cfg_bias and witness_cfg:
            args.cfg_scale = float(witness_cfg) * float(plan.witness.cfg_bias)
    return extra


def _apply_character_session(args: Any) -> Dict[str, Any]:
    path = str(getattr(args, "character_session", "") or "").strip()
    if not path:
        return {}
    session = load_character_session(path)
    additions = session.get("prompt_additions", "")
    if additions:
        base = (args.prompt or "").strip()
        args.prompt = f"{base}, {additions}" if base else additions
    refs = session.get("reference_images", [])
    if refs and not getattr(args, "dissect_refs", ""):
        args.dissect_refs = ",".join(str(r) for r in refs)
    neg = session.get("negative_prompt", "")
    if neg:
        base_neg = getattr(args, "negative_prompt", "") or ""
        args.negative_prompt = f"{base_neg}, {neg}" if base_neg else neg
    return {}


def _apply_attention_layout(args: Any, *, steps: int) -> Dict[str, Any]:
    if not getattr(args, "box_attn_layout", False):
        return {}
    box_spec = getattr(args, "_box_layout_spec", None)
    if box_spec is None:
        return {}
    from frontier.attention.layout_plan import build_attention_layout_plan

    frac = float(getattr(args, "box_attn_inject_frac", 0.4) or 0.4)
    strength = float(getattr(args, "box_attn_strength", 0.85) or 0.85)
    plan = build_attention_layout_plan(
        box_spec.regions,
        num_steps=steps,
        inject_frac=frac,
        strength=strength,
    )
    args._attention_layout_plan = plan
    return {"attention_layout_plan": plan}


def _apply_per_region_cads(args: Any) -> Dict[str, Any]:
    if not getattr(args, "per_region_cads", False):
        return {}
    box_spec = getattr(args, "_box_layout_spec", None)
    if box_spec is None:
        return {}
    from utils.generation.per_region_cads import PerRegionCADSConfig

    args._per_region_cads = PerRegionCADSConfig.from_box_spec(box_spec)
    return {}


def load_character_session(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"character session not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("character session must be a JSON object")
    return data


def save_character_session(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_region_fix_mask(
    box_spec: Any,
    region_name: str,
    *,
    latent_h: int,
    latent_w: int,
    device: Any = None,
) -> Tuple[Any, str]:
    """Return (mask tensor 1,1,H,W), regional prompt for ``region_name``."""
    import torch

    from utils.generation.regional_box_prompting import build_latent_region_masks

    name = region_name.strip().lower()
    target = None
    for r in box_spec.regions:
        if str(r.name).strip().lower() == name:
            target = r
            break
    if target is None:
        raise KeyError(f"region {region_name!r} not in box layout")
    region_masks, _bg = build_latent_region_masks(box_spec, latent_h, latent_w, device=device or torch.device("cpu"))
    idx = list(box_spec.regions).index(target)
    mask = region_masks[idx : idx + 1].clone()
    return mask, str(target.prompt or target.name)


def score_image_heuristic(image_rgb_u8: np.ndarray) -> float:
    """Cheap 0–1 quality proxy (edge sharpness + exposure balance)."""
    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] < 3:
        return 0.0
    gray = image_rgb_u8[..., :3].astype(np.float32).mean(axis=2) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    sharp = float((gx + gy) * 0.5)
    exp = 1.0 - abs(float(gray.mean()) - 0.45) * 2.0
    return max(0.0, min(1.0, 0.55 * sharp + 0.45 * max(0.0, exp)))


@dataclass
class AutoRefineResult:
    best_index: int
    scores: List[float]
    outputs: List[str]
    prompts: List[str] = field(default_factory=list)


def _replace_or_append_prompt(cmd: List[str], prompt: str) -> List[str]:
    """Return a copy of ``cmd`` with ``--prompt`` value replaced or appended."""
    out: List[str] = []
    replaced = False
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok == "--prompt" and i + 1 < len(cmd):
            out.extend(["--prompt", prompt])
            i += 2
            replaced = True
            continue
        if tok.startswith("--prompt="):
            out.append(f"--prompt={prompt}")
            i += 1
            replaced = True
            continue
        out.append(tok)
        i += 1
    if not replaced:
        out.extend(["--prompt", prompt])
    return out


def run_auto_refine_candidates(
    *,
    sample_cmd: Sequence[str],
    num_candidates: int = 3,
    seed_base: int = 42,
    out_stem: str = "refine",
    prompt_variants: Sequence[str] | None = None,
) -> AutoRefineResult:
    """
    Run ``sample.py`` multiple times; pick best by heuristic score.

    When ``prompt_variants`` is set, each candidate uses a different prompt
    (e.g. from ``--creative-mutate``). Otherwise only seed offsets vary.
    """
    from PIL import Image

    scores: List[float] = []
    outputs: List[str] = []
    prompts_used: List[str] = []
    best_i = 0
    best_s = -1.0
    n = max(1, int(num_candidates))
    variants = list(prompt_variants or [])

    for i in range(n):
        out = f"{out_stem}_{i}.png"
        cmd = list(sample_cmd)
        if variants:
            idx = i % len(variants)
            cmd = _replace_or_append_prompt(cmd, variants[idx])
            prompts_used.append(variants[idx])
        else:
            prompts_used.append(_extract_prompt_from_cmd(cmd) or "")

        cmd = cmd + ["--seed", str(seed_base + i), "--out", out]
        subprocess.run(cmd, check=True)
        outputs.append(out)
        arr = np.array(Image.open(out).convert("RGB"))
        s = score_image_heuristic(arr)
        scores.append(s)
        if s > best_s:
            best_s = s
            best_i = i
    return AutoRefineResult(best_index=best_i, scores=scores, outputs=outputs, prompts=prompts_used)


def _extract_prompt_from_cmd(cmd: Sequence[str]) -> str:
    for i, tok in enumerate(cmd):
        if tok == "--prompt" and i + 1 < len(cmd):
            return str(cmd[i + 1])
        if tok.startswith("--prompt="):
            return tok.split("=", 1)[1]
    return ""


def run_creative_refine(
    *,
    sample_cmd: Sequence[str],
    base_prompt: str,
    num_candidates: int = 4,
    seed_base: int = 42,
    out_stem: str = "creative",
    mutate_count: int | None = None,
    base_serendipity: float = 0.25,
) -> AutoRefineResult:
    """
    Build mutation variants from ``analyze_imagination``, run each, pick best.

    Falls back to seed-only diversity if no mutations requested.
    """
    from frontier.imagination import analyze_imagination

    count = int(mutate_count if mutate_count is not None else num_candidates)
    plan = analyze_imagination(
        base_prompt,
        base_serendipity=base_serendipity,
        mutate_count=max(count, num_candidates),
        mutate_seed=seed_base,
    )
    variants = plan.mutations if plan.mutations else None
    if variants:
        # Include augmented base as first candidate
        base_aug = plan.augmented_prompt or base_prompt
        if base_aug not in variants:
            variants = [base_aug] + list(variants)
    return run_auto_refine_candidates(
        sample_cmd=sample_cmd,
        num_candidates=num_candidates,
        seed_base=seed_base,
        out_stem=out_stem,
        prompt_variants=variants,
    )


def export_adherence_heatmap(
    attn_weights: Any,
    prompt: str,
    out_path: str | Path,
    *,
    latent_h: int = 64,
    latent_w: int = 64,
    box_spec: Any = None,
) -> Path:
    """Save PNG heatmap from cross-attention weights + optional box overlay."""
    import torch
    from models.prompt_adherence import PromptParser, attribute_binding_heatmap
    from PIL import Image, ImageDraw

    parser = PromptParser()
    parsed = parser.parse(prompt)
    heat = attribute_binding_heatmap(
        attn_weights if isinstance(attn_weights, torch.Tensor) else torch.as_tensor(attn_weights),
        parsed,
        spatial_h=latent_h,
        spatial_w=latent_w,
    )
    arr = heat.detach().float().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    rgb = (plt_colormap_inferno(arr) * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if box_spec is not None:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for r in getattr(box_spec, "regions", []) or []:
            x1, y1, x2, y2 = float(r.x1), float(r.y1), float(r.x2), float(r.y2)
            draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(0, 200, 255), width=2)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


def plt_colormap_inferno(arr: np.ndarray) -> np.ndarray:
    """Inferno-like colormap without matplotlib dependency."""
    x = np.clip(arr, 0.0, 1.0)
    r = np.clip(1.5 * x - 0.1, 0, 1)
    g = np.clip(1.2 * x - 0.2, 0, 1) * np.clip(x * 2, 0, 1)
    b = np.clip(0.8 - x, 0, 1) * np.clip(1.5 * x, 0, 1)
    return np.stack([r, g, b], axis=-1)


def append_benchmark_history(leaderboard_path: Path, history_path: Path) -> Dict[str, Any]:
    """Append ``leaderboard.json`` snapshot to rolling history file."""
    lb = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    hist: List[Any] = []
    if history_path.is_file():
        hist = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(hist, list):
            hist = []
    import datetime as _dt

    entry = {"timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"), "leaderboard": lb}
    hist.append(entry)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    return entry
