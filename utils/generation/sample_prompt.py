"""``sample.py`` prompt phases: feature planners (frontier, Krea, attention layout, auto-refine), the user's prompt, additions and the final encoder text."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.generation.sample_core import (
    TEXT_MAX_TOKENS,
    _apply_gender_swap,
    _build_size_tokens,
    _infer_expected_texts_from_prompt,
    _load_character_sheet,
    _maybe_append_text_says,
    _parse_expected_texts,
    _parse_scale_csv,
)
from utils.prompt.text import parse_prompt_emphasis
from utils.safety.minor_guard import require_allowed


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
    from utils.generation.sample_core import apply_krea_controls_to_args

    apply_krea_controls_to_args(args)


def apply_sample_feature_diffusion_phase(args: Any, *, steps: int = 50) -> dict[str, Any]:
    """Diffusion-time kwargs (after layout / regional plan)."""
    extra: dict[str, Any] = {}
    extra.update(_apply_frontier_perfect(args, steps=steps, diffusion_only=True))
    extra.update(_apply_frontier_creative(args, steps=steps, diffusion_only=True))
    extra.update(_apply_frontier(args, steps=steps, diffusion_only=True))
    extra.update(_apply_attention_layout(args, steps=steps))
    return extra


def _apply_frontier_perfect(args: Any, *, steps: int, diffusion_only: bool = False) -> dict[str, Any]:
    perfect = getattr(args, "frontier_perfect", False)
    subject_only = getattr(args, "frontier_subject", False)
    if not perfect and not subject_only:
        return {}
    if diffusion_only:
        return {}

    if perfect:
        from frontier.planning import PolicyTier, analyze_perfect, perfect_sample_kwargs

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
            from utils.safety.minor_guard import ProhibitedContentError

            raise ProhibitedContentError(
                f"Prompt refused: sexual content involving minors ({', '.join(plan.refuse_reasons)})."
            )
        kw = perfect_sample_kwargs(plan, base_negative=getattr(args, "negative_prompt", "") or "")
        args.prompt = kw.get("prompt", args.prompt)
        args.negative_prompt = kw.get("negative_prompt", getattr(args, "negative_prompt", ""))
        if kw.get("cfg_scale_multiplier") and getattr(args, "cfg_scale", None):
            args.cfg_scale = float(args.cfg_scale) * float(kw["cfg_scale_multiplier"])
        return {k: v for k, v in kw.items() if k not in ("prompt", "negative_prompt")}

    from frontier.planning import analyze_subject, subject_sample_kwargs

    sub = analyze_subject(args.prompt or "")
    args.subject_frontier_plan = sub
    kw = subject_sample_kwargs(sub, base_negative=getattr(args, "negative_prompt", "") or "")
    args.prompt = kw.get("prompt", args.prompt)
    args.negative_prompt = kw.get("negative_prompt", getattr(args, "negative_prompt", ""))
    if kw.get("cfg_scale_multiplier") and getattr(args, "cfg_scale", None):
        args.cfg_scale = float(args.cfg_scale) * float(kw["cfg_scale_multiplier"])
    return {k: v for k, v in kw.items() if k not in ("prompt", "negative_prompt")}


def _apply_frontier_creative(args: Any, *, steps: int, diffusion_only: bool = False) -> dict[str, Any]:
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
        extra: dict[str, Any] = {}
        if plan.step_emphasis:
            extra["step_emphasis"] = list(plan.step_emphasis)
        from frontier.engine import SerendipityInjector

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
) -> tuple[Path, str]:
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


def _apply_frontier(args: Any, *, steps: int, diffusion_only: bool = False) -> dict[str, Any]:
    if getattr(args, "frontier_perfect", False) or getattr(args, "frontier_subject", False):
        return {}
    if getattr(args, "frontier_creative", False):
        return {}
    if not getattr(args, "frontier", False):
        return {}
    from frontier.engine import apply_frontier_to_args, frontier_diffusion_hooks

    dial = float(getattr(args, "frontier_serendipity", 0.25) or 0.25)
    if diffusion_only and getattr(args, "frontier_plan", None) is not None:
        plan = args.frontier_plan
    else:
        from frontier.engine import FrontierEngine

        eng = FrontierEngine(num_steps=steps, serendipity_dial=dial)
        plan = apply_frontier_to_args(args, engine=eng)
        if getattr(args, "frontier_auto_resolve", False):
            from frontier.engine import ContradictionScanner

            args.prompt = ContradictionScanner().suggest_rewrite(args.prompt or "")
    hooks = frontier_diffusion_hooks(plan)
    extra: dict[str, Any] = {}
    if hooks.get("serendipity_scales"):
        extra["step_noise_scales"] = hooks["serendipity_scales"]
    if hooks.get("entropy_per_step"):
        extra["step_noise_scales"] = hooks.get("entropy_per_step")
    if not diffusion_only:
        witness_cfg = getattr(args, "cfg_scale", None)
        if plan.witness and plan.witness.cfg_bias and witness_cfg:
            args.cfg_scale = float(witness_cfg) * float(plan.witness.cfg_bias)
    return extra


def _apply_character_session(args: Any) -> dict[str, Any]:
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


def _apply_attention_layout(args: Any, *, steps: int) -> dict[str, Any]:
    if not getattr(args, "box_attn_layout", False):
        return {}
    box_spec = getattr(args, "_box_layout_spec", None)
    if box_spec is None:
        return {}
    from frontier.layout import build_attention_layout_plan

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


def load_character_session(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"character session not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("character session must be a JSON object")
    return data


def save_character_session(path: str | Path, data: dict[str, Any]) -> None:
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
) -> tuple[Any, str]:
    """Return (mask tensor 1,1,H,W), regional prompt for ``region_name``."""
    import torch

    from utils.generation.regional import build_latent_region_masks

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


@dataclass
class AutoRefineResult:
    best_index: int
    scores: list[float]
    outputs: list[str]
    prompts: list[str] = field(default_factory=list)


def _replace_or_append_prompt(cmd: list[str], prompt: str) -> list[str]:
    """Return a copy of ``cmd`` with ``--prompt`` value replaced or appended."""
    out: list[str] = []
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
    Run ``sample.py`` multiple times; pick the best with the model-free ``heuristic`` ranker preset.

    When ``prompt_variants`` is set, each candidate uses a different prompt
    (e.g. from ``--creative-mutate``). Otherwise only seed offsets vary.
    """
    from PIL import Image

    from utils.agentic.quality_loop import CompositeRanker

    outputs: list[str] = []
    prompts_used: list[str] = []
    images: list[np.ndarray] = []
    variants = list(prompt_variants or [])
    for i in range(max(1, int(num_candidates))):
        out = f"{out_stem}_{i}.png"
        cmd = list(sample_cmd)
        if variants:
            cmd = _replace_or_append_prompt(cmd, variants[i % len(variants)])
            prompts_used.append(variants[i % len(variants)])
        else:
            prompts_used.append(_extract_prompt_from_cmd(cmd) or "")
        cmd = cmd + ["--seed", str(seed_base + i), "--out", out]
        subprocess.run(cmd, check=True)
        outputs.append(out)
        images.append(np.array(Image.open(out).convert("RGB")))
    best_i, scores = CompositeRanker("heuristic").pick_best_index(
        images, prompt=_extract_prompt_from_cmd(sample_cmd) or "", device="cpu"
    )
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
    from models.conditioning import PromptParser, attribute_binding_heatmap
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


def append_benchmark_history(leaderboard_path: Path, history_path: Path) -> dict[str, Any]:
    """Append ``leaderboard.json`` snapshot to rolling history file."""
    lb = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    hist: list[Any] = []
    if history_path.is_file():
        hist = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(hist, list):
            hist = []
    import datetime as _dt

    entry = {"timestamp": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"), "leaderboard": lb}
    hist.append(entry)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    return entry


_COUNT_RE = re.compile(
    r"\b(exactly\s+\d+|\d+\s+(people|persons|person|characters?|girls?|boys?|coins?|candles?|windows?))\b"
)


@dataclass
class PromptExtras:
    """Negatives and flags gathered while composing the prompt, consumed by the prompt stack."""

    character_negative: str = ""
    scene_negative: str = ""
    scale_distortion: bool = False


@dataclass
class PromptPlan:
    """Final text for the encoders plus what later phases read from the prompt."""

    prompt: str
    negative: str
    emphasis_segments: list = field(default_factory=list)
    num_gen: int = 1
    expected_texts: list = field(default_factory=list)
    photo_grade: str = "none"
    photo_filter: str = "none"
    photo_grain: str = "none"
    photo_pack: str = "none"
    is_photo_prompt: bool = False
    auto_pick_metric: str = ""

    @property
    def expected_ocr(self) -> str:
        return str(self.expected_texts[0]) if self.expected_texts else ""

    @property
    def has_count(self) -> bool:
        return bool(_COUNT_RE.search(self.prompt.lower()))


def _join(*parts: str, sep: str = ", ") -> str:
    return sep.join(p for p in parts if p)


def load_box_layout(args: Any, prompt: str) -> str:
    """Load ``--box-layout`` into ``args._box_layout_spec``; returns the prompt with its global text merged."""
    try:
        from utils.generation.regional import layout_text_from_regions, load_box_layout_file

        spec = load_box_layout_file(str(args.box_layout).strip())
    except Exception as e:
        print(f"Warning: --box-layout failed: {e}", file=sys.stderr)
        return prompt
    args._box_layout_spec = spec
    layout_line = layout_text_from_regions(spec)
    if spec.global_prompt and spec.global_prompt.lower() not in prompt.lower():
        prompt = _join(spec.global_prompt, prompt)
    if not prompt:
        prompt = layout_line
    elif str(getattr(args, "box_layout_mode", "regional_cfg") or "regional_cfg").lower() == "text_only":
        prompt = f"{prompt}. {layout_line}"
    if spec.global_negative and not getattr(args, "negative_prompt", "").strip():
        args.negative_prompt = spec.global_negative
    print(f"Box layout: {len(spec.regions)} region(s) ({', '.join(r.name for r in spec.regions)})", file=sys.stderr)
    return prompt


def compose_user_prompt(args: Any, parser: Any) -> None:
    """Build the user's prompt from --prompt / --prompt-file / --tags / --lora-trigger / layouts; screen it."""
    has_tags = bool(getattr(args, "tags", "").strip() or getattr(args, "tags_file", "").strip())
    has_prompt_file = bool(getattr(args, "prompt_file", "").strip())
    has_prompt_layout = bool(getattr(args, "prompt_layout", "").strip())
    has_cast = bool(str(getattr(args, "cast_json", "") or "").strip())
    if not (args.prompt or has_tags or has_prompt_file or has_prompt_layout or args.box_layout.strip() or has_cast):
        parser.error(
            "Provide at least one of --prompt, --prompt-file, --tags, --tags-file, --prompt-layout, --box-layout "
            "or --cast-json"
        )

    prompt = (args.prompt or "").strip()
    if has_prompt_file:
        pf_path = Path(args.prompt_file)
        if pf_path.exists():
            prompt = pf_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not prompt:
                print(f"Warning: --prompt-file is empty: {pf_path}", file=sys.stderr)
    args._raw_prompt_before_compose = prompt
    tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()]
    if getattr(args, "tags_file", "").strip():
        tags_path = Path(args.tags_file)
        if tags_path.exists():
            raw = tags_path.read_text(encoding="utf-8", errors="ignore")
            tags += [t.strip() for line in raw.strip().split("\n") for t in line.split(",") if t.strip()]
        else:
            print(f"Warning: --tags-file not found: {tags_path}", file=sys.stderr)
    if tags:
        from data.caption_utils import prompt_from_tags

        prompt = _join(prompt_from_tags(tags), prompt)
    if getattr(args, "lora_trigger", "").strip() and args.lora:
        prompt = _join(args.lora_trigger.strip(), prompt)
    # A cast can supply the box layout, character sheets and moodboard, so it runs before they are read.
    args.prompt = prompt
    _apply_cast_and_corpus(args)
    prompt = str(args.prompt or "")

    args._prompt_layout_negative = ""
    args._used_prompt_layout = False
    args._layout_compiled = None
    args._box_layout_spec = None
    args._regional_cfg_plan = None
    if args.box_layout.strip():
        prompt = load_box_layout(args, prompt)
    if has_prompt_layout:
        try:
            from utils.prompt.text import load_prompt_layout_file, merge_prompt_with_layout

            compiled = load_prompt_layout_file(str(args.prompt_layout).strip())
            args._prompt_layout_negative = compiled.negative or ""
            args._used_prompt_layout = True
            args._layout_compiled = compiled
            prompt = merge_prompt_with_layout(compiled.positive, prompt, layout_first=True)
        except Exception as e:
            print(f"Warning: --prompt-layout failed: {e}", file=sys.stderr)

    if getattr(args, "subject_first", False) and prompt and not args._used_prompt_layout:
        from data.caption_utils import prompt_from_tags

        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        if len(parts) > 1:
            prompt = prompt_from_tags(parts)
    if prompt:
        args.prompt = prompt
    require_allowed(str(args.prompt or ""), str(getattr(args, "negative_prompt", "") or ""))
    # The prompt stack's token budget never drops these; everything added later competes for the rest.
    args._user_prompt = str(args.prompt or "")
    args._user_negative = str(getattr(args, "negative_prompt", "") or "")
    _expand_artist_mentions(args)


def _apply_cast_and_corpus(args: Any) -> None:
    """``--cast-json`` (multi-character cast) and ``--corpus-jsonl`` (retrieved captions + moodboard)."""
    try:
        from utils.generation.inventions import apply_cast_and_corpus_to_namespace

        apply_cast_and_corpus_to_namespace(args, work_dir="outputs/_invention")
    except Exception as e:
        print(f"Warning: --cast-json / --corpus-jsonl skipped: {e}", file=sys.stderr)


def _expand_artist_mentions(args: Any) -> None:
    """``@artist`` style mentions and ``+category`` blocks (characters, buildings, vehicles, ...)."""
    raw = str(getattr(args, "prompt", "") or "")
    if not raw:
        return
    strength = float(getattr(args, "artist_strength", 1.0) or 1.0)
    if getattr(args, "prompt_compose", True):
        if "@" not in raw and "+" not in raw:
            return
        from utils.prompt.text import compose_prompt

        idx = str(getattr(args, "artist_index", "") or "").strip() or None
        composed = compose_prompt(raw, artist_strength=strength, artist_index=idx)
        args.prompt = composed.positive
        if composed.artists:
            print(f"Artist style: {', '.join(composed.artists)}")
        if composed.blocks:
            print(f"Prompt blocks: {', '.join(sorted(composed.blocks))}")
    elif getattr(args, "artist_style", True) and "@" in raw:
        from utils.prompt.text import expand_artist_mentions

        args.prompt, artists = expand_artist_mentions(raw, strength=strength)
        if artists:
            print(f"Artist style: {', '.join(artists)}")


def compose_prompt_additions(args: Any) -> PromptExtras:
    """Character, scene and style additions; style packs go after the user's text so the budget can gate them."""
    extras = PromptExtras()
    anatomy = _parse_scale_csv(getattr(args, "anatomy_scale", ""))
    objects = _parse_scale_csv(getattr(args, "object_scale", ""))
    scene = _parse_scale_csv(getattr(args, "scene_scale", ""))
    extras.scale_distortion = bool(anatomy or objects or scene)
    if getattr(args, "gender_swap", False) and getattr(args, "prompt", ""):
        args.prompt = _apply_gender_swap(args.prompt)
    args.prompt = _join(getattr(args, "prompt", "") or "", _build_size_tokens(anatomy, objects, scene))

    auto_oc_negative = _auto_original_character(args)
    char_pos, char_neg = _character_sheets(args)
    char_neg = _join(char_neg, str(getattr(args, "character_negative_extra", "") or "").strip())
    char_neg = _join(char_neg, auto_oc_negative)
    char_pos = _join(char_pos, str(getattr(args, "character_prompt_extra", "") or "").strip())
    extras.character_negative = char_neg
    args.prompt = _join(args.prompt, char_pos)

    if getattr(args, "scene_blueprint", "").strip():
        try:
            from utils.prompt.scene import load_scene_blueprint

            scene_pos, extras.scene_negative = load_scene_blueprint(
                args.scene_blueprint, strength=float(getattr(args, "scene_blueprint_strength", 1.0))
            )
            args.prompt = _join(args.prompt, scene_pos)
        except Exception as e:
            print(f"Warning: failed to load --scene-blueprint: {e}", file=sys.stderr)

    args.prompt = _join(args.prompt, *_style_packs(args))

    if getattr(args, "originality", 0.0) and args.prompt:
        try:
            from utils.prompt.art_direction import inject_originality_tokens

            strength = max(0.0, min(1.0, float(args.originality)))
            rng = np.random.default_rng(int(args.seed) + int(1000 * strength))
            args.prompt = inject_originality_tokens(args.prompt, strength, rng)
            if getattr(args, "creativity", None) is None:
                args.creativity = strength
        except Exception:
            pass
    return extras


def _auto_original_character(args: Any) -> str:
    """Append a synthesised original-character block when the prompt asks for one; returns its negative."""
    if not (getattr(args, "auto_original_character", True) and getattr(args, "prompt", "")):
        return ""
    try:
        from utils.prompt.scene import infer_auto_original_character

        profile = infer_auto_original_character(
            str(args.prompt),
            seed=int(getattr(args, "seed", 0)) + int(getattr(args, "auto_oc_seed_offset", 0)),
            style_context=", ".join(
                str(getattr(args, k, "") or "") for k in ("style", "hard_style", "preset", "op_mode")
            ),
        )
    except Exception:
        return ""
    if profile is None or str(getattr(args, "character_sheet", "") or "").strip():
        return ""
    block = profile.to_prompt_block()
    if block:
        args.prompt = f"{args.prompt}, {block}".strip(", ")
    return str(getattr(profile, "negative_block", "") or "").strip()


def _character_sheets(args: Any) -> tuple[str, str]:
    """(positive, negative) additions from ``--character-sheet`` files, labelled when several share a scene."""
    sheet_paths = [s.strip() for s in str(getattr(args, "character_sheet", "") or "").split(",") if s.strip()]
    pos_all, neg_all = [], []
    for sp in sheet_paths:
        try:
            pos_i, neg_i = _load_character_sheet(
                sp,
                uncensored_mode=bool(getattr(args, "uncensored_mode", False)),
                character_strength=float(getattr(args, "character_strength", 1.0)),
            )
        except Exception as e:
            print(f"Warning: failed to load --character-sheet '{sp}': {e}", file=sys.stderr)
            continue
        if pos_i:
            pos_all.append(pos_i)
        if neg_i:
            neg_all.append(neg_i)
    use_labels = bool(getattr(args, "label_multi_character_sheets", False)) or (
        len(sheet_paths) >= 2 and str(getattr(args, "composition_mode", "none") or "none") == "multi_character"
    )
    if use_labels and len(pos_all) >= 2:
        from utils.prompt.scene import merge_character_sheet_positives, multi_sheet_extra_negatives_csv
        from utils.prompt.text import merge_csv_unique

        return (
            merge_character_sheet_positives(pos_all),
            _join(merge_csv_unique(*neg_all), multi_sheet_extra_negatives_csv()),
        )
    return ", ".join(pos_all).strip(", "), ", ".join(neg_all).strip(", ")


def _style_packs(args: Any) -> list[str]:
    """Text implied by --hard-style, --naturalize(-deep), --anti-bleed and --diversity."""
    from config.prompts import (
        CONCEPT_BLEEDING_POSITIVE,
        DIVERSITY_POSITIVE,
        HARD_STYLE_RECOMMENDED_PROMPTS,
        NATURAL_LOOK_POSITIVE,
        NATURAL_LOOK_POSITIVE_DEEP,
    )

    packs = []
    if getattr(args, "hard_style", None):
        packs.append(HARD_STYLE_RECOMMENDED_PROMPTS.get(args.hard_style, [None])[0] or "")
    if getattr(args, "naturalize", False):
        packs.append(NATURAL_LOOK_POSITIVE_DEEP if getattr(args, "naturalize_deep", False) else NATURAL_LOOK_POSITIVE)
    if getattr(args, "anti_bleed", False):
        packs.append(CONCEPT_BLEEDING_POSITIVE)
    if getattr(args, "diversity", False):
        packs.append(DIVERSITY_POSITIVE)
    return packs


def build_prompt_plan(args: Any, m: Any, extras: PromptExtras) -> PromptPlan:
    """Feature, enrichment and prompt-stack passes; returns the text the encoders see."""

    apply_sample_feature_prompt_phase(args, steps=int(getattr(args, "steps", 50) or 50))
    if bool(getattr(args, "apply_user_taste", False)):
        from utils.generation.planners import apply_taste_to_prompts, load_user_taste

        pos, neg = apply_taste_to_prompts(
            str(getattr(args, "prompt", "") or ""), str(getattr(args, "negative_prompt", "") or ""), load_user_taste()
        )
        args.prompt = pos
        if neg:
            args.negative_prompt = neg
    _apply_invention_stack(args)

    num_gen = max(1, getattr(args, "num", 1))
    if getattr(args, "explore_styles", False) and num_gen < int(getattr(args, "invent_styles", 3) or 3):
        num_gen = int(args.invent_styles)
        args.num = num_gen
    if str(getattr(args, "pick_best", "none") or "none").strip().lower() not in ("none", "") and num_gen < 3:
        num_gen = 3
        args.num = num_gen

    if "(" in args.prompt or "[" in args.prompt:
        prompt, emphasis = parse_prompt_emphasis(args.prompt)
    else:
        prompt, emphasis = args.prompt, []
    if getattr(args, "expand_prompt", False) and prompt.strip() and not getattr(args, "_skip_prompt_expansion", False):
        from utils.agentic.quality_loop import expand_prompt_heuristic

        prompt = expand_prompt_heuristic(prompt)
        print("Superior: expanded prompt heuristically.", file=sys.stderr)
    hm = str(getattr(args, "human_made", "none") or "none").lower().strip()
    if hm not in ("none", "off", "0", ""):
        from utils.quality.finishing import append_human_made_prompt_fragments

        prompt, args.negative_prompt = append_human_made_prompt_fragments(
            prompt, str(getattr(args, "negative_prompt", "") or ""), hm
        )
    prompt = _ground_and_design(args, prompt, m.device)
    if getattr(args, "boost_quality", False) and prompt.strip():
        from data.caption_utils import QUALITY_PREFIX

        if not prompt.strip().lower().startswith("masterpiece"):
            prompt = f"{QUALITY_PREFIX}{prompt}".strip()
    prompt, num_gen = _multi_instance(args, prompt, num_gen)
    prompt = _detailed_scene(args, prompt)
    prompt = _fix_region(args, prompt, m.image_size)

    expected_texts = _parse_expected_texts(getattr(args, "expected_text", ""))
    if not expected_texts and bool(getattr(args, "auto_expected_text", True)):
        expected_texts = _infer_expected_texts_from_prompt(prompt)
        if expected_texts:
            print(f"Inferred expected text from prompt: {expected_texts}", file=sys.stderr)
    if bool(getattr(args, "auto_constraint_boost", True)) and (expected_texts or _expected_count(args, prompt) > 0):
        if num_gen < 4:
            print(f"Auto constraint boost: num {num_gen} -> 4", file=sys.stderr)
            num_gen = 4
    if getattr(args, "ocr_fix", False) and expected_texts:
        args.text_in_image = True
        prompt = _maybe_append_text_says(prompt, expected_texts)
    args.prompt = prompt

    from utils.prompt.stack import apply_sample_prompt_stack

    prompt, negative = apply_sample_prompt_stack(
        args,
        prompt,
        character_negative_additions=extras.character_negative,
        scene_negative_additions=extras.scene_negative,
        apply_scale_distortion=extras.scale_distortion,
        tokenizer=m.tokenizer,
        max_tokens=TEXT_MAX_TOKENS,
    )
    args.prompt = prompt
    # The stack's photo-realism stage decides these, so they are read only after it ran.
    return PromptPlan(
        prompt=prompt,
        negative=negative,
        emphasis_segments=emphasis,
        num_gen=num_gen,
        expected_texts=expected_texts,
        photo_grade=str(getattr(args, "_pr_grade_ef", "none") or "none"),
        photo_filter=str(getattr(args, "_pr_filter_ef", "none") or "none"),
        photo_grain=str(getattr(args, "_pr_grain_ef", "none") or "none"),
        photo_pack=str(getattr(args, "_pr_pack_ef", "none") or "none"),
        is_photo_prompt=bool(getattr(args, "_is_photo_prompt", False)),
        auto_pick_metric=str(getattr(args, "_auto_pick_metric", "") or ""),
    )


def _apply_invention_stack(args: Any) -> None:
    """``--invention-stack`` then ``--hard-prompts`` (counts, relations, binding, text); a box layout either
    plans is loaded here because the user's layout was read earlier."""
    before = str(getattr(args, "box_layout", "") or "")
    try:
        from utils.generation.inventions import apply_invention_to_namespace

        apply_invention_to_namespace(args)
    except Exception as e:
        print(f"Warning: invention stack skipped: {e}", file=sys.stderr)
    try:
        from utils.prompt.realism import apply_hard_prompts_to_args

        apply_hard_prompts_to_args(args)
    except Exception as e:
        print(f"Warning: hard-prompt plan skipped: {e}", file=sys.stderr)
    if str(getattr(args, "box_layout", "") or "") != before and getattr(args, "_box_layout_spec", None) is None:
        args.prompt = load_box_layout(args, str(args.prompt or ""))


def _expected_count(args: Any, prompt: str) -> int:
    """``--expected-count`` or the people / object count the prompt asks for."""
    count = int(getattr(args, "expected_count", 0) or 0)
    if count > 0:
        return count
    from utils.quality.test_time_pick import infer_expected_object_count, infer_expected_people_count

    target = str(getattr(args, "expected_count_target", "auto") or "auto").lower().strip()
    if target == "objects":
        return infer_expected_object_count(prompt)[0]
    count = infer_expected_people_count(prompt)
    if count <= 0 and target == "auto":
        count = infer_expected_object_count(prompt)[0]
    return count


def _ground_and_design(args: Any, prompt: str, device: Any) -> str:
    """Local RAG, agentic facts, composition brief, visual-design packs and creative RAG."""
    local_rag = str(getattr(args, "local_rag_jsonl", "") or "").strip()
    if local_rag:
        try:
            from utils.agentic.quality_loop import SuperiorPromptStack

            stack = SuperiorPromptStack(rag_jsonl=local_rag, rag_top_k=int(getattr(args, "local_rag_top_k", 8) or 8))
            prompt = stack.enrich(prompt)
            print(f"Local RAG: enriched prompt from {local_rag}", file=sys.stderr)
        except Exception as e:
            print(f"Local RAG skipped: {e}", file=sys.stderr)
    prompt = _agentic_facts(args, prompt)

    brief = str(getattr(args, "composition_brief", "off") or "off")
    if brief != "off" and prompt.strip():
        try:
            from utils.prompt.art_direction import apply_composition_brief

            prompt = apply_composition_brief(prompt, brief)  # type: ignore[arg-type]
        except Exception as e:
            print(f"Composition brief skipped: {e}", file=sys.stderr)

    args._visual_design_negative = ""
    vd_preset = str(getattr(args, "visual_design_preset", "") or "").strip().lower()
    if vd_preset and prompt.strip():
        try:
            from utils.visual_design import apply_visual_design_preset_to_prompt

            prompt, args.visual_design_domain, args.visual_design_intensity = apply_visual_design_preset_to_prompt(
                prompt, vd_preset
            )
        except Exception as e:
            print(f"Visual design preset skipped: {e}", file=sys.stderr)
    vd_domain = str(getattr(args, "visual_design_domain", "none") or "none").lower().strip()
    if vd_domain != "none" and prompt.strip():
        try:
            from utils.visual_design import apply_visual_design_stage

            vd = apply_visual_design_stage(
                prompt,
                cli_domain=vd_domain,
                intensity=str(getattr(args, "visual_design_intensity", "standard") or "standard"),
                use_negative_pack=bool(getattr(args, "visual_design_negative_pack", False)),
                emit=lambda msg: print(msg, file=sys.stderr),
            )
            prompt = vd.prompt
            if vd.negative_addon:
                args._visual_design_negative = vd.negative_addon
        except Exception as e:
            print(f"Visual design pack skipped: {e}", file=sys.stderr)
    if getattr(args, "creative_rag", False) and prompt.strip():
        prompt = _creative_rag(args, prompt, device)
    return prompt


def _agentic_facts(args: Any, prompt: str) -> str:
    """Merge Gen-Searcher JSON / JSONL facts (``--agentic-facts-json``) into the prompt."""
    facts_path = str(getattr(args, "agentic_facts_json", "") or "").strip()
    if not facts_path:
        return prompt
    try:
        from utils.prompt.rag import load_facts_from_gen_searcher_json, load_facts_from_jsonl, merge_facts_into_prompt

        mode = str(getattr(args, "agentic_facts_format", "auto") or "auto").lower().strip()
        max_facts = max(1, int(getattr(args, "agentic_max_facts", 16) or 16))
        facts: list = []
        if mode in ("auto", "gen_searcher"):
            facts = load_facts_from_gen_searcher_json(facts_path, max_entries=max_facts)
        if not facts and mode in ("auto", "jsonl_text"):
            facts = load_facts_from_jsonl(facts_path, max_entries=max_facts)
        if not facts:
            print(f"Agentic grounding: no facts found in {facts_path}", file=sys.stderr)
            return prompt
        max_chars = max(256, int(getattr(args, "agentic_facts_max_chars", 2400) or 2400))
        print(f"Agentic grounding merged {len(facts)} facts from {facts_path}", file=sys.stderr)
        return merge_facts_into_prompt(prompt, facts, max_chars=max_chars)
    except Exception as e:
        print(f"Warning: agentic fact grounding failed: {e}", file=sys.stderr)
        return prompt


def _creative_rag(args: Any, prompt: str, device: Any) -> str:
    """Semantic conflict resolution plus intent-aware enrichment (moondream2 / Qwen2.5 when available)."""
    try:
        from utils.prompt.rag import CreativeRAGOptimizer, classify_prompt_intent, resolve_semantic_conflicts

        level = float(getattr(args, "creative_rag_level", 0.7) or 0.7)
        refs = [str(getattr(args, "creative_rag_image", "") or "").strip()]
        refs += str(getattr(args, "creative_rag_images", "") or "").split(",")
        refs = list(dict.fromkeys(r.strip() for r in refs if r.strip()))[:16]
        if bool(getattr(args, "creative_rag_resolve_conflicts", True)):
            resolved, removed = resolve_semantic_conflicts(prompt)
            if removed:
                print(
                    f"Creative RAG: resolved {len(removed)} semantic contradiction(s): "
                    + ", ".join(f"'{r}'" for r in removed[:4]),
                    file=sys.stderr,
                )
                prompt = resolved
        intent = classify_prompt_intent(prompt)
        print(f"Creative RAG: detected intent={intent}, creativity_level={level:.2f}", file=sys.stderr)
        if refs:
            print(
                f"Creative RAG: reference images={len(refs)} (facts/dissection up to 16; moondream up to 8)",
                file=sys.stderr,
            )
        result = CreativeRAGOptimizer(device=str(device)).optimize(
            prompt,
            reference_image_paths=refs or None,
            creativity_level=level,
            optimization_level="balanced",
            seed=int(getattr(args, "seed", 42)),
            use_rag=True,
        )
        enriched = str(result.get("optimized_prompt", "") or "")
        if enriched.strip():
            added = len(enriched.split(",")) - len(prompt.split(","))
            fallback = getattr(result.get("rag_result"), "fallback_used", True)
            print(
                f"Creative RAG ({'semantic fallback' if fallback else 'Qwen2.5 synthesis'}): "
                f"added {max(0, added)} enrichment(s) [intent={intent}]",
                file=sys.stderr,
            )
            prompt = enriched
    except Exception as e:
        print(f"Creative RAG failed (non-fatal): {e}", file=sys.stderr)
    return prompt


def _multi_instance(args: Any, prompt: str, num_gen: int) -> tuple[str, int]:
    """``--multi-instance-preset`` (+ ``--multi-instance-auto``) for crowds of near-identical subjects."""
    args._multi_instance_negative = ""
    preset = str(getattr(args, "multi_instance_preset", "none") or "none").lower().strip()
    if preset in ("none", "") or not prompt.strip():
        return prompt, num_gen
    try:
        from utils.prompt.scene import apply_multi_instance_preset, describe_limitation_short

        user_n = max(0, int(getattr(args, "multi_instance_count", 0) or 0))
        prompt, neg, min_n, expected, note = apply_multi_instance_preset(prompt, preset, user_expected_count=user_n)
        if neg:
            args._multi_instance_negative = neg
        if min_n > num_gen:
            num_gen = min_n
            args.num = num_gen
            print(
                f"Multi-instance preset '{preset}': raised --num to {num_gen} for broader search "
                f"({describe_limitation_short()})",
                file=sys.stderr,
            )
        if user_n > 0:
            args.expected_count = user_n
        elif expected is not None and expected > 0 and int(getattr(args, "expected_count", 0) or 0) <= 0:
            args.expected_count = int(expected)
        if note:
            print(f"Multi-instance: {note}", file=sys.stderr)
    except Exception as e:
        print(f"Multi-instance preset skipped: {e}", file=sys.stderr)
    if not bool(getattr(args, "multi_instance_auto", False)):
        return prompt, num_gen
    try:
        from utils.prompt.art_direction import apply_composition_brief
        from utils.prompt.scene import multi_instance_auto_settings, print_multi_instance_hints

        st = multi_instance_auto_settings(preset)
        if st.get("composition_brief_boost") and str(getattr(args, "composition_brief", "off")).lower() == "off":
            args.composition_brief = "auto"
            prompt = apply_composition_brief(prompt, "auto")
            print("Multi-instance auto: composition-brief -> auto", file=sys.stderr)
        if str(getattr(args, "pick_best", "none") or "none").strip().lower() in ("none", "") and st.get("pick_best"):
            args.pick_best = str(st["pick_best"])
            print(f"Multi-instance auto: pick-best -> {st['pick_best']}", file=sys.stderr)
        target = str(getattr(args, "expected_count_target", "auto") or "auto").lower().strip()
        if (
            int(getattr(args, "multi_instance_count", 0) or 0) > 0
            and target == "auto"
            and st.get("expected_count_target")
        ):
            args.expected_count_target = str(st["expected_count_target"])
            print(f"Multi-instance auto: expected-count-target -> {st['expected_count_target']}", file=sys.stderr)
        if str(getattr(args, "pick_best", "none") or "").strip().lower() not in ("none", "") and num_gen < 2:
            num_gen = 2
            args.num = num_gen
            print("Multi-instance auto: --num raised to >= 2 because pick-best is active", file=sys.stderr)
        print_multi_instance_hints(preset)
    except Exception as e:
        print(f"Multi-instance auto skipped: {e}", file=sys.stderr)
    return prompt, num_gen


def _detailed_scene(args: Any, prompt: str) -> str:
    """``--detailed-scene-boost``: restate the key clauses of long scene prompts."""
    args._detailed_scene_negative = ""
    mode = str(getattr(args, "detailed_scene_boost", "auto") or "auto")
    strength = str(getattr(args, "detailed_scene_strength", "lite") or "lite")
    if mode == "off" or not prompt.strip():
        return prompt
    try:
        from utils.prompt.scene import apply_detailed_scene_boost, detailed_scene_warrants_boost, extract_key_segments

        if mode == "on" or (mode == "auto" and detailed_scene_warrants_boost(prompt)):
            prompt, neg = apply_detailed_scene_boost(prompt, "on", strength=strength)  # type: ignore[arg-type]
            if neg:
                args._detailed_scene_negative = neg
            segs = extract_key_segments(prompt)
            if segs:
                print(
                    f"Detailed-scene boost ({strength}): {len(segs)} key clauses (e.g. {segs[:3]!r})", file=sys.stderr
                )
    except Exception as e:
        print(f"Detailed-scene boost skipped: {e}", file=sys.stderr)
    return prompt


def _fix_region(args: Any, prompt: str, image_size: int) -> str:
    """``--fix-region NAME`` with ``--init-image`` + ``--box-layout``: inpaint that box with its own prompt."""
    region = str(getattr(args, "fix_region", "") or "").strip()
    box_spec = getattr(args, "_box_layout_spec", None)
    if not (region and box_spec is not None and args.init_image and not args.init_latent and not args.mask):
        return prompt
    try:
        import tempfile

        mask_path, region_prompt = write_fix_region_mask(
            box_spec,
            region,
            image_size=image_size,
            out_path=Path(tempfile.gettempdir()) / f"sdx_fix_{region}.png",
        )
        args.mask = str(mask_path)
        args.inpaint_mode = "mdm"
        print(f"Fix-region: inpainting {region!r} via {args.mask}", file=sys.stderr)
        return _join(prompt, region_prompt)
    except Exception as e:
        print(f"Warning: --fix-region failed: {e}", file=sys.stderr)
        return prompt
