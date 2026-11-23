"""Sampler presets / OP modes, the prompt-routed quality policy, and the agentic + quality-flywheel stack defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.safety.minor_guard import is_sexual
from utils.safety.minor_guard import reasons as minor_guard_reasons


# Sampler presets (``sample.py --preset``) and OP modes (``--op-mode``).
@dataclass(slots=True)
class SamplerPreset:
    name: str
    description: str
    cfg_scale: float | None = None
    cfg_rescale: float | None = None
    scheduler: str | None = None  # ays_dit | ays | ddim | …
    solver: str | None = None  # dpmpp_2m | unipc | ddim | …
    flow_solver: str | None = None
    flow_schedule: str | None = None
    steps: int | None = None
    hard_style: str | None = None  # "3d" | "realistic" | "3d_realistic" | "style_mix"
    naturalize: bool | None = None
    naturalize_grain: float | None = None
    anti_bleed: bool | None = None
    diversity: bool | None = None
    anti_artifacts: bool | None = None
    strong_watermark: bool | None = None


PRESETS: dict[str, SamplerPreset] = {
    "sdxl": SamplerPreset(
        name="sdxl",
        description="SDXL-style preset: photorealistic, strong adherence, natural look.",
        cfg_scale=6.5,
        cfg_rescale=0.3,
        scheduler="ays_dit",
        solver="dpmpp_2m",
        steps=28,
        hard_style="realistic",
        naturalize=True,
        naturalize_grain=0.015,
        anti_bleed=True,
        diversity=False,
        anti_artifacts=True,
        strong_watermark=True,
    ),
    "flux": SamplerPreset(
        name="flux",
        description="Flux-style preset: realistic/photographic; avoids grid artifact and burn.",
        cfg_scale=3.5,
        cfg_rescale=0.0,
        scheduler="ays_dit",
        solver="dpmpp_2m",
        flow_solver="dpmpp_2m",
        flow_schedule="ays",
        steps=24,
        hard_style="realistic",
        naturalize=True,
        naturalize_grain=0.02,
        anti_bleed=True,
        diversity=True,
        anti_artifacts=True,
        strong_watermark=True,
    ),
    "anime": SamplerPreset(
        name="anime",
        description="Anime/stylized preset: strong quality tags, semi-realistic/2.5D support.",
        cfg_scale=7.0,
        cfg_rescale=0.0,
        scheduler="ays_dit",
        solver="dpmpp_2m",
        flow_solver="heun",
        flow_schedule="ays",
        steps=28,
        hard_style="style_mix",
        naturalize=False,
        naturalize_grain=0.0,
        anti_bleed=True,
        diversity=True,
        anti_artifacts=True,
        strong_watermark=True,
    ),
    "zit": SamplerPreset(
        name="zit",
        description="Z-Image-style preset: higher diversity, less centering, strong composition.",
        cfg_scale=6.0,
        cfg_rescale=0.3,
        scheduler="ays_dit",
        solver="dpmpp_2m",
        steps=32,
        hard_style=None,
        naturalize=True,
        naturalize_grain=0.015,
        anti_bleed=True,
        diversity=True,
        anti_artifacts=True,
        strong_watermark=True,
    ),
    "superior": SamplerPreset(
        name="superior",
        description="Superior Stack preset: strong adherence, natural look, anti-artifact; pair with --pick-best superior_composite.",
        cfg_scale=7.0,
        cfg_rescale=0.3,
        scheduler="ays_dit",
        solver="dpmpp_2m",
        steps=28,
        hard_style="realistic",
        naturalize=True,
        naturalize_grain=0.012,
        anti_bleed=True,
        diversity=True,
        anti_artifacts=True,
        strong_watermark=True,
    ),
    "fast": SamplerPreset(
        name="fast",
        description="Fast few-step preset: UniPC + AYS-DiT (~15 steps).",
        cfg_scale=6.5,
        cfg_rescale=0.3,
        scheduler="ays_dit",
        solver="unipc",
        flow_solver="heun",
        flow_schedule="karras",
        steps=15,
        naturalize=True,
        anti_bleed=True,
        anti_artifacts=True,
    ),
}


OP_MODES: dict[str, dict[str, Any]] = {
    "portrait": {
        "naturalize": True,
        "naturalize_grain": 0.02,
        "anti_bleed": True,
        "diversity": True,
        "anti_artifacts": True,
    },
    "fullbody": {
        "anti_bleed": True,
        "anti_artifacts": True,
    },
    "anime_char": {
        "anti_bleed": True,
        "diversity": True,
        "anti_artifacts": True,
    },
    "fast": {
        "solver": "unipc",
        "scheduler": "ays_dit",
        "steps": 15,
    },
    "quality": {
        "solver": "dpmpp_2m",
        "scheduler": "ays_dit",
        "steps": 28,
    },
}


def apply_preset_to_args(args: Any, preset_name: str) -> None:
    """Fill ``args`` from the preset wherever the value still looks unset (``None``/0 or the parser default)."""
    preset = PRESETS.get(preset_name)
    if preset is None:
        return

    def maybe_set(name: str, value: Any, unset_values: tuple) -> None:
        if value is not None and hasattr(args, name) and getattr(args, name) in unset_values:
            setattr(args, name, value)

    maybe_set("cfg_scale", preset.cfg_scale, (0.0, 7.5, None))
    maybe_set("cfg_rescale", preset.cfg_rescale, (0.0, None))
    maybe_set("scheduler", preset.scheduler, ("ddim", "ays_dit", None))
    maybe_set("solver", preset.solver, ("ddim", "dpmpp_2m", None))
    maybe_set("flow_solver", preset.flow_solver, ("euler", "dpmpp_2m", None))
    maybe_set("flow_schedule", preset.flow_schedule, ("linear", "ays", None))
    maybe_set("steps", preset.steps, (0, 50, None))
    maybe_set("hard_style", preset.hard_style, (None, ""))
    maybe_set("naturalize_grain", preset.naturalize_grain, (0.0, 0.015, None))

    for flag_name in ("naturalize", "anti_bleed", "diversity", "anti_artifacts", "strong_watermark"):
        if getattr(preset, flag_name) and hasattr(args, flag_name):
            setattr(args, flag_name, True)


def apply_op_mode_to_args(args: Any, mode_name: str) -> None:
    """Apply an OP mode on top of any preset: boolean flags are only switched on, other values override."""
    for key, value in OP_MODES.get(mode_name, {}).items():
        if not hasattr(args, key):
            continue
        if isinstance(value, bool):
            if value:
                setattr(args, key, True)
        else:
            setattr(args, key, value)


# Prompt-routed quality policy: per-profile soft defaults that ``quality_stack.apply_best_model_policy`` fills in.
@dataclass(frozen=True, slots=True)
class ProfileRecipe:
    """Soft defaults for one prompt profile (``None`` leaves that arg alone)."""

    name: str
    holy_grail: bool = True
    holy_grail_preset: str = "aggressive"
    cfg_scale: float | None = None
    # Guidance stack: APG (eta 0, reverse momentum -0.5, Sadat et al. ICLR 2025) removes the
    # oversaturating parallel component; FDG weights low frequencies below and high frequencies above
    # the CFG strength (paper's SD-XL setting is w_low=5, w_high=10). APG already handles saturation,
    # so the Lin et al. rescale stays off by default.
    cfg_rescale: float | None = 0.0
    apg_parallel_eta: float | None = 0.0
    apg_momentum_beta: float | None = -0.5
    fdg_low: float | None = 0.6
    fdg_high: float | None = 1.3
    qsilk_micrograin: float | None = 0.12
    steps: int | None = 32
    human_made: str = "strong"
    less_ai: bool = True
    naturalize: bool = True
    naturalize_deep: bool = True
    anti_ai_pack: str = "strong"
    human_media: str | None = "dslr"
    # Keyword-matched packs only; "all" floods every prompt with skin/hands/pixel/vector text.
    shortcomings_mitigation: str = "auto"
    anatomy_guidance: str = "lite"
    diversity: bool = True
    anti_bleed: bool = True
    anti_artifacts: bool = True
    strong_watermark: bool = True
    expand_prompt: bool = False
    anti_slop: bool = True
    photo_realism_prefer: str | None = "documentary"
    hand_mode: str | None = "detailed"
    pose_naturalness: str | None = "natural"
    typography_mode: str | None = None
    text_in_image: bool = False
    ocr_fix: bool = False
    num: int | None = 3
    pick_best: str | None = "superior_composite"
    frontier_subject: bool = False
    superior_self_correct: bool = True
    # Guidance interval in EDM sigma (Kynkäänniemi et al. 2024; their SD-XL band).
    cfg_interval: str | None = "0.28,5.42"
    token_emphasis: bool = True


# Keyed by the names ``match_profile_name`` returns.
PROFILES: dict[str, ProfileRecipe] = {
    "text_render": ProfileRecipe(
        name="text_render",
        holy_grail_preset="balanced",
        cfg_scale=6.5,
        steps=36,
        human_made="standard",
        naturalize=False,
        naturalize_deep=False,
        human_media=None,
        anatomy_guidance="none",
        typography_mode="poster",
        text_in_image=True,
        ocr_fix=True,
        photo_realism_prefer=None,
        hand_mode=None,
        pose_naturalness=None,
        pick_best="ocr",
        num=4,
        frontier_subject=False,
        anti_slop=False,
        fdg_low=0.85,
        fdg_high=1.15,
        cfg_interval=None,
    ),
    "photoreal": ProfileRecipe(
        name="photoreal",
        holy_grail_preset="photoreal",
        cfg_scale=3.8,
        steps=32,
        fdg_high=1.5,
        human_media="dslr",
        photo_realism_prefer="documentary",
        hand_mode="detailed",
        pose_naturalness="natural",
        frontier_subject=True,
        pick_best="combo_vit_realism",
        num=3,
    ),
    "portrait": ProfileRecipe(
        name="portrait",
        holy_grail_preset="photoreal",
        cfg_scale=3.8,
        steps=34,
        fdg_high=1.5,
        human_media="photographic",
        photo_realism_prefer="studio_portrait",
        hand_mode="detailed",
        pose_naturalness="intimate_natural",
        frontier_subject=True,
        pick_best="combo_vit_realism",
        num=3,
    ),
    "anime": ProfileRecipe(
        name="anime",
        holy_grail_preset="anime",
        cfg_scale=5.5,
        cfg_rescale=0.3,
        apg_parallel_eta=None,
        apg_momentum_beta=None,
        fdg_low=0.8,
        fdg_high=1.2,
        steps=28,
        human_made="standard",
        naturalize=False,
        naturalize_deep=False,
        human_media=None,
        anatomy_guidance="strong",
        photo_realism_prefer=None,
        hand_mode="stable",
        pose_naturalness="dynamic_natural",
        anti_slop=False,
        pick_best="combo_vit",
        num=3,
        frontier_subject=True,
    ),
    "pixai": ProfileRecipe(
        name="pixai",
        holy_grail_preset="anime",
        cfg_scale=6.0,
        steps=36,
        human_made="standard",
        naturalize=False,
        anatomy_guidance="strong",
        hand_mode="stable",
        pose_naturalness="dynamic_natural",
        pick_best="combo_vit",
        num=3,
        frontier_subject=True,
    ),
    "illustration": ProfileRecipe(
        name="illustration",
        holy_grail_preset="illustration",
        cfg_scale=6.5,
        steps=30,
        human_made="standard",
        naturalize=False,
        naturalize_deep=False,
        human_media=None,
        photo_realism_prefer=None,
        hand_mode="stable",
        anti_slop=False,
        pick_best="combo_vit",
        num=3,
    ),
    "people": ProfileRecipe(
        name="people",
        holy_grail_preset="aggressive",
        cfg_scale=6.5,
        steps=34,
        photo_realism_prefer="cinematic",
        hand_mode="grip",
        pose_naturalness="dynamic_natural",
        frontier_subject=True,
        pick_best="combo_vit_hq",
        num=4,
    ),
    "occlusion": ProfileRecipe(
        name="occlusion",
        holy_grail_preset="aggressive",
        anatomy_guidance="strong",
        pick_best="combo_occlusion",
        num=3,
        frontier_subject=True,
    ),
    "reflection": ProfileRecipe(
        name="reflection",
        holy_grail_preset="aggressive",
        anatomy_guidance="strong",
        pick_best="combo_reflection",
        num=3,
        frontier_subject=True,
    ),
    "sfw": ProfileRecipe(
        name="sfw",
        holy_grail_preset="photoreal",
        cfg_scale=5.5,
        anatomy_guidance="strong",
        hand_mode="stable",
        pose_naturalness="natural",
        pick_best="combo_vit_realism",
        num=3,
        frontier_subject=True,
    ),
    "nsfw": ProfileRecipe(
        name="nsfw",
        holy_grail_preset="aggressive",
        # Do not force DSLR / human-made / anti-slop photoreal onto hentai or silicon.
        human_made="none",
        less_ai=False,
        naturalize=False,
        naturalize_deep=False,
        anti_ai_pack="none",
        human_media=None,
        photo_realism_prefer=None,
        anti_slop=False,
        anatomy_guidance="strong",
        hand_mode="detailed",
        pose_naturalness="intimate_natural",
        pick_best="combo_contact",
        num=3,
        frontier_subject=True,
    ),
    "bind": ProfileRecipe(
        name="bind",
        holy_grail_preset="aggressive",
        anatomy_guidance="strong",
        pick_best="combo_spatial_bind",
        num=3,
        frontier_subject=True,
    ),
    "default": ProfileRecipe(
        name="default",
        holy_grail_preset="aggressive",
        cfg_scale=6.5,
        steps=32,
        photo_realism_prefer=None,
        hand_mode="stable",
        pose_naturalness="natural",
        pick_best="superior_composite",
        num=3,
        frontier_subject=True,
        anti_slop=True,
    ),
}


@dataclass(slots=True)
class QualityPolicy:
    """Global switches over the per-profile recipes."""

    enabled: bool = True
    aggression: str = "max"  # max | balanced (balanced skips the multi-candidate pick)
    enable_holy_grail: bool = True
    enable_pick_best: bool = True
    enable_ocr_auto: bool = True
    enable_frontier_subject: bool = True
    enable_token_emphasis: bool = True
    enable_cfg_interval: bool = True
    enable_box_attn: bool = True
    profiles: dict[str, ProfileRecipe] = field(default_factory=lambda: dict(PROFILES))


POLICY = QualityPolicy()


_TEXT_CUES = ("text that says", "text says", "sign that says", "storefront", "typograph", "lettering", "[text:")
_SFW_CUES = ("sfw", "safe for work", "work-safe", "family-friendly", "linkedin", "passport photo", "corporate headshot")
_PIXAI_CUES = ("pixai", "tsubaki", "serin", "moonlit garden", "full-body illustration")
_ANIME_CUES = ("anime", "manga", "cel shading", "waifu", "danbooru", "visual novel")
_ILLUSTRATION_CUES = ("illustration", "concept art", "painterly", "comic", "storybook", "watercolor", "oil paint")
_OCCLUSION_CUES = (
    "fence",
    "occlud",
    "chain-link",
    "through the gaps",
    "partly hidden behind",
    "visible through",
    "wine glass",
)
_REFLECTION_CUES = ("reflection", "reflected", "puddle", "mirror surface")
_BIND_COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black", "brown")
_PORTRAIT_CUES = ("portrait", "headshot", "beauty shot", "passport photo")
_PHOTO_CUES = (
    "photo",
    "photoreal",
    "dslr",
    "raw photo",
    "8k photo",
    "cinematic photo",
    "documentary",
    "street photo",
    "hyperreal",
    "realistic",
)
_PEOPLE_CUES = ("person", "people", "man", "woman", "girl", "boy", "hands", "full body", "character")


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(c in text for c in cues)


def match_profile_name(prompt: str, *, style: str = "") -> str:
    """First matching ``PROFILES`` key for this prompt (plain substring cues, checked in priority order)."""
    prompt = prompt or ""
    text = f"{prompt} {style}".lower()
    quoted = '"' in prompt
    # Text rendering needs an explicit quoted string or [text:...] span to draw.
    if "[text:" in text or (quoted and len(prompt) < 400) or ((quoted or "“" in prompt) and _has_any(text, _TEXT_CUES)):
        return "text_render"
    if is_sexual(text) and not minor_guard_reasons(text, training=True):
        return "nsfw"
    if _has_any(text, _SFW_CUES):
        return "sfw"
    if _has_any(text, _PIXAI_CUES):
        return "pixai"
    if _has_any(text, _ANIME_CUES):
        return "anime"
    if _has_any(text, _ILLUSTRATION_CUES):
        return "illustration"
    # Occlusion / reflection / bind before portrait so fence scenes pick physics.
    if _has_any(text, _OCCLUSION_CUES):
        return "occlusion"
    if _has_any(text, _REFLECTION_CUES) or ("chrome" in text and "granite" in text):
        return "reflection"
    if "on the left" in text and "on the right" in text:
        return "bind"
    if "left of" in text and sum(c in text for c in _BIND_COLORS) >= 2:
        return "bind"
    if _has_any(text, _PORTRAIT_CUES):
        return "portrait"
    if _has_any(text, _PHOTO_CUES):
        return "photoreal"
    if _has_any(text, _PEOPLE_CUES):
        return "people"
    return "default"


def recipe_for_prompt(prompt: str, *, style: str = "") -> ProfileRecipe:
    name = match_profile_name(prompt, style=style)
    return POLICY.profiles.get(name, POLICY.profiles["default"])


def soft_set(args: Any, name: str, value: Any, *, unset: tuple[Any, ...]) -> bool:
    """Set ``args.name = value`` only while its current value is in ``unset``; True when applied."""
    if value is None or not hasattr(args, name) or getattr(args, name) not in unset:
        return False
    setattr(args, name, value)
    return True


# Defaults for the agentic generate/reflect loop in ``utils/agentic`` and ``utils/brain``.
@dataclass(slots=True)
class AgenticStackDefaults:
    max_reflect_loops: int = 3
    num_candidates: int = 4
    pick_metric: str = "superior_composite"
    expand_prompt: bool = True
    self_correct: bool = True
    rag_top_k: int = 8
    min_composite_accept: float = 0.62
    min_clip_accept: float = 0.22
    qwen_path: str = ""
    trajectory_variants: int = 3
    extra_sample_args: list[str] = field(
        default_factory=lambda: [
            "--apg-parallel-eta",
            "0",
            "--cfg-zero-star",
            "--qsilk-micrograin",
            "0.1",
            "--human-made",
            "standard",
        ]
    )


# Defaults and plan for the quality flywheel (``utils/agentic/quality_loop.py``, ``run_flywheel``).
@dataclass(slots=True)
class SuperiorStackDefaults:
    """Knobs for the flywheel's curate and auto-improve (best-of-N + DPO + soup) steps."""

    preset: str = "sdxl"
    num_candidates: int = 4
    pick_best: str = "superior_composite"
    steps: int = 28
    cfg_scale: float = 7.0
    expand_prompt: bool = True
    self_correct: bool = True
    compile_inference: bool = False
    local_rag_top_k: int = 8
    dpo_steps: int = 500
    dpo_beta: float = 300.0
    dpo_timestep_weight: str = "high_noise"
    dpo_safeguard: float = 0.85
    dpo_ref_ema_alpha: float = 0.01
    qsilk_micrograin: float = 0.1
    taylor_cache: bool = False
    taylor_cache_order: int = 1
    block_cache_threshold: float = 0.18
    auto_loop_iterations: int = 1
    model_soup: bool = True
    soup_weight_base: float = 0.35
    soup_weight_dpo: float = 0.65
    benchmark_suite_pack: str = "top_contender_proxy_v1"
    preference_min_margin: float = 0.08
    dedup: str = "phash"
    min_caption_len: int = 8
    max_caption_len: int = 512


@dataclass(slots=True)
class FlywheelPlan:
    """End-to-end quality flywheel plan (curate → align → promote)."""

    base_ckpt: str
    work_dir: str = "flywheel_run"
    manifest_in: str | None = None
    manifest_out: str | None = None
    local_rag_jsonl: str | None = None
    vit_ckpt: str | None = None
    promote_path: str = "flywheel_run/best.pt"
    skip_curate: bool = False
    skip_align: bool = False
    defaults: SuperiorStackDefaults = field(default_factory=SuperiorStackDefaults)
    extra_auto_loop_args: list[str] = field(default_factory=list)
