"""
Model presets and OP modes for sample.py.

Each preset is a light-weight bundle of defaults for cfg-scale, cfg-rescale,
scheduler, solver, and helper flags (hard-style, naturalize, anti-bleed, diversity, etc.).

sample.py should treat these as **soft defaults**:
- Only apply a preset value when the user did NOT explicitly set that flag.
- Presets are advisory, not mandatory; they are meant to be safe, high-quality starting points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SamplerPreset:
    name: str
    description: str
    # Core sampler knobs
    cfg_scale: float | None = None
    cfg_rescale: float | None = None
    scheduler: str | None = None  # ays_dit | ays | ddim | …
    solver: str | None = None  # dpmpp_2m | unipc | ddim | …
    flow_solver: str | None = None
    flow_schedule: str | None = None
    steps: int | None = None
    # Hard styles / look controls
    hard_style: str | None = None  # "3d" | "realistic" | "3d_realistic" | "style_mix"
    naturalize: bool | None = None
    naturalize_grain: float | None = None
    anti_bleed: bool | None = None
    diversity: bool | None = None
    anti_artifacts: bool | None = None
    strong_watermark: bool | None = None


PRESETS: dict[str, SamplerPreset] = {
    # SDXL-style: strong prompt adherence, photorealistic by default.
    "sdxl": SamplerPreset(
        name="sdxl",
        description="SDXL-style preset: photorealistic, strong adherence, natural look.",
        cfg_scale=6.5,
        cfg_rescale=0.7,
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
    # Flux/Klein-style realistic photography; keep CFG modest to avoid grid/burn.
    "flux": SamplerPreset(
        name="flux",
        description="Flux-style preset: realistic/photographic; avoids grid artifact and burn.",
        cfg_scale=3.5,
        cfg_rescale=0.7,
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
    # Anime/stylized preset; assumes quality tags and hard-style mix are important.
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
    # Z-Image / diversity-focused preset: encourage variation and strong tag structure.
    "zit": SamplerPreset(
        name="zit",
        description="Z-Image-style preset: higher diversity, less centering, strong composition.",
        cfg_scale=6.0,
        cfg_rescale=0.7,
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
    # Superior Stack: balanced CFG + quality flags for composite pick / self-correct flows.
    "superior": SamplerPreset(
        name="superior",
        description="Superior Stack preset: strong adherence, natural look, anti-artifact; pair with --pick-best superior_composite.",
        cfg_scale=7.0,
        cfg_rescale=0.7,
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
    # Few-step quality via UniPC.
    "fast": SamplerPreset(
        name="fast",
        description="Fast few-step preset: UniPC + AYS-DiT (~15 steps).",
        cfg_scale=6.5,
        cfg_rescale=0.7,
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
    # Portrait photography: faces, skin, no AI slop.
    "portrait": {
        "naturalize": True,
        "naturalize_grain": 0.02,
        "anti_bleed": True,
        "diversity": True,
        "anti_artifacts": True,
    },
    # Full-body character / fashion.
    "fullbody": {
        "anti_bleed": True,
        "anti_artifacts": True,
    },
    # Anime character focus.
    "anime_char": {
        "anti_bleed": True,
        "diversity": True,
        "anti_artifacts": True,
    },
    # Few-step UniPC path.
    "fast": {
        "solver": "unipc",
        "scheduler": "ays_dit",
        "steps": 15,
    },
    # Max quality multistep path.
    "quality": {
        "solver": "dpmpp_2m",
        "scheduler": "ays_dit",
        "steps": 28,
    },
}


def apply_preset_to_args(args: Any, preset_name: str) -> None:
    """
    Mutate argparse args in-place using the given preset.
    Only set values if the corresponding arg looks "unset" (0, None, default bool False).
    """
    preset = PRESETS.get(preset_name)
    if not preset:
        return

    def maybe_set(name: str, value: Any, unset_values: tuple) -> None:
        if value is None:
            return
        if not hasattr(args, name):
            return
        current = getattr(args, name)
        if current in unset_values:
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

    # Booleans: only flip from False to True when preset says True.
    for flag_name, value in (
        ("naturalize", preset.naturalize),
        ("anti_bleed", preset.anti_bleed),
        ("diversity", preset.diversity),
        ("anti_artifacts", preset.anti_artifacts),
        ("strong_watermark", preset.strong_watermark),
    ):
        if value and hasattr(args, flag_name) and not getattr(args, flag_name):
            setattr(args, flag_name, True)


def apply_op_mode_to_args(args: Any, mode_name: str) -> None:
    """
    Apply a high-level OP mode on top of any preset.
    Like presets, only turns on flags that are currently off.
    """
    overrides = OP_MODES.get(mode_name)
    if not overrides:
        return
    _sampler_defaults = {
        "solver": {"ddim", "dpmpp_2m", "unipc", ""},
        "scheduler": {"ddim", "ays_dit", "ays", ""},
        "flow_solver": {"euler", "dpmpp_2m", "heun", ""},
        "flow_schedule": {"linear", "ays", "karras", ""},
    }
    for key, value in overrides.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if isinstance(value, bool):
            if value and not current:
                setattr(args, key, True)
        elif isinstance(value, str):
            allowed = _sampler_defaults.get(key)
            if allowed is not None:
                if current in allowed or current is None:
                    setattr(args, key, value)
            elif current in (None, ""):
                setattr(args, key, value)
        else:
            if current in (0, 0.0, None):
                setattr(args, key, value)
