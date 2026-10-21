"""Sampler presets (``sample.py --preset``) and OP modes (``--op-mode``).

Both are soft defaults: ``sample_main.keep_explicit`` restores every flag the user typed after they apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
