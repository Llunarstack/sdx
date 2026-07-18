"""Parallax Depth Budget — 2.5D layer separation hints for compositing pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["ParallaxLayer", "ParallaxPlan", "parse_parallax_config", "plan_parallax_budget"]


@dataclass(slots=True)
class ParallaxLayer:
    name: str
    depth: float  # 0 bg .. 1 fg
    motion_scale: float


@dataclass(slots=True)
class ParallaxPlan:
    shot_id: str
    layers: list[ParallaxLayer]
    prompt_suffix: str
    edit_overrides: dict[str, Any]


def parse_parallax_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", True)),
            "layers": int(raw.get("layers") or raw.get("layer_count") or 3),
        }
    return {"enabled": bool(raw)}


def plan_parallax_budget(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[ParallaxPlan]:
    if not config.get("enabled"):
        return []
    n = max(2, min(5, int(config.get("layers") or 3)))
    plans: list[ParallaxPlan] = []
    names = ["far_background", "midground", "subject_plane", "foreground_occluder", "ultra_fg"]
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        layers: list[ParallaxLayer] = []
        for i in range(n):
            depth = i / max(1, n - 1)
            motion = 0.15 + depth * 0.85
            layers.append(
                ParallaxLayer(name=names[min(i, len(names) - 1)], depth=round(depth, 2), motion_scale=round(motion, 2))
            )
        frag = f"parallax depth stack {n} layers, separated foreground midground background motion"
        plans.append(
            ParallaxPlan(
                shot_id=sid,
                layers=layers,
                prompt_suffix=frag,
                edit_overrides={"depth_interpolate": True, "region_motion": True},
            )
        )
    return plans
