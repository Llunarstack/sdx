"""Thumbnail-first rehearsal — cheap composition gates before full GPU render."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "ThumbnailConfig",
    "ThumbnailSpec",
    "ThumbnailRehearsalPlan",
    "parse_thumbnail_config",
    "plan_thumbnails",
    "thumbnail_edit_overrides",
    "thumbnail_gate_issues",
    "apply_thumbnail_timeline",
]


@dataclass(slots=True)
class ThumbnailConfig:
    enabled: bool = False
    size: int = 128
    frames_per_shot: int = 1
    gate: str = "off"  # off | warn | require_approval
    prompt_suffix: str = "storyboard thumbnail, strong silhouette, composition sketch"


@dataclass(slots=True)
class ThumbnailSpec:
    shot_id: str
    shot_index: int
    frame_role: str  # start | mid | end
    prompt: str
    width: int
    height: int
    approved: bool = False


@dataclass(slots=True)
class ThumbnailRehearsalPlan:
    enabled: bool
    config: ThumbnailConfig
    specs: List[ThumbnailSpec] = field(default_factory=list)
    gate_passed: bool = True
    pending_count: int = 0


def parse_thumbnail_config(
    data: Mapping[str, Any] | None,
    *,
    studio: Optional[Mapping[str, Any]] = None,
    edit: Optional[Mapping[str, Any]] = None,
) -> ThumbnailConfig:
    """Read thumbnail settings from continuity, studio, or edit blocks."""
    raw: Dict[str, Any] = {}
    if isinstance(data, Mapping):
        thumb = data.get("thumbnail") or data.get("thumbnail_first")
        if isinstance(thumb, Mapping):
            raw.update(thumb)
        elif thumb is True:
            raw["enabled"] = True
    if isinstance(studio, Mapping):
        if studio.get("thumbnail_first") or studio.get("thumbnail"):
            raw.setdefault("enabled", True)
        for k in ("thumbnail_size", "thumbnail_gate", "thumbnail_frames"):
            if k in studio:
                raw[k.replace("thumbnail_", "")] = studio[k]
    if isinstance(edit, Mapping):
        if edit.get("thumbnail_first") or edit.get("thumbnail_pass"):
            raw.setdefault("enabled", True)

    enabled = bool(raw.get("enabled", False))
    size = int(raw.get("size") or raw.get("thumbnail_size") or 128)
    size = max(32, min(512, size))
    frames = int(raw.get("frames_per_shot") or raw.get("frames") or 1)
    frames = max(1, min(3, frames))
    gate = str(raw.get("gate") or "off").lower()
    suffix = str(
        raw.get("prompt_suffix") or "storyboard thumbnail, strong silhouette, readable composition, minimal detail"
    )
    return ThumbnailConfig(enabled=enabled, size=size, frames_per_shot=frames, gate=gate, prompt_suffix=suffix)


def _frame_roles(n: int) -> List[str]:
    if n <= 1:
        return ["start"]
    if n == 2:
        return ["start", "end"]
    return ["start", "mid", "end"]


def plan_thumbnails(
    shots: Sequence[Any],
    *,
    config: ThumbnailConfig,
    base_prompt: str = "",
    aspect_width: int = 16,
    aspect_height: int = 9,
) -> ThumbnailRehearsalPlan:
    if not config.enabled:
        return ThumbnailRehearsalPlan(enabled=False, config=config, gate_passed=True)

    size = config.size
    if aspect_width >= aspect_height:
        w, h = size, max(32, int(size * aspect_height / max(1, aspect_width)))
    else:
        h, w = size, max(32, int(size * aspect_width / max(1, aspect_height)))

    specs: List[ThumbnailSpec] = []
    roles = _frame_roles(config.frames_per_shot)
    pending = 0

    for i, sh in enumerate(shots):
        sid = str(getattr(sh, "id", None) or f"shot_{i}")
        prompt = str(getattr(sh, "prompt", "") or base_prompt)
        if config.prompt_suffix and config.prompt_suffix.lower() not in prompt.lower():
            prompt = f"{prompt}, {config.prompt_suffix}".strip(", ")
        approved = bool(getattr(sh, "thumbnail_approved", False))
        for role in roles:
            specs.append(
                ThumbnailSpec(
                    shot_id=sid,
                    shot_index=i,
                    frame_role=role,
                    prompt=prompt,
                    width=w,
                    height=h,
                    approved=approved,
                )
            )
        if not approved:
            pending += 1

    gate_passed = pending == 0 or config.gate in ("off", "warn")
    if config.gate == "require_approval" and pending > 0:
        gate_passed = False

    return ThumbnailRehearsalPlan(
        enabled=True,
        config=config,
        specs=specs,
        gate_passed=gate_passed,
        pending_count=pending,
    )


def thumbnail_edit_overrides(config: ThumbnailConfig) -> Dict[str, Any]:
    """Cheap pass overrides when thumbnail_first is active."""
    return {
        "thumbnail_pass": True,
        "thumbnail_size": config.size,
        "keyframe_interval": 24,
        "edit_strength": 0.28,
        "frame_enhance": False,
        "quality_retry": False,
        "semantic_drift_repair": False,
        "flow_consistency": False,
        "deflicker": False,
        "post_grade": "",
    }


def apply_thumbnail_timeline(
    width: int,
    height: int,
    config: ThumbnailConfig,
) -> tuple[int, int]:
    """Downscale timeline for thumbnail pass."""
    if not config.enabled:
        return width, height
    size = config.size
    if width >= height:
        tw = size
        th = max(32, int(size * height / max(1, width)))
    else:
        th = size
        tw = max(32, int(size * width / max(1, height)))
    return tw, th


def thumbnail_gate_issues(plan: ThumbnailRehearsalPlan) -> List[str]:
    if not plan.enabled or plan.gate_passed:
        return []
    gate = plan.config.gate
    if gate == "require_approval":
        return [
            f"Thumbnail gate: {plan.pending_count} shot(s) lack thumbnail_approved: true "
            "(set on shots[] or run thumbnail pass first)"
        ]
    if gate == "warn" and plan.pending_count > 0:
        return [f"Thumbnail rehearsal: {plan.pending_count} shot(s) not marked approved (advisory)"]
    return []
