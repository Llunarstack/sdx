"""Layout control: Omost canvas, coordinate prompts, LAMIC schedules and metrics, Dense Diffusion-style
cross-attention biasing for box layouts (training-free: boosts attention between spatial patches and their
regional prompts during the first ``inject_frac`` of steps), and storyboard chains (a sequence of prompts
with carry-over world/character locks).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def bind_coordinates_to_prompt(
    prompt: str,
    box: tuple[float, float, float, float],
    *,
    prefix: bool = True,
) -> str:
    """Insert ``<loc_x1_y1_x2_y2>`` token before prompt text."""
    x1, y1, x2, y2 = (round(float(v), 3) for v in box)
    token = f"<loc_{x1}_{y1}_{x2}_{y2}>"
    p = (prompt or "").strip()
    if token.lower() in p.lower():
        return p
    return f"{token} {p}" if prefix else f"{p} {token}"


@dataclass(frozen=True)
class RegionFusionSchedule:
    """Controls how much regions blend with each other over steps."""

    isolate_until_frac: float = 0.45
    curve: str = "cosine"  # linear | cosine
    min_isolation: float = 0.85  # 1 = full isolation early
    max_fusion: float = 1.0


def fusion_weight_at_step(
    step_index: int,
    total_steps: int,
    schedule: RegionFusionSchedule,
) -> float:
    """
    Return cross-region fusion factor in ``[0, 1]``.

    Low early (isolated), high late (fused). Multiply regional mask overlap
    blending by this factor when implementing LAMIC-style attention hooks.
    """
    if total_steps <= 1:
        return schedule.max_fusion
    p = step_index / (total_steps - 1)
    frac = schedule.isolate_until_frac
    if p <= frac:
        iso = schedule.min_isolation + (1.0 - schedule.min_isolation) * _ease(p / max(1e-6, frac), schedule.curve)
        return 1.0 - iso
    return schedule.max_fusion * _ease((p - frac) / max(1e-6, 1.0 - frac), schedule.curve)


def _ease(t: float, curve: str) -> float:
    return t if curve == "linear" else 0.5 * (1.0 - math.cos(math.pi * t))


@dataclass(frozen=True)
class LayoutQualityReport:
    inclusion_ratio: float
    fill_ratio: float
    overlap_ratio: float
    background_ratio: float
    notes: str


def score_layout_masks(
    region_masks: torch.Tensor,
    bg_mask: torch.Tensor,
    *,
    target_boxes_fill: float = 0.55,
) -> LayoutQualityReport:
    """
    Score latent-region masks.

    ``region_masks``: (R, 1, H, W), ``bg_mask``: (1, 1, H, W).
    """
    if region_masks.numel() == 0:
        return LayoutQualityReport(0.0, 0.0, 0.0, 1.0, "no regions")

    rm = region_masks.float().clamp(0, 1)
    # Coverage beyond the first region at each pixel.
    overlap_ratio = float((rm.sum(dim=0) - 1.0).clamp(min=0).mean())
    per_region_mean = rm.flatten(1).mean(dim=1)
    inclusion_ratio = float((per_region_mean > 0.05).float().mean())
    fill_ratio = float(per_region_mean.mean())
    background_ratio = float(bg_mask.float().clamp(0, 1).mean())

    notes_parts: list[str] = []
    if fill_ratio < target_boxes_fill * 0.5:
        notes_parts.append("regions are very small; consider larger boxes")
    if overlap_ratio > 0.15:
        notes_parts.append("high overlap; check priority or feather")
    if background_ratio < 0.05:
        notes_parts.append("layout covers almost entire frame")

    return LayoutQualityReport(
        inclusion_ratio=inclusion_ratio,
        fill_ratio=fill_ratio,
        overlap_ratio=overlap_ratio,
        background_ratio=background_ratio,
        notes="; ".join(notes_parts) if notes_parts else "ok",
    )


# Omost uses 9 anchor boxes; we expose the same grid for LLM compatibility.
OMOST_GRID: dict[str, tuple[float, float, float, float]] = {
    "top_left": (0.0, 0.0, 0.33, 0.33),
    "top": (0.33, 0.0, 0.66, 0.33),
    "top_right": (0.66, 0.0, 1.0, 0.33),
    "left": (0.0, 0.33, 0.33, 0.66),
    "center": (0.33, 0.33, 0.66, 0.66),
    "right": (0.66, 0.33, 1.0, 0.66),
    "bottom_left": (0.0, 0.66, 0.33, 1.0),
    "bottom": (0.33, 0.66, 0.66, 1.0),
    "bottom_right": (0.66, 0.66, 1.0, 1.0),
}


@dataclass
class LocalDescription:
    prompt: str
    box: tuple[float, float, float, float]
    name: str
    negative: str = ""
    priority: int = 5
    reference: str = ""


@dataclass
class OmostCanvas:
    """Virtual canvas agent (Omost-compatible API surface)."""

    global_description: str = ""
    global_negative: str = ""
    local_regions: list[LocalDescription] = field(default_factory=list)
    feather: int = 8

    def set_global_description(self, text: str, *, negative: str = "") -> None:
        self.global_description = (text or "").strip()
        if negative:
            self.global_negative = negative.strip()

    def add_local_description(
        self,
        prompt: str,
        *,
        box: tuple[float, float, float, float] | None = None,
        anchor: str | None = None,
        name: str | None = None,
        negative: str = "",
        priority: int = 5,
        reference: str = "",
    ) -> None:
        if anchor:
            key = anchor.strip().lower().replace("-", "_").replace(" ", "_")
            if key not in OMOST_GRID:
                raise ValueError(f"unknown anchor {anchor!r}; choose from {list(OMOST_GRID)}")
            b = OMOST_GRID[key]
        elif box is not None:
            b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        else:
            raise ValueError("provide box=(x1,y1,x2,y2) or anchor='center' etc.")
        idx = len(self.local_regions)
        self.local_regions.append(
            LocalDescription(
                prompt=(prompt or "").strip(),
                box=b,
                name=name or f"region_{idx}",
                negative=negative,
                priority=priority,
                reference=reference,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        regions: list[dict[str, Any]] = []
        for loc in self.local_regions:
            r: dict[str, Any] = {
                "name": loc.name,
                "box": list(loc.box),
                "prompt": loc.prompt,
                "priority": loc.priority,
            }
            if loc.negative:
                r["negative"] = loc.negative
            if loc.reference:
                r["reference"] = loc.reference
            regions.append(r)
        out: dict[str, Any] = {
            "global_prompt": self.global_description,
            "feather": self.feather,
            "regions": regions,
        }
        if self.global_negative:
            out["global_negative"] = self.global_negative
        return out


def canvas_to_box_layout(canvas: OmostCanvas) -> dict[str, Any]:
    """Export dict suitable for ``json.dump`` → ``--box-layout``."""
    if not canvas.local_regions:
        raise ValueError("canvas has no local descriptions")
    return canvas.to_dict()


@dataclass
class AttentionLayoutPlan:
    region_names: tuple[str, ...]
    boxes: tuple[tuple[float, float, float, float], ...]
    enforce_steps: tuple[int, ...]
    strength: float = 0.85


def build_attention_layout_plan(
    regions: Sequence[Any],
    *,
    num_steps: int = 28,
    inject_frac: float = 0.4,
    strength: float = 0.85,
) -> AttentionLayoutPlan:
    """Plan from box-layout regions (objects with ``name`` and ``x1``…``y2``); enforce the first ``inject_frac`` steps."""
    names = tuple(str(getattr(r, "name", f"region_{i}")) for i, r in enumerate(regions))
    boxes = tuple(
        (
            float(getattr(r, "x1", 0)),
            float(getattr(r, "y1", 0)),
            float(getattr(r, "x2", 1)),
            float(getattr(r, "y2", 1)),
        )
        for r in regions
    )
    n_enforce = max(1, int(round(num_steps * inject_frac)))
    return AttentionLayoutPlan(
        region_names=names,
        boxes=boxes,
        enforce_steps=tuple(range(n_enforce)),
        strength=float(strength),
    )


def region_masks_from_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
    latent_h: int,
    latent_w: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Hard ``(R, 1, H, W)`` masks from normalized ``(x1, y1, x2, y2)`` boxes."""
    masks = torch.zeros(len(boxes), 1, latent_h, latent_w, device=device, dtype=dtype)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        y0, x0 = int(y1 * latent_h), int(x1 * latent_w)
        masks[i, :, y0 : max(y0 + 1, int(y2 * latent_h)), x0 : max(x0 + 1, int(x2 * latent_w))] = 1.0
    return masks


def _patch_grid(n: int, mask_h: int, mask_w: int) -> tuple[int, int] | None:
    """Token grid ``(h, w)`` with ``h * w == n`` and the masks' aspect ratio."""
    if n == mask_h * mask_w:
        return mask_h, mask_w
    h = round(math.sqrt(n * mask_h / mask_w))
    if h <= 0 or n % h:
        return None
    return h, n // h


def bias_cross_attention(
    attn_logits: torch.Tensor,
    *,
    region_masks: torch.Tensor,
    step_index: int,
    plan: Any,
    cond_batch_size: int | None = None,
    num_patch_tokens: int | None = None,
) -> torch.Tensor:
    """
    Add layout bias to cross-attention logits ``(B, H, N, L)``.

    Text tokens are split into one contiguous chunk per region (plan order); chunk ``r`` gains
    ``strength * mask_r`` on the patch queries inside region ``r``. ``region_masks`` ``(R, 1, h, w)``
    are area-pooled onto the patch grid. Only the first ``cond_batch_size`` rows (the conditional
    half of a CFG batch) and the first ``num_patch_tokens`` queries (registers excluded) are biased.
    Returns ``attn_logits`` itself when the plan does not apply at this step.
    """
    if plan is None or step_index not in set(getattr(plan, "enforce_steps", ())):
        return attn_logits
    b, _, n_query, n_text = attn_logits.shape
    n_regions = int(region_masks.shape[0])
    n = min(int(num_patch_tokens or n_query), n_query)
    rows = min(int(cond_batch_size if cond_batch_size is not None else b), b)
    grid = _patch_grid(n, int(region_masks.shape[-2]), int(region_masks.shape[-1]))
    if n_regions == 0 or n_text < n_regions or rows <= 0 or grid is None:
        return attn_logits
    masks = F.adaptive_avg_pool2d(region_masks.to(attn_logits.device, attn_logits.dtype), grid).flatten(1)
    chunks = torch.zeros(n_regions, n_text, device=attn_logits.device, dtype=attn_logits.dtype)
    for r in range(n_regions):
        chunks[r, r * n_text // n_regions : (r + 1) * n_text // n_regions] = 1.0
    bias = float(getattr(plan, "strength", 0.85)) * (masks.t() @ chunks)
    out = attn_logits.clone()
    out[:rows, :, :n] += bias
    return out


# World bible: characters / locations / locks the storyboard keeps consistent across beats.


@dataclass
class CharacterRecord:
    id: str
    display_name: str
    visual_tokens: str  # appended to prompt when character appears
    negative_tokens: str = ""
    reference_path: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class LocationRecord:
    id: str
    name: str
    visual_tokens: str
    palette: str = ""  # e.g. "teal shadows, amber highlights"
    tags: list[str] = field(default_factory=list)


@dataclass
class WorldBible:
    """Loadable world state for multi-shot / series generation."""

    world_id: str = "default"
    characters: dict[str, CharacterRecord] = field(default_factory=dict)
    locations: dict[str, LocationRecord] = field(default_factory=dict)
    lore: str = ""

    @classmethod
    def load(cls, path: str | Path) -> WorldBible:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        chars = {k: CharacterRecord(**v) for k, v in data.get("characters", {}).items()}
        locs = {k: LocationRecord(**v) for k, v in data.get("locations", {}).items()}
        return cls(
            world_id=data.get("world_id", p.stem),
            characters=chars,
            locations=locs,
            lore=data.get("lore", ""),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "world_id": self.world_id,
            "lore": self.lore,
            "characters": {k: asdict(v) for k, v in self.characters.items()},
            "locations": {k: asdict(v) for k, v in self.locations.items()},
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def detect_characters(self, prompt: str) -> list[CharacterRecord]:
        text = (prompt or "").lower()
        hits: list[CharacterRecord] = []
        for rec in self.characters.values():
            names = [rec.display_name.lower(), rec.id.lower(), *[t.lower() for t in rec.tags]]
            if any(n and n in text for n in names):
                hits.append(rec)
        return hits

    def detect_location(self, prompt: str) -> LocationRecord | None:
        text = (prompt or "").lower()
        for rec in self.locations.values():
            if rec.name.lower() in text or rec.id.lower() in text:
                return rec
            if any(t.lower() in text for t in rec.tags):
                return rec
        return None

    def apply_lock(self, prompt: str, lock: WorldLock) -> str:
        parts = [prompt.strip()] if prompt.strip() else []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.visual_tokens:
                parts.append(rec.visual_tokens)
        loc = self.locations.get(lock.location_id)
        if loc and loc.visual_tokens:
            parts.append(loc.visual_tokens)
        if loc and loc.palette:
            parts.append(loc.palette)
        for item in lock.must_preserve:
            parts.append(f"keep {item} consistent")
        return ", ".join(parts)

    def negative_for_lock(self, lock: WorldLock) -> str:
        negs: list[str] = []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.negative_tokens:
                negs.append(rec.negative_tokens)
        for item in lock.forbid:
            negs.append(item)
        return ", ".join(negs)

    def references_for_lock(self, lock: WorldLock) -> list[str]:
        paths: list[str] = []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.reference_path:
                paths.append(rec.reference_path)
        return paths


_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")


def guess_proper_nouns(prompt: str) -> list[str]:
    """Cheap proper-noun harvest for auto world-bible seeding."""
    return list(dict.fromkeys(_NAME_RE.findall(prompt or "")))


@dataclass
class WorldLock:
    """Hard constraints for one generation."""

    character_ids: list[str] = field(default_factory=list)
    location_id: str = ""
    must_preserve: list[str] = field(default_factory=list)  # "red scarf", "clock tower"
    forbid: list[str] = field(default_factory=list)


@dataclass
class StoryboardBeat:
    index: int
    prompt: str
    negative_prompt: str = ""
    lock: WorldLock = field(default_factory=WorldLock)
    seed_offset: int = 0
    notes: str = ""


@dataclass
class Storyboard:
    title: str = ""
    beats: list[StoryboardBeat] = field(default_factory=list)
    carry_tags: list[str] = field(default_factory=list)  # appended to every beat after first

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "carry_tags": self.carry_tags,
            "beats": [asdict(b) for b in self.beats],
        }

    @classmethod
    def load(cls, path: str | Path) -> Storyboard:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        beats = []
        for i, b in enumerate(data.get("beats", [])):
            lock = WorldLock(**b.get("lock", {}))
            beats.append(
                StoryboardBeat(
                    index=int(b.get("index", i)),
                    prompt=b.get("prompt", ""),
                    negative_prompt=b.get("negative_prompt", ""),
                    lock=lock,
                    seed_offset=int(b.get("seed_offset", 0)),
                    notes=b.get("notes", ""),
                )
            )
        return cls(title=data.get("title", ""), beats=beats, carry_tags=list(data.get("carry_tags", [])))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def expanded_prompts(self) -> list[str]:
        carry = ", ".join(self.carry_tags) if self.carry_tags else ""
        out: list[str] = []
        for i, beat in enumerate(self.beats):
            p = beat.prompt
            if i > 0 and carry:
                p = f"{p}, {carry}" if p else carry
            out.append(p)
        return out


class StoryboardPlanner:
    """Build storyboards from bullet lists or numbered scenes."""

    def from_scene_list(self, scenes: Sequence[str], *, title: str = "") -> Storyboard:
        beats = [StoryboardBeat(index=i, prompt=s.strip(), seed_offset=i) for i, s in enumerate(scenes) if s.strip()]
        return Storyboard(title=title, beats=beats)

    def with_carry(self, board: Storyboard, tags: Sequence[str]) -> Storyboard:
        board.carry_tags = list(tags)
        return board


__all__ = [
    "OmostCanvas",
    "canvas_to_box_layout",
    "bind_coordinates_to_prompt",
    "RegionFusionSchedule",
    "fusion_weight_at_step",
    "LayoutQualityReport",
    "score_layout_masks",
    "AttentionLayoutPlan",
    "bias_cross_attention",
    "build_attention_layout_plan",
    "region_masks_from_boxes",
    "Storyboard",
    "StoryboardBeat",
    "StoryboardPlanner",
]
