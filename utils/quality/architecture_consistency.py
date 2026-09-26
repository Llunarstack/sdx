"""
Architecture / perspective **consistency lock**.

Pure-2D diffusion invents rooms as texture, not geometry. Classic failure: a
grand hall where upper-story shelves sit *farther back* than the ground-floor
pillars (or a "second floor" request regenerates a different building).

This module gives sample-time helpers that do not require a trained 3D field:

1. Detect interior / multi-story / hall prompts.
2. Build **story-band box layouts** (aisle, ground shelves, balcony rail,
   recessed upper shelves, ceiling, far wall).
3. Emit a **synthetic depth control** where upper shelf faces are farther than
   the balcony rail / ground facade — the depth plane Google-style models miss.
4. Soft-enable dual-stage layout, box-attn, anti-perspective-drift, and a
   pick-best metric that prefers vanishing-line / depth-order sanity.
5. Detect **novel-view** asks ("from the second floor") and re-bias camera +
   layout instead of treating them as a brand-new scene.

Wire-in: ``apply_architecture_consistency(args)`` from sample prompt phase.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ArchitectureProfile",
    "prompt_needs_architecture_lock",
    "prompt_asks_novel_view",
    "infer_architecture_profile",
    "build_story_box_layout",
    "story_box_layout_dict",
    "write_story_box_layout",
    "synthetic_story_depth_map",
    "write_synthetic_depth_control",
    "estimate_depth_control_from_image",
    "architecture_prompt_fragments",
    "score_vanishing_symmetry",
    "score_horizon_level",
    "score_story_depth_order",
    "score_architecture_consistency",
    "apply_architecture_consistency",
]

# Multi-story / hall / interior cues — first-hit is enough to engage the lock.
_ARCH_KEYWORDS = (
    "library",
    "cathedral",
    "basilica",
    "palace",
    "castle hall",
    "great hall",
    "corridor",
    "hallway",
    "colonnade",
    "arcade",
    "cloister",
    "nave",
    "aisle",
    "balcony",
    "mezzanine",
    "gallery",
    "second floor",
    "third floor",
    "upper floor",
    "upper level",
    "multi-story",
    "multi story",
    "two-story",
    "two story",
    "bookshelves",
    "book shelves",
    "interior architecture",
    "grand hall",
    "vanishing point",
    "one-point perspective",
    "one point perspective",
    "vaulted ceiling",
    "gothic interior",
    "baroque interior",
    "ballroom",
    "opera house",
    "museum hall",
    "train station hall",
)

_NOVEL_VIEW = re.compile(
    r"(?:from\s+the\s+)?(?:second|third|upper)\s+(?:floor|level|gallery|balcony)"
    r"|looking\s+(?:down|along|from)\s+(?:the\s+)?(?:balcony|gallery|mezzanine|upper)"
    r"|novel\s+view|another\s+(?:angle|viewpoint|perspective)"
    r"|same\s+(?:room|library|hall|scene).{0,40}(?:angle|view|camera|floor)"
    r"|show\s+(?:me\s+)?(?:the\s+)?(?:second|upper)\s+floor",
    re.I,
)

_HALL_ONE_POINT = re.compile(
    r"library|great hall|cathedral|nave|corridor|hallway|colonnade|aisle|vanishing",
    re.I,
)


@dataclass(slots=True)
class ArchitectureProfile:
    """Inferred structural prior for one prompt."""

    kind: str = "none"  # none | hall | interior | novel_view
    stories: int = 1
    one_point: bool = False
    has_balcony: bool = False
    novel_view: bool = False
    focus_story: int = 0  # 0 = ground, 1 = upper, …
    reason: str = ""
    cues: tuple[str, ...] = ()


def prompt_needs_architecture_lock(prompt: str) -> bool:
    p = (prompt or "").lower()
    if not p.strip():
        return False
    return any(k in p for k in _ARCH_KEYWORDS) or bool(_NOVEL_VIEW.search(p))


def prompt_asks_novel_view(prompt: str) -> bool:
    return bool(_NOVEL_VIEW.search(prompt or ""))


def infer_architecture_profile(prompt: str) -> ArchitectureProfile:
    text = prompt or ""
    p = text.lower()
    cues = tuple(k for k in _ARCH_KEYWORDS if k in p)
    novel = prompt_asks_novel_view(text)
    if not cues and not novel:
        return ArchitectureProfile()

    stories = 1
    if any(k in p for k in ("third floor", "three-story", "three story", "3-story")):
        stories = 3
    elif any(
        k in p
        for k in (
            "second floor",
            "two-story",
            "two story",
            "upper floor",
            "upper level",
            "balcony",
            "mezzanine",
            "gallery",
            "multi-story",
            "multi story",
        )
    ):
        stories = 2

    has_balcony = any(k in p for k in ("balcony", "mezzanine", "gallery", "upper floor", "second floor"))
    one_point = bool(_HALL_ONE_POINT.search(text)) or "vanishing" in p or "perspective" in p
    focus = 1 if novel or any(k in p for k in ("second floor", "upper floor", "from the balcony")) else 0
    kind = "novel_view" if novel else ("hall" if one_point else "interior")
    reason = "novel_view" if novel else ("keyword:" + (cues[0] if cues else "architecture"))
    return ArchitectureProfile(
        kind=kind,
        stories=max(stories, 2 if has_balcony or novel else 1),
        one_point=one_point,
        has_balcony=has_balcony or stories >= 2,
        novel_view=novel,
        focus_story=focus,
        reason=reason,
        cues=cues[:8],
    )


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def build_story_box_layout(
    prompt: str,
    *,
    profile: ArchitectureProfile | None = None,
    global_negative: str = "warped perspective, melting architecture, floating shelves, inconsistent depth, broken vanishing lines, shelf walls drifting back incorrectly",
) -> dict[str, Any]:
    """
    Return a box-layout JSON dict with physically motivated story bands.

    Upper shelf faces are **inset** (smaller x-span) vs ground pillars so regional
    CFG + Dense Diffusion keep the upper wall on a deeper plane than the rail.
    """
    prof = profile or infer_architecture_profile(prompt)
    stories = max(1, int(prof.stories))
    focus = int(prof.focus_story)

    # Vertical bands (image y grows downward).
    ceiling_y1, ceiling_y2 = 0.0, 0.16
    if stories >= 3:
        upper_y1, upper_y2 = 0.10, 0.32
        mid_y1, mid_y2 = 0.30, 0.50
        ground_y1, ground_y2 = 0.48, 0.92
        rail_y1, rail_y2 = 0.46, 0.54
    elif stories >= 2:
        upper_y1, upper_y2 = 0.10, 0.42
        mid_y1, mid_y2 = 0.0, 0.0  # unused
        ground_y1, ground_y2 = 0.48, 0.95
        rail_y1, rail_y2 = 0.40, 0.50
    else:
        upper_y1, upper_y2 = 0.0, 0.0
        mid_y1, mid_y2 = 0.0, 0.0
        ground_y1, ground_y2 = 0.35, 0.95
        rail_y1, rail_y2 = 0.0, 0.0

    # Ground facade flush to aisle; upper shelves inset (recessed wall).
    ground_inset = 0.0
    upper_inset = 0.06  # upper wall face sits farther → narrower near vanishing axis
    aisle_x1, aisle_x2 = 0.34, 0.66

    def _side_box(side: str, y1: float, y2: float, inset: float) -> list[float]:
        if side == "left":
            return [_clamp01(0.0 + inset * 0.3), _clamp01(y1), _clamp01(aisle_x1 - inset), _clamp01(y2)]
        return [_clamp01(aisle_x2 + inset), _clamp01(y1), _clamp01(1.0 - inset * 0.3), _clamp01(y2)]

    base = (prompt or "").strip() or "grand architectural interior, coherent perspective"
    regions: list[dict[str, Any]] = [
        {
            "name": "ceiling_vault",
            "box": [0.0, ceiling_y1, 1.0, ceiling_y2],
            "priority": 4,
            "prompt": f"{base}, vaulted ceiling, consistent arches, single vanishing point",
            "negative": "floor furniture, floating bookshelves",
        },
        {
            "name": "center_aisle",
            "box": [aisle_x1, max(ground_y1, 0.55), aisle_x2, 1.0],
            "priority": 6,
            "prompt": f"{base}, stone aisle floor tiles, leading lines to vanishing point, clear depth",
            "negative": "bookshelf in center, railing across aisle",
        },
        {
            "name": "far_end_wall",
            "box": [0.38, 0.22, 0.62, 0.55],
            "priority": 5,
            "prompt": f"{base}, distant end wall, arched doorway, haze, far depth plane",
            "negative": "foreground pillars, close balcony rail",
        },
    ]

    # Ground-floor shelves / pillars — flush facade.
    for side in ("left", "right"):
        regions.append(
            {
                "name": f"{side}_ground_shelves",
                "box": _side_box(side, ground_y1, ground_y2, ground_inset),
                "priority": 10 if focus == 0 else 7,
                "prompt": (
                    f"{base}, ground floor arched bookshelves, pillars flush with aisle, "
                    "near depth plane, readable shelf faces"
                ),
                "negative": "receding incorrectly, floating upper gallery merge, warped pillars",
            }
        )

    if stories >= 2:
        for side in ("left", "right"):
            # Rail / walkway stays on the near plane (same depth as ground pillars).
            rail_box = _side_box(side, rail_y1, rail_y2, 0.0)
            # Nudge rail toward aisle so walkway reads in front of recessed shelves.
            if side == "left":
                rail_box[2] = _clamp01(aisle_x1 + 0.02)
            else:
                rail_box[0] = _clamp01(aisle_x2 - 0.02)
            regions.append(
                {
                    "name": f"{side}_balcony_rail",
                    "box": rail_box,
                    "priority": 11 if focus >= 1 else 9,
                    "prompt": (
                        f"{base}, wooden balcony railing, walkway in front of upper shelves, "
                        "same depth plane as ground pillars, clear separation from recessed wall"
                    ),
                    "negative": "shelves flush with rail, missing walkway, railing fused into books",
                }
            )
            regions.append(
                {
                    "name": f"{side}_upper_shelves",
                    "box": _side_box(side, upper_y1, upper_y2, upper_inset),
                    "priority": 12 if focus >= 1 else 8,
                    "prompt": (
                        f"{base}, upper story bookshelves recessed behind the balcony walkway, "
                        "deeper wall plane than the railing, stacked arches aligned with pillars below"
                    ),
                    "negative": "upper shelves farther forward than ground, wall thinner upstairs, melted arches",
                }
            )

    if stories >= 3 and mid_y2 > mid_y1:
        for side in ("left", "right"):
            regions.append(
                {
                    "name": f"{side}_mid_shelves",
                    "box": _side_box(side, mid_y1, mid_y2, upper_inset * 0.5),
                    "priority": 8,
                    "prompt": f"{base}, middle gallery shelves, consistent pillar alignment",
                    "negative": "depth jump, floating floor",
                }
            )

    if focus >= 1:
        # Novel-view bias: overweight upper walkway language in global prompt.
        global_prompt = (
            f"{base}, viewpoint from the upper gallery balcony, looking along the hall, "
            "walkway visible in front of recessed upper bookshelves, coherent multi-story depth, "
            "aligned pillars floor-to-floor, stable one-point perspective"
        )
    else:
        global_prompt = (
            f"{base}, multi-story interior, balcony walkway in front of recessed upper shelves, "
            "ground pillars aligned under upper arches, stable vanishing point, coherent depth"
        )

    return {
        "global_prompt": global_prompt,
        "global_negative": global_negative,
        "feather": 10,
        "overlap_mode": "priority",
        "regions": regions,
        "_architecture_profile": {
            "kind": prof.kind,
            "stories": stories,
            "focus_story": focus,
            "novel_view": prof.novel_view,
            "one_point": prof.one_point,
        },
    }


def story_box_layout_dict(prompt: str, **kwargs: Any) -> dict[str, Any]:
    """Alias for ``build_story_box_layout`` (public naming)."""
    return build_story_box_layout(prompt, **kwargs)


def write_story_box_layout(prompt: str, path: str | Path, **kwargs: Any) -> Path:
    data = build_story_box_layout(prompt, **kwargs)
    data.pop("_architecture_profile", None)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def synthetic_story_depth_map(
    height: int,
    width: int,
    *,
    profile: ArchitectureProfile | None = None,
    stories: int | None = None,
) -> np.ndarray:
    """
    Build a uint8 depth image (near=white, far=black) for ControlNet.

    Encodes the library failure mode fix: upper shelf bands are **farther**
    (darker) than the balcony rail / ground facade at the same image-x.
    """
    h, w = max(8, int(height)), max(8, int(width))
    n_stories = int(stories if stories is not None else (profile.stories if profile else 2))
    n_stories = max(1, n_stories)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = yy / max(h - 1, 1)
    x = xx / max(w - 1, 1)
    # Center aisle goes to vanishing point (farther).
    aisle = np.exp(-((x - 0.5) ** 2) / 0.045)
    # Base: bottom nearer, top/center farther.
    depth = 0.25 + 0.45 * (1.0 - y) + 0.35 * aisle  # higher = farther for now

    # Ground facade bands (sides, lower half) — nearer.
    ground_side = ((x < 0.34) | (x > 0.66)) & (y > 0.48)
    depth = np.where(ground_side, np.minimum(depth, 0.28), depth)

    if n_stories >= 2:
        # Balcony rail strip — same near plane as ground facade.
        rail = ((x < 0.36) | (x > 0.64)) & (y > 0.40) & (y < 0.50)
        depth = np.where(rail, np.minimum(depth, 0.30), depth)
        # Upper shelves inset + farther than rail.
        upper = ((x < 0.30) | (x > 0.70)) & (y > 0.10) & (y < 0.42)
        depth = np.where(upper, np.maximum(depth, 0.55), depth)

    if n_stories >= 3:
        mid = ((x < 0.32) | (x > 0.68)) & (y > 0.30) & (y < 0.48)
        depth = np.where(mid, np.maximum(depth, 0.45), depth)

    # Far end portal.
    portal = (x > 0.40) & (x < 0.60) & (y > 0.22) & (y < 0.52)
    depth = np.where(portal, np.maximum(depth, 0.75), depth)

    depth = np.clip(depth, 0.0, 1.0)
    # ControlNet convention in this repo: bright = near. Invert.
    near = 1.0 - depth
    return (near * 255.0).astype(np.uint8)


def write_synthetic_depth_control(
    path: str | Path,
    *,
    height: int = 512,
    width: int = 512,
    profile: ArchitectureProfile | None = None,
    stories: int | None = None,
) -> Path:
    from PIL import Image

    arr = synthetic_story_depth_map(height, width, profile=profile, stories=stories)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out)
    return out


def estimate_depth_control_from_image(
    image_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cuda",
    prefer: str = "small",
) -> Path:
    """
    Generate → estimate-depth → re-sample helper.

    Prefer Depth-Anything / Marigold when installed; fall back to a story-shaped
    synthetic map so the control path always returns a usable PNG.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from utils.modeling.hf_loaders import depth_map

        depth_map(str(image_path), str(out), device=device, prefer=prefer)
        if out.is_file():
            return out
    except Exception:
        pass
    try:
        from utils.modeling.hf_loaders import marigold_depth_map

        marigold_depth_map(str(image_path), str(out), device=device)
        if out.is_file():
            return out
    except Exception:
        pass
    # Fallback: keep architecture prior rather than failing open.
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    write_synthetic_depth_control(out, height=h, width=w, stories=2)
    return out


_ARCH_POSITIVE = (
    "coherent perspective",
    "stable vanishing point",
    "aligned pillars floor to floor",
    "balcony walkway in front of recessed upper shelves",
    "consistent architectural depth",
    "readable multi-story structure",
)
_ARCH_NEGATIVE = (
    "warped perspective",
    "melting architecture",
    "inconsistent depth",
    "upper shelves farther forward than ground",
    "floating balcony",
    "broken vanishing lines",
    "impossible geometry",
    "walls drifting between floors",
)
_NOVEL_POSITIVE = (
    "viewpoint from upper gallery",
    "looking along the hall from the balcony",
    "same building different camera",
    "continuous architecture",
)


def architecture_prompt_fragments(
    profile: ArchitectureProfile,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pos = list(_ARCH_POSITIVE)
    neg = list(_ARCH_NEGATIVE)
    if profile.novel_view or profile.focus_story >= 1:
        pos = list(_NOVEL_POSITIVE) + pos
    if profile.one_point:
        pos = ["one-point perspective", "leading lines to vanishing point", *pos]
    return tuple(pos), tuple(neg)


def score_vanishing_symmetry(rgb_uint8: np.ndarray) -> float:
    """
    Cheap one-point hall prior: left/right mirror agreement on edge energy.

    High when the image is roughly left-right symmetric (grand hall look).
    Low when one side invents a different depth plane.
    """
    img = np.asarray(rgb_uint8)
    if img.ndim != 3 or img.shape[0] < 16 or img.shape[1] < 16:
        return 0.5
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    # Simple gradient magnitude.
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    # Pad to align.
    edge = np.zeros_like(gray)
    edge[:, :-1] += gx
    edge[:-1, :] += gy
    left = edge[:, : edge.shape[1] // 2]
    right = np.flip(edge[:, edge.shape[1] - left.shape[1] :], axis=1)
    m = min(left.shape[1], right.shape[1])
    left, right = left[:, :m], right[:, :m]
    denom = float(np.mean(left) + np.mean(right)) + 1e-6
    diff = float(np.mean(np.abs(left - right))) / denom
    return float(np.clip(1.0 - diff, 0.0, 1.0))


def score_horizon_level(rgb_uint8: np.ndarray) -> float:
    """Penalize strong global tilt via row-energy centroid drift L↔R."""
    img = np.asarray(rgb_uint8)
    if img.ndim != 3 or img.shape[0] < 16:
        return 0.5
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    h, w = gy.shape
    left = gy[:, : w // 2]
    right = gy[:, w // 2 :]
    rows = np.arange(h, dtype=np.float32)

    def _centroid(band: np.ndarray) -> float:
        col = band.mean(axis=1)
        s = float(col.sum()) + 1e-6
        return float((rows * col).sum() / s)

    drift = abs(_centroid(left) - _centroid(right)) / max(h, 1)
    return float(np.clip(1.0 - drift * 4.0, 0.0, 1.0))


def score_story_depth_order(
    depth_near_white: np.ndarray,
    *,
    stories: int = 2,
) -> float:
    """
    Given a near=white depth map, score whether upper shelf bands are farther
    (darker) than the balcony rail band — the library bug check.
    """
    d = np.asarray(depth_near_white)
    if d.ndim == 3:
        d = d.mean(axis=2)
    d = d.astype(np.float32)
    if d.max() > 1.5:
        d = d / 255.0
    h, w = d.shape
    if h < 16 or w < 16 or stories < 2:
        return 0.5

    def _mean_band(x0: float, x1: float, y0: float, y1: float) -> float:
        xa, xb = int(x0 * w), int(x1 * w)
        ya, yb = int(y0 * h), int(y1 * h)
        xa, xb = max(0, xa), min(w, max(xa + 1, xb))
        ya, yb = max(0, ya), min(h, max(ya + 1, yb))
        return float(d[ya:yb, xa:xb].mean())

    # near=white → higher mean = nearer
    rail_l = _mean_band(0.02, 0.34, 0.40, 0.50)
    rail_r = _mean_band(0.66, 0.98, 0.40, 0.50)
    upper_l = _mean_band(0.02, 0.28, 0.12, 0.38)
    upper_r = _mean_band(0.72, 0.98, 0.12, 0.38)
    ground_l = _mean_band(0.02, 0.34, 0.55, 0.90)
    ground_r = _mean_band(0.66, 0.98, 0.55, 0.90)

    rail = 0.5 * (rail_l + rail_r)
    upper = 0.5 * (upper_l + upper_r)
    ground = 0.5 * (ground_l + ground_r)

    # Want: upper farther than rail (upper < rail in near-white), ground ≈ rail.
    upper_ok = float(np.clip((rail - upper) / 0.15 + 0.5, 0.0, 1.0))
    ground_ok = float(np.clip(1.0 - abs(ground - rail) / 0.20, 0.0, 1.0))
    return float(0.65 * upper_ok + 0.35 * ground_ok)


def score_architecture_consistency(
    rgb_uint8: np.ndarray,
    prompt: str = "",
    *,
    depth_near_white: np.ndarray | None = None,
) -> float:
    """Composite [0,1] score for pick-best / audits."""
    sym = score_vanishing_symmetry(rgb_uint8)
    level = score_horizon_level(rgb_uint8)
    parts = [sym, level]
    weights = [0.45, 0.25]
    prof = infer_architecture_profile(prompt) if prompt else ArchitectureProfile(stories=2)
    if depth_near_white is not None:
        order = score_story_depth_order(depth_near_white, stories=max(2, prof.stories))
        parts.append(order)
        weights.append(0.30)
    else:
        # Without depth, lean on symmetry/level only; still useful for halls.
        parts.append(sym)
        weights.append(0.30)
    wsum = sum(weights) or 1.0
    return float(sum(p * w for p, w in zip(parts, weights, strict=True)) / wsum)


def _soft_set(args: Any, name: str, value: Any, unset: tuple[Any, ...]) -> bool:
    if not hasattr(args, name):
        # Still set so downstream getattr works for attrs we introduce.
        if name.startswith("_"):
            setattr(args, name, value)
            return True
        return False
    cur = getattr(args, name)
    if cur in unset:
        setattr(args, name, value)
        return True
    return False


def _append_csv(base: str, additions: tuple[str, ...]) -> str:
    existing = {t.strip().lower() for t in (base or "").split(",") if t.strip()}
    out = [t.strip() for t in (base or "").split(",") if t.strip()]
    for a in additions:
        if a.strip().lower() not in existing:
            out.append(a.strip())
            existing.add(a.strip().lower())
    return ", ".join(out)


def apply_architecture_consistency(args: Any, *, prompt: str | None = None) -> dict[str, Any]:
    """
    Soft-wire architecture lock into sample args.

    Enabled when ``args.architecture_lock`` is true/auto and the prompt needs it,
    or when ``architecture_lock`` is forced ``on``.
    """
    mode = str(getattr(args, "architecture_lock", "auto") or "auto").lower().strip()
    text = prompt if prompt is not None else str(getattr(args, "prompt", "") or "")
    meta: dict[str, Any] = {"enabled": False, "mode": mode, "applied": []}

    if mode in ("off", "none", "0", "false"):
        meta["skipped"] = "disabled"
        return meta

    needs = prompt_needs_architecture_lock(text)
    if mode == "auto" and not needs:
        meta["skipped"] = "prompt_not_architectural"
        return meta
    if mode not in ("auto", "on", "true", "1", "force", "hall", "strict"):
        # Unknown → treat as auto
        if not needs:
            meta["skipped"] = "prompt_not_architectural"
            return meta

    prof = infer_architecture_profile(text)
    if mode in ("hall", "strict") and prof.stories < 2:
        prof = ArchitectureProfile(
            kind="hall",
            stories=2,
            one_point=True,
            has_balcony=True,
            novel_view=prof.novel_view,
            focus_story=prof.focus_story,
            reason="forced_hall",
            cues=prof.cues,
        )

    meta["enabled"] = True
    meta["profile"] = {
        "kind": prof.kind,
        "stories": prof.stories,
        "novel_view": prof.novel_view,
        "focus_story": prof.focus_story,
        "reason": prof.reason,
    }
    args._architecture_profile = prof

    # Prompt fragments
    pos, neg = architecture_prompt_fragments(prof)
    if _soft_set(args, "anti_perspective_drift", True, (False,)):
        meta["applied"].append("anti_perspective_drift")
    if hasattr(args, "prompt"):
        args.prompt = _append_csv(str(getattr(args, "prompt", "") or ""), pos)
        meta["applied"].append("prompt_fragments")
    if hasattr(args, "negative_prompt"):
        args.negative_prompt = _append_csv(str(getattr(args, "negative_prompt", "") or ""), neg)
        meta["applied"].append("negative_fragments")

    # Domain / composition soft-fills
    if _soft_set(args, "scene_domain", "architecture", ("none", "", None)):
        meta["applied"].append("scene_domain")
    if _soft_set(args, "artist_composition", "perspective", ("none", "", None)):
        meta["applied"].append("artist_composition")
    if _soft_set(args, "dual_stage_layout", True, (False,)):
        meta["applied"].append("dual_stage_layout")
    if _soft_set(args, "box_attn_layout", True, (False,)):
        meta["applied"].append("box_attn_layout")

    # Box layout — only if user did not supply one
    existing_box = str(getattr(args, "box_layout", "") or "").strip()
    if not existing_box and getattr(args, "_box_layout_spec", None) is None:
        layout = build_story_box_layout(text, profile=prof)
        # Persist to a temp JSON so sample_main's load path can reuse it if needed.
        tmp_dir = Path(tempfile.mkdtemp(prefix="sdx_arch_"))
        layout_path = tmp_dir / "story_box_layout.json"
        serial = {k: v for k, v in layout.items() if not k.startswith("_")}
        layout_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
        try:
            from utils.generation.regional_box_prompting import parse_box_layout

            spec = parse_box_layout(serial)
            args._box_layout_spec = spec
            if hasattr(args, "box_layout"):
                args.box_layout = str(layout_path)
            meta["applied"].append("story_box_layout")
            meta["box_layout_path"] = str(layout_path)
        except Exception as e:  # pragma: no cover - parse edge
            meta["box_layout_error"] = f"{type(e).__name__}: {e}"

    # Synthetic depth control when no control image is set
    ctrl_img = str(getattr(args, "control_image", "") or "").strip()
    ctrl_list = list(getattr(args, "control", []) or [])
    force_depth = bool(getattr(args, "architecture_depth_control", True))
    if (
        force_depth
        and not ctrl_img
        and not ctrl_list
        and mode in ("auto", "on", "true", "1", "force", "hall", "strict")
    ):
        img_size = int(getattr(args, "image_size", 512) or 512)
        tmp_dir = Path(getattr(args, "_architecture_tmp", "") or tempfile.mkdtemp(prefix="sdx_arch_"))
        args._architecture_tmp = str(tmp_dir)
        depth_path = tmp_dir / "story_depth.png"
        write_synthetic_depth_control(depth_path, height=img_size, width=img_size, profile=prof)
        depth_strength = float(getattr(args, "architecture_depth_strength", 0.7) or 0.7)
        control_str = f"{depth_path}:depth:{depth_strength}"
        if hasattr(args, "control"):
            args.control = [control_str]
            meta["applied"].append("synthetic_depth_control")
        elif hasattr(args, "control_image"):
            args.control_image = str(depth_path)
            if hasattr(args, "control_type"):
                _soft_set(args, "control_type", "depth", ("none", "", None, "auto", "edge", "canny"))
            meta["applied"].append("synthetic_depth_control")
        meta["depth_path"] = str(depth_path)
        args._architecture_depth_path = str(depth_path)

    # Pick-best soft preference for multi-candidate runs
    num = int(getattr(args, "num", 1) or 1)
    if num > 1 and _soft_set(args, "pick_best", "combo_architecture", ("none", "", None, "auto")):
        meta["applied"].append("pick_best_combo_architecture")

    if prof.novel_view:
        meta["novel_view_hint"] = (
            "For a true same-scene novel view, pass --init-image of the first shot "
            "plus this architecture lock (or wire frontier/multiview)."
        )

    return meta
