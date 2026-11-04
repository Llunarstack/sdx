"""Segment quality gates and post repairs: permanence, artifacts, identity, glyphs, drift, contact,
counts, shimmer, shutter, occlusion, secondary props, shot-chain cuts, and the superior pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.video.scene import SegmentQualityReport
from pipelines.video.video_io import read_frame_rgb, save_frame_rgb

# ==================================================================================================
# permanence: Object permanence score: do props blink in and out of a clip?
#
# Extracts cheap salient "slots" per frame (luma peaks / blobs), matches them across time (greedy), and counts
# abrupt appear / disappear events (GenVID's "Abrupt Appearance/Disappearance" artifact). Used as a segment
# quality gate (``min_permanence``) that triggers a retry.
# ==================================================================================================


@dataclass(slots=True)
class Slot:
    """One salient region in a frame."""

    cy: float
    cx: float
    mass: float
    color: tuple[float, float, float]
    bbox: tuple[int, int, int, int]  # y0,x0,y1,x1


@dataclass(slots=True)
class PermanenceReport:
    score: float
    """1 = perfect permanence, 0 = chaos of popping objects."""
    disappear_events: int = 0
    appear_events: int = 0
    notes: list[str] = field(default_factory=list)
    # Restored video feature switches (superior pass, physics, audio, neural engine, ...).
    repaired: int = 0


def _downsample(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(rgb[..., :3].astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32)


def extract_slots(rgb: np.ndarray, *, k: int = 8, grid: int = 8) -> list[Slot]:
    """
    Grid-peak slots: take the highest-contrast cell centers as entity proxies.

    No detector dependency — works offline and is stable enough to catch pops.
    """
    small = _downsample(rgb, 64)
    h, w, _ = small.shape
    cell = max(1, 64 // grid)
    cands: list[tuple[float, int, int, np.ndarray]] = []
    gray = small.mean(axis=2)
    # Local contrast vs neighborhood
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * cell, gx * cell
            y1, x1 = min(h, y0 + cell), min(w, x0 + cell)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            contrast = float(patch.std() + abs(patch.mean() - gray.mean()) * 0.25)
            cy = (y0 + y1) * 0.5
            cx = (x0 + x1) * 0.5
            color = small[int(cy), int(cx)]
            cands.append((contrast, int(cy), int(cx), color))
    cands.sort(key=lambda t: -t[0])
    slots: list[Slot] = []
    scale_y = rgb.shape[0] / 64.0
    scale_x = rgb.shape[1] / 64.0
    for contrast, cy, cx, color in cands[:k]:
        if contrast < 1.5:
            continue
        ry, rx = int(cy * scale_y), int(cx * scale_x)
        half = max(4, int(min(rgb.shape[0], rgb.shape[1]) * 0.04))
        y0, x0 = max(0, ry - half), max(0, rx - half)
        y1, x1 = min(rgb.shape[0], ry + half), min(rgb.shape[1], rx + half)
        mass = float(contrast)
        slots.append(
            Slot(
                cy=float(ry),
                cx=float(rx),
                mass=mass,
                color=(float(color[0]), float(color[1]), float(color[2])),
                bbox=(y0, x0, y1, x1),
            )
        )
    return slots


def _slot_dist(a: Slot, b: Slot, *, h: float, w: float) -> float:
    dy = (a.cy - b.cy) / max(h, 1.0)
    dx = (a.cx - b.cx) / max(w, 1.0)
    spatial = (dy * dy + dx * dx) ** 0.5
    col = (abs(a.color[0] - b.color[0]) + abs(a.color[1] - b.color[1]) + abs(a.color[2] - b.color[2])) / (3.0 * 255.0)
    return 0.75 * spatial + 0.25 * col


def match_slots(
    prev: list[Slot],
    curr: list[Slot],
    *,
    h: int,
    w: int,
    max_dist: float = 0.18,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy match → (pairs, unmatched_prev=disappear, unmatched_curr=appear)."""
    pairs: list[tuple[int, int]] = []
    used_p: set[int] = set()
    used_c: set[int] = set()
    scores: list[tuple[float, int, int]] = []
    for i, a in enumerate(prev):
        for j, b in enumerate(curr):
            scores.append((_slot_dist(a, b, h=float(h), w=float(w)), i, j))
    scores.sort(key=lambda t: t[0])
    for d, i, j in scores:
        if d > max_dist:
            break
        if i in used_p or j in used_c:
            continue
        pairs.append((i, j))
        used_p.add(i)
        used_c.add(j)
    disappear = [i for i in range(len(prev)) if i not in used_p]
    appear = [j for j in range(len(curr)) if j not in used_c]
    return pairs, disappear, appear


def score_object_permanence(
    frame_paths: list[Path] | list[str],
    *,
    sample_every: int = 2,
    k_slots: int = 8,
) -> PermanenceReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return PermanenceReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    if idxs[-1] != len(paths) - 1:
        idxs.append(len(paths) - 1)
    disappear = appear = 0
    transitions = 0
    prev_slots: list[Slot] | None = None
    h = w = 1
    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=k_slots)
        if prev_slots is not None:
            _pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w)
            # Only count massy slots as real entities
            dis_n = sum(1 for i in dis if prev_slots[i].mass > 3.0)
            app_n = sum(1 for j in app if slots[j].mass > 3.0)
            disappear += dis_n
            appear += app_n
            transitions += 1
        prev_slots = slots
    if transitions <= 0:
        return PermanenceReport(score=1.0)
    # Penalize events relative to transitions
    rate = (disappear + appear) / max(transitions * k_slots * 0.35, 1.0)
    score = float(np.clip(1.0 - rate, 0.0, 1.0))
    notes = []
    if disappear:
        notes.append(f"disappear_events={disappear}")
    if appear:
        notes.append(f"appear_events={appear}")
    return PermanenceReport(
        score=score,
        disappear_events=disappear,
        appear_events=appear,
        notes=notes,
    )


# ==================================================================================================
# artifact_critic: Artifact critic — GenVID-inspired axes, but actionable for SDX retry.
#
# Competitors hide failures behind pretty demos. We score Appearance / Motion /
# Camera explicitly and fail the segment when thresholds breach.
# ==================================================================================================


@dataclass(slots=True)
class ArtifactReport:
    """Higher scores = cleaner video (fewer artifacts)."""

    appearance: float
    motion: float
    camera: float
    overall: float
    flicker: float = 1.0
    color_drift: float = 1.0
    deformation: float = 1.0
    appear_disappear: float = 1.0
    shake: float = 1.0
    notes: list[str] = field(default_factory=list)


def _sample_paths(paths: list[Path], max_n: int = 24) -> list[Path]:
    if len(paths) <= max_n:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_n).astype(int)
    return [paths[i] for i in idx]


def score_appearance_artifacts(frame_paths: list[Path]) -> tuple[float, float, float, list[str]]:
    """Returns (appearance_score, flicker, color_drift, notes)."""
    paths = _sample_paths([Path(p) for p in frame_paths if Path(p).is_file()])
    notes: list[str] = []
    if len(paths) < 2:
        return 1.0, 1.0, 1.0, notes
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    # Flicker: high-frequency frame-to-frame energy after low-pass
    diffs = []
    for a, b in zip(frames[:-1], frames[1:], strict=False):
        diffs.append(float(np.mean(np.abs(a - b)) / 255.0))
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs))
    # Good motion has moderate mean_d; flicker has high var relative to mean
    flicker = float(np.clip(1.0 - (var_d / (mean_d + 1e-4)) * 0.35 - var_d * 8.0, 0.0, 1.0))
    # Alternating / soap-opera chatter: even↔odd oscillation with huge swing
    if len(frames) >= 4:
        even = np.stack([frames[i] for i in range(0, len(frames), 2)])
        odd = np.stack([frames[i] for i in range(1, len(frames), 2)])
        n = min(len(even), len(odd))
        parity = float(np.mean(np.abs(even[:n] - odd[:n])) / 255.0)
        # Penalize strong even/odd disagreement (classic gen flicker)
        if parity > 0.18:
            flicker = float(np.clip(flicker * (1.0 - parity) - parity * 0.5, 0.0, 1.0))
    # Color drift: mean RGB shift across clip
    means = [f.reshape(-1, 3).mean(axis=0) for f in frames]
    drift = float(np.mean(np.abs(means[-1] - means[0])) / 255.0)
    color = float(np.clip(1.0 - drift * 6.0, 0.0, 1.0))
    if flicker < 0.55:
        notes.append(f"flicker={flicker:.2f}")
    if color < 0.55:
        notes.append(f"color_drift={color:.2f}")
    appearance = 0.55 * flicker + 0.45 * color
    return appearance, flicker, color, notes


def score_motion_artifacts(frame_paths: list[Path]) -> tuple[float, float, float, list[str]]:
    """Returns (motion_score, deformation, appear_disappear, notes)."""

    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    notes: list[str] = []
    if len(paths) < 3:
        return 1.0, 1.0, 1.0, notes
    perm = score_object_permanence(paths)
    appear_disappear = float(perm.score)
    if perm.disappear_events or perm.appear_events:
        notes.extend(perm.notes)

    # Deformation proxy: local patch correlation collapse between frames
    sample = _sample_paths(paths, 12)
    corrs: list[float] = []
    for a_p, b_p in zip(sample[:-1], sample[1:], strict=False):
        a = read_frame_rgb(a_p).astype(np.float32)
        b = read_frame_rgb(b_p).astype(np.float32)
        # Center crop correlation
        h, w = a.shape[:2]
        y0, x0 = h // 4, w // 4
        y1, x1 = 3 * h // 4, 3 * w // 4
        pa = a[y0:y1, x0:x1].ravel()
        pb = b[y0:y1, x0:x1].ravel()
        pa = (pa - pa.mean()) / (pa.std() + 1e-6)
        pb = (pb - pb.mean()) / (pb.std() + 1e-6)
        corrs.append(float(np.mean(pa * pb)))
    # Extreme correlation drops without being near-zero global change → deform
    deform = float(np.clip(np.mean(corrs) * 0.5 + 0.5, 0.0, 1.0))
    if deform < 0.45:
        notes.append(f"deformation={deform:.2f}")
    motion = 0.55 * appear_disappear + 0.45 * deform
    return motion, deform, appear_disappear, notes


def score_camera_artifacts(frame_paths: list[Path]) -> tuple[float, float, list[str]]:
    """Returns (camera_score, shake, notes)."""
    paths = _sample_paths([Path(p) for p in frame_paths if Path(p).is_file()], 16)
    notes: list[str] = []
    if len(paths) < 3:
        return 1.0, 1.0, notes
    # Global translation proxy via phase-ish centroid of edges
    cents: list[tuple[float, float]] = []
    for p in paths:
        g = read_frame_rgb(p).astype(np.float32).mean(axis=2)
        gy = np.abs(np.diff(g, axis=0)).mean(axis=1)
        gx = np.abs(np.diff(g, axis=1)).mean(axis=0)
        ys = np.arange(len(gy), dtype=np.float32)
        xs = np.arange(len(gx), dtype=np.float32)
        cy = float((ys * gy).sum() / (gy.sum() + 1e-6))
        cx = float((xs * gx).sum() / (gx.sum() + 1e-6))
        cents.append((cy, cx))
    jumps = [
        ((cents[i][0] - cents[i - 1][0]) ** 2 + (cents[i][1] - cents[i - 1][1]) ** 2) ** 0.5
        for i in range(1, len(cents))
    ]
    # Smooth camera → low variance of jumps; shake → high variance
    jvar = float(np.var(jumps)) if jumps else 0.0
    shake = float(np.clip(1.0 - jvar / 50.0, 0.0, 1.0))
    # Static death: almost zero motion entire clip
    jmean = float(np.mean(jumps)) if jumps else 0.0
    static_pen = 0.0 if jmean > 0.4 else 0.25
    camera = float(np.clip(shake - static_pen, 0.0, 1.0))
    if shake < 0.5:
        notes.append(f"camera_shake={shake:.2f}")
    if static_pen > 0:
        notes.append("near_static_camera")
    return camera, shake, notes


def score_video_artifacts(frame_paths: list[Path] | list[str]) -> ArtifactReport:
    paths = [Path(p) for p in frame_paths]
    app, flicker, color, n1 = score_appearance_artifacts(paths)
    mot, deform, appear_d, n2 = score_motion_artifacts(paths)
    cam, shake, n3 = score_camera_artifacts(paths)
    overall = float(0.35 * app + 0.45 * mot + 0.20 * cam)
    notes = n1 + n2 + n3
    return ArtifactReport(
        appearance=app,
        motion=mot,
        camera=cam,
        overall=overall,
        flicker=flicker,
        color_drift=color,
        deformation=deform,
        appear_disappear=appear_d,
        shake=shake,
        notes=notes,
    )


# ==================================================================================================
# extremity_lock: Temporal hand / limb coherence score: melted fingers and structure that appears then dissolves.
#
# Scores each frame with a cheap hand-structure heuristic and flags temporal collapses (good to mush); the score
# gates segments (``min_extremity``).
# ==================================================================================================


@dataclass(slots=True)
class ExtremityReport:
    score: float
    mean_hand: float = 1.0
    stability: float = 1.0
    collapsed_frames: int = 0
    notes: list[str] = field(default_factory=list)
    repaired: int = 0


def _hand_score(rgb: np.ndarray) -> float:
    from utils.quality.critics import score_hand_structure

    return float(score_hand_structure(rgb))


def score_extremity_coherence(
    frame_paths: list[Path] | list[str],
    *,
    sample_every: int = 1,
    collapse_drop: float = 0.22,
) -> ExtremityReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ExtremityReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    scores = [_hand_score(read_frame_rgb(paths[i])) for i in idxs]
    mean_h = float(np.mean(scores))
    # Stability: 1 - normalized consecutive drop magnitude
    drops = [max(0.0, scores[i - 1] - scores[i]) for i in range(1, len(scores))]
    collapsed = sum(1 for d in drops if d >= collapse_drop)
    mean_drop = float(np.mean(drops)) if drops else 0.0
    stability = float(np.clip(1.0 - mean_drop * 2.5 - collapsed * 0.08, 0.0, 1.0))
    score = float(np.clip(0.55 * mean_h + 0.45 * stability, 0.0, 1.0))
    notes = []
    if collapsed:
        notes.append(f"hand_collapses={collapsed}")
    if mean_h < 0.4:
        notes.append(f"weak_hand_structure={mean_h:.2f}")
    return ExtremityReport(score=score, mean_hand=mean_h, stability=stability, collapsed_frames=collapsed, notes=notes)


# ==================================================================================================
# glyph_lock: Glyph lock — temporal logo / on-screen text stability.
#
# Product shots and title cards melt letters and warp logos. Finds high-frequency "glyph-ish" ROIs on the first
# frame and tracks their edge-hash distance over time (Hamming on binarized edges); the score gates segments
# (``min_glyph``).
# ==================================================================================================


@dataclass(slots=True)
class GlyphROI:
    y0: int
    x0: int
    y1: int
    x1: int
    edge_hash: np.ndarray  # flat binary


@dataclass(slots=True)
class GlyphReport:
    score: float
    melt_events: int = 0
    rois: int = 0
    notes: list[str] = field(default_factory=list)
    repaired: int = 0


def _edge_map(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
    return gx + gy


def _hash_patch(edge: np.ndarray, y0: int, x0: int, y1: int, x1: int, size: int = 16) -> np.ndarray:
    from PIL import Image

    patch = edge[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((size * size,), dtype=np.uint8)
    im = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    # Fixed mid threshold — mean-adaptive hashes are too invariant under mush
    thr = max(6.0, float(np.percentile(arr, 60)))
    return (arr > thr).astype(np.uint8).ravel()


def find_glyph_rois(rgb: np.ndarray, *, max_rois: int = 4, grid: int = 6) -> list[GlyphROI]:
    """Peak high-frequency cells as logo/text candidates."""
    edge = _edge_map(rgb)
    h, w = edge.shape
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
    cands: list[tuple[float, int, int, int, int]] = []
    global_mean = float(edge.mean()) + 1e-6
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * cell_h, gx * cell_w
            y1, x1 = min(h, y0 + cell_h), min(w, x0 + cell_w)
            patch = edge[y0:y1, x0:x1]
            energy = float(patch.mean())
            contrast = energy / global_mean
            # Glyphs: locally elevated edge energy vs frame
            if energy < 5.0 or contrast < 1.35:
                continue
            cands.append((energy * contrast, y0, x0, y1, x1))
    cands.sort(key=lambda t: -t[0])
    rois: list[GlyphROI] = []
    for _, y0, x0, y1, x1 in cands[:max_rois]:
        rois.append(GlyphROI(y0=y0, x0=x0, y1=y1, x1=x1, edge_hash=_hash_patch(edge, y0, x0, y1, x1)))
    return rois


def _hamming(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 1.0
    return float(np.mean(a[:n] != b[:n]))


def score_glyph_stability(
    frame_paths: list[Path] | list[str],
    *,
    melt_threshold: float = 0.28,
    sample_every: int = 2,
) -> GlyphReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return GlyphReport(score=1.0)
    donor = read_frame_rgb(paths[0])
    rois = find_glyph_rois(donor)
    if not rois:
        return GlyphReport(score=1.0, notes=["no_glyph_rois"])
    melts = 0
    dists: list[float] = []
    idxs = list(range(0, len(paths), max(1, sample_every)))
    for i in idxs[1:]:
        edge = _edge_map(read_frame_rgb(paths[i]))
        for roi in rois:
            h = _hash_patch(edge, roi.y0, roi.x0, roi.y1, roi.x1)
            d = _hamming(roi.edge_hash, h)
            dists.append(d)
            if d >= melt_threshold:
                melts += 1
    mean_d = float(np.mean(dists)) if dists else 0.0
    score = float(np.clip(1.0 - mean_d * 2.2 - melts * 0.04, 0.0, 1.0))
    notes = []
    if melts:
        notes.append(f"glyph_melts={melts}")
    return GlyphReport(score=score, melt_events=melts, rois=len(rois), notes=notes)


# ==================================================================================================
# identity_bind: Element Binding — multi-reference identity anchors (Kling-inspired, detector-free).
#
# Faces morph past ~60° turns and wardrobe color drifts mid-clip. A cheap multi-ref fingerprint
# (HSV hist + spatial chroma grid) scores every frame against the bound identity; the score gates segments
# (``min_identity``). No face detector required, so it also works on product/prop subjects.
# ==================================================================================================


@dataclass(slots=True)
class IdentityFingerprint:
    """Compact appearance prior from one or more reference crops."""

    hsv_hist: np.ndarray  # (48,)
    chroma_grid: np.ndarray  # (4, 4, 3) mean RGB in subject band
    band: tuple[float, float, float, float] = (0.12, 0.08, 0.88, 0.78)  # x0,y0,x1,y1 norm


@dataclass(slots=True)
class IdentityBindReport:
    score: float
    drift_frames: int = 0
    repaired: int = 0
    per_frame: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _subject_band(rgb: np.ndarray, band: tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = band
    xa, xb = int(w * x0), int(w * x1)
    ya, yb = int(h * y0), int(h * y1)
    xa, xb = max(0, min(w - 1, xa)), max(1, min(w, xb))
    ya, yb = max(0, min(h - 1, ya)), max(1, min(h, yb))
    return rgb[ya:yb, xa:xb]


def _hsv_hist(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(rgb[..., :3].astype(np.uint8)).convert("HSV")
    arr = np.asarray(im, dtype=np.float32)
    h = arr[..., 0].ravel()
    s = arr[..., 1].ravel()
    v = arr[..., 2].ravel()
    hh, _ = np.histogram(h, bins=bins, range=(0, 255), density=True)
    hs, _ = np.histogram(s, bins=bins, range=(0, 255), density=True)
    hv, _ = np.histogram(v, bins=bins, range=(0, 255), density=True)
    hist = np.concatenate([hh, hs, hv]).astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist


def _chroma_grid(rgb: np.ndarray, grid: int = 4) -> np.ndarray:
    from PIL import Image

    small = np.asarray(Image.fromarray(rgb[..., :3].astype(np.uint8)).resize((grid * 8, grid * 8)), dtype=np.float32)
    out = np.zeros((grid, grid, 3), dtype=np.float32)
    cell = 8
    for gy in range(grid):
        for gx in range(grid):
            patch = small[gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
            out[gy, gx] = patch.reshape(-1, 3).mean(axis=0)
    return out / 255.0


def build_fingerprint(rgb: np.ndarray, *, band: tuple[float, float, float, float] | None = None) -> IdentityFingerprint:
    b = band or (0.12, 0.08, 0.88, 0.78)
    crop = _subject_band(rgb, b)
    if crop.size == 0:
        crop = rgb
    return IdentityFingerprint(hsv_hist=_hsv_hist(crop), chroma_grid=_chroma_grid(crop), band=b)


def bind_from_refs(
    refs: list[str | Path] | list[np.ndarray],
    *,
    band: tuple[float, float, float, float] | None = None,
) -> IdentityFingerprint:
    """Average fingerprints across multiple angle/expression refs (Element Binding)."""
    fps: list[IdentityFingerprint] = []
    for r in refs:
        if isinstance(r, (str, Path)):
            rgb = read_frame_rgb(r)
        else:
            rgb = np.asarray(r)
        fps.append(build_fingerprint(rgb, band=band))
    if not fps:
        raise ValueError("bind_from_refs requires at least one reference")
    hist = np.mean([f.hsv_hist for f in fps], axis=0)
    hist /= hist.sum() + 1e-8
    grid = np.mean([f.chroma_grid for f in fps], axis=0)
    return IdentityFingerprint(hsv_hist=hist.astype(np.float32), chroma_grid=grid.astype(np.float32), band=fps[0].band)


def _similarity(fp: IdentityFingerprint, rgb: np.ndarray) -> float:
    other = build_fingerprint(rgb, band=fp.band)
    hist_sim = float(np.sqrt(fp.hsv_hist * other.hsv_hist).sum())
    chroma = 1.0 - float(np.mean(np.abs(fp.chroma_grid - other.chroma_grid)))
    return float(np.clip(0.55 * hist_sim + 0.45 * chroma, 0.0, 1.0))


def score_identity_bind(
    frame_paths: list[Path] | list[str],
    fingerprint: IdentityFingerprint | None = None,
    *,
    anchor: str | Path | None = None,
    refs: list[str | Path] | None = None,
    sample_every: int = 2,
    drift_threshold: float = 0.62,
) -> IdentityBindReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 1:
        return IdentityBindReport(score=1.0)
    if fingerprint is None:
        if refs:
            fingerprint = bind_from_refs(refs)
        elif anchor and Path(anchor).is_file():
            fingerprint = build_fingerprint(read_frame_rgb(anchor))
        else:
            fingerprint = build_fingerprint(read_frame_rgb(paths[0]))
    idxs = list(range(0, len(paths), max(1, sample_every)))
    if idxs[-1] != len(paths) - 1:
        idxs.append(len(paths) - 1)
    scores = [_similarity(fingerprint, read_frame_rgb(paths[i])) for i in idxs]
    full = [0.0] * len(paths)
    for j, i in enumerate(idxs):
        full[i] = scores[j]
    for i in range(len(paths)):
        if full[i] == 0.0:
            nearest = min(idxs, key=lambda k: abs(k - i))
            full[i] = full[nearest]
    drift = sum(1 for s in scores if s < drift_threshold)
    mean = float(np.mean(scores)) if scores else 1.0
    notes = []
    if drift:
        notes.append(f"identity_drift_frames={drift}")
    return IdentityBindReport(score=mean, drift_frames=drift, per_frame=full, notes=notes)


# ==================================================================================================
# physics_gate: Physics gate — Physical Invariance Score lite (NewtonGen / CausalMotion inspired).
#
# SOTA models still invent impossible motion: objects accelerate upward "falling",
# velocity reverses without braking, contacts ignore floors. We don't run a full
# sim — we score trajectory sanity on permanence slots and fail/retry when broken.
#
# Training-free, detector-light, actionable for segment_retry.
# ==================================================================================================


@dataclass(slots=True)
class PhysicsReport:
    """Higher = more physically plausible."""

    score: float
    velocity_flip_penalties: int = 0
    anti_gravity_events: int = 0
    jump_discontinuities: int = 0
    notes: list[str] = field(default_factory=list)


def score_physics_invariance(
    frame_paths: list[Path] | list[str],
    *,
    sample_every: int = 1,
    k_slots: int = 6,
) -> PhysicsReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return PhysicsReport(score=1.0)

    idxs = list(range(0, len(paths), max(1, sample_every)))
    if idxs[-1] != len(paths) - 1:
        idxs.append(len(paths) - 1)

    # Track primary slot (highest mass) trajectories via greedy match
    trajectories: dict[int, list[tuple[float, float]]] = {}  # track_id -> [(cy,cx), ...]
    next_id = 0
    prev_slots = None
    prev_ids: list[int] = []
    h = w = 1

    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=k_slots)
        if prev_slots is None:
            prev_ids = list(range(len(slots)))
            next_id = len(slots)
            for tid, s in zip(prev_ids, slots, strict=False):
                trajectories.setdefault(tid, []).append((s.cy, s.cx))
            prev_slots = slots
            continue
        pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w, max_dist=0.22)
        curr_ids = [-1] * len(slots)
        for pi, cj in pairs:
            tid = prev_ids[pi]
            curr_ids[cj] = tid
            trajectories.setdefault(tid, []).append((slots[cj].cy, slots[cj].cx))
        for cj in app:
            tid = next_id
            next_id += 1
            curr_ids[cj] = tid
            trajectories.setdefault(tid, []).append((slots[cj].cy, slots[cj].cx))
        prev_slots = slots
        prev_ids = curr_ids

    flips = anti_g = jumps = 0
    appear_pop = 0
    for pts in trajectories.values():
        if len(pts) < 3:
            continue
        vys = [pts[i][0] - pts[i - 1][0] for i in range(1, len(pts))]
        vxs = [pts[i][1] - pts[i - 1][1] for i in range(1, len(pts))]
        for i in range(1, len(vys)):
            # Abrupt direction reverse without near-zero speed
            speed_prev = (vys[i - 1] ** 2 + vxs[i - 1] ** 2) ** 0.5
            speed = (vys[i] ** 2 + vxs[i] ** 2) ** 0.5
            if speed_prev > 1.5 and speed > 1.5:
                # Dot product of consecutive velocities
                dot = vys[i - 1] * vys[i] + vxs[i - 1] * vxs[i]
                if dot < -0.5 * speed_prev * speed:
                    flips += 1
            # Anti-gravity: strong upward acceleration (cy decreases = up in image coords)
            ay = vys[i] - vys[i - 1]
            if ay < -3.0 and vys[i] < -2.0:  # accelerating upward hard
                anti_g += 1
            # Teleport discontinuity
            if speed > max(h, w) * 0.12:
                jumps += 1

    # Re-scan for pop teleports: massy slot disappears and another appears far away same step
    prev_slots = None
    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=k_slots)
        if prev_slots is not None:
            _pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w, max_dist=0.22)
            massy_dis = [prev_slots[i] for i in dis if prev_slots[i].mass > 3.0]
            massy_app = [slots[j] for j in app if slots[j].mass > 3.0]
            if massy_dis and massy_app:
                # If a disappeared and appeared slot are far apart → teleport/pop
                for a in massy_dis:
                    for b in massy_app:
                        dist = ((a.cy - b.cy) ** 2 + (a.cx - b.cx) ** 2) ** 0.5
                        if dist > max(h, w) * 0.25:
                            appear_pop += 1
                            jumps += 1
        prev_slots = slots

    # Normalize penalties by trajectory-steps
    steps = max(1, sum(max(0, len(p) - 2) for p in trajectories.values()) + appear_pop)
    rate = (flips * 1.2 + anti_g * 1.5 + jumps * 1.0) / steps
    score = float(np.clip(1.0 - rate * 3.0, 0.0, 1.0))
    notes = []
    if flips:
        notes.append(f"velocity_flips={flips}")
    if anti_g:
        notes.append(f"anti_gravity={anti_g}")
    if jumps:
        notes.append(f"teleports={jumps}")
    if appear_pop:
        notes.append(f"pop_teleports={appear_pop}")
    return PhysicsReport(
        score=score,
        velocity_flip_penalties=flips,
        anti_gravity_events=anti_g,
        jump_discontinuities=jumps,
        notes=notes,
    )


# ==================================================================================================
# quality
# ==================================================================================================


def score_frame_sharpness(rgb: np.ndarray) -> float:
    gray = rgb[..., :3].astype(np.float32).mean(axis=2) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return float(min(1.0, (gx + gy) * 0.5 * 4.0))


def score_temporal_consistency(frame_paths: Sequence[Path]) -> float:
    if len(frame_paths) < 2:
        return 1.0
    diffs: list[float] = []
    prev = read_frame_rgb(frame_paths[0]).astype(np.float32)
    for p in frame_paths[1:]:
        curr = read_frame_rgb(p).astype(np.float32)
        d = float(np.mean(np.abs(curr - prev)) / 255.0)
        diffs.append(d)
        prev = curr
    mean_d = float(np.mean(diffs))
    # Lower diff variance + moderate mean diff = good
    var_d = float(np.var(diffs))
    score = 1.0 - min(1.0, mean_d * 2.5 + var_d * 4.0)
    return max(0.0, score)


def score_segment_quality(
    frame_paths: Sequence[Path],
    *,
    segment_index: int = 0,
    min_temporal: float = 0.35,
    min_sharpness: float = 0.08,
    min_permanence: float = 0.0,
    min_artifact: float = 0.0,
    min_identity: float = 0.0,
    min_extremity: float = 0.0,
    min_physics: float = 0.0,
    min_glyph: float = 0.0,
) -> SegmentQualityReport:
    """Score a segment; an optional axis is only measured (and gated) when its ``min_*`` floor is > 0."""
    paths = list(frame_paths)
    temporal = score_temporal_consistency(paths)
    sharpness = float(np.mean([score_frame_sharpness(read_frame_rgb(p)) for p in paths[: min(8, len(paths))]]))
    notes: list[str] = []
    if temporal < min_temporal:
        notes.append(f"low temporal consistency ({temporal:.3f})")
    if sharpness < min_sharpness:
        notes.append(f"low sharpness ({sharpness:.3f})")
    passed = temporal >= min_temporal and sharpness >= min_sharpness

    axes = (
        ("permanence", min_permanence, lambda: score_object_permanence(paths).score),
        ("artifact", min_artifact, lambda: score_video_artifacts(paths).overall),
        ("identity", min_identity, lambda: score_identity_bind(paths).score),
        ("extremity", min_extremity, lambda: score_extremity_coherence(paths).score),
        ("physics", min_physics, lambda: score_physics_invariance(paths).score),
        ("glyph", min_glyph, lambda: score_glyph_stability(paths).score),
    )
    axis_scores: dict[str, float] = {}
    for name, floor, scorer in axes:
        if float(floor) <= 0.0:
            continue
        score = float(np.clip(scorer(), 0.0, 1.0))
        axis_scores[f"{name}_score"] = score
        if score < float(floor):
            notes.append(f"low {name} ({score:.3f})")
            passed = False

    return SegmentQualityReport(
        segment_index=segment_index,
        temporal_score=temporal,
        sharpness_score=sharpness,
        passed=passed,
        notes=notes,
        **axis_scores,
    )


# ==================================================================================================
# drift_repair: Appearance drift vs the anchor frame (color histogram + structure proxy) and neighbor-blend repair.
# ==================================================================================================


@dataclass(slots=True)
class DriftRepairReport:
    drift_indices: list[int]
    repaired_count: int
    max_drift: float
    mean_drift: float


def _color_histogram(rgb: np.ndarray, *, bins: int = 16) -> np.ndarray:
    hists = [np.histogram(rgb[..., c], bins=bins, range=(0, 256))[0].astype(np.float32) for c in range(3)]
    return np.concatenate([h / (h.sum() + 1e-6) for h in hists])


def _histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 = identical, 1 = very different."""
    return min(1.0, float(np.sum(np.abs(a - b)) * 0.5))


def _centered_luma(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    return g - g.mean()


def _structure_distance_centered(ga: np.ndarray, gb: np.ndarray) -> float:
    corr = float(np.sum(ga * gb)) / float(np.sqrt(np.sum(ga * ga) * np.sum(gb * gb)) + 1e-6)
    return 1.0 - min(1.0, max(0.0, (corr + 1.0) * 0.5))


def _structure_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - rescaled luma correlation (cheap SSIM stand-in)."""
    return _structure_distance_centered(_centered_luma(a), _centered_luma(b))


def _drift_from_parts(h_cur, g_cur, h_ref, g_ref, h_prev) -> float:
    score = 0.55 * _histogram_distance(h_cur, h_ref) + 0.45 * _structure_distance_centered(g_cur, g_ref)
    if h_prev is not None:
        score = 0.7 * score + 0.3 * _histogram_distance(h_cur, h_prev)
    return float(min(1.0, max(0.0, score)))


def frame_drift_score(frame_rgb: np.ndarray, reference_rgb: np.ndarray, *, prev_rgb: np.ndarray | None = None) -> float:
    """Higher = more drift from the reference (and the previous frame when given)."""
    h_prev = _color_histogram(prev_rgb) if prev_rgb is not None else None
    return _drift_from_parts(
        _color_histogram(frame_rgb),
        _centered_luma(frame_rgb),
        _color_histogram(reference_rgb),
        _centered_luma(reference_rgb),
        h_prev,
    )


def score_sequence_drift(frame_paths: Sequence[Path], anchor_path: str | Path | None = None) -> list[float]:
    if not frame_paths:
        return []
    ref = read_frame_rgb(anchor_path) if anchor_path and Path(anchor_path).is_file() else read_frame_rgb(frame_paths[0])
    h_ref, g_ref = _color_histogram(ref), _centered_luma(ref)
    h_prev = None
    out: list[float] = []
    for fp in frame_paths:
        rgb = read_frame_rgb(fp)
        h_cur = _color_histogram(rgb)
        out.append(_drift_from_parts(h_cur, _centered_luma(rgb), h_ref, g_ref, h_prev))
        h_prev = h_cur
    return out


def find_drift_frames(
    frame_paths: Sequence[Path], *, anchor_path: str | Path | None = None, threshold: float = 0.55
) -> list[int]:
    return [i for i, s in enumerate(score_sequence_drift(frame_paths, anchor_path)) if s >= threshold]


def repair_sequence_drift(
    frame_paths: list[Path],
    *,
    anchor_path: str | Path | None = None,
    threshold: float = 0.55,
    blend_strength: float = 0.45,
) -> DriftRepairReport:
    """Blend frames whose drift is >= ``threshold`` toward their temporal neighbors (65%) and the anchor (35%)."""
    scores = score_sequence_drift(frame_paths, anchor_path)
    drift_idx = [i for i, s in enumerate(scores) if s >= threshold] if len(frame_paths) >= 2 else []
    rgbs = [read_frame_rgb(p) for p in frame_paths] if drift_idx else []
    anchor = read_frame_rgb(anchor_path) if anchor_path and Path(anchor_path).is_file() else (rgbs[0] if rgbs else None)
    b = float(max(0.0, min(1.0, blend_strength)))
    for i in drift_idx:
        refs = [rgbs[j].astype(np.float32) for j in (i - 1, i + 1) if 0 <= j < len(rgbs)]
        neighbor = np.mean(refs, axis=0)
        fixed = rgbs[i].astype(np.float32) * (1.0 - b) + neighbor * (b * 0.65) + anchor.astype(np.float32) * (b * 0.35)
        rgbs[i] = np.clip(fixed, 0, 255).astype(np.uint8)
        save_frame_rgb(frame_paths[i], rgbs[i])
    return DriftRepairReport(
        drift_indices=drift_idx,
        repaired_count=len(drift_idx),
        max_drift=float(max(scores, default=0.0)),
        mean_drift=float(np.mean(scores)) if scores else 0.0,
    )


# ==================================================================================================
# identity_lock: Lock identity regions across frames (face/body from anchor).
# ==================================================================================================


def composite_with_mask(
    frame_rgb: np.ndarray,
    anchor_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.85,
) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() > 1.0:
        m = m / 255.0
    if m.shape[:2] != frame_rgb.shape[:2]:
        import cv2

        m = cv2.resize(m, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    s = float(max(0.0, min(1.0, strength)))
    m3 = m[..., None]
    out = frame_rgb.astype(np.float32) * (1.0 - m3 * s) + anchor_rgb.astype(np.float32) * (m3 * s)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_identity_lock_to_sequence(
    frame_paths: list[Path],
    anchor_path: str | Path,
    mask_path: str | Path | None = None,
    *,
    box: tuple[float, float, float, float] | None = None,
    strength: float = 0.85,
) -> list[Path]:
    """Re-apply anchor pixels inside mask/box on every frame."""
    from pipelines.video.motion import load_mask_binary, rasterize_box_mask

    anchor = read_frame_rgb(anchor_path)
    if not frame_paths:
        return []
    h, w = read_frame_rgb(frame_paths[0]).shape[:2]
    if mask_path and Path(mask_path).is_file():
        mask0 = load_mask_binary(mask_path, shape=(h, w))
    elif box is not None:
        mask0 = rasterize_box_mask(w, h, box)
    else:
        # Default: center subject box
        mask0 = rasterize_box_mask(w, h, (0.25, 0.05, 0.75, 0.95))
    for fp in frame_paths:
        rgb = read_frame_rgb(fp)
        locked = composite_with_mask(rgb, anchor, mask0, strength=strength)
        save_frame_rgb(fp, locked)
    return frame_paths


def apply_propagated_identity_lock(
    frame_paths: list[Path],
    anchor_path: str | Path,
    *,
    mask_path: str | Path | None = None,
    box: tuple[float, float, float, float] | None = None,
    strength: float = 0.85,
    work_dir: Path | None = None,
) -> list[Path]:
    """Per-frame mask propagation then anchor composite (tracks moving subjects)."""
    from pipelines.video.motion import load_mask_binary, propagate_mask_sequence, rasterize_box_mask

    if not frame_paths:
        return []
    anchor = read_frame_rgb(anchor_path)
    h, w = read_frame_rgb(frame_paths[0]).shape[:2]
    if mask_path and Path(mask_path).is_file():
        mask0 = load_mask_binary(mask_path, shape=(h, w))
    elif box is not None:
        mask0 = rasterize_box_mask(w, h, box)
    else:
        mask0 = rasterize_box_mask(w, h, (0.25, 0.05, 0.75, 0.95))

    mask_dir = (work_dir or frame_paths[0].parent) / "_prop_masks"
    if mask_dir.exists():
        import shutil

        shutil.rmtree(mask_dir)
    mask_paths = propagate_mask_sequence(mask0, frame_paths, out_dir=mask_dir)
    for fp, mp in zip(frame_paths, mask_paths[: len(frame_paths)], strict=False):
        m = load_mask_binary(mp, shape=(h, w))
        rgb = read_frame_rgb(fp)
        locked = composite_with_mask(rgb, anchor, m, strength=strength)
        save_frame_rgb(fp, locked)
    return frame_paths


# ==================================================================================================
# Restored from b654ae9:pipelines/video/permanence.py
# ==================================================================================================
def _paste_bbox(dst: np.ndarray, src: np.ndarray, bbox: tuple[int, int, int, int], *, alpha: float) -> None:
    y0, x0, y1, x1 = bbox
    patch = src[y0:y1, x0:x1]
    if patch.size == 0:
        return
    region = dst[y0:y1, x0:x1].astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    dst[y0:y1, x0:x1] = np.clip(region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255).astype(np.uint8)


def repair_permanence_breaks(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
    k_slots: int = 8,
) -> PermanenceReport:
    """
    When a massive slot disappears with little global motion, reinject last patch.
    """
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return PermanenceReport(score=1.0)
    repaired = 0
    disappear = appear = 0
    prev_rgb = read_frame_rgb(paths[0])
    prev_slots = extract_slots(prev_rgb, k=k_slots)
    h, w = prev_rgb.shape[:2]
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        slots = extract_slots(curr, k=k_slots)
        pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w)
        disappear += len(dis)
        appear += len(app)
        # Global motion proxy
        gmotion = float(np.mean(np.abs(curr.astype(np.float32) - prev_rgb.astype(np.float32))) / 255.0)
        if gmotion < 0.12:  # mostly static camera — pops are suspicious
            changed = False
            for i in dis:
                slot = prev_slots[i]
                if slot.mass < 4.0:
                    continue
                _paste_bbox(curr, prev_rgb, slot.bbox, alpha=strength)
                repaired += 1
                changed = True
            if changed:
                save_frame_rgb(p, curr)
                curr = read_frame_rgb(p)
        prev_rgb = curr
        prev_slots = extract_slots(curr, k=k_slots)
    base = score_object_permanence(paths)
    base.repaired = repaired
    if repaired:
        base.notes.append(f"repaired={repaired}")
    return base


def apply_permanence_pass(
    frame_paths: list[Path],
    *,
    repair: bool = True,
    strength: float = 0.55,
) -> PermanenceReport:
    if repair:
        return repair_permanence_breaks(frame_paths, strength=strength)
    return score_object_permanence(frame_paths)


# ==================================================================================================
# Restored from b654ae9:pipelines/video/identity_bind.py
# ==================================================================================================
def apply_identity_bind(
    frame_paths: list[Path],
    *,
    fingerprint: IdentityFingerprint | None = None,
    anchor: str | Path | None = None,
    refs: list[str | Path] | None = None,
    strength: float = 0.45,
    drift_threshold: float = 0.62,
) -> IdentityBindReport:
    """Re-anchor drifted frames toward the best-matching frame's subject band."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return IdentityBindReport(score=1.0)
    report = score_identity_bind(
        paths, fingerprint, anchor=anchor, refs=refs, sample_every=1, drift_threshold=drift_threshold
    )
    fp = fingerprint
    if fp is None:
        if refs:
            fp = bind_from_refs(refs)
        elif anchor and Path(anchor).is_file():
            fp = build_fingerprint(read_frame_rgb(anchor))
        else:
            fp = build_fingerprint(read_frame_rgb(paths[0]))
    donor_i = int(np.argmax(report.per_frame)) if report.per_frame else 0
    donor = read_frame_rgb(paths[donor_i])
    donor_band = _subject_band(donor, fp.band)
    repaired = 0
    from PIL import Image

    for i, p in enumerate(paths):
        if report.per_frame[i] >= drift_threshold:
            continue
        curr = read_frame_rgb(p)
        h, w = curr.shape[:2]
        x0, y0, x1, y1 = fp.band
        xa, xb = int(w * x0), int(w * x1)
        ya, yb = int(h * y0), int(h * y1)
        patch = np.asarray(Image.fromarray(donor_band).resize((max(1, xb - xa), max(1, yb - ya)), Image.BILINEAR))
        region = curr[ya:yb, xa:xb].astype(np.float32)
        a = float(np.clip(strength, 0.0, 1.0))
        if patch.shape[:2] == region.shape[:2]:
            curr[ya:yb, xa:xb] = np.clip(region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255).astype(np.uint8)
            save_frame_rgb(p, curr)
            repaired += 1
    after = score_identity_bind(paths, fp, sample_every=1, drift_threshold=drift_threshold)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"identity_bind_repaired={repaired}")
    return after


# ==================================================================================================
# Restored from b654ae9:pipelines/video/extremity_lock.py
# ==================================================================================================
def _extremity_mask(h: int, w: int) -> np.ndarray:
    """Lower 45% + side margins — where hands/arms usually live."""
    m = np.zeros((h, w), dtype=np.float32)
    m[int(h * 0.55) :, :] = 1.0
    m[:, : max(1, w // 4)] = np.maximum(m[:, : max(1, w // 4)], 0.85)
    m[:, -max(1, w // 4) :] = np.maximum(m[:, -max(1, w // 4) :], 0.85)
    return m


def apply_extremity_lock(
    frame_paths: list[Path],
    *,
    strength: float = 0.5,
    collapse_drop: float = 0.22,
) -> ExtremityReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ExtremityReport(score=1.0)
    scores = [_hand_score(read_frame_rgb(p)) for p in paths]
    donor_i = int(np.argmax(scores))
    donor = read_frame_rgb(paths[donor_i]).astype(np.float32)
    h, w = donor.shape[:2]
    mask = _extremity_mask(h, w)[..., None]
    repaired = 0
    for i, p in enumerate(paths):
        if i == 0:
            continue
        drop = scores[i - 1] - scores[i]
        if drop < collapse_drop and scores[i] >= 0.35:
            continue
        # Prefer previous good frame if better than global donor for local continuity
        src_i = i - 1 if scores[i - 1] > scores[donor_i] * 0.9 else donor_i
        src = read_frame_rgb(paths[src_i]).astype(np.float32)
        curr = read_frame_rgb(p).astype(np.float32)
        a = float(np.clip(strength, 0.0, 1.0))
        out = curr * (1.0 - mask * a) + src * (mask * a)
        save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
        repaired += 1
    after = score_extremity_coherence(paths, collapse_drop=collapse_drop)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"extremity_repaired={repaired}")
    return after


# ==================================================================================================
# Restored from b654ae9:pipelines/video/glyph_lock.py
# ==================================================================================================
def apply_glyph_lock(
    frame_paths: list[Path],
    *,
    strength: float = 0.7,
    melt_threshold: float = 0.28,
) -> GlyphReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return GlyphReport(score=1.0)
    donor = read_frame_rgb(paths[0])
    rois = find_glyph_rois(donor)
    if not rois:
        return GlyphReport(score=1.0, notes=["no_glyph_rois"])
    repaired = 0
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        edge = _edge_map(curr)
        changed = False
        for roi in rois:
            h = _hash_patch(edge, roi.y0, roi.x0, roi.y1, roi.x1)
            if _hamming(roi.edge_hash, h) < melt_threshold:
                continue
            patch = donor[roi.y0 : roi.y1, roi.x0 : roi.x1]
            region = curr[roi.y0 : roi.y1, roi.x0 : roi.x1].astype(np.float32)
            a = float(np.clip(strength, 0.0, 1.0))
            curr[roi.y0 : roi.y1, roi.x0 : roi.x1] = np.clip(
                region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255
            ).astype(np.uint8)
            changed = True
            repaired += 1
        if changed:
            save_frame_rgb(p, curr)
    after = score_glyph_stability(paths, melt_threshold=melt_threshold)
    after.repaired = repaired
    after.rois = len(rois)
    if repaired:
        after.notes.append(f"glyph_repaired={repaired}")
    return after


# ==================================================================================================
# Restored from b654ae9:pipelines/video/physics_gate.py
# ==================================================================================================


@dataclass(slots=True)
class SuperiorPassResult:
    ops: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


def run_superior_pass(
    frame_paths: list[Path],
    opts: Any,
    *,
    prompt: str = "",
    anchor: str = "",
    identity_refs: list[str] | None = None,
) -> SuperiorPassResult:
    """Apply enabled superiority repairs in a stable order; collect scores."""
    out = SuperiorPassResult()
    paths = list(frame_paths)
    if len(paths) < 2:
        return out

    if bool(getattr(opts, "permanence_repair", True)):
        r = apply_permanence_pass(
            paths, repair=True, strength=float(getattr(opts, "permanence_strength", 0.55) or 0.55)
        )
        if r.repaired:
            out.ops.append(f"permanence_repair:{r.repaired}")
        out.scores["permanence"] = float(r.score)
        out.ops.append(f"permanence_score:{r.score:.2f}")

    refs = list(identity_refs or getattr(opts, "identity_refs", ()) or ())
    if bool(getattr(opts, "identity_bind", True)):
        r = apply_identity_bind(
            paths,
            anchor=anchor or None,
            refs=refs or None,
            strength=float(getattr(opts, "identity_bind_strength", 0.45) or 0.45),
        )
        if r.repaired:
            out.ops.append(f"identity_bind:{r.repaired}")
        out.scores["identity"] = float(r.score)
        out.ops.append(f"identity_score:{r.score:.2f}")

    if bool(getattr(opts, "extremity_lock", True)):
        r = apply_extremity_lock(paths, strength=float(getattr(opts, "extremity_strength", 0.50) or 0.50))
        if r.repaired:
            out.ops.append(f"extremity_lock:{r.repaired}")
        out.scores["extremity"] = float(r.score)

    glyph_on = bool(getattr(opts, "glyph_lock", False)) or str(getattr(opts, "motion_grammar", "")) == "product"
    if glyph_on:
        r = apply_glyph_lock(paths, strength=float(getattr(opts, "glyph_strength", 0.70) or 0.70))
        if r.repaired:
            out.ops.append(f"glyph_lock:{r.repaired}")
        out.scores["glyph"] = float(r.score)

    if bool(getattr(opts, "contact_ground", True)):
        r = apply_contact_ground(paths, strength=float(getattr(opts, "contact_strength", 0.55) or 0.55))
        if r.repaired:
            out.ops.append(f"contact_ground:{r.repaired}")
        out.scores["contact"] = float(r.score)

    if bool(getattr(opts, "secondary_track", True)):
        r = apply_secondary_track(paths, strength=float(getattr(opts, "secondary_strength", 0.55) or 0.55))
        if r.repaired:
            out.ops.append(f"secondary_track:{r.repaired}")
        out.scores["secondary"] = float(r.score)

    if bool(getattr(opts, "occlusion_resolve", False)):
        r = apply_occlusion_resolve(paths, strength=float(getattr(opts, "occlusion_strength", 0.40) or 0.40))
        if r.repaired:
            out.ops.append(f"occlusion_resolve:{r.repaired}")
        out.scores["occlusion"] = float(r.score)

    if bool(getattr(opts, "hf_deshimmer", True)):
        r = apply_hf_deshimmer(paths, strength=float(getattr(opts, "shimmer_strength", 0.55) or 0.55))
        if r.repaired:
            out.ops.append(f"hf_deshimmer:{r.repaired}")
        out.scores["shimmer"] = float(r.score)

    mg = str(getattr(opts, "motion_grammar", "") or "")
    shutter_on = bool(getattr(opts, "motion_shutter", False)) or mg in ("film", "realistic")
    if shutter_on and len(paths) >= 3:
        r = apply_motion_shutter(paths, amount=float(getattr(opts, "shutter_amount", 0.45) or 0.45))
        if r.applied:
            out.ops.append(f"motion_shutter:{r.applied}")
        out.scores["shutter"] = float(r.score)

    if bool(getattr(opts, "physics_gate", True)) and len(paths) >= 3:
        r = score_physics_invariance(paths)
        out.scores["physics"] = float(r.score)
        out.ops.append(f"physics_score:{r.score:.2f}")
        out.ops.extend(r.notes[:3])

    if bool(getattr(opts, "count_bind", True)):
        r = score_count_stability(paths, prompt=prompt)
        out.scores["count"] = float(r.score)
        if r.notes:
            out.ops.extend(r.notes[:2])

    return out


@dataclass(slots=True)
class ContactReport:
    score: float
    float_events: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _foot_band(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Lower 22% of frame as candidate foot/ground interface."""
    h, w = rgb.shape[:2]
    y0 = int(h * 0.78)
    return rgb[y0:, :], y0, h


def _subject_mask_lower(rgb: np.ndarray) -> np.ndarray:
    """Cheap subject proxy: pixels darker/contrasty vs floor median."""
    band, _, _ = _foot_band(rgb)
    gray = band.astype(np.float32).mean(axis=2)
    med = float(np.median(gray))
    # Subject feet usually darker or much brighter than flat floor
    mask = (np.abs(gray - med) > 18.0).astype(np.float32)
    return mask


def score_contact_ground(frame_paths: list[Path] | list[str], *, sample_every: int = 2) -> ContactReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        return ContactReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    floats = 0
    scores: list[float] = []
    for i in idxs:
        rgb = read_frame_rgb(paths[i])
        band, y0, h = _foot_band(rgb)
        gray = band.astype(np.float32).mean(axis=2)
        mask = _subject_mask_lower(rgb)
        if mask.mean() < 0.02:
            scores.append(0.7)
            continue
        # Halo just below subject mass (contact AO zone)
        bh, bw = gray.shape
        # Bottom row of subject columns
        cols = np.where(mask.max(axis=0) > 0.5)[0]
        if cols.size == 0:
            scores.append(0.7)
            continue
        # For each subject column, compare pixel under foot tip vs floor further down
        aos: list[float] = []
        for c in cols[:: max(1, len(cols) // 8)]:
            rows = np.where(mask[:, c] > 0.5)[0]
            if rows.size == 0:
                continue
            tip = int(rows.max())
            under = tip + 1
            far = min(bh - 1, tip + max(3, bh // 6))
            if under >= bh:
                continue
            aos.append(float(gray[far, c] - gray[under, c]) / 255.0)
        if not aos:
            scores.append(0.6)
            continue
        ao = float(np.mean(aos))
        # Positive ao = under tip darker than far floor = contact shadow present
        s = float(np.clip(0.4 + ao * 5.0, 0.0, 1.0))
        if ao < 0.015:
            floats += 1
        scores.append(s)
    mean = float(np.mean(scores)) if scores else 1.0
    notes = [f"float_events={floats}"] if floats else []
    return ContactReport(score=mean, float_events=floats, notes=notes)


def apply_contact_ground(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
) -> ContactReport:
    """Paint soft elliptical contact shadows under lower-band subject mass."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        return ContactReport(score=1.0)
    repaired = 0
    a = float(np.clip(strength, 0.0, 1.0))
    for p in paths:
        rgb = read_frame_rgb(p).astype(np.float32)
        h, w = rgb.shape[:2]
        mask = _subject_mask_lower(rgb.astype(np.uint8))
        if mask.mean() < 0.015:
            continue
        # Centroid of foot mass
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        # Shadow sits just below foot mass in full frame coords
        y0 = int(h * 0.78)
        cy = y0 + float(ys.mean()) + max(2.0, h * 0.01)
        rx = max(4.0, (xs.max() - xs.min()) * 0.55)
        ry = max(2.0, h * 0.012)
        yy, xx = np.mgrid[0:h, 0:w]
        ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        shadow = np.clip(1.0 - ell, 0.0, 1.0)[..., None] * (0.35 * a)
        # Only darken, never brighten
        out = rgb * (1.0 - shadow)
        save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
        repaired += 1
    after = score_contact_ground(paths, sample_every=1)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"contact_repaired={repaired}")
    return after


@dataclass(slots=True)
class CountReport:
    score: float
    expected: int = 0
    median_count: float = 0.0
    drop_events: int = 0
    notes: list[str] = field(default_factory=list)


def expected_count_from_prompt(prompt: str) -> int:
    """Best-effort expected subject count from prompt text."""
    try:
        from utils.quality.critics import analyze_prompt_counts

        rep = analyze_prompt_counts(prompt or "")
        if rep.expected_counts:
            return max(int(v) for v in rep.expected_counts.values())
        return max(1, int(rep.expected_subjects))
    except Exception:
        text = (prompt or "").lower()
        for word, n in (
            ("five ", 5),
            ("four ", 4),
            ("three ", 3),
            ("two ", 2),
            ("a pair", 2),
            ("couple", 2),
            ("group of", 3),
        ):
            if word in text:
                return n
        return 1


def _massy_slot_count(rgb: np.ndarray, *, mass_min: float = 3.5) -> int:
    slots = extract_slots(rgb, k=12)
    return sum(1 for s in slots if s.mass >= mass_min)


def score_count_stability(
    frame_paths: list[Path] | list[str],
    *,
    expected: int = 0,
    prompt: str = "",
    sample_every: int = 2,
    drop_tol: float = 0.45,
) -> CountReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return CountReport(score=1.0, expected=expected)
    exp = int(expected) if expected > 0 else expected_count_from_prompt(prompt)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    counts = [_massy_slot_count(read_frame_rgb(paths[i])) for i in idxs]
    med = float(np.median(counts)) if counts else 0.0
    drops = 0
    for i in range(1, len(counts)):
        if counts[i - 1] >= 2 and counts[i] <= counts[i - 1] * (1.0 - drop_tol):
            drops += 1
    # Score vs expected + temporal stability
    if exp > 0 and med > 0:
        ratio = min(med, exp) / max(med, exp)
    else:
        ratio = 1.0
    stab = float(np.clip(1.0 - drops * 0.2, 0.0, 1.0))
    score = float(np.clip(0.55 * ratio + 0.45 * stab, 0.0, 1.0))
    notes = []
    if drops:
        notes.append(f"count_drops={drops}")
    if exp > 0 and abs(med - exp) >= 1.5:
        notes.append(f"count_median={med:.1f}_expected={exp}")
    return CountReport(score=score, expected=exp, median_count=med, drop_events=drops, notes=notes)


@dataclass(slots=True)
class ShimmerReport:
    score: float
    shimmer_energy: float = 0.0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _hf_map(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    # Laplacian-ish high frequency
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return np.abs(lap)


def score_hf_shimmer(frame_paths: list[Path] | list[str], *, sample_every: int = 1) -> ShimmerReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return ShimmerReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    maps = [_hf_map(read_frame_rgb(paths[i])) for i in idxs]
    # Temporal variance of HF energy
    stack = np.stack(maps, axis=0)
    tvar = stack.var(axis=0)
    # Global motion proxy
    rgbs = [read_frame_rgb(paths[i]).astype(np.float32) for i in idxs]
    gmot = float(np.mean([np.mean(np.abs(rgbs[i] - rgbs[i - 1])) / 255.0 for i in range(1, len(rgbs))]))
    shimmer = float(tvar.mean())
    # Penalize high HF temporal variance when global motion is low
    pen = shimmer / (8.0 + gmot * 40.0)
    score = float(np.clip(1.0 - pen, 0.0, 1.0))
    notes = []
    if score < 0.7:
        notes.append(f"hf_shimmer={shimmer:.2f}")
    return ShimmerReport(score=score, shimmer_energy=shimmer, notes=notes)


def apply_hf_deshimmer(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
    window: int = 3,
) -> ShimmerReport:
    """Temporal median on high-frequency residual (keeps low-freq structure)."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return ShimmerReport(score=1.0)
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    w = max(3, int(window) | 1)  # odd
    half = w // 2
    a = float(np.clip(strength, 0.0, 1.0))
    repaired = 0
    for i in range(len(frames)):
        lo = max(0, i - half)
        hi = min(len(frames), i + half + 1)
        neighborhood = np.stack(frames[lo:hi], axis=0)
        med = np.median(neighborhood, axis=0)
        # Only blend where HF temporal variance is high
        hf_stack = np.stack([_hf_map(frames[j].astype(np.uint8)) for j in range(lo, hi)], axis=0)
        tvar = hf_stack.var(axis=0)
        thr = float(np.percentile(tvar, 70))
        mask = (tvar >= thr).astype(np.float32)[..., None] * a
        out = frames[i] * (1.0 - mask) + med * mask
        if float(mask.mean()) > 0.02:
            save_frame_rgb(paths[i], np.clip(out, 0, 255).astype(np.uint8))
            repaired += 1
    after = score_hf_shimmer(paths)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"deshimmered={repaired}")
    return after


@dataclass(slots=True)
class ShutterReport:
    score: float = 1.0
    applied: int = 0
    mean_flow: float = 0.0
    notes: list[str] = field(default_factory=list)


def _block_flow(a: np.ndarray, b: np.ndarray, *, block: int = 16) -> tuple[float, float]:
    """Coarse translational flow via SSD on center crop."""
    ha, wa = a.shape[:2]
    y0, x0 = ha // 4, wa // 4
    y1, x1 = 3 * ha // 4, 3 * wa // 4
    pa = a[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
    pb = b[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
    best = 1e18
    bdx = bdy = 0
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            shifted = np.roll(np.roll(pb, dy, axis=0), dx, axis=1)
            err = float(np.mean((pa - shifted) ** 2))
            if err < best:
                best = err
                bdx, bdy = dx, dy
    return float(bdx), float(bdy)


def _directional_blur(rgb: np.ndarray, dx: float, dy: float, *, amount: float) -> np.ndarray:
    """Cheap multi-tap blur along (dx, dy)."""
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < 0.4 or amount < 0.05:
        return rgb
    taps = max(2, min(7, int(round(mag * amount * 1.5))))
    acc = rgb.astype(np.float32) * 0.0
    wsum = 0.0
    for t in range(-taps, taps + 1):
        wt = 1.0 - abs(t) / (taps + 1)
        ox = int(round(dx * t / max(taps, 1) * amount))
        oy = int(round(dy * t / max(taps, 1) * amount))
        shifted = np.roll(np.roll(rgb, oy, axis=0), ox, axis=1).astype(np.float32)
        acc += shifted * wt
        wsum += wt
    return np.clip(acc / max(wsum, 1e-6), 0, 255).astype(np.uint8)


def apply_motion_shutter(
    frame_paths: list[Path],
    *,
    amount: float = 0.45,
    min_flow: float = 0.8,
) -> ShutterReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ShutterReport()
    applied = 0
    flows: list[float] = []
    prev = read_frame_rgb(paths[0])
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        dx, dy = _block_flow(prev, curr)
        mag = (dx * dx + dy * dy) ** 0.5
        flows.append(mag)
        if mag >= min_flow:
            out = _directional_blur(curr, dx, dy, amount=amount)
            save_frame_rgb(p, out)
            applied += 1
            curr = out
        prev = curr
    mean_f = float(np.mean(flows)) if flows else 0.0
    # Score: applied when motion warrants it
    score = 1.0 if mean_f < min_flow else float(np.clip(applied / max(len(paths) - 1, 1), 0.4, 1.0))
    notes = [f"shutter_applied={applied}"] if applied else ["shutter_skip_low_motion"]
    return ShutterReport(score=score, applied=applied, mean_flow=mean_f, notes=notes)


def score_motion_shutter(frame_paths: list[Path] | list[str]) -> ShutterReport:
    """Soft score: reward moderate directional streaking vs isotropic mush."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ShutterReport(score=1.0)
    # Proxy: if consecutive frames differ anisotropically, shutter looks "alive"
    a = read_frame_rgb(paths[0]).astype(np.float32)
    b = read_frame_rgb(paths[min(1, len(paths) - 1)]).astype(np.float32)
    d = np.abs(a - b).mean(axis=2)
    gy = np.abs(np.diff(d, axis=0)).mean()
    gx = np.abs(np.diff(d, axis=1)).mean()
    aniso = abs(gx - gy) / (gx + gy + 1e-6)
    score = float(np.clip(0.5 + aniso, 0.0, 1.0))
    return ShutterReport(score=score, notes=[f"aniso={aniso:.2f}"])


@dataclass(slots=True)
class OcclusionReport:
    score: float
    conflict_frames: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _depth_proxy(rgb: np.ndarray) -> np.ndarray:
    """Higher = closer. Ground-plane: lower in frame = closer; slight brightness bias."""
    h, w = rgb.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    gray = rgb.astype(np.float32).mean(axis=2) / 255.0
    return 0.7 * yy + 0.3 * (1.0 - gray)


def _edge_energy(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
    return gx + gy


def score_occlusion_coherence(frame_paths: list[Path] | list[str], *, sample_every: int = 2) -> OcclusionReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return OcclusionReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    conflicts = 0
    scores: list[float] = []
    for i in idxs:
        rgb = read_frame_rgb(paths[i])
        e = _edge_energy(rgb)
        # Soft collisions: high edge energy in mid band with low local contrast coherence
        h, w = e.shape
        mid = e[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        dens = float(mid.mean())
        # Penalize extremely dense mid edges (mushy overlaps)
        s = float(np.clip(1.0 - max(0.0, dens - 25.0) / 40.0, 0.0, 1.0))
        if dens > 40.0:
            conflicts += 1
        scores.append(s)
    mean = float(np.mean(scores)) if scores else 1.0
    notes = [f"occlusion_conflicts={conflicts}"] if conflicts else []
    return OcclusionReport(score=mean, conflict_frames=conflicts, notes=notes)


def apply_occlusion_resolve(
    frame_paths: list[Path],
    *,
    strength: float = 0.4,
) -> OcclusionReport:
    """
    Prefer nearer (lower-in-frame) structure when blending with previous frame
    in high-conflict mid regions — reduces silhouette mush across overlaps.
    """
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return OcclusionReport(score=1.0)
    a = float(np.clip(strength, 0.0, 1.0))
    repaired = 0
    prev = read_frame_rgb(paths[0]).astype(np.float32)
    prev_d = _depth_proxy(prev.astype(np.uint8))
    for p in paths[1:]:
        curr = read_frame_rgb(p).astype(np.float32)
        d = _depth_proxy(curr.astype(np.uint8))
        e = _edge_energy(curr.astype(np.uint8))
        thr = float(np.percentile(e, 75))
        conflict = (e >= thr).astype(np.float32)
        # Where current is closer than previous, keep current; else soft-blend prev structure
        closer = (d >= prev_d).astype(np.float32)
        w = conflict * a * (1.0 - closer)
        w3 = w[..., None]
        out = curr * (1.0 - w3) + prev * w3
        if float(w.mean()) > 0.03:
            save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
            repaired += 1
            curr = out
        prev = curr
        prev_d = _depth_proxy(curr.astype(np.uint8))
    after = score_occlusion_coherence(paths, sample_every=1)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"occlusion_resolved={repaired}")
    return after


@dataclass(slots=True)
class SecondaryReport:
    score: float
    detach_events: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _primary_secondary(slots: list[Slot]) -> tuple[Slot | None, Slot | None]:
    if not slots:
        return None, None
    ordered = sorted(slots, key=lambda s: -s.mass)
    primary = ordered[0]
    secondary = None
    best = 1e9
    for s in ordered[1:6]:
        dist = ((s.cy - primary.cy) ** 2 + (s.cx - primary.cx) ** 2) ** 0.5
        # Prefer nearby mid-mass companions (props), not far background
        if 4.0 < dist < 80.0 and s.mass > 2.0 and dist < best:
            best = dist
            secondary = s
    return primary, secondary


def score_secondary_track(frame_paths: list[Path] | list[str], *, sample_every: int = 1) -> SecondaryReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return SecondaryReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    detach = 0
    dists: list[float] = []
    prev_sec: Slot | None = None
    h = w = 1
    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=10)
        prim, sec = _primary_secondary(slots)
        if prim is None or sec is None:
            continue
        d = ((sec.cy - prim.cy) ** 2 + (sec.cx - prim.cx) ** 2) ** 0.5
        dists.append(d)
        if prev_sec is not None:
            # Secondary jumped far from previous secondary without primary moving much
            jump = ((sec.cy - prev_sec.cy) ** 2 + (sec.cx - prev_sec.cx) ** 2) ** 0.5
            if jump > max(h, w) * 0.2:
                detach += 1
        prev_sec = sec
    if not dists:
        return SecondaryReport(score=0.85, notes=["no_secondary_slot"])
    stab = float(np.clip(1.0 - float(np.std(dists)) / (max(h, w) * 0.15 + 1e-6), 0.0, 1.0))
    score = float(np.clip(stab - detach * 0.12, 0.0, 1.0))
    notes = [f"detach_events={detach}"] if detach else []
    return SecondaryReport(score=score, detach_events=detach, notes=notes)


def apply_secondary_track(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
) -> SecondaryReport:
    """Reinject last-seen secondary patch when it detaches from the hero."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return SecondaryReport(score=1.0)
    repaired = 0
    prev_rgb = read_frame_rgb(paths[0])
    prev_slots = extract_slots(prev_rgb, k=10)
    _, prev_sec = _primary_secondary(prev_slots)
    h, w = prev_rgb.shape[:2]
    a = float(np.clip(strength, 0.0, 1.0))
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        slots = extract_slots(curr, k=10)
        prim, sec = _primary_secondary(slots)
        if prev_sec is None:
            prev_rgb = curr
            prev_slots = slots
            prev_sec = sec
            continue
        lost = False
        if sec is None and prim is not None:
            lost = True
        elif sec is not None and prev_sec is not None:
            jump = ((sec.cy - prev_sec.cy) ** 2 + (sec.cx - prev_sec.cx) ** 2) ** 0.5
            if jump > max(h, w) * 0.2:
                lost = True
        if lost and prev_sec is not None:
            y0, x0, y1, x1 = prev_sec.bbox
            # Offset bbox toward current primary if available
            if prim is not None:
                dy = int(prim.cy - (prev_slots[0].cy if prev_slots else prim.cy))
                dx = int(prim.cx - (prev_slots[0].cx if prev_slots else prim.cx))
                y0, y1 = max(0, y0 + dy), min(h, y1 + dy)
                x0, x1 = max(0, x0 + dx), min(w, x1 + dx)
            patch = prev_rgb[prev_sec.bbox[0] : prev_sec.bbox[2], prev_sec.bbox[1] : prev_sec.bbox[3]]
            th, tw = y1 - y0, x1 - x0
            if patch.size and th > 0 and tw > 0:
                from PIL import Image

                patch_r = np.asarray(Image.fromarray(patch).resize((tw, th), Image.BILINEAR))
                region = curr[y0:y1, x0:x1].astype(np.float32)
                curr[y0:y1, x0:x1] = np.clip(region * (1.0 - a) + patch_r.astype(np.float32) * a, 0, 255).astype(
                    np.uint8
                )
                save_frame_rgb(p, curr)
                repaired += 1
        prev_rgb = curr
        prev_slots = extract_slots(curr, k=10)
        _, prev_sec = _primary_secondary(prev_slots)
    after = score_secondary_track(paths)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"secondary_repaired={repaired}")
    return after


@dataclass(slots=True)
class ShotChainReport:
    score: float
    delta_luma: float = 0.0
    delta_chroma: float = 0.0
    notes: list[str] = field(default_factory=list)


def chain_anchor_from_segment(frame_paths: list[Path] | list[str], *, out_path: str | Path) -> Path:
    """Write the last frame of a segment as the next shot's start image."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        raise FileNotFoundError("chain_anchor_from_segment: empty segment")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = read_frame_rgb(paths[-1])
    save_frame_rgb(out, rgb)
    return out


def score_cut_continuity(
    prev_frames: list[Path] | list[str],
    next_frames: list[Path] | list[str],
) -> ShotChainReport:
    prev = [Path(p) for p in prev_frames if Path(p).is_file()]
    nxt = [Path(p) for p in next_frames if Path(p).is_file()]
    if not prev or not nxt:
        return ShotChainReport(score=1.0)
    a = read_frame_rgb(prev[-1]).astype(np.float32)
    b = read_frame_rgb(nxt[0]).astype(np.float32)
    if a.shape != b.shape:
        from PIL import Image

        b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0])), dtype=np.float32)
    luma_a = a.mean()
    luma_b = b.mean()
    chroma_a = a.reshape(-1, 3).mean(axis=0)
    chroma_b = b.reshape(-1, 3).mean(axis=0)
    dl = abs(float(luma_a - luma_b)) / 255.0
    dc = float(np.mean(np.abs(chroma_a - chroma_b))) / 255.0
    score = float(np.clip(1.0 - dl * 3.0 - dc * 4.0, 0.0, 1.0))
    notes = []
    if dl > 0.12:
        notes.append(f"cut_luma_jump={dl:.2f}")
    if dc > 0.10:
        notes.append(f"cut_chroma_jump={dc:.2f}")
    return ShotChainReport(score=score, delta_luma=dl, delta_chroma=dc, notes=notes)


def harmonize_cut(
    next_frames: list[Path],
    prev_anchor: str | Path,
    *,
    strength: float = 0.35,
    n_frames: int = 4,
) -> ShotChainReport:
    """
    Soft-match exposure/chroma of the first N frames of the next shot to the
    previous shot's last frame (bridge the cut without hard paste).
    """
    paths = [Path(p) for p in next_frames if Path(p).is_file()]
    anchor_p = Path(prev_anchor)
    if not paths or not anchor_p.is_file():
        return ShotChainReport(score=1.0)
    anchor = read_frame_rgb(anchor_p).astype(np.float32)
    a_mean = anchor.reshape(-1, 3).mean(axis=0) + 1e-3
    a = float(np.clip(strength, 0.0, 1.0))
    for p in paths[: max(1, n_frames)]:
        rgb = read_frame_rgb(p).astype(np.float32)
        b_mean = rgb.reshape(-1, 3).mean(axis=0) + 1e-3
        scale = a_mean / b_mean
        matched = np.clip(rgb * (1.0 - a) + rgb * scale * a, 0, 255)
        # Mild residual blend toward anchor luminance
        out = matched * (1.0 - 0.15 * a) + anchor.mean() * (0.15 * a)
        # Keep spatial content from matched
        out = matched
        save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
    return score_cut_continuity([anchor_p], paths)


@dataclass(slots=True)
class CountBindVideoReport:
    score: float
    expected: int = 0
    mean_count: float = 0.0
    drift_frames: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _blob_count(rgb: np.ndarray, *, k: int = 12) -> int:
    slots = extract_slots(rgb, k=k)
    return sum(1 for s in slots if s.mass > 4.0)


def _expected_count_words(prompt: str) -> int:
    """Cheap numeral / word-count parse without heavy NLP."""
    text = (prompt or "").lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "a pair": 2,
        "couple": 2,
        "trio": 3,
    }
    for w, n in words.items():
        if w in text:
            return n
    import re

    m = re.search(r"\b([2-9]|1[0-2])\b", text)
    if m:
        return int(m.group(1))
    try:
        from utils.quality.critics import analyze_prompt_counts

        rep = analyze_prompt_counts(prompt)
        if rep.expected_counts:
            return max(rep.expected_counts.values())
        return max(1, int(rep.expected_subjects))
    except Exception:
        return 0


def score_count_bind(
    frame_paths: list[Path] | list[str],
    *,
    expected: int = 0,
    prompt: str = "",
    sample_every: int = 2,
) -> CountBindVideoReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        return CountBindVideoReport(score=1.0)
    exp = int(expected) if expected > 0 else _expected_count_words(prompt)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    counts = [_blob_count(read_frame_rgb(paths[i])) for i in idxs]
    mean_c = float(np.mean(counts)) if counts else 0.0
    if exp <= 0:
        # Stability-only: variance of count
        var = float(np.var(counts)) if len(counts) > 1 else 0.0
        score = float(np.clip(1.0 - var * 0.35, 0.0, 1.0))
        return CountBindVideoReport(score=score, expected=0, mean_count=mean_c, notes=["count_stability_only"])
    drift = sum(1 for c in counts if abs(c - exp) > 1)
    err = float(np.mean([abs(c - exp) for c in counts]))
    score = float(np.clip(1.0 - err / max(exp, 1) - drift * 0.05, 0.0, 1.0))
    notes = []
    if drift:
        notes.append(f"count_drift_frames={drift}")
    return CountBindVideoReport(score=score, expected=exp, mean_count=mean_c, drift_frames=drift, notes=notes)


def apply_count_bind(
    frame_paths: list[Path],
    *,
    expected: int = 0,
    prompt: str = "",
    strength: float = 0.4,
) -> CountBindVideoReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return CountBindVideoReport(score=1.0)
    exp = int(expected) if expected > 0 else _expected_count_words(prompt)
    counts = [_blob_count(read_frame_rgb(p)) for p in paths]
    if exp <= 0:
        exp = int(round(float(np.median(counts)))) if counts else 0
    # Donor = frame closest to expected count
    donor_i = int(np.argmin([abs(c - exp) for c in counts])) if counts else 0
    donor = read_frame_rgb(paths[donor_i])
    donor_slots = [s for s in extract_slots(donor, k=12) if s.mass > 4.0]
    repaired = 0
    a = float(np.clip(strength, 0.0, 1.0))
    for i, p in enumerate(paths):
        if abs(counts[i] - exp) <= 1:
            continue
        curr = read_frame_rgb(p)
        # Paste missing donor slots that aren't near any current slot
        curr_slots = extract_slots(curr, k=12)
        for ds in donor_slots:
            near = any(
                ((ds.cy - cs.cy) ** 2 + (ds.cx - cs.cx) ** 2) ** 0.5 < 12.0 for cs in curr_slots if cs.mass > 3.0
            )
            if near:
                continue
            y0, x0, y1, x1 = ds.bbox
            patch = donor[y0:y1, x0:x1]
            region = curr[y0:y1, x0:x1].astype(np.float32)
            if patch.size == 0 or region.size == 0:
                continue
            curr[y0:y1, x0:x1] = np.clip(region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255).astype(np.uint8)
            repaired += 1
            break  # one reinject per frame to avoid soup
        if repaired:
            save_frame_rgb(p, curr)
    after = score_count_bind(paths, expected=exp, sample_every=1)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"count_repaired={repaired}")
    return after


@dataclass(slots=True)
class PhysicsRepairReport:
    score_before: float
    score_after: float = 1.0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def apply_physics_repair(
    frame_paths: list[Path] | list[str],
    *,
    strength: float = 0.45,
    trigger_below: float = 0.55,
) -> PhysicsRepairReport:
    """Temporal blend frames where primary-slot velocity flips / jumps hard.

    Training-free: when physics_gate score is poor, blend mid frames toward
    neighbors to kill anti-gravity pops and teleport discontinuities.
    """

    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    before = score_physics_invariance(paths)
    report = PhysicsRepairReport(score_before=float(before.score))
    if len(paths) < 3 or before.score >= float(trigger_below):
        report.score_after = float(before.score)
        report.notes.append("physics_ok_or_short")
        return report

    s = float(np.clip(strength, 0.05, 0.85))
    repaired = 0
    # Read once
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    h, w = frames[0].shape[:2]
    prev_slots = extract_slots(frames[0].astype(np.uint8), k=6)
    for i in range(1, len(frames) - 1):
        slots = extract_slots(frames[i].astype(np.uint8), k=6)
        pairs, dis, _app = match_slots(prev_slots, slots, h=h, w=w, max_dist=0.35)
        jump = bool(dis)
        for pi, cj in pairs:
            if pi >= len(prev_slots) or cj >= len(slots):
                continue
            a, b = prev_slots[pi], slots[cj]
            dy = (a.cy - b.cy) / max(float(h), 1.0)
            dx = (a.cx - b.cx) / max(float(w), 1.0)
            # >18% of frame diagonal in one step → teleport / physics break
            if (dy * dy + dx * dx) ** 0.5 > 0.18:
                jump = True
                break
        if before.anti_gravity_events or before.velocity_flip_penalties or before.jump_discontinuities or jump:
            blended = (1.0 - s) * frames[i] + (s * 0.5) * frames[i - 1] + (s * 0.5) * frames[i + 1]
            frames[i] = np.clip(blended, 0, 255)
            repaired += 1
        prev_slots = slots

    if repaired:
        for p, fr in zip(paths, frames, strict=False):
            save_frame_rgb(p, fr.astype(np.uint8))
    after = score_physics_invariance(paths)
    report.score_after = float(after.score)
    report.repaired = repaired
    report.notes.append(f"blended_frames={repaired}")
    return report


@dataclass(slots=True)
class AdherenceReport:
    score: float
    structure_score: float = 1.0
    count_score: float = 1.0
    color_bind_score: float = 1.0
    clip_score: float = 1.0
    notes: list[str] = field(default_factory=list)


_COLOR_RGB = {
    "red": (200, 40, 40),
    "blue": (40, 40, 200),
    "green": (40, 180, 40),
    "yellow": (220, 200, 40),
    "orange": (230, 120, 30),
    "purple": (140, 40, 180),
    "pink": (230, 100, 160),
    "black": (20, 20, 20),
    "white": (230, 230, 230),
    "brown": (120, 70, 40),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


def _color_presence(rgb: np.ndarray, name: str) -> float:
    target = _COLOR_RGB.get(name.lower())
    if target is None:
        return 0.5
    t = np.asarray(target, dtype=np.float32)
    pix = rgb.astype(np.float32).reshape(-1, 3)
    # Fraction of pixels within distance of target
    dist = np.linalg.norm(pix - t[None], axis=1)
    hit = float((dist < 90.0).mean())
    return float(np.clip(hit * 8.0, 0.0, 1.0))  # small regions still score


def score_temporal_adherence(
    frame_paths: list[Path] | list[str],
    prompt: str,
    *,
    sample_every: int = 2,
    use_clip: bool = False,
) -> AdherenceReport:
    from pipelines.video.neural_sample import parse_prompt_ground

    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    g = parse_prompt_ground(prompt)
    notes: list[str] = []
    if len(paths) < 1:
        return AdherenceReport(score=1.0)

    idxs = list(range(0, len(paths), max(1, sample_every)))
    # Color attribute binding: required colors should appear
    color_scores: list[float] = []
    required_colors = []
    for e in g.entities:
        if e.negated:
            continue
        for a in e.attributes:
            if a.lower() in _COLOR_RGB:
                required_colors.append(a.lower())
    if required_colors:
        for i in idxs:
            rgb = read_frame_rgb(paths[i])
            color_scores.append(float(np.mean([_color_presence(rgb, c) for c in required_colors])))
        color_bind = float(np.mean(color_scores)) if color_scores else 0.5
        if color_bind < 0.25:
            notes.append(f"weak_color_bind={color_bind:.2f}:{required_colors}")
    else:
        color_bind = 1.0

    # Negated colors should be rare
    for neg_c in [n for n in g.negations if n.lower() in _COLOR_RGB]:
        rgb = read_frame_rgb(paths[idxs[0]])
        if _color_presence(rgb, neg_c) > 0.35:
            notes.append(f"negation_leak:{neg_c}")
            color_bind *= 0.7

    count_rep = score_count_stability(paths, prompt=prompt, sample_every=sample_every)
    count_score = float(count_rep.score)
    notes.extend(count_rep.notes[:3])

    # Structure: entity density prior
    structure = 1.0
    if g.entities:
        structure = float(np.clip(0.6 + 0.1 * len([e for e in g.entities if not e.negated]), 0, 1))

    clip_s = 1.0
    if use_clip:
        try:
            import torch
            from utils.generation.guidance import clip_image_text_cosine

            rgb = read_frame_rgb(paths[idxs[len(idxs) // 2]])
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            sim = clip_image_text_cosine(t, g.rewritten or prompt, "openai/clip-vit-base-patch32", torch.device("cpu"))
            # Map typical CLIP sim ~0.2–0.35 → 0–1
            clip_s = float(np.clip((sim - 0.15) / 0.25, 0.0, 1.0))
            notes.append(f"clip_sim={sim:.3f}")
        except Exception as e:
            notes.append(f"clip_skip:{type(e).__name__}")

    score = float(np.clip(0.35 * color_bind + 0.30 * count_score + 0.15 * structure + 0.20 * clip_s, 0.0, 1.0))
    return AdherenceReport(
        score=score,
        structure_score=structure,
        count_score=count_score,
        color_bind_score=color_bind,
        clip_score=clip_s,
        notes=notes,
    )
