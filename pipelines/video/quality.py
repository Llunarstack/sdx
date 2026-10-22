"""Segment quality gates: temporal consistency, sharpness, and optional superiority axes.

Also holds what used to be: permanence, artifact_critic, extremity_lock, glyph_lock, identity_bind, physics_gate, drift_repair, identity_lock.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import SegmentQualityReport
from .video_io import read_frame_rgb, save_frame_rgb

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
    for a, b in zip(frames[:-1], frames[1:]):
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
    for a_p, b_p in zip(sample[:-1], sample[1:]):
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
    from utils.quality.hand_verifier import score_hand_structure

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
            for tid, s in zip(prev_ids, slots):
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


def _structure_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - rescaled luma correlation (cheap SSIM stand-in)."""
    ga = a.astype(np.float32).mean(axis=2)
    gb = b.astype(np.float32).mean(axis=2)
    ga -= ga.mean()
    gb -= gb.mean()
    corr = float(np.sum(ga * gb)) / float(np.sqrt(np.sum(ga * ga) * np.sum(gb * gb)) + 1e-6)
    return 1.0 - min(1.0, max(0.0, (corr + 1.0) * 0.5))


def frame_drift_score(frame_rgb: np.ndarray, reference_rgb: np.ndarray, *, prev_rgb: np.ndarray | None = None) -> float:
    """Higher = more drift from the reference (and the previous frame when given)."""
    h_cur = _color_histogram(frame_rgb)
    score = 0.55 * _histogram_distance(h_cur, _color_histogram(reference_rgb)) + 0.45 * _structure_distance(
        frame_rgb, reference_rgb
    )
    if prev_rgb is not None:
        score = 0.7 * score + 0.3 * _histogram_distance(h_cur, _color_histogram(prev_rgb))
    return float(min(1.0, max(0.0, score)))


def score_sequence_drift(frame_paths: Sequence[Path], anchor_path: str | Path | None = None) -> list[float]:
    if not frame_paths:
        return []
    ref = read_frame_rgb(anchor_path) if anchor_path and Path(anchor_path).is_file() else read_frame_rgb(frame_paths[0])
    prev = None
    out: list[float] = []
    for fp in frame_paths:
        rgb = read_frame_rgb(fp)
        out.append(frame_drift_score(rgb, ref, prev_rgb=prev))
        prev = rgb
    return out


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
    from .motion import load_mask_binary, rasterize_box_mask

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
    from .motion import load_mask_binary, propagate_mask_sequence, rasterize_box_mask

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
    for fp, mp in zip(frame_paths, mask_paths[: len(frame_paths)]):
        m = load_mask_binary(mp, shape=(h, w))
        rgb = read_frame_rgb(fp)
        locked = composite_with_mask(rgb, anchor, m, strength=strength)
        save_frame_rgb(fp, locked)
    return frame_paths


__all__ = [
    "PermanenceReport",
    "Slot",
    "extract_slots",
    "match_slots",
    "score_object_permanence",
    "ArtifactReport",
    "score_appearance_artifacts",
    "score_motion_artifacts",
    "score_camera_artifacts",
    "score_video_artifacts",
    "ExtremityReport",
    "score_extremity_coherence",
    "GlyphReport",
    "find_glyph_rois",
    "score_glyph_stability",
    "IdentityFingerprint",
    "IdentityBindReport",
    "build_fingerprint",
    "bind_from_refs",
    "score_identity_bind",
    "PhysicsReport",
    "score_physics_invariance",
    "score_segment_quality",
    "score_temporal_consistency",
    "score_frame_sharpness",
    "DriftRepairReport",
    "frame_drift_score",
    "repair_sequence_drift",
    "score_sequence_drift",
    "apply_identity_lock_to_sequence",
    "apply_propagated_identity_lock",
    "composite_with_mask",
]


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
