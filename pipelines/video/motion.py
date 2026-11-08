"""Motion analysis and temporal synthesis: optical flow, motion brushes/templates, stabilization,
keyframe scheduling and picking, frame interpolation.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.video.video_io import frame_count_for_duration, read_frame_rgb, save_frame_rgb

# ==================================================================================================
# motion
# ==================================================================================================


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python required for motion helpers") from exc


def compute_flow_magnitude(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Dense optical flow magnitude between consecutive RGB frames."""
    cv2 = _require_cv2()
    g0 = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return flow, float(np.mean(mag))


def warp_frame_with_flow(
    source_rgb: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Warp source by flow field (motion-guided propagation)."""
    cv2 = _require_cv2()
    h, w = source_rgb.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0] * alpha).astype(np.float32)
    map_y = (grid_y + flow[..., 1] * alpha).astype(np.float32)
    warped = cv2.remap(
        cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


# ==================================================================================================
# motion_beats: Place keyframes on motion peaks in reference footage.
# ==================================================================================================


def detect_motion_beats(frame_paths: Sequence[Path], *, min_gap: int = 4) -> list[int]:
    """Return frame indices where motion magnitude is a local peak."""
    n = len(frame_paths)
    if n < 3:
        return [0] if n else []
    mags: list[float] = [0.0]
    for i in range(1, n):
        try:
            _, m = compute_flow_magnitude(read_frame_rgb(frame_paths[i - 1]), read_frame_rgb(frame_paths[i]))
            mags.append(m)
        except Exception:
            mags.append(mags[-1] if mags else 0.0)
    arr = np.array(mags, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    peaks: list[int] = [0, n - 1]
    gap = max(1, int(min_gap))
    for i in range(1, n - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1] and arr[i] > 0.35:
            if not peaks or i - peaks[-1] >= gap:
                peaks.append(i)
    return sorted(set(peaks))


def merge_beat_indices(base: Sequence[int], beats: Sequence[int], total: int) -> list[int]:
    """Union keyframe cadence with motion beat anchors."""
    idx = set(int(i) for i in base if 0 <= int(i) < total)
    idx.update(int(b) for b in beats if 0 <= int(b) < total)
    idx.add(0)
    if total > 0:
        idx.add(total - 1)
    return sorted(idx)


# ==================================================================================================
# mask_propagate: Propagate entity masks across frames using optical flow.
# ==================================================================================================


def load_mask_binary(path: str | Path, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    from PIL import Image

    m = np.array(Image.open(path).convert("L"))
    if shape is not None and (m.shape[0], m.shape[1]) != shape:
        import cv2

        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.float32)


def rasterize_box_mask(
    width: int,
    height: int,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=np.float32)
    ix0 = int(max(0, min(width - 1, x0 * width)))
    iy0 = int(max(0, min(height - 1, y0 * height)))
    ix1 = int(max(0, min(width, x1 * width)))
    iy1 = int(max(0, min(height, y1 * height)))
    mask[iy0:iy1, ix0:ix1] = 1.0
    return mask


def propagate_mask_sequence(
    mask0: np.ndarray,
    frame_paths: Sequence[Path],
    *,
    out_dir: Path,
) -> list[Path]:
    """Forward-warp initial mask through each frame pair's optical flow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    paths: list[Path] = []
    curr = mask0.astype(np.float32)
    prev_rgb = read_frame_rgb(frame_paths[0])
    h, w = prev_rgb.shape[:2]
    if curr.shape[:2] != (h, w):
        curr = cv2.resize(curr, (w, h), interpolation=cv2.INTER_NEAREST)
    p0 = out_dir / "mask_000001.png"
    save_frame_rgb(p0, (curr * 255).astype(np.uint8)[..., None].repeat(3, axis=2))
    paths.append(p0)
    for i in range(1, len(frame_paths)):
        a = read_frame_rgb(frame_paths[i - 1])
        b = read_frame_rgb(frame_paths[i])
        flow, _ = compute_flow_magnitude(a, b)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(curr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        curr = np.clip(warped, 0, 1)
        fp = out_dir / f"mask_{i + 1:06d}.png"
        save_frame_rgb(fp, (curr * 255).astype(np.uint8)[..., None].repeat(3, axis=2))
        paths.append(fp)
    return paths


# ==================================================================================================
# motion_brush: Motion brush — restrict optical flow to painted regions (Runway Motion Brush style).
# ==================================================================================================


@dataclass(slots=True)
class MotionBrushSpec:
    mask_path: str = ""
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    mode: str = "motion_only"  # motion_only | freeze_brush | background_only
    feather: int = 8


def parse_motion_brush(raw: Any) -> MotionBrushSpec | None:
    if not raw:
        return None
    if isinstance(raw, str):
        return MotionBrushSpec(mask_path=raw)
    if not isinstance(raw, Mapping):
        return None
    box_raw = raw.get("box") or raw.get("region")
    box = (0.0, 0.0, 0.0, 0.0)
    if isinstance(box_raw, (list, tuple)) and len(box_raw) >= 4:
        box = tuple(float(x) for x in box_raw[:4])
    return MotionBrushSpec(
        mask_path=str(raw.get("mask") or raw.get("mask_path") or ""),
        box=box,
        mode=str(raw.get("mode") or "motion_only").lower(),
        feather=int(raw.get("feather") or 8),
    )


def load_motion_brush(
    spec: MotionBrushSpec,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Return float mask 0–1."""
    if spec.mask_path and Path(spec.mask_path).is_file():
        if spec.mask_path.endswith(".json"):
            data = json.loads(Path(spec.mask_path).read_text(encoding="utf-8"))
            inner = parse_motion_brush(data)
            if inner:
                return load_motion_brush(inner, width=width, height=height)
        m = load_mask_binary(spec.mask_path, shape=(height, width))
    elif spec.box != (0.0, 0.0, 0.0, 0.0):
        m = rasterize_box_mask(width, height, spec.box)
    else:
        m = np.zeros((height, width), dtype=np.float32)
    if spec.feather > 0:
        try:
            import cv2

            k = max(3, spec.feather * 2 + 1)
            m = cv2.GaussianBlur(m, (k, k), 0)
        except ImportError:
            pass
    return np.clip(m, 0, 1).astype(np.float32)


def apply_brush_to_flow(
    flow: np.ndarray,
    brush: np.ndarray,
    *,
    mode: str = "motion_only",
) -> np.ndarray:
    """Zero or attenuate flow outside/inside brush depending on mode."""
    m = brush.astype(np.float32)
    m3 = m[..., None] if m.ndim == 2 else m
    out = flow.astype(np.float32).copy()
    if mode == "background_only":
        out = out * (1.0 - m3)
    elif mode == "freeze_brush":
        out = out * m3
    else:
        out = out * m3
    return out


def warp_with_motion_brush(
    source_rgb: np.ndarray,
    flow: np.ndarray,
    brush: np.ndarray,
    *,
    mode: str = "motion_only",
    alpha: float = 1.0,
) -> np.ndarray:

    masked_flow = apply_brush_to_flow(flow, brush, mode=mode)
    warped = warp_frame_with_flow(source_rgb, masked_flow, alpha=alpha).astype(np.float32)
    if mode == "motion_only":
        m = brush[..., None] if brush.ndim == 2 else brush
        base = source_rgb.astype(np.float32)
        return np.clip(base * (1.0 - m) + warped * m, 0, 255).astype(np.uint8)
    return warped.astype(np.uint8)


# ==================================================================================================
# motion_transfer: Apply reference clip motion to edited anchor frames (motion template path).
# ==================================================================================================


def retime_frame_list(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    if not frames or target_count <= 0:
        return []
    if len(frames) == target_count:
        return frames
    idx = np.linspace(0, len(frames) - 1, target_count).astype(int)
    return [frames[i] for i in idx]


def motion_template_sequence(
    anchor_path: str | Path,
    source_frame_paths: Sequence[Path],
    out_dir: str | Path,
    *,
    target_count: int,
    blend_source: float = 0.15,
    motion_brush: Any | None = None,
) -> list[Path]:
    """
    Warp edited anchor forward using optical flow from source clip.

    Keeps motion from reference while preserving anchor appearance.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not source_frame_paths:
        rgb = read_frame_rgb(anchor_path)
        paths = []
        for i in range(target_count):
            fp = out / f"frame_{i + 1:06d}.png"
            save_frame_rgb(fp, rgb)
            paths.append(fp)
        return paths

    brush_arr = None
    brush_mode = "motion_only"
    if motion_brush is not None:
        spec = motion_brush if hasattr(motion_brush, "mask_path") else parse_motion_brush(motion_brush)
        if spec:
            h0, w0 = read_frame_rgb(anchor_path).shape[:2]
            brush_arr = load_motion_brush(spec, width=w0, height=h0)
            brush_mode = spec.mode

    curr = read_frame_rgb(anchor_path).astype(np.float32)
    acc: list[np.ndarray] = [curr.astype(np.uint8)]
    n = min(len(source_frame_paths), max(target_count, 2))
    for i in range(1, n):
        a = read_frame_rgb(source_frame_paths[i - 1])
        b = read_frame_rgb(source_frame_paths[i])
        try:
            flow, _ = compute_flow_magnitude(a, b)
            if brush_arr is not None:
                warped = warp_with_motion_brush(
                    curr.astype(np.uint8), flow, brush_arr, mode=brush_mode, alpha=1.0
                ).astype(np.float32)
            else:
                warped = warp_frame_with_flow(curr.astype(np.uint8), flow, alpha=1.0).astype(np.float32)
            if blend_source > 0:
                src = b.astype(np.float32)
                if src.shape == warped.shape:
                    warped = (1.0 - blend_source) * warped + blend_source * src
            curr = warped
        except Exception:
            curr = read_frame_rgb(source_frame_paths[i]).astype(np.float32)
        acc.append(np.clip(curr, 0, 255).astype(np.uint8))

    acc = retime_frame_list(acc, target_count)
    paths: list[Path] = []
    for i, rgb in enumerate(acc):
        fp = out / f"frame_{i + 1:06d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths


# ==================================================================================================
# region_motion: Per-rig-box regional motion: limbs move independently from reference flow.
# ==================================================================================================


def load_rig_boxes(rig_path: str | Path) -> list[tuple[str, tuple[float, float, float, float], bool]]:
    data = json.loads(Path(rig_path).read_text(encoding="utf-8"))
    out: list[tuple[str, tuple[float, float, float, float], bool]] = []
    for r in data.get("regions", []):
        if "box" not in r:
            continue
        box = tuple(float(x) for x in r["box"][:4])
        lock = str(r.get("reference_mode", "")).lower() == "identity" or bool(r.get("lock"))
        out.append((str(r.get("name", "part")), box, lock))
    return out


def feather_mask(mask: np.ndarray, radius: int = 6) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    try:
        import cv2

        k = max(3, radius * 2 + 1)
        return cv2.GaussianBlur(m, (k, k), 0)
    except ImportError:
        return m


def _crop_box(rgb: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, int, int, int, int]:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    ix0 = int(max(0, min(w - 1, x0 * w)))
    iy0 = int(max(0, min(h - 1, y0 * h)))
    ix1 = int(max(ix0 + 1, min(w, x1 * w)))
    iy1 = int(max(iy0 + 1, min(h, y1 * h)))
    return rgb[iy0:iy1, ix0:ix1], ix0, iy0, ix1, iy1


def _paste_box(base: np.ndarray, patch: np.ndarray, ix0: int, iy0: int, ix1: int, iy1: int, alpha: np.ndarray) -> None:
    ph, pw = patch.shape[:2]
    bh, bw = iy1 - iy0, ix1 - ix0
    if ph != bh or pw != bw:
        try:
            import cv2

            patch = cv2.resize(patch, (bw, bh), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            return
    a = alpha[:bh, :bw]
    if a.ndim == 2:
        a = a[..., None]
    region = base[iy0:iy1, ix0:ix1].astype(np.float32)
    blended = region * (1.0 - a) + patch.astype(np.float32) * a
    base[iy0:iy1, ix0:ix1] = np.clip(blended, 0, 255).astype(np.uint8)


def apply_regional_motion(
    edited_anchor_path: str | Path,
    source_frame_paths: Sequence[Path],
    out_dir: str | Path,
    rig_path: str | Path,
    *,
    target_count: int,
    locked_only: bool = False,
) -> list[Path]:
    """
    Warp rig regions independently using source clip flow, composite onto anchor.

    Unlocked regions follow full-frame motion; locked regions get independent warp.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    boxes = load_rig_boxes(rig_path)
    if not boxes or not source_frame_paths:
        return motion_template_sequence(edited_anchor_path, source_frame_paths, out, target_count=target_count)

    anchor = read_frame_rgb(edited_anchor_path).astype(np.float32)
    h, w = anchor.shape[:2]
    part_states: dict[str, np.ndarray] = {}
    part_masks: dict[str, np.ndarray] = {}
    for name, box, lock in boxes:
        if locked_only and not lock:
            continue
        crop, ix0, iy0, ix1, iy1 = _crop_box(anchor.astype(np.uint8), box)
        part_states[name] = crop.astype(np.float32)
        m = rasterize_box_mask(w, h, box)
        part_masks[name] = feather_mask(m, radius=8)

    acc: list[np.ndarray] = []
    n = min(len(source_frame_paths), max(target_count, 2))
    curr_full = anchor.copy()
    acc.append(np.clip(curr_full, 0, 255).astype(np.uint8))

    for i in range(1, n):
        a = read_frame_rgb(source_frame_paths[i - 1])
        b = read_frame_rgb(source_frame_paths[i])
        try:
            flow, _ = compute_flow_magnitude(a, b)
        except Exception:
            flow = None
        frame = curr_full.copy()
        for name, box, lock in boxes:
            if locked_only and not lock:
                continue
            if name not in part_states:
                continue
            crop = part_states[name]
            ch, cw = crop.shape[:2]
            if flow is not None:
                # Regional mean flow inside box
                m = part_masks[name] > 0.05
                if m.any():
                    fx = float(np.mean(flow[..., 0][m]))
                    fy = float(np.mean(flow[..., 1][m]))
                else:
                    fx, fy = 0.0, 0.0
                try:
                    import cv2

                    mtx = np.float32([[1, 0, fx], [0, 1, fy]])
                    warped = cv2.warpAffine(crop.astype(np.uint8), mtx, (cw, ch), borderMode=cv2.BORDER_REPLICATE)
                    part_states[name] = warped.astype(np.float32)
                except ImportError:
                    pass
            _, ix0, iy0, ix1, iy1 = _crop_box(frame.astype(np.uint8), box)
            _paste_box(frame, part_states[name].astype(np.uint8), ix0, iy0, ix1, iy1, part_masks[name])
        curr_full = frame.astype(np.float32)
        acc.append(np.clip(curr_full, 0, 255).astype(np.uint8))

    acc = retime_frame_list(acc, target_count)
    paths: list[Path] = []
    for j, rgb in enumerate(acc):
        fp = out / f"frame_{j + 1:06d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths


# ==================================================================================================
# motion_library: Reusable motion packs — decouple motion from appearance.
# ==================================================================================================


@dataclass(slots=True)
class MotionPack:
    id: str
    clip: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    camera: str = ""
    description: str = ""
    loop: bool = False


@dataclass(slots=True)
class MotionLibrary:
    packs: dict[str, MotionPack] = field(default_factory=dict)

    def get(self, pack_id: str) -> MotionPack | None:
        return self.packs.get(pack_id)


def parse_motion_library(raw: Any) -> MotionLibrary:
    lib = MotionLibrary()
    if not isinstance(raw, Mapping):
        return lib
    for k, v in raw.items():
        pid = str(k)
        if isinstance(v, str):
            lib.packs[pid] = MotionPack(id=pid, clip=v)
            continue
        if not isinstance(v, Mapping):
            continue
        tags = v.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        lib.packs[pid] = MotionPack(
            id=pid,
            clip=str(v.get("clip") or v.get("video") or v.get("path") or ""),
            tags=[str(t) for t in tags],
            category=str(v.get("category") or v.get("type") or "general"),
            camera=str(v.get("camera") or ""),
            description=str(v.get("description") or ""),
            loop=bool(v.get("loop")),
        )
    return lib


def resolve_motion_clip(
    library: MotionLibrary,
    pack_ids: Sequence[str],
    *,
    fallback: str = "",
) -> str:
    for pid in pack_ids:
        p = library.get(pid)
        if p and p.clip and Path(p.clip).is_file():
            return p.clip
    return fallback


# ==================================================================================================
# flow_consistency: Forward-backward flow consistency repair for interpolated frames.
# ==================================================================================================


def _fb_error(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    h, w = flow_fwd.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_fwd[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_fwd[..., 1]).astype(np.float32)
    try:
        import cv2

        bx = cv2.remap(flow_bwd[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        by = cv2.remap(flow_bwd[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        err = np.sqrt((flow_fwd[..., 0] + bx) ** 2 + (flow_fwd[..., 1] + by) ** 2)
        return err.astype(np.float32)
    except ImportError:
        return np.zeros((h, w), dtype=np.float32)


def repair_flow_consistency(frame_paths: list[Path], *, threshold: float = 2.5, blend: float = 0.45) -> list[Path]:
    """Blend inconsistent pixels toward previous frame (reduces ghosting)."""
    if len(frame_paths) < 2:
        return frame_paths
    prev = read_frame_rgb(frame_paths[0])
    for i in range(1, len(frame_paths)):
        curr = read_frame_rgb(frame_paths[i])
        try:
            f_fwd, _ = compute_flow_magnitude(prev, curr)
            f_bwd, _ = compute_flow_magnitude(curr, prev)
            err = _fb_error(f_fwd, f_bwd)
            mask = (err > threshold).astype(np.float32)
            if mask.max() > 0:
                m3 = mask[..., None]
                b = float(max(0.0, min(1.0, blend)))
                fixed = curr.astype(np.float32) * (1.0 - m3 * b) + prev.astype(np.float32) * (m3 * b)
                curr = np.clip(fixed, 0, 255).astype(np.uint8)
                save_frame_rgb(frame_paths[i], curr)
        except Exception:
            pass
        prev = curr
    return frame_paths


# ==================================================================================================
# camera_stabilize: Stabilize shaky reference footage before keyframe extraction.
# ==================================================================================================


def _shift_rgb(rgb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    try:
        import cv2

        h, w = rgb.shape[:2]
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(rgb, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    except ImportError:
        return rgb


def stabilize_rgb_sequence(frames: Sequence[np.ndarray], *, strength: float = 0.65) -> list[np.ndarray]:
    """Compensate cumulative camera pan using mean optical flow."""
    if len(frames) < 2:
        return list(frames)
    s = float(max(0.0, min(1.0, strength)))
    out: list[np.ndarray] = [frames[0]]
    cum_x, cum_y = 0.0, 0.0
    prev = frames[0]
    for curr in frames[1:]:
        try:
            flow, _ = compute_flow_magnitude(prev, curr)
            dx = float(np.mean(flow[..., 0]))
            dy = float(np.mean(flow[..., 1]))
            cum_x += dx * s
            cum_y += dy * s
            stabilized = _shift_rgb(curr, -cum_x, -cum_y)
        except Exception:
            stabilized = curr
        out.append(stabilized)
        prev = curr
    return out


def stabilize_frame_paths(frame_paths: list[Path], *, strength: float = 0.65) -> list[Path]:
    if len(frame_paths) < 2:
        return frame_paths
    rgbs = [read_frame_rgb(p) for p in frame_paths]
    stable = stabilize_rgb_sequence(rgbs, strength=strength)
    for p, rgb in zip(frame_paths, stable, strict=False):
        save_frame_rgb(p, rgb)
    return frame_paths


# ==================================================================================================
# keyframes: Keyframe selection cadence for segment editing.
# ==================================================================================================


@dataclass(slots=True)
class KeyframeSchedule:
    segment_index: int
    frame_indices: list[int]
    frame_paths: list[Path]
    interval: int


def pick_keyframe_indices(total_frames: int, *, interval: int = 6, always_include: Sequence[int] = ()) -> list[int]:
    """Every ``interval`` frames plus first/last and optional anchors."""
    if total_frames <= 0:
        return []
    interval = max(1, int(interval))
    indices = {0, total_frames - 1}
    for i in range(0, total_frames, interval):
        indices.add(i)
    for a in always_include:
        if 0 <= int(a) < total_frames:
            indices.add(int(a))
    return sorted(indices)


def schedule_keyframes_for_segment(
    frame_paths: list[Path],
    *,
    segment_index: int,
    duration_sec: float,
    fps: float,
    interval: int = 6,
    beat_indices: Sequence[int] | None = None,
) -> KeyframeSchedule:
    target = frame_count_for_duration(duration_sec, fps)
    if len(frame_paths) > target:
        # subsample path list to target count first
        step = max(1, len(frame_paths) // max(1, target))
        subsampled = [frame_paths[i] for i in range(0, len(frame_paths), step)][:target]
    else:
        subsampled = frame_paths
    idxs = pick_keyframe_indices(len(subsampled), interval=interval)
    if beat_indices:
        idxs = merge_beat_indices(idxs, beat_indices, len(subsampled))
    return KeyframeSchedule(
        segment_index=segment_index,
        frame_indices=idxs,
        frame_paths=[subsampled[i] for i in idxs],
        interval=interval,
    )


# ==================================================================================================
# Restored from b654ae9:pipelines/video/motion.py
# ==================================================================================================
def estimate_motion_score(frame_paths: list[Path], *, sample_pairs: int = 8) -> float:
    """0–1 motion intensity proxy from sampled frame pairs."""
    if len(frame_paths) < 2:
        return 0.0
    step = max(1, len(frame_paths) // max(1, sample_pairs))
    mags: list[float] = []
    for i in range(0, len(frame_paths) - 1, step):
        a = read_frame_rgb(frame_paths[i])
        b = read_frame_rgb(frame_paths[min(len(frame_paths) - 1, i + step)])
        _, m = compute_flow_magnitude(a, b)
        mags.append(m)
    if not mags:
        return 0.0
    return float(min(1.0, np.mean(mags) / 8.0))


# ==================================================================================================
# interpolate
# ==================================================================================================


def _write_retimed(all_paths: Sequence[Path], out: Path, produced: set[Path]) -> list[Path]:
    """Write ``out/frame_NNNNNN.png``: frames this module wrote are byte-copied, other sources are decoded once."""
    decoded: dict[Path, np.ndarray] = {}
    final: list[Path] = []
    for i, src in enumerate(all_paths):
        fp = out / f"frame_{i + 1:06d}.png"
        if src.resolve() == fp.resolve():
            final.append(src)
            continue
        if src in produced:
            shutil.copyfile(src, fp)
        else:
            if src not in decoded:
                decoded[src] = read_frame_rgb(src)
            save_frame_rgb(fp, decoded[src])
        final.append(fp)
    return final


def blend_frames(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(max(0.0, min(1.0, t)))
    return np.clip((1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32), 0, 255).astype(np.uint8)


def interpolate_between_keyframes(
    frame_a_path: str | Path,
    frame_b_path: str | Path,
    out_dir: str | Path,
    *,
    count: int,
    use_flow: bool = True,
    prefix: str = "mid",
) -> list[Path]:
    """Generate ``count`` intermediate frames between A and B."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = read_frame_rgb(frame_a_path)
    b = read_frame_rgb(frame_b_path)
    flow = None
    if use_flow:
        try:
            flow, _ = compute_flow_magnitude(a, b)
        except Exception:
            flow = None
    paths: list[Path] = []
    for i in range(1, count + 1):
        t = i / (count + 1)
        if flow is not None:
            warped = warp_frame_with_flow(a, flow, alpha=t)
            rgb = blend_frames(warped, b, t * 0.35)
        else:
            rgb = blend_frames(a, b, t)
        fp = out / f"{prefix}_{i:04d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths


def interpolate_sequence(
    keyframe_paths: Sequence[str | Path],
    target_frame_count: int,
    out_dir: str | Path,
    *,
    use_flow: bool = True,
) -> list[Path]:
    """
    Expand sparse keyframes to ``target_frame_count`` frames via segment interpolation.
    """
    keys = [Path(p) for p in keyframe_paths if Path(p).is_file()]
    if not keys:
        return []
    if len(keys) == 1:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / f"frame_{i + 1:06d}.png" for i in range(target_frame_count)]
        if paths:
            save_frame_rgb(paths[0], read_frame_rgb(keys[0]))
            for fp in paths[1:]:
                shutil.copyfile(paths[0], fp)
        return paths
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments = len(keys) - 1
    per_seg = max(1, target_frame_count // segments)
    all_paths: list[Path] = [keys[0]]
    produced: set[Path] = set()
    for s in range(segments):
        mids = interpolate_between_keyframes(
            keys[s],
            keys[s + 1],
            out / f"seg_{s:02d}",
            count=max(0, per_seg - 2),
            use_flow=use_flow,
            prefix="mid",
        )
        produced.update(mids)
        all_paths.extend(mids)
        all_paths.append(keys[s + 1])
    # Retime to exact count
    if len(all_paths) > target_frame_count:
        idx = np.linspace(0, len(all_paths) - 1, target_frame_count).astype(int)
        all_paths = [all_paths[i] for i in idx]
    elif len(all_paths) < target_frame_count:
        last = all_paths[-1]
        while len(all_paths) < target_frame_count:
            all_paths.append(last)
    return _write_retimed(all_paths, out, produced)


# ==================================================================================================
# depth_interpolate: Edge-aware depth-proxy interpolation between keyframes.
# ==================================================================================================


def depth_proxy(rgb: np.ndarray) -> np.ndarray:
    """Cheap depth stand-in: inverse luminance + Sobel edge emphasis."""
    gray = rgb[..., :3].astype(np.float32).mean(axis=2) / 255.0
    inv = 1.0 - gray
    try:
        import cv2

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)
        depth = np.clip(0.65 * inv + 0.35 * edge, 0, 1)
    except ImportError:
        depth = inv
    return depth.astype(np.float32)


def depth_aware_blend(
    a: np.ndarray, b: np.ndarray, t: float, *, da: np.ndarray | None = None, db: np.ndarray | None = None
) -> np.ndarray:
    """Occlusion-aware blend using depth proxy (closer layer wins at mid-t); pass ``da``/``db`` to reuse depths."""
    t = float(max(0.0, min(1.0, t)))
    da = depth_proxy(a) if da is None else da
    db = depth_proxy(b) if db is None else db
    # Soft winner-take-most: nearer surface dominates cross-fade
    wa = (1.0 - t) * (1.0 + da * 0.75)
    wb = t * (1.0 + db * 0.75)
    wsum = wa + wb + 1e-6
    wa, wb = wa / wsum, wb / wsum
    out = wa[..., None] * a.astype(np.float32) + wb[..., None] * b.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def interpolate_sequence_depth(
    keyframe_paths: Sequence[str | Path],
    target_frame_count: int,
    out_dir: str | Path,
    *,
    use_flow: bool = True,
) -> list[Path]:
    """Like ``interpolate_sequence`` but mid-frames use depth-aware blending."""

    keys = [Path(p) for p in keyframe_paths if Path(p).is_file()]
    if len(keys) <= 1:
        return interpolate_sequence(keyframe_paths, target_frame_count, out_dir, use_flow=use_flow)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments = len(keys) - 1
    per_seg = max(1, target_frame_count // segments)
    all_paths: list[Path] = [keys[0]]
    produced: set[Path] = set()
    for s in range(segments):
        a = read_frame_rgb(keys[s])
        b = read_frame_rgb(keys[s + 1])
        db = depth_proxy(b)
        seg_dir = out / f"seg_{s:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        mids: list[Path] = []
        count = max(0, per_seg - 2)
        flow = None
        if use_flow:
            try:
                flow, _ = compute_flow_magnitude(a, b)
            except Exception:
                flow = None
        da = depth_proxy(a) if flow is None else None
        for i in range(1, count + 1):
            t = i / (count + 1)
            if flow is not None:
                warped = warp_frame_with_flow(a, flow, alpha=t)
                rgb = depth_aware_blend(warped, b, t * 0.45, db=db)
            else:
                rgb = depth_aware_blend(a, b, t, da=da, db=db)
            fp = seg_dir / f"mid_{i:04d}.png"
            save_frame_rgb(fp, rgb)
            mids.append(fp)
        produced.update(mids)
        all_paths.extend(mids)
        all_paths.append(keys[s + 1])

    if len(all_paths) > target_frame_count:
        idx = np.linspace(0, len(all_paths) - 1, target_frame_count).astype(int)
        all_paths = [all_paths[i] for i in idx]
    elif len(all_paths) < target_frame_count:
        all_paths.extend([all_paths[-1]] * (target_frame_count - len(all_paths)))

    return _write_retimed(all_paths[:target_frame_count], out, produced)


# ==================================================================================================
# flf2v: First–last-frame video (FLF2V) interpolation between anchor endpoints.
# ==================================================================================================


def prepare_flf2v_keyframes(
    start_path: str | Path,
    end_path: str | Path,
    out_dir: str | Path,
    *,
    mid_keyframe_paths: Sequence[Path] | None = None,
) -> list[Path]:
    """Ordered keyframes: start → mids → end."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_p = out / "flf_start.png"
    end_p = out / "flf_end.png"
    save_frame_rgb(start_p, read_frame_rgb(start_path))
    save_frame_rgb(end_p, read_frame_rgb(end_path))
    keys = [start_p]
    if mid_keyframe_paths:
        keys.extend([Path(p) for p in mid_keyframe_paths if Path(p).is_file()])
    keys.append(end_p)
    return keys


def interpolate_flf2v_sequence(
    start_path: str | Path,
    end_path: str | Path,
    target_frame_count: int,
    out_dir: str | Path,
    *,
    mid_keyframe_paths: Sequence[Path] | None = None,
    source_frame_paths: Sequence[Path] | None = None,
    use_depth: bool = True,
    use_flow: bool = True,
) -> list[Path]:
    """
    Constrain segment between ``start_path`` and ``end_path``.

    Optional source frames guide mid-segment motion; endpoints are pinned.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if target_frame_count <= 0:
        return []

    start_rgb = read_frame_rgb(start_path)
    end_rgb = read_frame_rgb(end_path)
    end_depth = depth_proxy(end_rgb) if use_depth else None
    keys = prepare_flf2v_keyframes(start_path, end_path, out / "keys", mid_keyframe_paths=mid_keyframe_paths)

    if len(keys) == 2 and source_frame_paths and len(source_frame_paths) >= 3 and use_flow:
        n = target_frame_count
        paths: list[Path] = []
        for i in range(n):
            t = i / max(1, n - 1)
            if i == 0:
                rgb = start_rgb
            elif i == n - 1:
                rgb = end_rgb
            else:
                src_i = min(len(source_frame_paths) - 1, int(round(t * (len(source_frame_paths) - 1))))
                src_rgb = read_frame_rgb(source_frame_paths[src_i])
                try:
                    flow_s, _ = compute_flow_magnitude(start_rgb, src_rgb)
                    warped = warp_frame_with_flow(start_rgb, flow_s, alpha=t)
                    if use_depth:
                        rgb = depth_aware_blend(warped, end_rgb, t, db=end_depth)
                    else:
                        rgb = depth_aware_blend(warped, src_rgb, 0.5)
                except Exception:
                    rgb = (
                        depth_aware_blend(start_rgb, end_rgb, t, db=end_depth)
                        if use_depth
                        else ((1 - t) * start_rgb.astype(np.float32) + t * end_rgb.astype(np.float32)).astype(np.uint8)
                    )
            fp = out / f"frame_{i + 1:06d}.png"
            save_frame_rgb(fp, rgb)
            paths.append(fp)
        return paths

    if use_depth and len(keys) >= 2:
        return interpolate_sequence_depth(keys, target_frame_count, out, use_flow=use_flow)
    return interpolate_sequence(keys, target_frame_count, out, use_flow=use_flow)


def pixel_continuity_score(candidate: np.ndarray, previous: np.ndarray) -> float:
    """
    Higher = more similar to previous keyframe (less flicker risk).

    Uses downsampled RGB L2 in [0, 1].
    """
    a = np.asarray(candidate, dtype=np.float32)
    b = np.asarray(previous, dtype=np.float32)
    if a.shape != b.shape:
        from PIL import Image

        b = np.asarray(
            Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
    # Downsample for speed
    step = max(1, min(a.shape[0], a.shape[1]) // 64)
    a_s = a[::step, ::step, :3] / 255.0
    b_s = b[::step, ::step, :3] / 255.0
    rms = float(np.sqrt(np.mean((a_s - b_s) ** 2)))
    return float(np.clip(1.0 - rms * 2.5, 0.0, 1.0))


def pick_keyframe_with_continuity(
    candidate_paths: list[str | Path],
    *,
    previous_keyframe: str | Path | None = None,
    quality_scores: list[float] | None = None,
    continuity_weight: float = 0.35,
) -> tuple[int, list[float]]:
    """
    Pick best candidate index blending quality scores with continuity to previous KF.

    If ``quality_scores`` is None, uses edge sharpness as a cheap proxy.
    """
    paths = [Path(p) for p in candidate_paths]
    n = len(paths)
    if n == 0:
        return 0, []
    if n == 1:
        return 0, [1.0]

    if quality_scores is None:
        from pipelines.video.quality import score_frame_sharpness

        quality_scores = [score_frame_sharpness(read_frame_rgb(p)) for p in paths]
    q = np.asarray(quality_scores, dtype=np.float64)
    if q.size != n:
        q = np.resize(q, n)
    q_min, q_max = float(q.min()), float(q.max())
    qn = (q - q_min) / (q_max - q_min + 1e-8) if q_max > q_min else np.ones(n)

    w = float(max(0.0, min(1.0, continuity_weight)))
    if previous_keyframe and Path(previous_keyframe).is_file() and w > 1e-6:
        prev = read_frame_rgb(previous_keyframe)
        cont = np.asarray(
            [pixel_continuity_score(read_frame_rgb(p), prev) for p in paths],
            dtype=np.float64,
        )
        combined = (1.0 - w) * qn + w * cont
    else:
        combined = qn

    best = int(np.argmax(combined))
    return best, [float(x) for x in combined]


def harmonize_keyframe_pair(
    current_path: str | Path,
    previous_path: str | Path,
    *,
    strength: float = 0.25,
) -> Path:
    """
    Soft-blend current keyframe toward previous for identity continuity
    (cross-keyframe identity before interpolate).
    """
    cur_p = Path(current_path)
    prev_p = Path(previous_path)
    if not cur_p.is_file() or not prev_p.is_file():
        return cur_p
    s = float(max(0.0, min(0.6, strength)))
    if s < 1e-6:
        return cur_p
    cur = read_frame_rgb(cur_p).astype(np.float32)
    prev = read_frame_rgb(prev_p).astype(np.float32)
    if prev.shape != cur.shape:
        from PIL import Image

        prev = np.asarray(
            Image.fromarray(prev.astype(np.uint8)).resize((cur.shape[1], cur.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
    out = cur * (1.0 - s) + prev * s
    save_frame_rgb(cur_p, np.clip(out, 0, 255).astype(np.uint8))
    return cur_p
