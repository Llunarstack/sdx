"""Exact-pixel finishing for game imagery: grid snap, palette, transparent background, seamless tiles, critics.

All functions take RGB arrays in [0, 255]; pixel-art output is uint8 so saving can't dither it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _edge_profile(img: np.ndarray, axis: int) -> np.ndarray:
    """Share of rows (axis=1) or columns (axis=0) with a real colour step between neighbours.

    Counting steps above the noise floor, instead of averaging them, keeps sparse sprites (a few big
    shapes on a flat background) from drowning in sensor-like noise.
    """
    d = np.abs(np.diff(img.astype(np.float32), axis=axis)).sum(-1)
    return (d > 3.0 * float(np.median(d)) + 8.0).mean(axis=1 - axis)


def _phase(profile: np.ndarray, p: int) -> tuple[int, float]:
    """(cell start offset, recall) at period p: recall = share of all edges within 1 px of the best lattice."""
    per_phase = profile[: (len(profile) // p) * p].reshape(-1, p).sum(0)
    win = per_phase + np.roll(per_phase, 1) + np.roll(per_phase, -1)  # blurred edges spread over +-1 px
    o = int((win + 0.25 * per_phase).argmax())  # sharp edges tie across the window; prefer the exact step
    # Step o sits between pixel o and o + 1, so cells start at o + 1.
    return (o + 1) % p, float(win[o] / (per_phase.sum() + 1e-6))


def estimate_pixel_size(img: np.ndarray, max_period: int = 64) -> tuple[int, int, int, float]:
    """(pixel size, x phase, y phase, confidence) of an upscaled pixel-art image.

    The true pixel lattice holds nearly all edges; divisors of it do too, multiples miss most. So the
    answer is the largest period whose lattice holds >= 80% of edges on both axes. Confidence = that
    recall over chance (3/p for a +-1 px window): well above 1 for pixel art, about 1 for photos/paintings.
    """
    x = np.asarray(img, dtype=np.float32)[..., :3]
    # A central window keeps this cheap on big canvases; 384 px holds >= 6 cells even at 64 px pixels.
    H, W = x.shape[:2]
    y0, x0 = max(0, (H - 384) // 2), max(0, (W - 384) // 2)
    x = x[y0 : y0 + 384, x0 : x0 + 384]
    prof_x, prof_y = _edge_profile(x, 1), _edge_profile(x, 0)
    best = (1, 0, 0, 1.0)
    for p in range(4, min(max_period, min(x.shape[:2]) // 4) + 1):
        ox, rec_x = _phase(prof_x, p)
        oy, rec_y = _phase(prof_y, p)
        if min(rec_x, rec_y) >= 0.8:
            best = (p, (ox + x0) % p, (oy + y0) % p, min(rec_x, rec_y) * p / 3.0)
    return best


def cell_colors(img: np.ndarray, p: int, ox: int = 0, oy: int = 0) -> np.ndarray:
    """One colour per p x p cell: the median of the cell's inner pixels, so anti-aliased edges don't bleed in."""
    h, w = (img.shape[0] - oy) // p, (img.shape[1] - ox) // p
    cells = img[oy : oy + h * p, ox : ox + w * p].astype(np.float32).reshape(h, p, w, p, 3)
    m = p // 4
    inner = cells[:, m : p - m, :, m : p - m] if p - 2 * m >= 1 else cells
    return np.median(inner.transpose(0, 2, 1, 3, 4).reshape(h, w, -1, 3), axis=2)


def quantize_palette(img: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> np.ndarray:
    """k-means palette (k-means++ init, fixed seed) -> uint8 image using at most k colours."""
    px = img.reshape(-1, 3).astype(np.float32)
    if k <= 0 or len(np.unique(np.rint(px), axis=0)) <= k:
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)
    rng = np.random.default_rng(seed)
    centers = [px[rng.integers(len(px))]]
    d2 = ((px - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        centers.append(px[rng.choice(len(px), p=d2 / d2.sum())] if d2.sum() > 0 else px[rng.integers(len(px))])
        d2 = np.minimum(d2, ((px - centers[-1]) ** 2).sum(1))
    c = np.stack(centers)

    def nearest(c: np.ndarray) -> np.ndarray:
        return ((c**2).sum(1)[None] - 2.0 * px @ c.T).argmin(1)  # |x|^2 is the same for every centre

    for _ in range(iters):
        lab = nearest(c)
        counts = np.bincount(lab, minlength=k)
        sums = np.zeros_like(c)
        np.add.at(sums, lab, px)
        moved = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], c)
        if np.allclose(moved, c, atol=0.25):
            break
        c = moved
    # k-means splits one real colour into near-identical shades; fold those into the larger cluster.
    counts = np.bincount(nearest(c), minlength=k)
    keep: list[int] = []
    for i in np.argsort(-counts):
        if counts[i] and all(np.abs(c[i] - c[j]).max() > 12.0 for j in keep):
            keep.append(int(i))
    c = c[keep]
    return np.clip(np.rint(c), 0, 255).astype(np.uint8)[nearest(c)].reshape(img.shape)


def despeckle(grid: np.ndarray, max_delta: float = 48.0) -> np.ndarray:
    """Merge lone pixels into their uniform 8-neighbourhood when the colours are close (AA leftovers).

    High-contrast lone pixels (eye glints, sparkles) are kept on purpose.
    """
    g = grid.astype(np.int32)
    pad = np.pad(g, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = g.shape[:2]
    nbrs = [pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    first = nbrs[0]
    uniform = np.all([np.all(n == first, axis=-1) for n in nbrs[1:]], axis=0)
    close = np.sqrt(((g - first) ** 2).sum(-1)) < max_delta
    out = np.where((uniform & close & np.any(g != first, axis=-1))[..., None], first, g)
    return out.astype(np.uint8)


def _border_color(img: np.ndarray) -> np.ndarray:
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])[:, :3].astype(np.int32)
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return colors[counts.argmax()]


def background_mask(img: np.ndarray, tol: float = 24.0) -> np.ndarray:
    """True where the pixel is flat background: the dominant border colour, connected to the border."""
    near = np.sqrt(((img[..., :3].astype(np.int32) - _border_color(img)) ** 2).sum(-1)) <= tol
    lab, _ = ndimage.label(near)
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge_labels[edge_labels > 0])


def pixel_snap(
    img: np.ndarray,
    *,
    pixel_size: int = 0,
    grid: int = 0,
    palette_colors: int = 32,
    transparent_bg: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a generated "pixel art" image into real pixel art.

    Returns (full-size uint8 image on an exact grid, native-resolution sprite); both RGBA when
    ``transparent_bg``. ``grid`` (32 for "32x32 sprite") wins over ``pixel_size``; with neither, the pixel
    size is estimated from the image.
    """
    img = np.asarray(img, dtype=np.float32)[..., :3]
    H, W = img.shape[:2]
    ox = oy = 0
    if grid > 0:
        p = max(1, min(H, W) // grid)
        ox, oy = (W - (W // p) * p) // 2, (H - (H // p) * p) // 2
    elif pixel_size > 0:
        p = pixel_size
    else:
        p, ox, oy, _ = estimate_pixel_size(img)
        if p < 4:
            # No visible lattice: use a classic 1/64-of-the-short-side pixel so it still reads as pixel art.
            p, ox, oy = max(2, min(H, W) // 64), 0, 0
    small = despeckle(quantize_palette(cell_colors(img, p, ox, oy), palette_colors, seed=seed))
    h, w = small.shape[:2]
    if transparent_bg:
        alpha = np.where(background_mask(small), 0, 255).astype(np.uint8)
        small = np.dstack([small, alpha])
    full = np.repeat(np.repeat(small, p, axis=0), p, axis=1)
    # Centre the exact grid on the requested canvas; the margin is background (or transparent).
    out = np.zeros((H, W, small.shape[2]), np.uint8)
    if not transparent_bg:
        out[:] = _border_color(small)
    y0, x0 = (H - h * p) // 2, (W - w * p) // 2
    out[y0 : y0 + h * p, x0 : x0 + w * p] = full
    return out, small


def make_tileable(img: np.ndarray, blend: float = 0.25) -> np.ndarray:
    """Offset-and-blend: the borders come from the image's own continuous middle, so it wraps without seams."""
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape[:2]
    rolled = np.roll(img, (H // 2, W // 2), axis=(0, 1))

    def ramp(n: int) -> np.ndarray:
        return np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / max(1.0, blend * n), 0.0, 1.0)

    wgt = np.minimum(ramp(H)[:, None], ramp(W)[None, :])[..., None]
    return img * wgt + rolled * (1.0 - wgt)


# --- critics: higher is better, roughly 0..1 ---


def seam_score(img: np.ndarray) -> float:
    """How well the image wraps: ordinary neighbour steps vs wrap-around steps (1 = indistinguishable)."""
    x = np.asarray(img, dtype=np.float32)[..., :3]
    inner = (np.abs(np.diff(x, axis=1)).mean() + np.abs(np.diff(x, axis=0)).mean()) / 2
    wrap = (np.abs(x[:, 0] - x[:, -1]).mean() + np.abs(x[0] - x[-1]).mean()) / 2
    return float(np.clip((inner + 1e-3) / (wrap + 1e-3), 0.0, 1.0))


def grid_score(img: np.ndarray) -> float:
    """How clearly the image sits on one pixel lattice (0 = no lattice / mush, 1 = crisp grid)."""
    x = np.asarray(img, dtype=np.float32)[..., :3]
    p, ox, oy, _ = estimate_pixel_size(x)
    if p < 4:
        return 0.0
    # Flat cells: how little the colour varies inside each cell vs across the image.
    h, w = (x.shape[0] - oy) // p, (x.shape[1] - ox) // p
    cells = x[oy : oy + h * p, ox : ox + w * p].reshape(h, p, w, p, 3)
    within = float(cells.std(axis=(1, 3)).mean())
    return float(np.clip(1.0 - within / (float(x.std()) + 1e-3), 0.0, 1.0))


def icon_score(img: np.ndarray) -> float:
    """Readability at 32 px: a centred subject that fills a sensible share and contrasts with its background."""
    from PIL import Image

    small = np.asarray(Image.fromarray(np.clip(img[..., :3], 0, 255).astype(np.uint8)).resize((32, 32), Image.BOX))
    bg = background_mask(small, tol=32.0)
    fill = 1.0 - bg.mean()
    if fill <= 0.02 or fill >= 0.98:
        return 0.0
    fill_ok = 1.0 - min(1.0, abs(fill - 0.45) / 0.45)
    lum = small.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    contrast = abs(lum[~bg].mean() - lum[bg].mean()) / 255.0
    ys, xs = np.nonzero(~bg)
    centred = 1.0 - min(1.0, np.hypot(ys.mean() - 15.5, xs.mean() - 15.5) / 16.0)
    return float(np.clip(0.4 * fill_ok + 0.35 * min(1.0, contrast * 2) + 0.25 * centred, 0.0, 1.0))


def game_score(img: np.ndarray, critic: str) -> float:
    fn = {"pixel": grid_score, "seam": seam_score, "icon": icon_score}.get(critic)
    return fn(img) if fn else 0.0
