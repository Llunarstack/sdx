"""MultiDiffusion-style tiled denoising: a latent bigger than the DiT's native grid is predicted as overlapping
native-size tiles (same timestep and conditioning) blended back with cosine ramps, so attention cost stays per tile."""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
from diffusion.sampling import dit_patch_size, snap_to_multiple

AUTO_TOKEN_RATIO = 1.5  # tile past this many native token counts (full attention is O(N^2) in memory)


def tile_starts(length: int, tile: int, overlap: int, align: int = 1) -> list[int]:
    """Tile offsets along one axis: first at 0, last flush with the end, neighbours overlapping by >= ``overlap``."""
    if length <= tile:
        return [0]
    n = math.ceil((length - overlap) / max(1, tile - overlap))
    last = length - tile
    return sorted({min(last, round(last * i / max(1, n - 1) / align) * align) for i in range(max(2, n))})


def _ramp(n: int) -> torch.Tensor:
    return 0.5 - 0.5 * torch.cos(math.pi * (torch.arange(n, dtype=torch.float32) + 0.5) / n)


def axis_weights(starts: list[int], tile: int, length: int) -> list[torch.Tensor]:
    """Per-tile 1-D weights: 1 inside, cosine ramps across each overlap (complementary, so neighbours sum to 1)."""
    size = min(tile, length)
    out = []
    for k, s in enumerate(starts):
        w = torch.ones(size)
        if k and (o := starts[k - 1] + size - s) > 0:
            w[:o] = torch.minimum(w[:o], _ramp(o))
        if k + 1 < len(starts) and (o := s + size - starts[k + 1]) > 0:
            w[-o:] = torch.minimum(w[-o:], _ramp(o).flip(0))
        out.append(w)
    return out


def tile_layout(h: int, w: int, tile: int, overlap: int, align: int = 1) -> list[tuple[int, int, torch.Tensor]]:
    """``(y, x, weight)`` per tile; the weights are normalized so they sum to 1 at every latent cell."""
    ys, xs = tile_starts(h, tile, overlap, align), tile_starts(w, tile, overlap, align)
    wy, wx = axis_weights(ys, tile, h), axis_weights(xs, tile, w)
    th, tw = min(tile, h), min(tile, w)
    tiles = [(y, x, a[:, None] * b[None, :]) for y, a in zip(ys, wy) for x, b in zip(xs, wx)]
    total = torch.zeros(h, w)
    for y, x, wt in tiles:
        total[y : y + th, x : x + tw] += wt
    return [(y, x, wt / total[y : y + th, x : x + tw]) for y, x, wt in tiles]


class TiledModel:
    """Model proxy: ``model(x, t, **kw)`` on overlapping ``tile x tile`` crops of ``x``, blended back seamlessly.

    4-D tensor kwargs with ``x``'s batch (control images) are cropped to the same relative region; other per-sample
    tensors are repeated per tile. ``block_cache`` is dropped: step caches key on one grid, not on many tiles.
    """

    def __init__(self, model: Any, tile: int, overlap: int, *, align: int = 1, max_batch: int = 4):
        self.model, self.tile, self.overlap, self.align, self.max_batch = model, tile, overlap, align, max_batch

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kw: Any) -> torch.Tensor:
        b, h, w = x.shape[0], x.shape[-2], x.shape[-1]
        if h <= self.tile and w <= self.tile:
            return self.model(x, t, **kw)
        kw.pop("block_cache", None)
        th, tw = min(self.tile, h), min(self.tile, w)
        layout = tile_layout(h, w, self.tile, self.overlap, self.align)
        out = None
        for i in range(0, len(layout), self.max_batch):
            chunk = layout[i : i + self.max_batch]
            n = len(chunk)
            xs = torch.cat([x[..., y : y + th, x0 : x0 + tw] for y, x0, _ in chunk])
            ts = t.repeat(n) if torch.is_tensor(t) and t.ndim == 1 and t.shape[0] == b else t
            kws = {k: _tile_kwarg(v, b, h, w, [(y, x0) for y, x0, _ in chunk], th, tw) for k, v in kw.items()}
            pred = self.model(xs, ts, **kws)
            if out is None:
                out = pred.new_zeros(b, pred.shape[1], h, w)
            for j, (y, x0, wt) in enumerate(chunk):
                out[..., y : y + th, x0 : x0 + tw] += pred[j * b : (j + 1) * b] * wt.to(pred)
        return out


def _tile_kwarg(v: Any, b: int, h: int, w: int, origins: list[tuple[int, int]], th: int, tw: int) -> Any:
    if not torch.is_tensor(v) or v.ndim == 0 or v.shape[0] != b:
        return v
    if v.ndim == 4:
        sy, sx = v.shape[-2] / h, v.shape[-1] / w
        crops = [v[..., round(y * sy) : round((y + th) * sy), round(x * sx) : round((x + tw) * sx)] for y, x in origins]
        return torch.cat(crops)
    return torch.cat([v] * len(origins))


def maybe_tiled(m: Any, lat_h: int, lat_w: int, *, force: bool = False, overlap_px: int = 0) -> TiledModel | None:
    """A :class:`TiledModel` over ``m.model`` for an ``lat_h x lat_w`` grid when ``force`` or past
    :data:`AUTO_TOKEN_RATIO` x the native token count; ``None`` (use the plain model) otherwise."""
    native = int(m.latent_size)
    if lat_h <= native and lat_w <= native:
        return None
    if not force and lat_h * lat_w <= AUTO_TOKEN_RATIO * native * native:
        return None
    patch = dit_patch_size(m.model)
    overlap = overlap_px // 8 if overlap_px > 0 else native // 4
    overlap = min(snap_to_multiple(overlap, patch, mode="floor"), native - patch)
    tiled = TiledModel(m.model, native, overlap, align=patch)
    n = len(tile_starts(lat_h, native, overlap, patch)) * len(tile_starts(lat_w, native, overlap, patch))
    print(
        f"Tiled denoise: latent {lat_h}x{lat_w} as {n} tiles of {native}x{native} (overlap {overlap}).", file=sys.stderr
    )
    return tiled
