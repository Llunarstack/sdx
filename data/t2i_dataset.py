"""Text-to-image training dataset (image folders or JSONL manifests) and its collate function."""

import json
import math
import random
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config.defaults.style_artists import extract_style_from_text
from models.controlnet import control_type_to_id
from PIL import Image
from torch.utils.data import Dataset
from utils.safety.minor_guard import reasons as minor_guard_reasons
from utils.training.part_aware_training import (
    PartAwareCaptionConfig,
    apply_part_aware_caption_pipeline,
    foveated_random_crop_box,
)

from .caption_utils import (
    add_anti_blending_and_count,
    apply_tag_emphasis,
    apply_training_guidance_to_caption_pair,
    boost_domain_tags,
    boost_hard_style_tags,
    boost_quality_tags,
    merge_region_captions_into_caption,
    normalize_tag_order,
    prepend_adherence_boost,
    truncate_caption_at_comma_boundary,
)
from .manifest_utils import iter_manifest_rows, manifest_row_text, negative_caption_from_row

try:
    from sdx_native.text_hygiene import normalize_caption_for_training as _normalize_caption_unicode
except ImportError:  # pragma: no cover
    _normalize_caption_unicode = None  # type: ignore[misc, assignment]

_TRAINABLE_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})

_SKIP_IMAGE_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    struct.error,
    Image.UnidentifiedImageError,
    Image.DecompressionBombError,
)

CropOffset = tuple[float, float] | None


def _looks_like_trainable_image(path: str) -> bool:
    return Path(path).suffix.lower() in _TRAINABLE_IMAGE_EXTS


def _open_train_image(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except _SKIP_IMAGE_LOAD_ERRORS as exc:
        raise Image.UnidentifiedImageError(f"failed to load image: {path}") from exc


def _cover_crop(pil_image: Image.Image, H: int, W: int, offset: CropOffset = None) -> Image.Image:
    """BOX-halve while the image is at least twice the target, bicubic-resize to cover ``H x W``, then crop.

    ``offset`` is the relative ``(y, x)`` crop position in ``[0, 1)``; ``None`` centres the crop.
    """
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid crop size H={H}, W={W}")
    iw, ih = pil_image.size
    while min(iw, ih) >= 2 * min(H, W):
        pil_image = pil_image.resize((iw // 2, ih // 2), resample=Image.BOX)
        iw, ih = pil_image.size
    scale = max(W / iw, H / ih)
    arr = np.array(pil_image.resize((round(iw * scale), round(ih * scale)), resample=Image.BICUBIC))
    dy, dx = max(0, arr.shape[0] - H), max(0, arr.shape[1] - W)
    if offset is None:
        cy, cx = dy // 2, dx // 2
    else:
        cy, cx = int(offset[0] * (dy + 1)), int(offset[1] * (dx + 1))
    return Image.fromarray(arr[cy : cy + H, cx : cx + W])


def center_crop(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Square ADM-style center crop (resize to cover, keep the middle)."""
    return _cover_crop(pil_image, image_size, image_size)


def _to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """RGB image -> ``(3, H, W)`` float tensor in ``[-1, 1]``."""
    arr = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy((arr - 0.5) / 0.5).permute(2, 0, 1)


def nearest_aspect_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> int:
    """Aspect-ratio bucketing (NovelAI 2022; SDXL, Podell et al. 2023): index of the ``(H, W)`` bucket nearest to the
    image in log aspect ratio, so the cover-crop removes as little of the frame as possible (no beheaded portraits).
    Among buckets of that aspect: the largest the image fills without upscaling, else the smallest."""
    ar = math.log(max(width, 1) / max(height, 1))
    dist = [abs(math.log(w / h) - ar) for h, w in buckets]
    same = [i for i, d in enumerate(dist) if d - min(dist) < 1e-6]
    fits = [i for i in same if buckets[i][0] <= height and buckets[i][1] <= width]
    area = lambda i: buckets[i][0] * buckets[i][1]  # noqa: E731
    return max(fits, key=area) if fits else min(same, key=area)


class Text2ImageDataset(Dataset):
    """Image/caption pairs from either a folder or a JSONL manifest.

    - Folder: ``data_path`` (and its direct subdirectories) holding ``img.png`` + ``img.txt`` / ``img.caption``;
      line 1 of the caption file is the caption, line 2 the optional negative.
    - Manifest: JSONL rows with ``image_path``/``path``/``image`` and ``caption``/``text`` plus optional
      negative, style, control, init image, regions, grounding and layout fields.

    Rows whose text involves minors are dropped at scan time (``n_blocked``). ``crop_mode`` is
    ``center`` | ``random`` | ``largest_center`` (cover-resize, then centre crop; same as ``center``).
    """

    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        caption_ext: str = ".txt",
        max_caption_length: int | None = None,
        shuffle_caption_parts: bool = False,
        use_tag_emphasis: bool = True,
        use_tag_order: bool = True,
        use_quality_boost: bool = True,
        use_adherence_boost: bool = False,
        train_shortcomings_mitigation: str = "none",
        train_shortcomings_2d: bool = False,
        train_art_guidance_mode: str = "none",
        train_art_guidance_photography: bool = True,
        train_anatomy_guidance: str = "none",
        train_style_guidance_mode: str = "none",
        train_style_guidance_artists: bool = True,
        caption_unicode_normalize: bool = False,
        use_anti_blending: bool = True,
        latent_cache_dir: str | None = None,
        crop_mode: str = "center",
        extract_style_from_caption: bool = True,
        region_caption_mode: str = "append",
        region_layout_tag: str = "[layout]",
        resolution_buckets: list[tuple[int, int]] | None = None,
        use_hierarchical_captions: bool = False,
        hierarchical_caption_separator: str = " | ",
        hierarchical_drop_global_p: float = 0.0,
        hierarchical_drop_local_p: float = 0.0,
        foveated_train_prob: float = 0.0,
        foveated_crop_frac: float = 0.55,
        grounding_mask_soft: bool = False,
        data_root: str | None = None,
    ):
        self.data_path = Path(data_path)
        self.data_root = Path(data_root).resolve() if data_root else None
        self.extract_style_from_caption = extract_style_from_caption
        self.image_size = image_size
        self.latent_cache_dir = Path(latent_cache_dir) if latent_cache_dir else None
        self.crop_mode = crop_mode
        self.resolution_buckets = resolution_buckets or []
        self.caption_ext = caption_ext
        self.max_caption_length = max_caption_length
        self.shuffle_caption_parts = shuffle_caption_parts
        self.use_tag_emphasis = use_tag_emphasis
        self.use_tag_order = use_tag_order
        self.use_quality_boost = use_quality_boost
        self.use_adherence_boost = use_adherence_boost
        self.train_shortcomings_mitigation = str(train_shortcomings_mitigation or "none").strip().lower()
        self.train_shortcomings_2d = bool(train_shortcomings_2d)
        self.train_art_guidance_mode = str(train_art_guidance_mode or "none").strip().lower()
        self.train_art_guidance_photography = bool(train_art_guidance_photography)
        self.train_anatomy_guidance = str(train_anatomy_guidance or "none").strip().lower()
        self.train_style_guidance_mode = str(train_style_guidance_mode or "none").strip().lower()
        self.train_style_guidance_artists = bool(train_style_guidance_artists)
        self.caption_unicode_normalize = caption_unicode_normalize
        self.use_anti_blending = use_anti_blending
        self.region_caption_mode = region_caption_mode
        self.region_layout_tag = region_layout_tag
        self.use_hierarchical_captions = bool(use_hierarchical_captions)
        self.hierarchical_caption_separator = hierarchical_caption_separator
        self.hierarchical_drop_global_p = float(hierarchical_drop_global_p)
        self.hierarchical_drop_local_p = float(hierarchical_drop_local_p)
        self.foveated_train_prob = float(foveated_train_prob)
        self.foveated_crop_frac = float(foveated_crop_frac)
        self.grounding_mask_soft = bool(grounding_mask_soft)
        self._partaware_caption_cfg = (
            PartAwareCaptionConfig(
                use_hierarchical_merge=True,
                hierarchical_separator=self.hierarchical_caption_separator,
                hierarchical_drop_global_p=self.hierarchical_drop_global_p,
                hierarchical_drop_local_p=self.hierarchical_drop_local_p,
            )
            if self.use_hierarchical_captions
            else None
        )
        self.samples: list[dict[str, Any]] = []
        self._scan()
        self._bucket_assign: list[int] = (
            [self._aspect_bucket(s) for s in self.samples] if self.resolution_buckets else []
        )

    def _aspect_bucket(self, sample: dict[str, Any]) -> int:
        """Fixed bucket of a sample from its image size (manifest ``width``/``height`` or the file header)."""
        size = sample.get("size")
        if size is None:
            try:
                with Image.open(self._resolve_media_path(sample["path"])) as img:
                    size = img.size
            except _SKIP_IMAGE_LOAD_ERRORS:
                size = (self.image_size, self.image_size)
        return nearest_aspect_bucket(int(size[0]), int(size[1]), self.resolution_buckets)

    def _hw_for_index(self, idx: int) -> tuple[int, int]:
        if self.resolution_buckets:
            return self.resolution_buckets[self._bucket_assign[idx]]
        return (self.image_size, self.image_size)

    def _crop_image(self, pil: Image.Image, idx: int, offset: CropOffset = None) -> Image.Image:
        H, W = self._hw_for_index(idx)
        return _cover_crop(pil, H, W, offset)

    def _resolve_media_path(self, rel: str | Path | None) -> Path | None:
        """Resolve manifest-relative paths: as given, then ``data_root`` (e.g. SDX_DATA), then the manifest dir."""
        if rel is None:
            return None
        s = str(rel).strip()
        if not s:
            return None
        p = Path(s)
        if p.is_absolute():
            return p
        if p.is_file():
            return p.resolve()
        if self.data_root is not None:
            cand = (self.data_root / p).resolve()
            if cand.is_file():
                return cand
        if self.data_path.suffix.lower() == ".jsonl":
            cand = (self.data_path.parent / p).resolve()
            if cand.is_file():
                return cand
        if self.data_root is not None:
            return (self.data_root / p).resolve()
        return (self.data_path / p).resolve() if self.data_path.is_dir() else p.resolve()

    def _maybe_foveate(
        self,
        pil: Image.Image,
        mask_l: Image.Image | None,
        idx: int,
        offset: CropOffset,
        *,
        skip_foveate: bool = False,
    ) -> tuple[Image.Image, Image.Image | None]:
        """Crop image and mask together: usually the regular crop, with ``foveated_train_prob`` a random sub-region."""
        if skip_foveate or self.foveated_train_prob <= 0 or random.random() >= self.foveated_train_prob:
            mask = self._crop_image(mask_l, idx, offset) if mask_l is not None else None
            return self._crop_image(pil, idx, offset), mask
        H, W = self._hw_for_index(idx)
        y0, x0, y1, x1 = foveated_random_crop_box(pil.height, pil.width, crop_frac=self.foveated_crop_frac)
        box = (x0, y0, x1, y1)
        pil_out = pil.convert("RGB").crop(box).resize((W, H), Image.BICUBIC)
        mask_out = mask_l.convert("L").crop(box).resize((W, H), Image.NEAREST) if mask_l is not None else None
        return pil_out, mask_out

    def _mask_image_to_tensor(self, mask_l: Image.Image) -> torch.Tensor:
        m = np.array(mask_l).astype(np.float32) / 255.0
        m = np.clip(m, 0.0, 1.0) if self.grounding_mask_soft else (m > 0.5).astype(np.float32)
        return torch.from_numpy(m).unsqueeze(0)

    def _open_mask(self, mask_rel: str) -> Image.Image | None:
        rp = self._resolve_media_path(mask_rel)
        if rp is None or not rp.exists():
            return None
        try:
            with Image.open(rp) as mask:
                return mask.convert("L")
        except _SKIP_IMAGE_LOAD_ERRORS:
            return None

    def _aux_image_tensor(self, path: str, idx: int, offset: CropOffset) -> torch.Tensor | None:
        """Control / init image cropped exactly like the main image; ``None`` when it cannot be read."""
        try:
            with Image.open(path) as img:
                return _to_tensor(self._crop_image(img.convert("RGB"), idx, offset))
        except _SKIP_IMAGE_LOAD_ERRORS:
            return None

    def _scan(self):
        self.n_blocked = 0
        self._scan_rows()
        if self.n_blocked:
            print(f"Text2ImageDataset: dropped {self.n_blocked} row(s) whose captions involve minors.")

    def _scan_rows(self):
        if self.data_path.suffix.lower() == ".jsonl":
            for d in iter_manifest_rows(self.data_path):
                if not isinstance(d, dict):
                    continue
                path = d.get("image_path") or d.get("path") or d.get("image")
                cap = d.get("caption") or d.get("text")
                if not (path and cap):
                    continue
                neg = negative_caption_from_row(d)
                screened = manifest_row_text(d)
                pl_file = self._resolve_media_path(d.get("prompt_layout_path"))
                if pl_file is not None and pl_file.is_file():
                    screened += ", " + pl_file.read_text(encoding="utf-8", errors="ignore")
                if minor_guard_reasons(screened, neg, training=True):
                    self.n_blocked += 1
                    continue
                parts = d.get("parts")
                rc = d.get("region_captions") or d.get("segments")
                if isinstance(parts, dict):
                    regions: Any = {"parts": parts, "region_captions": rc} if rc is not None else parts
                else:
                    regions = rc
                gmask = d.get("grounding_mask") or d.get("mask_path") or ""
                cg, cl, ec = d.get("caption_global"), d.get("caption_local"), d.get("entity_captions")
                pl_inline = d.get("prompt_layout")
                pl_path = d.get("prompt_layout_path") or ""
                size = (d.get("width"), d.get("height"))
                self.samples.append(
                    {
                        "path": path,
                        "caption": cap,
                        "negative_caption": neg,
                        "style": d.get("style") or "",
                        "control_image": d.get("control_image") or d.get("control_path") or "",
                        "control_type": d.get("control_type")
                        or d.get("control_kind")
                        or d.get("controlnet_type")
                        or "",
                        "init_image": d.get("init_image") or d.get("init_image_path") or d.get("source_image") or "",
                        "weight": float(d.get("weight", d.get("aesthetic_score", 1.0))),
                        "difficulty": float(d.get("difficulty", 0.5)),
                        "regions": regions,
                        "grounding_mask": gmask if isinstance(gmask, str) else "",
                        "caption_global": cg if isinstance(cg, str) else None,
                        "caption_local": cl if isinstance(cl, str) else None,
                        "entity_captions": ec if isinstance(ec, dict) else None,
                        "prompt_layout": pl_inline if isinstance(pl_inline, dict) else None,
                        "prompt_layout_path": pl_path if isinstance(pl_path, str) else "",
                        "size": tuple(map(int, size))
                        if all(isinstance(v, (int, float)) and v > 0 for v in size)
                        else None,
                    }
                )
            return
        # Both layouts work: data_path/<subdir>/img+caption and a flat data_path/img+caption folder.
        scan_dirs = [self.data_path] + [d for d in self.data_path.iterdir() if d.is_dir()]
        for subdir in scan_dirs:
            for img_path in subdir.glob("*"):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                cap_path = img_path.with_suffix(self.caption_ext)
                if not cap_path.exists():
                    cap_path = img_path.with_name(img_path.stem + ".caption")
                if not cap_path.exists():
                    continue
                lines = cap_path.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
                caption = lines[0].strip()
                negative_caption = lines[1].strip() if len(lines) > 1 else ""
                if not caption:
                    continue
                if minor_guard_reasons(caption, negative_caption, training=True):
                    self.n_blocked += 1
                    continue
                self.samples.append(
                    {
                        "path": str(img_path),
                        "caption": caption,
                        "negative_caption": negative_caption,
                        "style": "",
                        "control_image": "",
                        "control_type": "",
                        "init_image": "",
                        "weight": 1.0,
                        "difficulty": 0.5,
                        "regions": None,
                        "grounding_mask": "",
                        "caption_global": None,
                        "caption_local": None,
                        "entity_captions": None,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _process_caption(self, caption: str, negative_caption: str = "") -> tuple[str, str]:
        if self.caption_unicode_normalize and _normalize_caption_unicode is not None:
            caption = _normalize_caption_unicode(caption)
            if (negative_caption or "").strip():
                negative_caption = _normalize_caption_unicode(negative_caption)
        if self.use_tag_emphasis:
            caption = apply_tag_emphasis(caption)
        if self.use_tag_order:
            caption = normalize_tag_order(caption)
        caption = boost_hard_style_tags(caption)
        if self.use_quality_boost:
            caption = boost_quality_tags(caption)
        caption = boost_domain_tags(caption)
        if self.use_adherence_boost:
            caption = prepend_adherence_boost(caption)
        if self.use_anti_blending:
            caption, negative_caption = add_anti_blending_and_count(caption, negative_caption)
        if (
            self.train_shortcomings_mitigation in ("auto", "all")
            or self.train_art_guidance_mode in ("auto", "all")
            or self.train_anatomy_guidance in ("lite", "strong")
            or self.train_style_guidance_mode in ("auto", "all")
        ):
            caption, negative_caption = apply_training_guidance_to_caption_pair(
                caption,
                negative_caption,
                shortcomings_mode=self.train_shortcomings_mitigation,
                shortcomings_2d=self.train_shortcomings_2d,
                art_guidance_mode=self.train_art_guidance_mode,
                anatomy_guidance=self.train_anatomy_guidance,
                style_guidance_mode=self.train_style_guidance_mode,
                style_guidance_artists=self.train_style_guidance_artists,
                include_art_guidance_photography=self.train_art_guidance_photography,
            )
        if self.shuffle_caption_parts:
            parts = [p.strip() for p in caption.split(",") if p.strip()]
            random.shuffle(parts)
            caption = ", ".join(parts)
        if self.max_caption_length:
            caption = truncate_caption_at_comma_boundary(caption, self.max_caption_length)
        return caption.strip(), (negative_caption or "").strip()

    def _latent_path(self, path: str) -> Path | None:
        """Cache file of a precomputed latent: the image stem + ``.pt`` inside ``latent_cache_dir``."""
        if not self.latent_cache_dir:
            return None
        return self.latent_cache_dir / (Path(path).stem + ".pt")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_err: BaseException | None = None
        n = len(self.samples)
        for off in range(min(32, n)):
            try:
                # A replacement for an unreadable file is cropped to *this* index's bucket so the batch still stacks.
                return self._getitem_impl((idx + off) % n, hw_idx=idx)
            except Exception as exc:
                # Scraped datasets include corrupt PNGs, zips, truncated files, etc.
                last_err = exc
        raise RuntimeError(f"could not load any image near dataset index {idx}") from last_err

    def _load_latent(self, path: str) -> torch.Tensor | None:
        lp = self._latent_path(path)
        if lp is None or not lp.exists() or self.resolution_buckets:
            return None
        try:
            return torch.load(lp, map_location="cpu", weights_only=True)
        except Exception:  # a corrupt cache entry falls back to encoding the pixels
            return None

    def _getitem_impl(self, idx: int, hw_idx: int | None = None) -> dict[str, Any]:
        hw_idx = idx if hw_idx is None else hw_idx
        s = self.samples[idx]
        resolved = self._resolve_media_path(s["path"])
        path = str(resolved if resolved is not None else s["path"])
        if not _looks_like_trainable_image(path):
            raise Image.UnidentifiedImageError(f"not a trainable image: {path}")
        raw_caption = s["caption"]
        if self._partaware_caption_cfg is not None:
            raw_caption = apply_part_aware_caption_pipeline(raw_caption, s, self._partaware_caption_cfg, rng=random)
        raw_caption = merge_region_captions_into_caption(
            raw_caption,
            s.get("regions"),
            mode=self.region_caption_mode,
            layout_tag=self.region_layout_tag,
        )
        caption, negative_caption = self._process_caption(raw_caption, s.get("negative_caption", ""))
        out: dict[str, Any] = {
            "caption": caption,
            "negative_caption": negative_caption,
            "path": path,
            "weight": s.get("weight", 1.0),
            "difficulty": s.get("difficulty", 0.5),
        }
        style_text = s.get("style", "")
        if not style_text and self.extract_style_from_caption:
            style_text = extract_style_from_text(s.get("caption", "")) or ""
        if style_text:
            out["style"] = style_text
        pl = s.get("prompt_layout")
        if isinstance(pl, dict):
            out["prompt_layout"] = pl
        else:
            rp = self._resolve_media_path(s.get("prompt_layout_path"))
            if rp is not None and rp.is_file():
                try:
                    raw = json.loads(rp.read_text(encoding="utf-8", errors="ignore"))
                except (OSError, ValueError):
                    raw = None
                if isinstance(raw, dict):
                    out["prompt_layout"] = raw

        ctrl_path = s.get("control_image", "")
        if ctrl_path:
            rp = self._resolve_media_path(ctrl_path)
            ctrl_path = str(rp) if rp is not None else ctrl_path
            out["control_image_path"] = ctrl_path
        init_path = s.get("init_image", "")
        if init_path:
            rp = self._resolve_media_path(init_path)
            init_path = str(rp) if rp is not None else init_path
            out["init_image_path"] = init_path
        mask_rel = s.get("grounding_mask") or ""

        latent = self._load_latent(path)
        if latent is not None:
            # Cached latents are full-frame centre crops, so every aux input is centre-cropped and never foveated.
            offset: CropOffset = None
            h_i, w_i = self._hw_for_index(hw_idx)
            out["latent_values"] = latent
            out["pixel_values"] = torch.zeros(3, h_i, w_i)
            mask_l = self._open_mask(mask_rel) if mask_rel else None
            if mask_l is not None:
                out["grounding_mask"] = self._mask_image_to_tensor(self._crop_image(mask_l, hw_idx))
        else:
            # One random offset for every input so image, mask, control and init stay pixel-aligned.
            offset = (random.random(), random.random()) if self.crop_mode == "random" else None
            pil = _open_train_image(path)
            mask_l = self._open_mask(mask_rel) if mask_rel else None
            # Control / init inputs are not foveated, so their presence disables it to keep them aligned.
            skip_foveate = bool(ctrl_path) or bool(init_path)
            pil, mask_l = self._maybe_foveate(pil, mask_l, hw_idx, offset, skip_foveate=skip_foveate)
            out["pixel_values"] = _to_tensor(pil)
            if mask_l is not None:
                out["grounding_mask"] = self._mask_image_to_tensor(mask_l)

        if init_path:
            init = self._aux_image_tensor(init_path, hw_idx, offset)
            if init is not None:
                out["init_pixel_values"] = init
        if ctrl_path:
            ctrl = self._aux_image_tensor(ctrl_path, hw_idx, offset)
            if ctrl is not None:
                out["control_image"] = ctrl
                out["control_type_id"] = int(control_type_to_id(s.get("control_type", "")))
        return out


def collate_t2i(batch: list[dict]) -> dict[str, Any]:
    """Stack pixel/latent/control/init tensors (when every item has them); captions and styles stay lists."""
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "negative_captions": [b.get("negative_caption", "") for b in batch],
        "styles": [b.get("style", "") for b in batch],
        "sample_weights": torch.tensor([b.get("weight", 1.0) for b in batch], dtype=torch.float32),
    }
    if all("difficulty" in b for b in batch):
        out["difficulty"] = torch.tensor([b["difficulty"] for b in batch], dtype=torch.float32)
    if all("latent_values" in b for b in batch):
        out["latent_values"] = torch.stack([b["latent_values"] for b in batch])
    if all("control_image" in b for b in batch):
        out["control_image"] = torch.stack([b["control_image"] for b in batch])
        out["control_type_id"] = torch.tensor([int(b.get("control_type_id", 0)) for b in batch], dtype=torch.long)
    if all("init_pixel_values" in b for b in batch):
        out["init_pixel_values"] = torch.stack([b["init_pixel_values"] for b in batch])
    if any("grounding_mask" in b for b in batch):
        _, h, w = batch[0]["pixel_values"].shape
        empty = torch.zeros(1, h, w, dtype=torch.float32)
        out["grounding_mask"] = torch.stack([b.get("grounding_mask", empty) for b in batch])
        out["grounding_mask_valid"] = torch.tensor(["grounding_mask" in b for b in batch], dtype=torch.bool)
    if any("prompt_layout" in b for b in batch):
        out["prompt_layout"] = [b.get("prompt_layout") for b in batch]
    return out
