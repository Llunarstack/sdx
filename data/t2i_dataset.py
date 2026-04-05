# Text-to-image dataset: PixAI-style tag prompts + ReVe-style long/complex captions.
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from models.controlnet import control_type_to_id
from PIL import Image
from torch.utils.data import Dataset
from utils.training.part_aware_training import (
    PartAwareCaptionConfig,
    apply_part_aware_caption_pipeline,
    foveated_random_crop_box,
)

from .caption_utils import (
    add_anti_blending_and_count,
    apply_art_guidance_to_caption_pair,
    apply_pixai_emphasis,
    apply_shortcomings_to_caption_pair,
    apply_style_guidance_to_caption_pair,
    boost_domain_tags,
    boost_hard_style_tags,
    boost_quality_tags,
    merge_region_captions_into_caption,
    normalize_tag_order,
    prepend_adherence_boost,
)

try:
    from sdx_native.text_hygiene import normalize_caption_for_training as _normalize_caption_unicode
except ImportError:  # pragma: no cover
    _normalize_caption_unicode = None  # type: ignore[misc, assignment]


def _center_crop(pil_image, image_size: int):
    """Center crop to image_size (ADM-style)."""
    w, h = pil_image.size
    while min(w, h) >= 2 * image_size:
        pil_image = pil_image.resize((w // 2, h // 2), resample=Image.BOX)
        w, h = pil_image.size
    scale = image_size / min(w, h)
    pil_image = pil_image.resize((round(w * scale), round(h * scale)), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[:2]
    crop_y = (h - image_size) // 2
    crop_x = (w - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _random_crop(pil_image, image_size: int):
    """Random crop to image_size (data augmentation)."""
    w, h = pil_image.size
    while min(w, h) >= 2 * image_size:
        pil_image = pil_image.resize((w // 2, h // 2), resample=Image.BOX)
        w, h = pil_image.size
    scale = image_size / min(w, h)
    pil_image = pil_image.resize((round(w * scale), round(h * scale)), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[:2]
    if h == image_size and w == image_size:
        return Image.fromarray(arr)
    crop_y = random.randint(0, max(0, h - image_size))
    crop_x = random.randint(0, max(0, w - image_size))
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _largest_center_crop(pil_image, image_size: int):
    """Resize so short side = image_size (cover), then center crop. Preserves aspect; no stretch."""
    w, h = pil_image.size
    while min(w, h) >= 2 * image_size:
        pil_image = pil_image.resize((w // 2, h // 2), resample=Image.BOX)
        w, h = pil_image.size
    scale = image_size / min(w, h)  # scale so min(w,h) = image_size
    new_w, new_h = round(w * scale), round(h * scale)
    pil_image = pil_image.resize((new_w, new_h), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[:2]
    crop_y = (h - image_size) // 2
    crop_x = (w - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


# --- IMPROVEMENTS §1.1: rectangular targets (H, W) for resolution / aspect buckets ---
def _cover_center_crop_hw(pil_image: Image.Image, H: int, W: int) -> Image.Image:
    """Resize to cover (W,H), then center-crop to exactly H×W."""
    iw, ih = pil_image.size
    while min(iw, ih) >= 2 * min(H, W):
        pil_image = pil_image.resize((iw // 2, ih // 2), resample=Image.BOX)
        iw, ih = pil_image.size
    scale = max(W / iw, H / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    pil_image = pil_image.resize((nw, nh), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[:2]
    cy = max(0, (h - H) // 2)
    cx = max(0, (w - W) // 2)
    return Image.fromarray(arr[cy : cy + H, cx : cx + W])


def _cover_random_crop_hw(pil_image: Image.Image, H: int, W: int) -> Image.Image:
    iw, ih = pil_image.size
    while min(iw, ih) >= 2 * min(H, W):
        pil_image = pil_image.resize((iw // 2, ih // 2), resample=Image.BOX)
        iw, ih = pil_image.size
    scale = max(W / iw, H / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    pil_image = pil_image.resize((nw, nh), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[:2]
    cy = random.randint(0, max(0, h - H))
    cx = random.randint(0, max(0, w - W))
    return Image.fromarray(arr[cy : cy + H, cx : cx + W])


def _largest_center_crop_hw(pil_image: Image.Image, H: int, W: int) -> Image.Image:
    """Match short-side scaling idea for non-square targets: min side -> min(H,W), then center crop."""
    iw, ih = pil_image.size
    target_min = min(H, W)
    while min(iw, ih) >= 2 * target_min:
        pil_image = pil_image.resize((iw // 2, ih // 2), resample=Image.BOX)
        iw, ih = pil_image.size
    imin = min(iw, ih)
    scale = target_min / max(1, imin)
    nw, nh = round(iw * scale), round(ih * scale)
    pil_image = pil_image.resize((nw, nh), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[:2]
    cy = max(0, (h - H) // 2)
    cx = max(0, (w - W) // 2)
    return Image.fromarray(arr[cy : cy + H, cx : cx + W])


def _crop_to_hw(pil_image: Image.Image, H: int, W: int, crop_mode: str) -> Image.Image:
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid crop size H={H}, W={W}")
    if crop_mode == "random":
        return _cover_random_crop_hw(pil_image, H, W)
    if crop_mode == "largest_center":
        return _largest_center_crop_hw(pil_image, H, W)
    return _cover_center_crop_hw(pil_image, H, W)


def normalize_to_latent_range(x: torch.Tensor, scale: float = 0.18215) -> torch.Tensor:
    """Scale pixel tensor for VAE latent (e.g. SD convention)."""
    return x * scale


# --- PixAI-style: tag-based prompts (comma-separated, optional emphasis) ---
def format_as_tags(tokens: List[str], emphasis: bool = False) -> str:
    """Format list of tokens as comma-separated tags. Optionally wrap in () for emphasis."""
    if emphasis and random.random() < 0.3:
        return ", ".join(f"({t})" if random.random() < 0.4 else t for t in tokens)
    return ", ".join(tokens)


def structured_caption(parts: Dict[str, str], order: Optional[List[str]] = None) -> str:
    """ReVe/DetailMaster-style: structured caption for better adherence.
    order e.g. ['subject', 'setting', 'style', 'camera'].
    """
    if order is None:
        order = ["subject", "setting", "aesthetics", "style", "camera"]
    return ", ".join(parts.get(k, "") for k in order if parts.get(k))


class Text2ImageDataset(Dataset):
    """
    Expects either:
    - data_path = dir with subdirs, each containing images + captions (same name .txt or .caption)
    - Or a manifest: JSONL with keys {"image_path": "...", "caption": "..."} or {"path", "text"}
    """

    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        normalize_latent_scale: float = 0.18215,
        caption_ext: str = ".txt",
        use_struct: bool = False,
        tag_style_prob: float = 0.5,
        max_caption_length: Optional[int] = None,
        shuffle_caption_parts: bool = False,
        use_pixai_emphasis: bool = True,
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
        latent_cache_dir: Optional[str] = None,
        crop_mode: str = "center",  # "center" | "random" | "largest_center" (IMPROVEMENTS 1.2)
        extract_style_from_caption: bool = True,  # Auto-fill style from artist/style tags (PixAI, Danbooru)
        region_caption_mode: str = "append",  # "append" | "prefix" | "off" — merge JSONL regions/parts into T5 caption
        region_layout_tag: str = "[layout]",
        resolution_buckets: Optional[List[Tuple[int, int]]] = None,
        bucket_seed: int = 42,
        bucket_fixed_assign: bool = False,
        # Part-aware / grounding (JSONL: grounding_mask, caption_global, caption_local, entity_captions)
        use_hierarchical_captions: bool = False,
        hierarchical_caption_separator: str = " | ",
        hierarchical_drop_global_p: float = 0.0,
        hierarchical_drop_local_p: float = 0.0,
        foveated_train_prob: float = 0.0,
        foveated_crop_frac: float = 0.55,
        grounding_mask_soft: bool = False,
    ):
        self.data_path = Path(data_path)
        self.extract_style_from_caption = extract_style_from_caption
        self.image_size = image_size
        self.latent_cache_dir = Path(latent_cache_dir) if latent_cache_dir else None
        self.crop_mode = crop_mode
        self.resolution_buckets = resolution_buckets or []
        self.bucket_seed = int(bucket_seed)
        self.bucket_fixed_assign = bool(bucket_fixed_assign)
        self.normalize_latent_scale = normalize_latent_scale
        self.caption_ext = caption_ext
        self.use_struct = use_struct
        self.tag_style_prob = tag_style_prob
        self.max_caption_length = max_caption_length
        self.shuffle_caption_parts = shuffle_caption_parts
        self.use_pixai_emphasis = use_pixai_emphasis
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
        self._crop_fn = {"center": _center_crop, "random": _random_crop, "largest_center": _largest_center_crop}.get(
            crop_mode, _center_crop
        )
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
        self.samples: List[Dict[str, Any]] = []
        self._scan()
        self._bucket_assign: List[int] = [0] * len(self.samples)
        self.set_epoch(0)

    def set_epoch(self, epoch: int = 0) -> None:
        """Refresh per-index bucket ids when using ``resolution_buckets`` (IMPROVEMENTS §1.1)."""
        if not self.resolution_buckets:
            return
        nb = len(self.resolution_buckets)
        e = int(epoch)
        for i in range(len(self.samples)):
            if self.bucket_fixed_assign:
                self._bucket_assign[i] = (i * 100003 + self.bucket_seed) % nb
            else:
                self._bucket_assign[i] = (i * 100003 + e * 10007 + self.bucket_seed) % nb

    def _hw_for_index(self, idx: int) -> Tuple[int, int]:
        if self.resolution_buckets:
            b = self._bucket_assign[idx]
            return self.resolution_buckets[b]
        return (self.image_size, self.image_size)

    def _crop_image(self, pil: Image.Image, idx: int) -> Image.Image:
        H, W = self._hw_for_index(idx)
        if H == self.image_size and W == self.image_size and not self.resolution_buckets:
            return self._crop_fn(pil, self.image_size)
        return _crop_to_hw(pil, H, W, self.crop_mode)

    def _resolve_aux_path(self, rel: Optional[Union[str, Path]]) -> Optional[Path]:
        if rel is None:
            return None
        s = str(rel).strip()
        if not s:
            return None
        p = Path(s)
        if p.is_absolute():
            return p
        base = self.data_path.parent if self.data_path.suffix.lower() == ".jsonl" else self.data_path
        return (base / p).resolve()

    def _crop_mask_pil(self, pil_l: Image.Image, idx: int) -> Image.Image:
        return self._crop_image(pil_l.convert("RGB"), idx).split()[0]

    def _maybe_foveate(
        self,
        pil: Image.Image,
        mask_l: Optional[Image.Image],
        idx: int,
        *,
        skip_foveate: bool = False,
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        H, W = self._hw_for_index(idx)
        if (
            skip_foveate
            or self.foveated_train_prob <= 0
            or random.random() >= self.foveated_train_prob
        ):
            return self._crop_image(pil, idx), (self._crop_mask_pil(mask_l, idx) if mask_l is not None else None)
        arr = np.array(pil.convert("RGB"))
        ih, iw = arr.shape[:2]
        y0, x0, y1, x1 = foveated_random_crop_box(ih, iw, crop_frac=self.foveated_crop_frac)
        crop = arr[y0:y1, x0:x1]
        pil_out = Image.fromarray(crop).resize((W, H), Image.BICUBIC)
        mask_out: Optional[Image.Image] = None
        if mask_l is not None:
            ma = np.array(mask_l.convert("L")).astype(np.float32) / 255.0
            mc = ma[y0:y1, x0:x1]
            mask_out = Image.fromarray((np.clip(mc, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
            mask_out = mask_out.resize((W, H), Image.NEAREST)
        return pil_out, mask_out

    def _mask_image_to_tensor(self, mask_l: Image.Image) -> torch.Tensor:
        m = np.array(mask_l).astype(np.float32) / 255.0
        if self.grounding_mask_soft:
            m = np.clip(m, 0.0, 1.0)
        else:
            m = (m > 0.5).astype(np.float32)
        return torch.from_numpy(m).unsqueeze(0)

    def _load_grounding_mask_tensor(
        self, idx: int, image_path: str, mask_rel: str, *, skip_foveate: bool = False
    ) -> Optional[torch.Tensor]:
        rp = self._resolve_aux_path(mask_rel)
        if rp is None or not rp.exists():
            return None
        try:
            pil = Image.open(image_path).convert("RGB")
            mask_l = Image.open(rp).convert("L")
            pil, mask_l = self._maybe_foveate(pil, mask_l, idx, skip_foveate=skip_foveate)
            del pil  # spatial alignment only
            return self._mask_image_to_tensor(mask_l)
        except Exception:
            return None

    def _scan(self):
        if self.data_path.suffix.lower() == ".jsonl":
            import json

            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    path = d.get("image_path") or d.get("path") or d.get("image")
                    cap = d.get("caption") or d.get("text") or d.get("caption")
                    parts = d.get("parts")
                    rc = d.get("region_captions") or d.get("segments")
                    regions: Any = None
                    if isinstance(parts, dict) and rc is not None:
                        regions = {"parts": parts, "region_captions": rc}
                    elif isinstance(parts, dict):
                        regions = parts
                    else:
                        regions = rc
                    if path and cap:
                        neg = d.get("negative_caption") or d.get("negative_prompt") or ""
                        style = d.get("style") or ""
                        ctrl = d.get("control_image") or d.get("control_path") or ""
                        ctrl_type = d.get("control_type") or d.get("control_kind") or d.get("controlnet_type") or ""
                        w = float(d.get("weight", d.get("aesthetic_score", 1.0)))
                        init_img = d.get("init_image") or d.get("init_image_path") or d.get("source_image") or ""
                        difficulty = float(d.get("difficulty", 0.5))
                        gmask = d.get("grounding_mask") or d.get("mask_path") or ""
                        cg = d.get("caption_global")
                        cl = d.get("caption_local")
                        ec = d.get("entity_captions")
                        self.samples.append(
                            {
                                "path": path,
                                "caption": cap,
                                "negative_caption": neg,
                                "style": style,
                                "control_image": ctrl,
                                "control_type": ctrl_type,
                                "init_image": init_img,
                                "weight": w,
                                "difficulty": difficulty,
                                "regions": regions,
                                "grounding_mask": gmask if isinstance(gmask, str) else "",
                                "caption_global": cg if isinstance(cg, str) else None,
                                "caption_local": cl if isinstance(cl, str) else None,
                                "entity_captions": ec if isinstance(ec, dict) else None,
                            }
                        )
            return
        for subdir in self.data_path.iterdir():
            if not subdir.is_dir():
                continue
            for img_path in subdir.glob("*"):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                cap_path = img_path.with_suffix(self.caption_ext)
                if not cap_path.exists():
                    cap_path = img_path.with_name(img_path.stem + ".caption")
                if not cap_path.exists():
                    continue
                lines = cap_path.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
                caption = lines[0].strip() if lines else ""
                negative_caption = lines[1].strip() if len(lines) > 1 else ""
                if not caption:
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

    def _process_caption(self, caption: str, negative_caption: str = "") -> Tuple[str, str]:
        if self.caption_unicode_normalize and _normalize_caption_unicode is not None:
            caption = _normalize_caption_unicode(caption)
            if (negative_caption or "").strip():
                negative_caption = _normalize_caption_unicode(negative_caption)
        if self.use_pixai_emphasis:
            caption = apply_pixai_emphasis(caption)
        if self.use_tag_order:
            caption = normalize_tag_order(caption)
        caption = boost_hard_style_tags(caption, repeat_factor=3)
        if self.use_quality_boost:
            caption = boost_quality_tags(caption, repeat_factor=3)
        caption = boost_domain_tags(caption, repeat_factor=2)
        if self.use_adherence_boost:
            caption = prepend_adherence_boost(caption, repeat_factor=2)
        if self.use_anti_blending:
            caption, negative_caption = add_anti_blending_and_count(caption, negative_caption)
        if self.train_shortcomings_mitigation in ("auto", "all"):
            caption, negative_caption = apply_shortcomings_to_caption_pair(
                caption,
                negative_caption,
                mode=self.train_shortcomings_mitigation,
                include_2d=self.train_shortcomings_2d,
            )
        if self.train_art_guidance_mode in ("auto", "all") or self.train_anatomy_guidance in ("lite", "strong"):
            caption, negative_caption = apply_art_guidance_to_caption_pair(
                caption,
                negative_caption,
                mode=self.train_art_guidance_mode,
                include_photography=self.train_art_guidance_photography,
                anatomy_mode=self.train_anatomy_guidance,
            )
        if self.train_style_guidance_mode in ("auto", "all"):
            caption, negative_caption = apply_style_guidance_to_caption_pair(
                caption,
                negative_caption,
                mode=self.train_style_guidance_mode,
                include_artist_refs=self.train_style_guidance_artists,
            )
        if self.max_caption_length:
            parts = caption.split(",")
            if self.shuffle_caption_parts and len(parts) > 1:
                random.shuffle(parts)
            caption = ",".join(parts)[: self.max_caption_length]
        return caption.strip(), (negative_caption or "").strip()

    def _latent_path(self, path: str) -> Optional[Path]:
        """Cache key: same name as image but .pt in latent_cache_dir."""
        if not self.latent_cache_dir:
            return None
        name = Path(path).stem + ".pt"
        return self.latent_cache_dir / name

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        path = s["path"]
        raw_caption = s["caption"]
        if self._partaware_caption_cfg is not None:
            raw_caption = apply_part_aware_caption_pipeline(raw_caption, s, self._partaware_caption_cfg, rng=random)
        regions = s.get("regions")
        raw_caption = merge_region_captions_into_caption(
            raw_caption,
            regions,
            mode=self.region_caption_mode,
            layout_tag=self.region_layout_tag,
        )
        caption, negative_caption = self._process_caption(raw_caption, s.get("negative_caption", ""))
        out = {
            "caption": caption,
            "negative_caption": negative_caption,
            "path": path,
            "weight": s.get("weight", 1.0),
            "difficulty": s.get("difficulty", 0.5),
        }
        style_text = s.get("style", "")
        # Auto-fill style from caption when missing (artist/style tags from PixAI, Danbooru, etc.)
        if not style_text and self.extract_style_from_caption:
            try:
                from config.style_artists import extract_style_from_text

                style_text = extract_style_from_text(s.get("caption", "")) or ""
            except Exception:
                pass
        if style_text:
            out["style"] = style_text
        ctrl_path = s.get("control_image", "")
        ctrl_type = s.get("control_type", "")
        if ctrl_path:
            out["control_image_path"] = ctrl_path
        init_path = s.get("init_image", "")
        if init_path:
            out["init_image_path"] = init_path
        mask_rel = s.get("grounding_mask") or ""
        skip_fov = bool(ctrl_path) or bool(init_path)
        # Latent cache: load precomputed latent to skip VAE encode
        lp = self._latent_path(path)
        h_i, w_i = self._hw_for_index(idx)
        if lp is not None and lp.exists() and not self.resolution_buckets:
            try:
                latent = torch.load(lp, map_location="cpu", weights_only=True)
                out["latent_values"] = latent
                out["pixel_values"] = torch.zeros(3, h_i, w_i)
                if init_path:
                    try:
                        init_pil = Image.open(init_path).convert("RGB")
                        init_pil = self._crop_image(init_pil, idx)
                        init_arr = np.array(init_pil).astype(np.float32) / 255.0
                        init_arr = (init_arr - 0.5) / 0.5
                        out["init_pixel_values"] = torch.from_numpy(init_arr).permute(2, 0, 1)
                    except Exception:
                        pass
                if ctrl_path:
                    try:
                        ctrl_pil = Image.open(ctrl_path).convert("RGB")
                        ctrl_pil = self._crop_image(ctrl_pil, idx)
                        ctrl_arr = np.array(ctrl_pil).astype(np.float32) / 255.0
                        ctrl_arr = (ctrl_arr - 0.5) / 0.5
                        out["control_image"] = torch.from_numpy(ctrl_arr).permute(2, 0, 1)
                        out["control_type_id"] = int(control_type_to_id(ctrl_type))
                    except Exception:
                        pass
                if mask_rel:
                    gm = self._load_grounding_mask_tensor(idx, path, mask_rel, skip_foveate=skip_fov)
                    if gm is not None:
                        out["grounding_mask"] = gm
                return out
            except Exception:
                pass
        pil = Image.open(path).convert("RGB")
        mask_l: Optional[Image.Image] = None
        if mask_rel:
            rp = self._resolve_aux_path(mask_rel)
            if rp is not None and rp.exists():
                try:
                    mask_l = Image.open(rp).convert("L")
                except Exception:
                    mask_l = None
        pil, mask_l = self._maybe_foveate(pil, mask_l, idx, skip_foveate=skip_fov)
        img = np.array(pil).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        out["pixel_values"] = img
        if mask_l is not None:
            out["grounding_mask"] = self._mask_image_to_tensor(mask_l)
        if init_path:
            try:
                init_pil = Image.open(init_path).convert("RGB")
                init_pil = self._crop_image(init_pil, idx)
                init_arr = np.array(init_pil).astype(np.float32) / 255.0
                init_arr = (init_arr - 0.5) / 0.5
                out["init_pixel_values"] = torch.from_numpy(init_arr).permute(2, 0, 1)
            except Exception:
                pass
        if ctrl_path:
            try:
                ctrl_pil = Image.open(ctrl_path).convert("RGB")
                ctrl_pil = self._crop_image(ctrl_pil, idx)
                ctrl_arr = np.array(ctrl_pil).astype(np.float32) / 255.0
                ctrl_arr = (ctrl_arr - 0.5) / 0.5
                out["control_image"] = torch.from_numpy(ctrl_arr).permute(2, 0, 1)
                out["control_type_id"] = int(control_type_to_id(ctrl_type))
            except Exception:
                pass
        return out


def collate_t2i(batch: List[Dict]) -> Dict[str, Any]:
    """Stack pixel_values and/or latent_values; leave captions as lists; include style, control_image when present."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    negative_captions = [b.get("negative_caption", "") for b in batch]
    styles = [b.get("style", "") for b in batch]
    weights = torch.tensor([b.get("weight", 1.0) for b in batch], dtype=torch.float32)
    out = {
        "pixel_values": pixel_values,
        "captions": captions,
        "negative_captions": negative_captions,
        "styles": styles,
        "sample_weights": weights,
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
        masks: List[torch.Tensor] = []
        valid: List[bool] = []
        for b in batch:
            if "grounding_mask" in b:
                masks.append(b["grounding_mask"])
                valid.append(True)
            else:
                masks.append(torch.zeros(1, h, w, dtype=torch.float32))
                valid.append(False)
        out["grounding_mask"] = torch.stack(masks)
        out["grounding_mask_valid"] = torch.tensor(valid, dtype=torch.bool)
    return out
