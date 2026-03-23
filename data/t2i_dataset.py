# Text-to-image dataset: PixAI-style tag prompts + ReVe-style long/complex captions.
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .caption_utils import (
    add_anti_blending_and_count,
    apply_pixai_emphasis,
    boost_domain_tags,
    boost_hard_style_tags,
    boost_quality_tags,
    merge_region_captions_into_caption,
    normalize_tag_order,
)


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
        use_anti_blending: bool = True,
        latent_cache_dir: Optional[str] = None,
        crop_mode: str = "center",  # "center" | "random" | "largest_center" (IMPROVEMENTS 1.2)
        extract_style_from_caption: bool = True,  # Auto-fill style from artist/style tags (PixAI, Danbooru)
        region_caption_mode: str = "append",  # "append" | "prefix" | "off" — merge JSONL regions/parts into T5 caption
        region_layout_tag: str = "[layout]",
        resolution_buckets: Optional[List[Tuple[int, int]]] = None,
        bucket_seed: int = 42,
        bucket_fixed_assign: bool = False,
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
        self.use_anti_blending = use_anti_blending
        self._crop_fn = {"center": _center_crop, "random": _random_crop, "largest_center": _largest_center_crop}.get(
            crop_mode, _center_crop
        )
        self.region_caption_mode = region_caption_mode
        self.region_layout_tag = region_layout_tag
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
                        w = float(d.get("weight", d.get("aesthetic_score", 1.0)))
                        init_img = d.get("init_image") or d.get("init_image_path") or d.get("source_image") or ""
                        difficulty = float(d.get("difficulty", 0.5))
                        self.samples.append(
                            {
                                "path": path,
                                "caption": cap,
                                "negative_caption": neg,
                                "style": style,
                                "control_image": ctrl,
                                "init_image": init_img,
                                "weight": w,
                                "difficulty": difficulty,
                                "regions": regions,
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
                        "init_image": "",
                        "weight": 1.0,
                        "difficulty": 0.5,
                        "regions": None,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _process_caption(self, caption: str, negative_caption: str = "") -> Tuple[str, str]:
        if self.use_pixai_emphasis:
            caption = apply_pixai_emphasis(caption)
        if self.use_tag_order:
            caption = normalize_tag_order(caption)
        caption = boost_hard_style_tags(caption, repeat_factor=3)
        if self.use_quality_boost:
            caption = boost_quality_tags(caption, repeat_factor=3)
        caption = boost_domain_tags(caption, repeat_factor=2)
        if self.use_anti_blending:
            caption, negative_caption = add_anti_blending_and_count(caption, negative_caption)
        if self.max_caption_length:
            parts = caption.split(",")
            if self.shuffle_caption_parts and len(parts) > 1:
                random.shuffle(parts)
            caption = ",".join(parts)[: self.max_caption_length]
        return caption.strip(), (negative_caption or "").strip()

    def _latent_path(self, path: str) -> Path:
        """Cache key: same name as image but .pt in latent_cache_dir."""
        if not self.latent_cache_dir:
            return None
        name = Path(path).stem + ".pt"
        return self.latent_cache_dir / name

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        path = s["path"]
        raw_caption = s["caption"]
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
        if ctrl_path:
            out["control_image_path"] = ctrl_path
        init_path = s.get("init_image", "")
        if init_path:
            out["init_image_path"] = init_path
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
                    except Exception:
                        pass
                return out
            except Exception:
                pass
        pil = Image.open(path).convert("RGB")
        pil = self._crop_image(pil, idx)
        img = np.array(pil).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        out["pixel_values"] = img
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
    if all("init_pixel_values" in b for b in batch):
        out["init_pixel_values"] = torch.stack([b["init_pixel_values"] for b in batch])
    return out
