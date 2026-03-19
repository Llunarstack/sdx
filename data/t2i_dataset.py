# Text-to-image dataset: PixAI-style tag prompts + ReVe-style long/complex captions.
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .caption_utils import (
    apply_pixai_emphasis,
    normalize_tag_order,
    boost_hard_style_tags,
    boost_quality_tags,
    boost_domain_tags,
    add_anti_blending_and_count,
)


def _center_crop(pil_image, image_size: int):
    """Center crop to image_size (ADM-style)."""
    w, h = pil_image.size
    while min(w, h) >= 2 * image_size:
        pil_image = pil_image.resize((w // 2, h // 2), resample=Image.BOX)
        w, h = pil_image.size
    scale = image_size / min(w, h)
    pil_image = pil_image.resize(
        (round(w * scale), round(h * scale)), resample=Image.BICUBIC
    )
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
    pil_image = pil_image.resize(
        (round(w * scale), round(h * scale)), resample=Image.BICUBIC
    )
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
    ):
        self.data_path = Path(data_path)
        self.extract_style_from_caption = extract_style_from_caption
        self.image_size = image_size
        self.latent_cache_dir = Path(latent_cache_dir) if latent_cache_dir else None
        self.crop_mode = crop_mode
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
        self._crop_fn = {"center": _center_crop, "random": _random_crop, "largest_center": _largest_center_crop}.get(crop_mode, _center_crop)
        self.samples: List[Dict[str, Any]] = []
        self._scan()

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
                    if path and cap:
                        neg = d.get("negative_caption") or d.get("negative_prompt") or ""
                        style = d.get("style") or ""
                        ctrl = d.get("control_image") or d.get("control_path") or ""
                        w = float(d.get("weight", d.get("aesthetic_score", 1.0)))
                        init_img = d.get("init_image") or d.get("init_image_path") or d.get("source_image") or ""
                        difficulty = float(d.get("difficulty", 0.5))
                        self.samples.append({
                            "path": path, "caption": cap, "negative_caption": neg,
                            "style": style, "control_image": ctrl, "init_image": init_img, "weight": w,
                            "difficulty": difficulty,
                        })
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
                self.samples.append({
                    "path": str(img_path),
                    "caption": caption,
                    "negative_caption": negative_caption,
                    "style": "",
                    "control_image": "",
                    "init_image": "",
                    "weight": 1.0,
                    "difficulty": 0.5,
                })

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
        caption, negative_caption = self._process_caption(s["caption"], s.get("negative_caption", ""))
        out = {"caption": caption, "negative_caption": negative_caption, "path": path, "weight": s.get("weight", 1.0), "difficulty": s.get("difficulty", 0.5)}
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
        if lp is not None and lp.exists():
            try:
                latent = torch.load(lp, map_location="cpu", weights_only=True)
                out["latent_values"] = latent
                out["pixel_values"] = torch.zeros(3, self.image_size, self.image_size)
                if init_path:
                    try:
                        init_pil = Image.open(init_path).convert("RGB")
                        init_pil = self._crop_fn(init_pil, self.image_size)
                        init_arr = np.array(init_pil).astype(np.float32) / 255.0
                        init_arr = (init_arr - 0.5) / 0.5
                        out["init_pixel_values"] = torch.from_numpy(init_arr).permute(2, 0, 1)
                    except Exception:
                        pass
                if ctrl_path:
                    try:
                        ctrl_pil = Image.open(ctrl_path).convert("RGB")
                        ctrl_pil = self._crop_fn(ctrl_pil, self.image_size)
                        ctrl_arr = np.array(ctrl_pil).astype(np.float32) / 255.0
                        ctrl_arr = (ctrl_arr - 0.5) / 0.5
                        out["control_image"] = torch.from_numpy(ctrl_arr).permute(2, 0, 1)
                    except Exception:
                        pass
                return out
            except Exception:
                pass
        pil = Image.open(path).convert("RGB")
        pil = self._crop_fn(pil, self.image_size)
        img = np.array(pil).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        out["pixel_values"] = img
        if init_path:
            try:
                init_pil = Image.open(init_path).convert("RGB")
                init_pil = self._crop_fn(init_pil, self.image_size)
                init_arr = np.array(init_pil).astype(np.float32) / 255.0
                init_arr = (init_arr - 0.5) / 0.5
                out["init_pixel_values"] = torch.from_numpy(init_arr).permute(2, 0, 1)
            except Exception:
                pass
        if ctrl_path:
            try:
                ctrl_pil = Image.open(ctrl_path).convert("RGB")
                ctrl_pil = self._crop_fn(ctrl_pil, self.image_size)
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
