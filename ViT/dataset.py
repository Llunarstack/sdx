from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.architecture.ar_dit_vit import ar_conditioning_vector, parse_num_ar_blocks_from_row

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", flags=re.ASCII)


def text_feature_vector(text: str) -> torch.Tensor:
    """
    Cheap text feature vector for adherence head conditioning.
    This is intentionally lightweight and deterministic.
    """
    text = (text or "").strip()
    toks = _TOKEN_RE.findall(text)
    n_chars = float(len(text))
    n_toks = float(len(toks))
    n_commas = float(text.count(","))
    n_emph_up = float(text.count("(") + text.count(")"))
    n_emph_down = float(text.count("[") + text.count("]"))
    n_digits = float(sum(ch.isdigit() for ch in text))
    n_upper = float(sum(ch.isupper() for ch in text))
    avg_tok_len = float(sum(len(t) for t in toks) / max(1, len(toks)))
    v = torch.tensor(
        [
            n_chars / 300.0,
            n_toks / 100.0,
            n_commas / 50.0,
            n_emph_up / 20.0,
            n_emph_down / 20.0,
            n_digits / 50.0,
            n_upper / 50.0,
            avg_tok_len / 20.0,
        ],
        dtype=torch.float32,
    )
    return v


class ViTManifestDataset(Dataset):
    """
    Manifest dataset for ViT quality + adherence heads.

    Expected JSONL fields:
      - image_path (or path/image)
      - caption (or text)
      - quality_label (optional; 0/1)
      - adherence_score (optional; float in [0, 1])
      - num_ar_blocks / dit_num_ar_blocks / ar_blocks (optional; 0, 2, or 4 for DiT AR — see docs/AR.md)
    """

    def __init__(
        self,
        manifest_jsonl: str,
        image_root: str = "",
        image_size: int = 224,
        *,
        training_augment: bool = False,
    ):
        self.manifest_path = Path(manifest_jsonl)
        self.image_root = Path(image_root) if image_root else None
        self.samples: List[Dict[str, object]] = []
        if training_augment:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        self._load()

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if self.image_root is not None:
            path = self.image_root / path
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        return path

    def _load(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                try:
                    row = json.loads(t)
                except Exception:
                    continue
                p = row.get("image_path") or row.get("path") or row.get("image")
                cap = row.get("caption") or row.get("text") or ""
                if not isinstance(p, str) or not p.strip():
                    continue
                p_res = self._resolve_path(p.strip())
                if not p_res.exists():
                    continue
                q = row.get("quality_label", row.get("quality"))
                a = row.get("adherence_score", row.get("prompt_adherence"))
                ar_b = parse_num_ar_blocks_from_row(row)
                sample = {
                    "image_path": str(p_res),
                    "caption": str(cap),
                    "quality_label": None if q is None else float(q),
                    "adherence_score": None if a is None else float(a),
                    "num_ar_blocks": int(ar_b),
                }
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        x = self.transform(img)
        cap = str(s["caption"])
        tf = text_feature_vector(cap)
        quality_label = s["quality_label"]
        adherence_score = s["adherence_score"]
        ar_vec = ar_conditioning_vector(int(s["num_ar_blocks"]), dtype=torch.float32)
        return {
            "image": x,
            "caption": cap,
            "text_features": tf,
            "ar_conditioning": ar_vec,
            "quality_label": None if quality_label is None else torch.tensor(float(quality_label), dtype=torch.float32),
            "adherence_score": None
            if adherence_score is None
            else torch.tensor(float(adherence_score), dtype=torch.float32),
            "image_path": s["image_path"],
        }


def collate_vit_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    text_features = torch.stack([b["text_features"] for b in batch], dim=0)
    ar_conditioning = torch.stack([b["ar_conditioning"] for b in batch], dim=0)
    out: Dict[str, object] = {
        "images": images,
        "text_features": text_features,
        "ar_conditioning": ar_conditioning,
        "captions": [b["caption"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }
    q_vals: List[Optional[torch.Tensor]] = [b["quality_label"] for b in batch]
    a_vals: List[Optional[torch.Tensor]] = [b["adherence_score"] for b in batch]
    if all(v is not None for v in q_vals):
        out["quality_labels"] = torch.stack([v for v in q_vals if v is not None], dim=0)
    else:
        out["quality_labels"] = None
    if all(v is not None for v in a_vals):
        out["adherence_scores"] = torch.stack([v for v in a_vals if v is not None], dim=0)
    else:
        out["adherence_scores"] = None
    return out
