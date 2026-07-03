"""WD EVA02 Large Tagger — supplementary booru-style tags for scraped images.

WD/JoyTag are **not** trusted for character identity or artist attribution (see
``docs/guides/IMAGE_CAPTIONING.md``). This module adds general/scene descriptors
(hair, pose, clothing, background) on top of API ``character_tags`` /
``artist_tags`` from danbooru/rule34.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None  # type: ignore[assignment,misc]

__all__ = [
    "WDTagger",
    "default_tagger_dir",
    "merge_wd_tags_into_caption",
    "merge_wd_tags_into_row",
]

_IDENTITY_CATEGORIES = frozenset({"artist", "copyright", "character", "meta", "rating"})
_DEFAULT_THRESHOLD = 0.35
_DEFAULT_MAX_TAGS = 48
_MODEL_INPUT_SIZE = 448


def default_tagger_dir() -> Path:
    root = os.environ.get("SDX_PRETRAINED", "").strip()
    if root:
        return Path(root) / "WD-EVA02-Large-Tagger"
    return Path(__file__).resolve().parents[2] / "pretrained" / "WD-EVA02-Large-Tagger"


@dataclass(frozen=True)
class WDTag:
    name: str
    category: str
    score: float


def _slug_tag(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def merge_wd_tags_into_caption(
    caption: str,
    wd_tags: Sequence[str],
    *,
    identity_tags: Optional[Iterable[str]] = None,
    max_add: int = 32,
) -> str:
    """Append WD tags not already present; never override booru identity tags."""
    cap = (caption or "").strip()
    seen: Set[str] = {t.strip().lower().replace("_", " ") for t in cap.split(",") if t.strip()}
    for t in identity_tags or ():
        t = str(t).strip().replace("_", " ")
        if t:
            seen.add(t.lower())
    added: List[str] = []
    for raw in wd_tags:
        tag = str(raw).strip().replace("_", " ")
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        added.append(tag)
        if len(added) >= max_add:
            break
    if not added:
        return cap
    return (cap + ", " + ", ".join(added)).strip(", ")


def merge_wd_tags_into_row(row: dict, wd_tag_names: Sequence[str]) -> dict:
    """Patch a manifest row with merged caption + ``wd_tags`` metadata."""
    merged = dict(row)
    identity: List[str] = []
    for key in ("artist_tags", "character_tags", "copyright_tags"):
        for t in row.get(key) or []:
            identity.append(str(t))
    cap = str(row.get("caption") or row.get("tags") or "").strip()
    merged["caption"] = merge_wd_tags_into_caption(cap, wd_tag_names, identity_tags=identity)
    merged["wd_tags"] = list(wd_tag_names)
    sources = [str(s) for s in (row.get("tag_sources") or [])]
    if "wd_tagger" not in sources:
        sources.append("wd_tagger")
    merged["tag_sources"] = sources
    if row.get("caption") and not merged.get("booru_caption"):
        merged["booru_caption"] = row["caption"]
    return merged


class WDTagger:
    """ONNX WD tagger (SmilingWolf/wd-eva02-large-tagger-v3)."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        max_tags: int = _DEFAULT_MAX_TAGS,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_tagger_dir()
        self.threshold = float(threshold)
        self.max_tags = int(max_tags)
        self._session = None
        self._tag_names: List[str] = []
        self._tag_categories: List[str] = []

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if np is None or Image is None:
            raise ModuleNotFoundError("numpy and Pillow are required for WD tagger inference.")
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ModuleNotFoundError(
                "onnxruntime is required for WD tagger. Install: pip install onnxruntime-gpu"
            ) from e

        onnx_path = self.model_dir / "model.onnx"
        csv_path = self.model_dir / "selected_tags.csv"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"WD tagger ONNX not found: {onnx_path}\n"
                "Run: python setup/download_pretrained.py --only WD-EVA02-Large-Tagger"
            )
        if not csv_path.is_file():
            raise FileNotFoundError(f"WD tagger tag list not found: {csv_path}")

        with csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("name") or row.get("tag") or "").strip()
                cat = (row.get("category") or "general").strip().lower()
                if name:
                    self._tag_names.append(name)
                    self._tag_categories.append(cat)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    @staticmethod
    def _preprocess(image: "Image.Image", size: int) -> "np.ndarray":
        image = image.convert("RGB")
        w, h = image.size
        if w == 0 or h == 0:
            raise ValueError("empty image")
        scale = size / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(image, ((size - nw) // 2, (size - nh) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, axis=0)

    def predict(self, image_path: str | os.PathLike[str]) -> List[WDTag]:
        self._ensure_loaded()
        assert np is not None and Image is not None
        img = Image.open(image_path)
        batch = self._preprocess(img, _MODEL_INPUT_SIZE)
        out = self._session.run(None, {self._input_name: batch})[0]
        probs = 1.0 / (1.0 + np.exp(-out[0]))
        tags: List[WDTag] = []
        for i, score in enumerate(probs):
            if score < self.threshold:
                continue
            name = self._tag_names[i] if i < len(self._tag_names) else str(i)
            cat = self._tag_categories[i] if i < len(self._tag_categories) else "general"
            if cat in _IDENTITY_CATEGORIES:
                continue
            tags.append(WDTag(name=name, category=cat, score=float(score)))
        tags.sort(key=lambda t: t.score, reverse=True)
        return tags[: self.max_tags]

    def predict_names(self, image_path: str | os.PathLike[str]) -> List[str]:
        return [t.name for t in self.predict(image_path)]
