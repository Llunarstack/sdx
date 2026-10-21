"""WD EVA02 Large Tagger — supplementary booru-style tags for scraped images.

WD/JoyTag are **not** trusted for character identity or artist attribution (see
``docs/guides/IMAGE_CAPTIONING.md``). This module adds general/scene descriptors
(hair, pose, clothing, background) on top of API ``character_tags`` /
``artist_tags`` from danbooru/rule34.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
# selected_tags.csv stores Danbooru category ids; 9 is the tagger's rating head.
_CATEGORY_NAMES = {"0": "general", "1": "artist", "3": "copyright", "4": "character", "5": "meta", "9": "rating"}
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


def merge_wd_tags_into_caption(
    caption: str,
    wd_tags: Sequence[str],
    *,
    identity_tags: Iterable[str] | None = None,
    max_add: int = 32,
) -> str:
    """Append WD tags not already present; never override booru identity tags."""
    cap = (caption or "").strip()
    seen: set[str] = {t.strip().lower().replace("_", " ") for t in cap.split(",") if t.strip()}
    for t in identity_tags or ():
        t = str(t).strip().replace("_", " ")
        if t:
            seen.add(t.lower())
    added: list[str] = []
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
    identity: list[str] = []
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
        self._tag_names: list[str] = []
        self._tag_categories: list[str] = []

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
                cat = _CATEGORY_NAMES.get(cat, cat)
                if name:
                    self._tag_names.append(name)
                    self._tag_categories.append(cat)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_size = inp.shape[1] if isinstance(inp.shape[1], int) else _MODEL_INPUT_SIZE

    @property
    def tag_names(self) -> list[str]:
        self._ensure_loaded()
        return list(self._tag_names)

    @staticmethod
    def _preprocess(image: Image.Image, size: int) -> np.ndarray:
        """SmilingWolf's tagger input: alpha over white, white-padded square, bicubic, BGR NHWC in [0, 255]."""
        image = image.convert("RGBA")
        w, h = image.size
        if w == 0 or h == 0:
            raise ValueError("empty image")
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        side = max(w, h)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(canvas.convert("RGB"), ((side - w) // 2, (side - h) // 2))
        if side != size:
            square = square.resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(square, dtype=np.float32)[:, :, ::-1]
        return np.ascontiguousarray(arr[None])

    def scores(self, image_path: str | os.PathLike[str]) -> list[WDTag]:
        """Every tag with its probability, highest first (no threshold or category filter)."""
        self._ensure_loaded()
        assert np is not None and Image is not None
        with Image.open(image_path) as img:
            batch = self._preprocess(img, self._input_size)
        probs = self._session.run(None, {self._input_name: batch})[0][0]  # the graph ends in a sigmoid
        tags = [
            WDTag(name=n, category=c, score=float(p)) for n, c, p in zip(self._tag_names, self._tag_categories, probs)
        ]
        tags.sort(key=lambda t: t.score, reverse=True)
        return tags

    def select(self, tags: Sequence[WDTag]) -> list[WDTag]:
        """Caption tags: above ``threshold``, no identity/rating tags, at most ``max_tags``."""
        kept = [t for t in tags if t.score >= self.threshold and t.category not in _IDENTITY_CATEGORIES]
        return kept[: self.max_tags]
