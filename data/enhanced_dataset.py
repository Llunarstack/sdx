"""Text2ImageDataset plus the EnhancedDiT annotations (layout boxes, anatomy masks, in-image text, slot ids).

Extra JSONL fields (all optional):
  "spatial_layout":   [[x1, y1, x2, y2], ...]  normalised bbox per object
  "anatomy_mask":     path to a grayscale mask (white = human region)
  "text_tokens":      ["word1", "word2", ...]  strings rendered in the image
  "text_positions":   [[cx, cy], ...]  normalised centre per text token
  "character_id":     int  character slot index (consistency training)
  "style_id":         int  style slot index
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from data.manifest_utils import iter_manifest_rows
from data.t2i_dataset import Text2ImageDataset, collate_t2i

_ENHANCED_KEYS = ("spatial_layout", "anatomy_mask", "text_tokens", "text_positions", "character_id", "style_id")


class EnhancedT2IDataset(Text2ImageDataset):
    """Adds the optional enhanced annotations to each item; samples without them simply omit those keys."""

    def __init__(self, *args, max_objects: int = 10, max_text_tokens: int = 8, **kwargs):
        self.max_objects = int(max_objects)
        self.max_text_tokens = int(max_text_tokens)
        super().__init__(*args, **kwargs)

    def _scan_rows(self):
        """Base scan (including its safety screening), then attach the enhanced fields of each kept manifest row."""
        super()._scan_rows()
        if self.data_path.suffix.lower() != ".jsonl":
            return
        extras: dict[str, dict[str, Any]] = {}
        for d in iter_manifest_rows(self.data_path):
            if isinstance(d, dict):
                path = d.get("image_path") or d.get("path") or d.get("image")
                if path:
                    extras[str(path)] = {k: d[k] for k in _ENHANCED_KEYS if k in d}
        for s in self.samples:
            s.update(extras.get(str(s["path"]), {}))

    def _load_spatial_layout(self, raw: Any) -> torch.Tensor | None:
        """``(max_objects, 4)`` zero-padded boxes, or ``None`` when absent or malformed."""
        if not raw:
            return None
        try:
            bboxes = [[float(v) for v in box[:4]] for box in raw[: self.max_objects]]
        except (TypeError, ValueError):
            return None
        if not bboxes:
            return None
        t = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        for i, box in enumerate(bboxes):
            t[i] = torch.tensor(box, dtype=torch.float32)
        return t

    def _load_anatomy_mask(self, mask_rel: str, idx: int) -> torch.Tensor | None:
        """Binary ``(1, H, W)`` mask cropped with the image's (centre) crop geometry."""
        mask_l = self._open_mask(mask_rel)
        if mask_l is None:
            return None
        m = np.array(self._crop_image(mask_l, idx), dtype=np.float32) / 255.0
        return torch.from_numpy((m > 0.5).astype(np.float32)).unsqueeze(0)

    def _load_text_tokens(self, tokens: Any, positions: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``(ids, positions)``: string length as a tokenizer-free id proxy (capped at 255) and ``(cx, cy)`` centres."""
        if not tokens:
            return None, None
        try:
            tok_list = [str(t) for t in tokens[: self.max_text_tokens]]
            pos_list = positions[: len(tok_list)] if positions else []
        except (TypeError, AttributeError):
            return None, None
        ids = torch.zeros(self.max_text_tokens, dtype=torch.int64)
        pos = torch.zeros(self.max_text_tokens, 2, dtype=torch.float32)
        for i, tok in enumerate(tok_list):
            ids[i] = min(len(tok), 255)
        for i, p in enumerate(pos_list):
            try:
                pos[i, 0] = float(p[0])
                pos[i, 1] = float(p[1])
            except (TypeError, ValueError, IndexError):
                pass
        return ids, pos

    def _getitem_impl(self, idx: int, hw_idx: int | None = None) -> dict[str, Any]:
        # Annotations come from the sample actually loaded (``idx``), cropped to the requested bucket (``hw_idx``).
        out = super()._getitem_impl(idx, hw_idx)
        hw_idx = idx if hw_idx is None else hw_idx
        s = self.samples[idx]
        layout = self._load_spatial_layout(s.get("spatial_layout"))
        if layout is not None:
            out["spatial_layout"] = layout
        anat_rel = str(s.get("anatomy_mask") or "").strip()
        if anat_rel:
            anat = self._load_anatomy_mask(anat_rel, hw_idx)
            if anat is not None:
                out["anatomy_mask"] = anat
        tok_ids, tok_pos = self._load_text_tokens(s.get("text_tokens"), s.get("text_positions"))
        if tok_ids is not None:
            out["text_tokens"] = tok_ids
            out["text_positions"] = tok_pos
        for key in ("character_id", "style_id"):
            try:
                out[key] = int(s[key])
            except (KeyError, TypeError, ValueError):
                pass
        return out


def collate_enhanced(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """:func:`collate_t2i` plus the enhanced tensors; a partly annotated batch is zero-padded with a ``*_valid`` mask."""
    out = collate_t2i(batch)
    for key in ("spatial_layout", "anatomy_mask", "text_tokens", "text_positions"):
        if all(key in b for b in batch):
            out[key] = torch.stack([b[key] for b in batch])
        elif any(key in b for b in batch):
            ref = next(b[key] for b in batch if key in b)
            out[key] = torch.stack([b.get(key, torch.zeros_like(ref)) for b in batch])
            out[f"{key}_valid"] = torch.tensor([key in b for b in batch], dtype=torch.bool)
    for key in ("character_id", "style_id"):
        if any(key in b for b in batch):
            out[key] = torch.tensor([b.get(key, -1) for b in batch], dtype=torch.long)
    return out
