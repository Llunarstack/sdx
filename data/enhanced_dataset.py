"""
Enhanced dataset for advanced DiT training.

Extends Text2ImageDataset with spatial layout annotations, anatomy masks,
text-in-image positions, and character consistency data.

JSONL fields (all optional, extend the base t2i_dataset.py schema):
  "spatial_layout":   [[x1,y1,x2,y2], ...]  normalised bbox per object
  "anatomy_mask":     path to grayscale mask (white = human region)
  "text_tokens":      ["word1", "word2", ...]  text strings present in image
  "text_positions":   [[cx, cy], ...]  normalised centre per text token
  "character_id":     int  character slot index (for consistency training)
  "style_id":         int  style slot index
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Re-use the base dataset's crop helpers
from data.t2i_dataset import (
    Text2ImageDataset,
    collate_t2i,
)


class EnhancedT2IDataset(Text2ImageDataset):
    """
    Enhanced Text-to-Image dataset that adds spatial, anatomy, text, and
    character-consistency annotations on top of the base dataset.

    All enhanced fields are optional — samples without them return None
    for those keys so the training loop can skip the corresponding losses.
    """

    def __init__(self, *args, max_objects: int = 10, max_text_tokens: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_objects = int(max_objects)
        self.max_text_tokens = int(max_text_tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_spatial_layout(self, raw: Any) -> Optional[torch.Tensor]:
        """
        Parse spatial layout from JSONL field.
        raw: list of [x1, y1, x2, y2] normalised bboxes, or None.
        Returns (max_objects, 4) float32 tensor, zero-padded.
        """
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

    def _load_anatomy_mask(self, mask_rel: str, image_path: str, idx: int) -> Optional[torch.Tensor]:
        """Load anatomy mask and resize to patch grid. Returns (1, H, W) float32."""
        rp = self._resolve_aux_path(mask_rel)
        if rp is None or not rp.exists():
            return None
        try:
            pil = Image.open(image_path).convert("RGB")
            mask_l = Image.open(rp).convert("L")
            # Resize mask to match image crop
            pil_crop = self._crop_image(pil, idx)
            h, w = pil_crop.size[1], pil_crop.size[0]
            mask_crop = mask_l.resize((w, h), Image.NEAREST)
            m = np.array(mask_crop, dtype=np.float32) / 255.0
            m = (m > 0.5).astype(np.float32)
            return torch.from_numpy(m).unsqueeze(0)  # (1, H, W)
        except Exception:
            return None

    def _load_text_tokens(
        self,
        tokens: Any,
        positions: Any,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Parse text tokens and positions.
        tokens: list of strings
        positions: list of [cx, cy] normalised centres
        Returns:
            token_ids: (max_text_tokens,) int64 — character-count proxy (length of each string)
            text_positions: (max_text_tokens, 2) float32
        """
        if not tokens:
            return None, None
        try:
            tok_list = [str(t) for t in tokens[: self.max_text_tokens]]
            pos_list = positions[: self.max_text_tokens] if positions else []
        except (TypeError, AttributeError):
            return None, None

        n = len(tok_list)
        # Use string length as a simple token-id proxy (no tokenizer dependency here)
        ids = torch.zeros(self.max_text_tokens, dtype=torch.int64)
        pos = torch.zeros(self.max_text_tokens, 2, dtype=torch.float32)
        for i, tok in enumerate(tok_list):
            ids[i] = min(len(tok), 255)
        for i, p in enumerate(pos_list[:n]):
            try:
                pos[i, 0] = float(p[0])
                pos[i, 1] = float(p[1])
            except (TypeError, IndexError):
                pass
        return ids, pos

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        s = self.samples[idx]

        # Spatial layout
        layout = self._load_spatial_layout(s.get("spatial_layout"))
        if layout is not None:
            out["spatial_layout"] = layout

        # Anatomy mask
        anat_rel = str(s.get("anatomy_mask") or "").strip()
        if anat_rel:
            anat = self._load_anatomy_mask(anat_rel, s["path"], idx)
            if anat is not None:
                out["anatomy_mask"] = anat

        # Text tokens + positions
        tok_ids, tok_pos = self._load_text_tokens(
            s.get("text_tokens"),
            s.get("text_positions"),
        )
        if tok_ids is not None:
            out["text_tokens"] = tok_ids
            out["text_positions"] = tok_pos

        # Character / style slot IDs
        char_id = s.get("character_id")
        if char_id is not None:
            try:
                out["character_id"] = int(char_id)
            except (TypeError, ValueError):
                pass

        style_id = s.get("style_id")
        if style_id is not None:
            try:
                out["style_id"] = int(style_id)
            except (TypeError, ValueError):
                pass

        return out


def collate_enhanced(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for EnhancedT2IDataset.
    Stacks enhanced tensors when all samples have them; otherwise omits the key.
    Falls back to the base collate for standard fields.
    """
    out = collate_t2i(batch)

    for key in ("spatial_layout", "anatomy_mask", "text_tokens", "text_positions"):
        if all(key in b for b in batch):
            out[key] = torch.stack([b[key] for b in batch])
        elif any(key in b for b in batch):
            # Partial batch: pad missing entries with zeros matching the first present shape
            ref = next(b[key] for b in batch if key in b)
            tensors = [b.get(key, torch.zeros_like(ref)) for b in batch]
            out[key] = torch.stack(tensors)
            out[f"{key}_valid"] = torch.tensor(
                [key in b for b in batch], dtype=torch.bool
            )

    for key in ("character_id", "style_id"):
        if any(key in b for b in batch):
            ids = [b.get(key, -1) for b in batch]
            out[key] = torch.tensor(ids, dtype=torch.long)

    return out


# Keep the old name as an alias for backward compatibility
EnhancedDataset = EnhancedT2IDataset
