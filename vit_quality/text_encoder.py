"""
Text featurizers for the ViT quality/adherence model.

The original adherence head was conditioned on an 8-D hand-crafted vector of
character/comma/bracket counts (``dataset.text_feature_vector``). That vector
carries *shape* of the prompt but zero semantics, so the "prompt adherence"
head could not actually see the prompt. This module adds a real semantic
featurizer backed by a frozen CLIP text encoder, selectable per-checkpoint.

Design goals:
  * **Back-compatible.** ``mode="handcrafted"`` reproduces the exact 8-D vector,
    so existing checkpoints (which stored ``text_feat_dim=8``) load and score
    identically. New training opts in with ``mode="clip"``.
  * **Single source of truth.** Every consumer (train / infer / test_time_pick /
    export_embeddings) builds its featurizer from the checkpoint config via
    :func:`featurizer_from_config`, so the training-time and inference-time text
    pathways can never silently diverge.
  * **Graceful.** If ``transformers`` or the CLIP weights are unavailable, the
    CLIP featurizer raises a clear error at construction (fail loud during
    training) rather than silently degrading.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import torch
from utils.generation.clip_alignment import _clip_feature_tensor

from vit_quality.dataset import text_feature_vector

# Default CLIP text encoder. clip-vit-base-patch32 -> projection_dim 512;
# clip-vit-large-patch14 -> 768. Both are commonly cached in this repo.
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@runtime_checkable
class TextFeaturizer(Protocol):
    """A callable ``caption -> (dim,) float32 tensor`` with a known ``dim``."""

    dim: int

    def __call__(self, text: str) -> torch.Tensor: ...

    def embed_many(self, texts: list[str]) -> torch.Tensor: ...


class HandcraftedTextFeaturizer:
    """Wraps the legacy 8-D deterministic vector (no model, no semantics)."""

    mode = "handcrafted"

    def __init__(self) -> None:
        self.dim = int(text_feature_vector("").numel())

    def __call__(self, text: str) -> torch.Tensor:
        return text_feature_vector(text)

    def embed_many(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([text_feature_vector(t) for t in texts], dim=0)


# CLIP model cache shared across featurizer instances (keyed by model id + device).
_CLIP_CACHE: dict[tuple[str, str], tuple[object, object, int]] = {}


def _load_clip(model_id: str, device: torch.device) -> tuple[object, object, int]:
    key = (model_id, str(device))
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from transformers import CLIPModel, CLIPTokenizerFast
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            "CLIP text featurizer requires `transformers`. Install it or use text_embed_mode='handcrafted'."
        ) from e
    try:
        tok = CLIPTokenizerFast.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CLIP text encoder '{model_id}'. Ensure the weights are "
            "downloaded (see scripts/download/), or use text_embed_mode='handcrafted'."
        ) from e
    for p in model.parameters():
        p.requires_grad_(False)
    proj_dim = int(getattr(model.config, "projection_dim", 512))
    bundle = (model, tok, proj_dim)
    _CLIP_CACHE[key] = bundle
    return bundle


class ClipTextFeaturizer:
    """
    Frozen CLIP text encoder -> L2-normalized pooled text embedding.

    Embeddings are memoized per caption (captions repeat across a manifest), so
    a training epoch pays the CLIP forward cost once per unique prompt.
    """

    mode = "clip"

    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL_ID,
        device: torch.device | str = "cpu",
        max_length: int = 77,
    ) -> None:
        self.model_id = str(model_id)
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self._model, self._tok, self.dim = _load_clip(self.model_id, self.device)
        self._cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    @torch.inference_mode()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self._tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _clip_feature_tensor(self._model.get_text_features(**inputs))
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.float().cpu()

    def __call__(self, text: str) -> torch.Tensor:
        k = self._key(text)
        v = self._cache.get(k)
        if v is None:
            v = self._encode([text or ""])[0]
            self._cache[k] = v
        return v

    def embed_many(self, texts: list[str]) -> torch.Tensor:
        out: list[torch.Tensor | None] = [None] * len(texts)
        todo_idx: list[int] = []
        todo_txt: list[str] = []
        for i, t in enumerate(texts):
            k = self._key(t)
            cached = self._cache.get(k)
            if cached is None:
                todo_idx.append(i)
                todo_txt.append(t or "")
            else:
                out[i] = cached
        if todo_txt:
            enc = self._encode(todo_txt)
            for j, i in enumerate(todo_idx):
                v = enc[j]
                self._cache[self._key(texts[i])] = v
                out[i] = v
        return torch.stack([o for o in out if o is not None], dim=0)


def build_text_featurizer(
    mode: str = "handcrafted",
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    device: torch.device | str = "cpu",
) -> TextFeaturizer:
    """Construct a featurizer by mode. ``mode`` in {"handcrafted", "clip"}."""
    m = (mode or "handcrafted").strip().lower()
    if m in ("handcrafted", "hand", "legacy", "8d", "none"):
        return HandcraftedTextFeaturizer()
    if m in ("clip", "semantic"):
        return ClipTextFeaturizer(model_id=model_id, device=device)
    raise ValueError(f"Unknown text_embed_mode '{mode}' (expected 'handcrafted' or 'clip').")


def featurizer_from_config(cfg: dict, device: torch.device | str = "cpu") -> TextFeaturizer:
    """
    Build the featurizer described by a (checkpoint) config dict.

    Missing keys default to the legacy handcrafted path, so checkpoints trained
    before this module load and score exactly as before.
    """
    mode = str(cfg.get("text_embed_mode", "handcrafted"))
    model_id = str(cfg.get("text_encoder_model_id", DEFAULT_CLIP_MODEL_ID))
    return build_text_featurizer(mode=mode, model_id=model_id, device=device)


__all__ = [
    "DEFAULT_CLIP_MODEL_ID",
    "TextFeaturizer",
    "HandcraftedTextFeaturizer",
    "ClipTextFeaturizer",
    "build_text_featurizer",
    "featurizer_from_config",
]
