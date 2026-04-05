"""Encode a reference PIL image with CLIP vision (for DiT reference tokens)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

_vision_bundle: Optional[Tuple[Any, Any, str]] = None


def _get_clip_vision(model_id: str, device: torch.device) -> Tuple[Any, Any]:
    global _vision_bundle
    if _vision_bundle is not None and _vision_bundle[2] == model_id:
        return _vision_bundle[0], _vision_bundle[1]
    import transformers as tr

    proc = tr.CLIPImageProcessor.from_pretrained(model_id)
    model = tr.CLIPVisionModelWithProjection.from_pretrained(model_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    _vision_bundle = (model, proc, model_id)
    return model, proc


@torch.no_grad()
def encode_reference_image_pil(
    pil_image,
    *,
    device: torch.device,
    model_id: str = "openai/clip-vit-large-patch14",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, int]:
    """
    Returns (image_embeds, embed_dim) where image_embeds is (1, embed_dim).
    """
    model, proc = _get_clip_vision(model_id, device)
    inputs = proc(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    emb = out.image_embeds
    dim = int(emb.shape[-1])
    return emb.to(dtype=dtype), dim
