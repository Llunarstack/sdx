"""
Parse ``(emphasis)`` / ``[de-emphasis]`` in prompts and build per–T5-token weights.

Matches ``sample.py`` / DiT ``token_weights`` (see ``models/dit_text.py``): ``(word)`` → 1.2,
``[word]`` → 0.8. Training can use the same rules via ``--train-prompt-emphasis`` in ``train.py``.

**Triple text mode:** T5 sequence is ``max_length`` tokens plus **two** CLIP-derived tokens;
weights for those are set to **1.0**.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch

__all__ = [
    "parse_prompt_emphasis",
    "token_weights_from_cleaned_segments",
    "batch_encoder_token_weights",
]


def parse_prompt_emphasis(prompt: str) -> Tuple[str, list]:
    """
    Parse ``(word)`` → weight 1.2, ``[word]`` → 0.8.

    Returns ``(cleaned_prompt, segments)`` where *segments* are
    ``[(char_start, char_end, weight), ...]`` in *cleaned* string coordinates.
    """
    cleaned = ""
    segments: list = []
    parts = re.split(r"(\([^)]*\)|\[[^\]]*\])", prompt)
    for p in parts:
        if p.startswith("(") and p.endswith(")"):
            content = p[1:-1]
            start = len(cleaned)
            cleaned += content
            segments.append((start, len(cleaned), 1.2))
        elif p.startswith("[") and p.endswith("]"):
            content = p[1:-1]
            start = len(cleaned)
            cleaned += content
            segments.append((start, len(cleaned), 0.8))
        else:
            start = len(cleaned)
            cleaned += p
            if p:
                segments.append((start, len(cleaned), 1.0))
    return cleaned.strip(), segments


def token_weights_from_cleaned_segments(
    cleaned: str,
    segments: list,
    tokenizer,
    max_length: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """
    Return ``(max_length,)`` per-token weights using tokenizer ``offset_mapping``, or ``None``
    if the tokenizer does not support it.
    """
    try:
        enc = tokenizer(
            [cleaned],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
    except TypeError:
        return None
    offset_mapping = enc.get("offset_mapping")
    if offset_mapping is None:
        return None
    offset_mapping = offset_mapping[0]
    weights: List[float] = []
    for s, e in offset_mapping:
        if s == 0 and e == 0:  # padding
            weights.append(1.0)
            continue
        w = 1.0
        for seg_start, seg_end, seg_w in segments:
            if s < seg_end and e > seg_start:
                w = seg_w
                break
        weights.append(w)
    t = torch.tensor(weights, dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def batch_encoder_token_weights(
    captions: List[str],
    tokenizer,
    max_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    text_bundle=None,
) -> Tuple[List[str], Optional[torch.Tensor]]:
    """
    For each caption, strip emphasis brackets for T5 text and build aligned weights.

    Returns:
        ``(cleaned_captions, token_weights)`` with ``token_weights`` shape ``(B, L)`` where
        ``L == max_length`` for T5-only, or ``max_length + 2`` for triple text (extra two = 1.0).

    If offset mapping is unavailable for any row, returns ``(cleaned_captions, None)`` — caller
    should still encode *cleaned_captions* but omit ``token_weights`` in ``model_kwargs``.
    """
    cleaned_caps: List[str] = []
    segment_lists: List[Tuple[str, list]] = []
    for c in captions:
        cl, segs = parse_prompt_emphasis(c or "")
        cleaned_caps.append(cl)
        segment_lists.append((cl, segs))

    rows: List[torch.Tensor] = []
    for cl, segs in segment_lists:
        w = token_weights_from_cleaned_segments(cl, segs, tokenizer, max_length, device=None, dtype=torch.float32)
        if w is None:
            return cleaned_caps, None
        rows.append(w)

    stacked = torch.stack(rows, dim=0).to(device=device, dtype=dtype)
    if (
        text_bundle is not None
        and getattr(text_bundle, "mode", "") == "triple"
        and getattr(text_bundle, "fusion", None) is not None
    ):
        b = stacked.shape[0]
        stacked = torch.cat([stacked, torch.ones(b, 2, device=device, dtype=dtype)], dim=1)
    return cleaned_caps, stacked
