"""
Concatenated-segment T5 encoding: tokenize each segment, join input ids, pad/truncate, one forward.

Used when ``--t5-layout-encode segmented`` so section boundaries survive tokenization a bit better
than one long comma-separated string (still a single frozen T5; no fine-tuning).
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def encode_t5_segment_concat(
    segment_texts: Sequence[str],
    tokenizer,
    text_encoder: torch.nn.Module,
    device: torch.device,
    max_length: int = 300,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns ``last_hidden_state`` of shape ``(1, max_length, hidden)`` (batch 1).
    """
    pieces: List[torch.Tensor] = []
    for s in segment_texts:
        st = (s or "").strip()
        if not st:
            continue
        t = tokenizer(
            st,
            add_special_tokens=True,
            return_attention_mask=False,
            return_tensors="pt",
        )
        ids = t["input_ids"][0]
        pieces.append(ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    if not pieces:
        tok = tokenizer(
            "",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        attn = tok.attention_mask.to(device)
        with torch.no_grad():
            out = text_encoder(input_ids=input_ids, attention_mask=attn)
        return out.last_hidden_state.to(dtype)

    full = torch.cat(pieces, dim=0)
    if full.numel() > max_length:
        full = full[:max_length]
    seq_len = int(full.numel())
    if seq_len < max_length:
        pad = torch.full((max_length - seq_len,), int(pad_id), dtype=full.dtype)
        full = torch.cat([full, pad], dim=0)
    input_ids = full.unsqueeze(0).to(device)
    attention_mask = torch.zeros(1, max_length, dtype=torch.long, device=device)
    attention_mask[0, :seq_len] = 1
    with torch.no_grad():
        out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state.to(dtype)
