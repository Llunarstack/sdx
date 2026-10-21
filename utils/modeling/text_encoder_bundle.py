"""
T5-only, triple, or penta text conditioning.

- **t5:** T5-XXL sequence only.
- **triple:** T5 sequence + CLIP-L + CLIP-bigG pooled tokens (2 extra cross-attn tokens).
- **penta:** T5 sequence + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L pooled tokens (4 extra tokens).

Fusion weights are saved in checkpoints as ``text_encoder_fusion``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from utils.modeling.model_paths import (
    default_clip_bigg_path,
    default_clip_h14_path,
    default_clip_l_path,
    default_longclip_l_path,
    default_t5_path,
)
from utils.runtime import json_loads
from utils.safety.minor_guard import require_encodable


def encode_t5_segment_concat(
    segment_texts: Sequence[str],
    tokenizer,
    text_encoder: torch.nn.Module,
    device: torch.device,
    max_length: int = 300,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    ``--t5-layout-encode segmented``: tokenize each segment, join the ids, pad/truncate, one frozen T5 forward.

    Section boundaries survive better than in one long comma-joined string. Returns ``(1, max_length, hidden)``.
    """
    pieces: list[torch.Tensor] = []
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


def encode_text(
    captions,
    tokenizer,
    text_encoder,
    device,
    max_length: int = 300,
    dtype: torch.dtype = torch.float32,
    text_bundle: TextEncoderBundle | None = None,
    train_fusion: bool = False,
    clip_captions=None,
    long_clip_captions=None,
    segment_texts=None,
) -> torch.Tensor:
    """Encode captions to ``(B, L, text_dim)``; triple / penta bundles append fused CLIP tokens.

    Shared by ``train.py`` and ``sample.py``; every caption is screened by ``require_encodable`` first.
    """
    require_encodable(captions, clip_captions, long_clip_captions, segment_texts)
    device = torch.device(device)
    if text_bundle is not None:
        return text_bundle.encode(
            captions,
            device,
            max_length=max_length,
            dtype=dtype,
            train_fusion=train_fusion,
            clip_captions=clip_captions,
            long_clip_captions=long_clip_captions,
            segment_texts=segment_texts,
        )
    if segment_texts is not None:
        return encode_t5_segment_concat(
            segment_texts, tokenizer, text_encoder, device, max_length=max_length, dtype=dtype
        )
    nbc = device.type == "cuda"
    tok = tokenizer(captions, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = text_encoder(
            input_ids=tok.input_ids.to(device, non_blocking=nbc),
            attention_mask=tok.attention_mask.to(device, non_blocking=nbc),
        )
    return out.last_hidden_state.to(dtype)


FusionModule = Union["TripleTextFusion", "PentaTextFusion"]


def _read_clip_text_hidden_size(model_dir: str) -> int:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return 768
    try:
        with config_path.open("r", encoding="utf-8") as file_handle:
            config_data = json_loads(file_handle.read())
        if "text_config" in config_data and isinstance(config_data["text_config"], dict):
            return int(config_data["text_config"].get("hidden_size", 768))
        return int(config_data.get("hidden_size", 768))
    except Exception:
        return 768


class TripleTextFusion(nn.Module):
    """Append CLIP-L and CLIP-bigG as two extra tokens (dim=out_dim)."""

    extra_token_count: int = 2

    def __init__(self, clip_l_dim: int, clip_bg_dim: int, out_dim: int = 4096):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj_l = nn.Linear(int(clip_l_dim), self.out_dim)
        self.proj_bg = nn.Linear(int(clip_bg_dim), self.out_dim)
        nn.init.normal_(self.proj_l.weight, std=0.02)
        nn.init.zeros_(self.proj_l.bias)
        nn.init.normal_(self.proj_bg.weight, std=0.02)
        nn.init.zeros_(self.proj_bg.bias)

    def forward(
        self,
        t5_hidden: torch.Tensor,
        clip_l_pooled: torch.Tensor,
        clip_bg_pooled: torch.Tensor,
    ) -> torch.Tensor:
        extra = torch.stack(
            [self.proj_l(clip_l_pooled), self.proj_bg(clip_bg_pooled)],
            dim=1,
        )
        return torch.cat([t5_hidden, extra], dim=1)


class PentaTextFusion(nn.Module):
    """Append CLIP-L, CLIP-bigG, CLIP-H, and LongCLIP-L as four extra tokens."""

    extra_token_count: int = 4

    def __init__(
        self,
        clip_l_dim: int,
        clip_bg_dim: int,
        clip_h_dim: int,
        clip_long_dim: int,
        out_dim: int = 4096,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj_l = nn.Linear(int(clip_l_dim), self.out_dim)
        self.proj_bg = nn.Linear(int(clip_bg_dim), self.out_dim)
        self.proj_h = nn.Linear(int(clip_h_dim), self.out_dim)
        self.proj_long = nn.Linear(int(clip_long_dim), self.out_dim)
        for proj in (self.proj_l, self.proj_bg, self.proj_h, self.proj_long):
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        t5_hidden: torch.Tensor,
        clip_l_pooled: torch.Tensor,
        clip_bg_pooled: torch.Tensor,
        clip_h_pooled: torch.Tensor,
        clip_long_pooled: torch.Tensor,
    ) -> torch.Tensor:
        extra = torch.stack(
            [
                self.proj_l(clip_l_pooled),
                self.proj_bg(clip_bg_pooled),
                self.proj_h(clip_h_pooled),
                self.proj_long(clip_long_pooled),
            ],
            dim=1,
        )
        return torch.cat([t5_hidden, extra], dim=1)


@dataclass(slots=True)
class TextEncoderBundle:
    mode: str  # "t5" | "triple" | "penta"
    tokenizer: object
    text_encoder: nn.Module
    clip_tokenizer_l: object | None = None
    clip_text_l: nn.Module | None = None
    clip_tokenizer_bg: object | None = None
    clip_text_bg: nn.Module | None = None
    clip_tokenizer_h: object | None = None
    clip_text_h: nn.Module | None = None
    clip_tokenizer_long: object | None = None
    clip_text_long: nn.Module | None = None
    fusion: FusionModule | None = None

    def extra_fusion_tokens(self) -> int:
        if self.fusion is None:
            return 0
        return int(getattr(self.fusion, "extra_token_count", 0))

    def _t5_hidden(
        self,
        captions: list[str],
        device: torch.device,
        max_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tok = self.tokenizer(
            captions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        nbc = device.type == "cuda"
        input_ids = tok.input_ids.to(device, non_blocking=nbc)
        attention_mask = tok.attention_mask.to(device, non_blocking=nbc)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state.to(dtype)

    @staticmethod
    def _clip_pooled(
        model: nn.Module,
        tokenizer,
        captions: list[str],
        device: torch.device,
        dtype: torch.dtype,
        *,
        max_length: int = 77,
    ) -> torch.Tensor:
        inputs = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        nbc = device.type == "cuda"
        inputs = {k: v.to(device, non_blocking=nbc) for k, v in inputs.items()}
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output.to(dtype)
        hs = out.last_hidden_state
        return hs[:, -1].to(dtype)

    def encode(
        self,
        captions: list[str],
        device: torch.device,
        max_length: int = 300,
        dtype: torch.dtype = torch.bfloat16,
        *,
        train_fusion: bool = False,
        clip_captions: list[str] | None = None,
        long_clip_captions: list[str] | None = None,
        segment_texts: list[str] | None = None,
    ) -> torch.Tensor:
        clip_src = clip_captions if clip_captions is not None else captions
        long_src = long_clip_captions if long_clip_captions is not None else captions

        def _t5_out() -> torch.Tensor:
            if segment_texts is not None:
                return encode_t5_segment_concat(
                    segment_texts,
                    self.tokenizer,
                    self.text_encoder,
                    device,
                    max_length=max_length,
                    dtype=dtype,
                )
            return self._t5_hidden(captions, device, max_length, dtype)

        # no_grad, not inference_mode: these frozen outputs feed trainable modules (fusion, DiT), and autograd cannot
        # save inference tensors for its backward.
        if self.mode not in ("triple", "penta") or self.fusion is None:
            with torch.no_grad():
                return _t5_out()

        with torch.no_grad():
            t5_hidden = _t5_out()
            clip_l_pooled = self._clip_pooled(self.clip_text_l, self.clip_tokenizer_l, clip_src, device, dtype)
            clip_bg_pooled = self._clip_pooled(self.clip_text_bg, self.clip_tokenizer_bg, clip_src, device, dtype)

        if self.mode == "triple":
            fused = self.fusion(t5_hidden, clip_l_pooled, clip_bg_pooled)
        else:
            with torch.no_grad():
                clip_h_pooled = self._clip_pooled(self.clip_text_h, self.clip_tokenizer_h, clip_src, device, dtype)
                long_max = min(int(getattr(self.clip_tokenizer_long, "model_max_length", 248) or 248), 248)
                clip_long_pooled = self._clip_pooled(
                    self.clip_text_long,
                    self.clip_tokenizer_long,
                    long_src,
                    device,
                    dtype,
                    max_length=long_max,
                )
            fused = self.fusion(t5_hidden, clip_l_pooled, clip_bg_pooled, clip_h_pooled, clip_long_pooled)

        if train_fusion:
            return fused
        return fused.detach()


def attach_fusion_weights(bundle: TextEncoderBundle, fusion_sd: dict) -> None:
    """Load trained fusion weights into an existing bundle (sampling / resume)."""
    if bundle.fusion is None:
        raise ValueError("bundle has no fusion module")
    bundle.fusion.load_state_dict(fusion_sd)
    bundle.fusion.eval()


def _load_frozen_clip_pair(transformers_lib, path: str, device: torch.device):
    clip_tokenizer = transformers_lib.CLIPTokenizer.from_pretrained(path)
    clip_model = transformers_lib.CLIPTextModel.from_pretrained(path)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_tokenizer, clip_model.to(device)


def load_text_encoder_bundle(
    cfg: object,
    device: torch.device,
    *,
    out_dim: int = 4096,
) -> TextEncoderBundle | None:
    """
    Load multi-encoder bundle when ``cfg.text_encoder_mode`` is ``triple`` or ``penta``.
    Otherwise return None (caller uses T5 only).
    """
    mode = str(getattr(cfg, "text_encoder_mode", "t5") or "t5").lower()
    if mode not in ("triple", "penta"):
        return None

    import transformers as transformers_lib

    t5_path = getattr(cfg, "text_encoder", None) or default_t5_path()
    clip_l_path = (getattr(cfg, "clip_text_encoder_l", None) or "").strip() or default_clip_l_path()
    clip_bg_path = (getattr(cfg, "clip_text_encoder_bigg", None) or "").strip() or default_clip_bigg_path()

    tokenizer = transformers_lib.AutoTokenizer.from_pretrained(t5_path)
    t5 = transformers_lib.T5EncoderModel.from_pretrained(t5_path)
    t5.eval()
    for p in t5.parameters():
        p.requires_grad = False
    t5 = t5.to(device)

    clip_tokenizer_l, clip_l = _load_frozen_clip_pair(transformers_lib, clip_l_path, device)
    clip_tokenizer_bg, clip_bg = _load_frozen_clip_pair(transformers_lib, clip_bg_path, device)

    clip_l_hidden_dim = _read_clip_text_hidden_size(clip_l_path)
    clip_bg_hidden_dim = _read_clip_text_hidden_size(clip_bg_path)

    clip_tokenizer_h = None
    clip_h = None
    clip_tokenizer_long = None
    clip_long = None
    fusion: FusionModule

    if mode == "penta":
        clip_h_path = (getattr(cfg, "clip_text_encoder_h", None) or "").strip() or default_clip_h14_path()
        clip_long_path = (getattr(cfg, "clip_text_encoder_long", None) or "").strip() or default_longclip_l_path()
        clip_tokenizer_h, clip_h = _load_frozen_clip_pair(transformers_lib, clip_h_path, device)
        clip_tokenizer_long, clip_long = _load_frozen_clip_pair(transformers_lib, clip_long_path, device)
        clip_h_hidden_dim = _read_clip_text_hidden_size(clip_h_path)
        clip_long_hidden_dim = _read_clip_text_hidden_size(clip_long_path)
        fusion = PentaTextFusion(
            clip_l_hidden_dim,
            clip_bg_hidden_dim,
            clip_h_hidden_dim,
            clip_long_hidden_dim,
            out_dim=out_dim,
        ).to(device)
    else:
        fusion = TripleTextFusion(clip_l_hidden_dim, clip_bg_hidden_dim, out_dim=out_dim).to(device)

    return TextEncoderBundle(
        mode=mode,
        tokenizer=tokenizer,
        text_encoder=t5,
        clip_tokenizer_l=clip_tokenizer_l,
        clip_text_l=clip_l,
        clip_tokenizer_bg=clip_tokenizer_bg,
        clip_text_bg=clip_bg,
        clip_tokenizer_h=clip_tokenizer_h,
        clip_text_h=clip_h,
        clip_tokenizer_long=clip_tokenizer_long,
        clip_text_long=clip_long,
        fusion=fusion,
    )
