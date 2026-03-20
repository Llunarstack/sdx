"""
T5-only or triple text conditioning: T5-XXL sequence + CLIP-L + CLIP-bigG pooled tokens
projected to `out_dim` and appended as two extra cross-attention tokens.

Requires training with the same mode; fusion weights are saved in checkpoints as `text_encoder_fusion`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from utils.model_paths import default_clip_bigg_path, default_clip_l_path, default_t5_path


def _read_clip_text_hidden_size(model_dir: str) -> int:
    p = Path(model_dir) / "config.json"
    if not p.is_file():
        return 768
    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        # HF CLIPTextModel
        if "text_config" in cfg and isinstance(cfg["text_config"], dict):
            return int(cfg["text_config"].get("hidden_size", 768))
        return int(cfg.get("hidden_size", 768))
    except Exception:
        return 768


class TripleTextFusion(nn.Module):
    """Append CLIP-L and CLIP-bigG as two extra tokens (dim=out_dim)."""

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
        # t5_hidden: (B, L, D)  clip_*: (B, d)
        extra = torch.stack(
            [self.proj_l(clip_l_pooled), self.proj_bg(clip_bg_pooled)],
            dim=1,
        )
        return torch.cat([t5_hidden, extra], dim=1)


@dataclass
class TextEncoderBundle:
    mode: str  # "t5" | "triple"
    tokenizer: object
    text_encoder: nn.Module
    clip_tokenizer_l: Optional[object] = None
    clip_text_l: Optional[nn.Module] = None
    clip_tokenizer_bg: Optional[object] = None
    clip_text_bg: Optional[nn.Module] = None
    fusion: Optional[TripleTextFusion] = None

    def _t5_hidden(
        self,
        captions: List[str],
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
        input_ids = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state.to(dtype)

    @staticmethod
    def _clip_pooled(
        model: nn.Module,
        tokenizer,
        captions: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        inputs = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output.to(dtype)
        hs = out.last_hidden_state
        return hs[:, -1].to(dtype)

    def encode(
        self,
        captions: List[str],
        device: torch.device,
        max_length: int = 300,
        dtype: torch.dtype = torch.bfloat16,
        *,
        train_fusion: bool = False,
    ) -> torch.Tensor:
        if self.mode != "triple" or self.fusion is None:
            with torch.no_grad():
                return self._t5_hidden(captions, device, max_length, dtype)
        with torch.no_grad():
            t5_h = self._t5_hidden(captions, device, max_length, dtype)
            cl = self._clip_pooled(self.clip_text_l, self.clip_tokenizer_l, captions, device, dtype)
            cbg = self._clip_pooled(self.clip_text_bg, self.clip_tokenizer_bg, captions, device, dtype)
        fused = self.fusion(t5_h, cl, cbg)
        if train_fusion:
            return fused
        return fused.detach()


def attach_fusion_weights(bundle: TextEncoderBundle, fusion_sd: dict) -> None:
    """Load trained fusion weights into an existing bundle (sampling / resume)."""
    if bundle.fusion is None:
        raise ValueError("bundle has no fusion module")
    bundle.fusion.load_state_dict(fusion_sd)
    bundle.fusion.eval()


def load_fusion_from_state_dict(sd: dict, device: torch.device) -> TripleTextFusion:
    """Rebuild TripleTextFusion from a checkpoint state dict."""
    w = sd["proj_l.weight"]
    clip_l_dim, out_dim = int(w.shape[1]), int(w.shape[0])
    clip_bg_dim = int(sd["proj_bg.weight"].shape[1])
    m = TripleTextFusion(clip_l_dim, clip_bg_dim, out_dim=out_dim)
    m.load_state_dict(sd)
    return m.to(device).eval()


def load_text_encoder_bundle(
    cfg: object,
    device: torch.device,
    *,
    out_dim: int = 4096,
) -> Optional[TextEncoderBundle]:
    """
    If cfg.text_encoder_mode == 'triple', load T5 + CLIP-L + CLIP-bigG + fusion.
    Otherwise return None (caller uses T5 only).
    """
    mode = str(getattr(cfg, "text_encoder_mode", "t5") or "t5").lower()
    if mode != "triple":
        return None

    import transformers as tr

    t5_path = getattr(cfg, "text_encoder", None) or default_t5_path()
    clip_l_path = (getattr(cfg, "clip_text_encoder_l", None) or "").strip() or default_clip_l_path()
    clip_bg_path = (getattr(cfg, "clip_text_encoder_bigg", None) or "").strip() or default_clip_bigg_path()

    tokenizer = tr.AutoTokenizer.from_pretrained(t5_path)
    t5 = tr.T5EncoderModel.from_pretrained(t5_path)
    t5.eval()
    for p in t5.parameters():
        p.requires_grad = False
    t5 = t5.to(device)

    clip_tok_l = tr.CLIPTokenizer.from_pretrained(clip_l_path)
    clip_l = tr.CLIPTextModel.from_pretrained(clip_l_path)
    clip_l.eval()
    for p in clip_l.parameters():
        p.requires_grad = False
    clip_l = clip_l.to(device)

    clip_tok_bg = tr.CLIPTokenizer.from_pretrained(clip_bg_path)
    clip_bg = tr.CLIPTextModel.from_pretrained(clip_bg_path)
    clip_bg.eval()
    for p in clip_bg.parameters():
        p.requires_grad = False
    clip_bg = clip_bg.to(device)

    d_l = _read_clip_text_hidden_size(clip_l_path)
    d_bg = _read_clip_text_hidden_size(clip_bg_path)
    fusion = TripleTextFusion(d_l, d_bg, out_dim=out_dim).to(device)

    return TextEncoderBundle(
        mode="triple",
        tokenizer=tokenizer,
        text_encoder=t5,
        clip_tokenizer_l=clip_tok_l,
        clip_text_l=clip_l,
        clip_tokenizer_bg=clip_tok_bg,
        clip_text_bg=clip_bg,
        fusion=fusion,
    )
