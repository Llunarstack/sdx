"""Text-encoder bundles (T5 / triple / penta), their readiness checks and caption routing, and the training/sampling autoencoder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Union

import torch
import torch.nn as nn

from utils.modeling.model_paths import (
    default_clip_bigg_path,
    default_clip_h14_path,
    default_clip_l_path,
    default_longclip_l_path,
    default_t5_path,
    has_local_weights,
    model_dir,
    resolve_model_path,
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


StackMode = Literal["t5", "triple", "penta"]

TRIPLE_CATALOG = ("T5-XXL", "CLIP-ViT-L-14", "CLIP-ViT-bigG-14")
PENTA_CATALOG = TRIPLE_CATALOG + ("CLIP-ViT-H-14", "LongCLIP-L")

_CATALOG_FALLBACK = {
    "T5-XXL": ("google/t5-v1_1-xxl", default_t5_path),
    "CLIP-ViT-L-14": ("openai/clip-vit-large-patch14", default_clip_l_path),
    "CLIP-ViT-bigG-14": ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", default_clip_bigg_path),
    "CLIP-ViT-H-14": ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", default_clip_h14_path),
    "LongCLIP-L": ("creative-graphic-design/LongCLIP-L", default_longclip_l_path),
}


@dataclass(slots=True)
class EncoderSlotStatus:
    name: str
    resolved: str
    local: bool
    has_weights: bool

    @property
    def ready(self) -> bool:
        return self.has_weights or not self.local


@dataclass(slots=True)
class TextEncoderStackStatus:
    mode: StackMode
    slots: list[EncoderSlotStatus]

    @property
    def ready(self) -> bool:
        return all(s.ready for s in self.slots)

    @property
    def local_count(self) -> int:
        return sum(1 for s in self.slots if s.local)

    @property
    def weights_count(self) -> int:
        return sum(1 for s in self.slots if s.has_weights)


def catalog_for_mode(mode: str) -> tuple[str, ...]:
    m = str(mode or "t5").lower()
    if m == "penta":
        return PENTA_CATALOG
    if m == "triple":
        return TRIPLE_CATALOG
    return ("T5-XXL",)


def _slot_status(name: str) -> EncoderSlotStatus:
    hub_id, default_fn = _CATALOG_FALLBACK[name]
    resolved = resolve_model_path(name, hub_id)
    local_path = Path(resolved)
    is_local = local_path.is_absolute() and local_path.is_dir() and any(local_path.iterdir())
    if not is_local:
        local_path = model_dir() / name
        is_local = local_path.is_dir() and any(local_path.iterdir())
        if is_local:
            resolved = str(local_path)
    has_w = has_local_weights(Path(resolved)) if is_local else False
    if not has_w and not is_local:
        resolved = default_fn()
    return EncoderSlotStatus(name=name, resolved=resolved, local=is_local, has_weights=has_w)


def stack_status(mode: str = "penta") -> TextEncoderStackStatus:
    """Return per-encoder local/weights status for ``t5``, ``triple``, or ``penta``."""
    m = str(mode or "t5").lower()
    if m not in ("t5", "triple", "penta"):
        m = "t5"
    names = catalog_for_mode(m)
    return TextEncoderStackStatus(mode=m, slots=[_slot_status(n) for n in names])


def stack_status_lines(mode: str = "penta") -> list[str]:
    """Markdown bullet lines for eval reports and CLI."""
    st = stack_status(mode)
    if st.mode == "t5":
        s = st.slots[0]
        w = "weights" if s.has_weights else ("config-only" if s.local else "hub fallback")
        return [f"- T5: `{s.resolved}` ({w})"]

    lines = [
        f"- Mode: **{st.mode}** — {st.weights_count}/{len(st.slots)} with local weights, "
        f"{st.local_count}/{len(st.slots)} local folders"
    ]
    for s in st.slots:
        if s.has_weights:
            tag = "weights"
        elif s.local:
            tag = "config-only"
        else:
            tag = "hub fallback"
        lines.append(f"  - {s.name}: `{s.resolved}` ({tag})")
    if not st.ready:
        missing = [s.name for s in st.slots if s.local and not s.has_weights]
        if missing:
            lines.append(
                f"- Note: config-only locally ({', '.join(missing)}); runtime uses HF hub unless weights downloaded."
            )
    return lines


def stack_download_hint(mode: str = "penta") -> str:
    m = str(mode or "t5").lower()
    if m == "penta":
        return (
            "python scripts/download/download_models.py --penta-text-encoders\n"
            "python scripts/download/download_hf_scaffold.py --penta"
        )
    if m == "triple":
        return "python scripts/download/download_models.py --triple-text-encoders"
    return "python scripts/download/download_models.py --t5"


def text_encoder_mode_label(mode: str) -> str:
    return " + ".join(catalog_for_mode(mode))


def checkpoint_text_stack(ckpt_path: str) -> tuple[StackMode, bool]:
    """``(text_encoder_mode, has_fusion_weights)`` of a DiT checkpoint; ``("t5", False)`` when unreadable."""
    import torch

    p = Path(str(ckpt_path).strip())
    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False) if p.is_file() else None
    except Exception:
        ckpt = None
    if not isinstance(ckpt, dict):
        return "t5", False
    mode = str(getattr(ckpt.get("config"), "text_encoder_mode", "t5") or "t5").lower()
    fusion = ckpt.get("text_encoder_fusion")
    return (mode if mode in ("triple", "penta") else "t5"), isinstance(fusion, dict) and bool(fusion)  # type: ignore[return-value]


LayoutSpec = None | str | Path | dict


def _compile_layout_spec(spec: LayoutSpec) -> Any:
    """Return :class:`~utils.prompt.text.CompiledPromptLayout`."""
    from utils.prompt.text import compile_prompt_layout, load_prompt_layout_file

    if spec is None:
        raise ValueError("layout spec is None")
    if isinstance(spec, dict):
        return compile_prompt_layout(spec)
    p = Path(spec)
    if p.is_file():
        return load_prompt_layout_file(p)
    if isinstance(spec, str) and spec.strip().startswith("{"):
        raw = json.loads(spec)
        if isinstance(raw, dict):
            return compile_prompt_layout(raw)
    raise ValueError(f"invalid prompt layout spec: {spec!r}")


def clip_and_long_captions_for_layout(
    flat_caption: str,
    layout_spec: LayoutSpec,
) -> tuple[str, str]:
    """
    CLIP string (labeled sections, 77-token friendly) and LongCLIP string (full flat text).
    """
    from utils.prompt.text import multi_clip_caption

    flat = (flat_caption or "").strip()
    if layout_spec is None:
        return flat, flat
    try:
        compiled = _compile_layout_spec(layout_spec)
        clip = multi_clip_caption(compiled, flat or compiled.positive)
        long_cap = flat or compiled.positive
        return clip, long_cap
    except Exception:
        return flat, flat


def prepare_multi_encoder_kwargs(
    t5_captions: list[str],
    text_bundle: Any = None,
    *,
    layout_specs: list[LayoutSpec | None] | None = None,
    flat_full_prompts: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Build ``clip_captions`` / ``long_clip_captions`` kwargs for multi-encoder bundles.

    When ``layout_specs[i]`` is set, CLIP encoders use the compiled layout caption; LongCLIP
    (penta) uses ``flat_full_prompts[i]`` or ``t5_captions[i]``.
    """
    if text_bundle is None or getattr(text_bundle, "fusion", None) is None:
        return {}
    mode = str(getattr(text_bundle, "mode", "t5") or "t5").lower()
    if mode not in ("triple", "penta"):
        return {}

    n = len(t5_captions)
    flats = list(flat_full_prompts) if flat_full_prompts is not None else list(t5_captions)
    if len(flats) < n:
        flats.extend([t5_captions[i] for i in range(len(flats), n)])
    specs: list[LayoutSpec | None] = list(layout_specs) if layout_specs is not None else [None] * n
    if len(specs) < n:
        specs.extend([None] * (n - len(specs)))

    clip_caps: list[str] = []
    long_caps: list[str] = []
    for i in range(n):
        flat = flats[i] if i < len(flats) else t5_captions[i]
        spec = specs[i] if i < len(specs) else None
        clip_s, long_s = clip_and_long_captions_for_layout(flat, spec)
        clip_caps.append(clip_s)
        long_caps.append(long_s)

    out: dict[str, list[str]] = {"clip_captions": clip_caps}
    if mode == "penta":
        out["long_clip_captions"] = long_caps
    return out


def encode_kwargs_for_captions(
    captions: list[str],
    text_bundle: Any = None,
    *,
    layout_specs: list[LayoutSpec | None] | None = None,
    flat_full_prompts: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Extra kwargs for :func:`train.encode_text` / :class:`TextEncoderBundle.encode`.

    Pass ``layout_specs`` from a training batch (JSONL ``prompt_layout`` fields) for
    layout-aware CLIP / LongCLIP routing.
    """
    if layout_specs is not None or flat_full_prompts is not None:
        return prepare_multi_encoder_kwargs(
            captions,
            text_bundle,
            layout_specs=layout_specs,
            flat_full_prompts=flat_full_prompts,
        )
    if text_bundle is None or getattr(text_bundle, "fusion", None) is None:
        return {}
    mode = str(getattr(text_bundle, "mode", "t5") or "t5").lower()
    if mode not in ("triple", "penta"):
        return {}
    caps = list(captions)
    out: dict[str, list[str]] = {"clip_captions": caps}
    if mode == "penta":
        out["long_clip_captions"] = caps
    return out


def load_text_bundle_for_training(
    cfg: object,
    device: Any,
    fusion_sd: dict | None = None,
    *,
    out_dim: int = 4096,
):
    """Load triple/penta bundle and attach checkpoint fusion weights when provided."""
    bundle = load_text_encoder_bundle(cfg, device, out_dim=out_dim)
    if bundle is not None and fusion_sd is not None:
        attach_fusion_weights(bundle, fusion_sd)
    return bundle


SD_LATENT_SCALE = 0.18215  # SD 1.x VAE; what checkpoints without latent fields were trained with


def get_autoencoder_class(ae_type: str):
    """Return ``AutoencoderKL`` for "kl" (default) or ``AutoencoderRAE`` for "rae"."""
    if str(ae_type or "kl").strip().lower() == "rae":
        try:
            from diffusers import AutoencoderRAE
        except ImportError as e:
            raise ImportError(
                "autoencoder_type='rae' requires diffusers>=0.37.0 (AutoencoderRAE). "
                "Upgrade with: pip install -U 'diffusers>=0.37.0'"
            ) from e
        return AutoencoderRAE
    from diffusers import AutoencoderKL

    return AutoencoderKL


@dataclass(frozen=True, slots=True)
class LatentSpec:
    channels: int = 4
    scale: float = SD_LATENT_SCALE
    shift: float = 0.0


def latent_spec(vae: Any) -> LatentSpec:
    """Latent space a KL VAE was trained for, from its diffusers config (SD 1.x values when absent)."""
    cfg = getattr(vae, "config", None)
    return LatentSpec(
        channels=int(getattr(cfg, "latent_channels", None) or 4),
        scale=float(getattr(cfg, "scaling_factor", None) or SD_LATENT_SCALE),
        shift=float(getattr(cfg, "shift_factor", None) or 0.0),
    )


class _ShiftedPosterior:
    """KL posterior whose samples are offset by ``-shift``."""

    def __init__(self, dist: Any, shift: float) -> None:
        self._dist, self._shift = dist, shift

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return self._dist.sample(generator) - self._shift

    def mode(self) -> torch.Tensor:
        return self._dist.mode() - self._shift

    @property
    def mean(self) -> torch.Tensor:
        return self._dist.mean - self._shift

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)


class ShiftedAutoencoder(torch.nn.Module):
    """A KL VAE with ``shift_factor`` applied inside ``encode`` / ``decode``; everything else passes through."""

    def __init__(self, vae: torch.nn.Module, shift: float) -> None:
        super().__init__()
        self.vae = vae
        self.shift = float(shift)

    def encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        out = self.vae.encode(x, *args, **kwargs)
        dist = out[0] if isinstance(out, tuple) else out.latent_dist
        return SimpleNamespace(latent_dist=_ShiftedPosterior(dist, self.shift))

    def decode(self, z: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.vae.decode(z + self.shift, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["vae"], name)


def load_autoencoder(
    ae_type: str, model_id: str, *, subfolder: str = "", device: Any = None, dtype: torch.dtype | None = None
) -> torch.nn.Module:
    """Frozen autoencoder in eval mode; KL VAEs with a ``shift_factor`` come back as :class:`ShiftedAutoencoder`."""
    kwargs: dict[str, Any] = {"subfolder": subfolder} if subfolder else {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    ae: torch.nn.Module = get_autoencoder_class(ae_type).from_pretrained(model_id, **kwargs)
    if str(ae_type or "kl").strip().lower() == "kl":
        shift = latent_spec(ae).shift
        if shift:
            ae = ShiftedAutoencoder(ae, shift)
    ae.eval().requires_grad_(False)
    return ae.to(device) if device is not None else ae


def apply_latent_spec(cfg: Any, vae: Any) -> LatentSpec:
    """Write the KL VAE's latent space onto ``cfg`` (saved into checkpoints, so sampling decodes the same way)."""
    spec = latent_spec(vae)
    cfg.latent_channels, cfg.latent_scale, cfg.latent_shift = spec.channels, spec.scale, spec.shift
    return spec
