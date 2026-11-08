"""ViT quality/adherence model, losses, text featurizers and checkpoint loading (ViTScorer is the one inference path)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modeling.architecture import AR_COND_DIM, default_unknown_ar_batch


class ViTQualityAdherenceModel(nn.Module):
    """
    ViT backbone with two heads:
    - quality_logit: binary quality classification (good/bad)
    - adherence_score: regression in [0,1] for prompt adherence

    Optional DiT AR regime conditioning (``use_ar_conditioning``): concatenates a 4-D one-hot
    (full DiT / AR 2x2 / AR 4x4 / unknown) so scores match how the generator was trained.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        text_feat_dim: int = 8,
        hidden_dim: int = 256,
        *,
        use_ar_conditioning: bool = True,
        ar_cond_dim: int = AR_COND_DIM,
        fuse_dropout: float = 0.1,
        text_proj_dropout: float = 0.0,
        backbone_grad_checkpointing: bool = False,
        timm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.use_ar_conditioning = bool(use_ar_conditioning)
        self.ar_cond_dim = int(ar_cond_dim) if self.use_ar_conditioning else 0
        self.text_proj_dropout = float(text_proj_dropout)
        tk: dict[str, Any] = dict(timm_kwargs or {})
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            **tk,
        )
        if backbone_grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        feat_dim = int(self.backbone.num_features)
        cond_in = int(text_feat_dim) + self.ar_cond_dim
        self.text_proj = nn.Sequential(
            nn.Linear(cond_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        fd = float(fuse_dropout)
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(fd),
        )
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.adherence_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        ar_conditioning: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        img_feat = self.backbone(images)
        if self.use_ar_conditioning:
            if ar_conditioning is None:
                ar_conditioning = default_unknown_ar_batch(images.shape[0], images.device, dtype=text_features.dtype)
            x_txt = torch.cat([text_features, ar_conditioning], dim=-1)
        else:
            x_txt = text_features
        txt_feat = self.text_proj(x_txt)
        if self.text_proj_dropout > 0 and self.training:
            txt_feat = F.dropout(txt_feat, p=self.text_proj_dropout, training=True)
        fused = self.fuse(torch.cat([img_feat, txt_feat], dim=1))
        quality_logit = self.quality_head(fused).squeeze(1)
        adherence_score = self.adherence_head(fused).squeeze(1)
        return {
            "quality_logit": quality_logit,
            "adherence_score": adherence_score,
            "embedding": fused,
        }


def build_vit_model(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    text_feat_dim: int = 8,
    hidden_dim: int = 256,
    *,
    use_ar_conditioning: bool = True,
    ar_cond_dim: int = AR_COND_DIM,
    fuse_dropout: float = 0.1,
    text_proj_dropout: float = 0.0,
    backbone_grad_checkpointing: bool = False,
    timm_kwargs: dict[str, Any] | None = None,
) -> ViTQualityAdherenceModel:
    return ViTQualityAdherenceModel(
        model_name=model_name,
        pretrained=pretrained,
        text_feat_dim=text_feat_dim,
        hidden_dim=hidden_dim,
        use_ar_conditioning=use_ar_conditioning,
        ar_cond_dim=ar_cond_dim,
        fuse_dropout=fuse_dropout,
        text_proj_dropout=text_proj_dropout,
        backbone_grad_checkpointing=backbone_grad_checkpointing,
        timm_kwargs=timm_kwargs,
    )


@dataclass(slots=True)
class ViTConfig:
    # Data
    manifest_jsonl: str = ""
    image_root: str = ""
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    training_augment: bool = False

    # Model
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    # Optional extra kwargs for timm.create_model (e.g. qk_norm, reg tokens on supported archs).
    timm_kwargs: dict | None = None
    text_feat_dim: int = 8
    # "handcrafted" = legacy 8-D caption stats; "clip" = frozen CLIP text embeddings (see text_encoder.py).
    text_embed_mode: str = "handcrafted"
    text_encoder_model_id: str = "openai/clip-vit-base-patch32"
    hidden_dim: int = 256
    fuse_dropout: float = 0.1
    text_proj_dropout: float = 0.0
    backbone_grad_checkpointing: bool = False
    # DiT block-AR regime (0/2/4) as extra ViT conditioner; see utils/modeling/architecture.py
    use_ar_conditioning: bool = True
    ar_cond_dim: int = 4

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    quality_loss_weight: float = 1.0
    adherence_loss_weight: float = 1.0
    # 0 = plain BCE; >0 focal gamma (imbalanced quality labels)
    focal_loss_gamma: float = 0.0
    focal_loss_alpha: float | None = None
    # If True, SmoothL1 on adherence instead of MSE (outlier-robust)
    adherence_smooth_l1: bool = False
    adherence_smooth_l1_beta: float = 0.1

    # Runtime
    device: str = "cuda"
    seed: int = 42
    out_dir: str = "vit_runs"


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    """
    Focal loss for imbalanced binary quality labels (down-weights easy examples).
    """
    gamma = float(gamma)
    if gamma <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    with torch.no_grad():
        prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal = (1.0 - p_t.clamp(min=1e-6, max=1.0 - 1e-6)) ** gamma * bce
    if alpha is not None:
        a = float(alpha)
        w = a * targets + (1.0 - a) * (1.0 - targets)
        focal = focal * w
    return focal.mean()


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.15,
    min_target_gap: float = 0.05,
) -> torch.Tensor:
    """
    Pairwise ranking for quality/adherence ordering.
    If target[i] > target[j] by at least min_target_gap, enforce pred[i] > pred[j] + margin.
    """
    b = int(pred.shape[0])
    if b < 2:
        return pred.new_zeros(())

    pred_i = pred.unsqueeze(1)
    pred_j = pred.unsqueeze(0)
    target_i = target.unsqueeze(1)
    target_j = target.unsqueeze(0)
    gap = target_i - target_j
    mask = gap > float(min_target_gap)
    if not mask.any():
        return pred.new_zeros(())

    dif = pred_i - pred_j
    losses = F.relu(float(margin) - dif)
    return losses[mask].mean()


# Legacy 8-D handcrafted caption features.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", flags=re.ASCII)


def text_feature_vector(text: str) -> torch.Tensor:
    """
    Cheap text feature vector for adherence head conditioning.
    This is intentionally lightweight and deterministic.
    """
    text = (text or "").strip()
    toks = _TOKEN_RE.findall(text)
    n_chars = float(len(text))
    n_toks = float(len(toks))
    n_commas = float(text.count(","))
    n_emph_up = float(text.count("(") + text.count(")"))
    n_emph_down = float(text.count("[") + text.count("]"))
    n_digits = float(sum(ch.isdigit() for ch in text))
    n_upper = float(sum(ch.isupper() for ch in text))
    avg_tok_len = float(sum(len(t) for t in toks) / max(1, len(toks)))
    v = torch.tensor(
        [
            n_chars / 300.0,
            n_toks / 100.0,
            n_commas / 50.0,
            n_emph_up / 20.0,
            n_emph_down / 20.0,
            n_digits / 50.0,
            n_upper / 50.0,
            avg_tok_len / 20.0,
        ],
        dtype=torch.float32,
    )
    return v


# Text featurizers: ``handcrafted`` is the legacy 8-D vector (old checkpoints score identically), ``clip`` a
# frozen CLIP text embedding. Train and infer both build theirs via :func:`featurizer_from_config`.
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
        from utils.generation.guidance import _clip_feature_tensor

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


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def peek_vit_quality_config(ckpt_path: str | Path) -> dict[str, Any]:
    """
    Load only the embedded ``config`` dict from a ViT quality checkpoint (no model build).

    Used for book-pipeline preflight and DiT/ViT AR alignment checks. Returns ``{}`` if missing
    or unreadable.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get("config") or {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def load_vit_quality_checkpoint(
    ckpt_path: str | Path,
    *,
    use_ema: bool = False,
) -> tuple[ViTQualityAdherenceModel, dict[str, Any]]:
    """
    Load a ViT ``best.pt``-style checkpoint and return ``(model, config_dict)`` on CPU.
    """
    path = Path(ckpt_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}

    model_name = cfg.get("model_name", "vit_base_patch16_224")
    text_feat_dim = int(cfg.get("text_feat_dim", 8))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    fuse_dropout = float(cfg.get("fuse_dropout", 0.1))
    text_proj_dropout = float(cfg.get("text_proj_dropout", 0.0))
    backbone_grad_checkpointing = bool(cfg.get("backbone_grad_checkpointing", False))
    # Older checkpoints omit these -> full bidirectional DiT path (no AR side-info).
    use_ar_conditioning = bool(cfg.get("use_ar_conditioning", False))
    ar_cond_dim = int(cfg.get("ar_cond_dim", 4))
    timm_kw = cfg.get("timm_kwargs")
    if timm_kw is not None and not isinstance(timm_kw, dict):
        timm_kw = None

    model = build_vit_model(
        model_name=str(model_name),
        pretrained=False,
        text_feat_dim=text_feat_dim,
        hidden_dim=hidden_dim,
        use_ar_conditioning=use_ar_conditioning,
        ar_cond_dim=ar_cond_dim,
        fuse_dropout=fuse_dropout,
        text_proj_dropout=text_proj_dropout,
        backbone_grad_checkpointing=backbone_grad_checkpointing,
        timm_kwargs=timm_kw,
    )
    if use_ema and bool(ckpt.get("ema_state_dict")):
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["state_dict"], strict=True)

    model.eval()
    return model, cfg


def vit_model_parameter_report(model: ViTQualityAdherenceModel) -> dict[str, int]:
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


class ViTScorer:
    """A loaded checkpoint with the text featurizer and image normalisation it was trained with."""

    def __init__(self, ckpt_path: str | Path, device: str = "cuda", *, use_ema: bool = False) -> None:
        from torchvision import transforms

        wants_cuda = str(device).startswith("cuda")
        self.device = torch.device(device if torch.cuda.is_available() or not wants_cuda else "cpu")
        model, self.cfg = load_vit_quality_checkpoint(ckpt_path, use_ema=use_ema)
        self.model = model.to(self.device).eval()
        self.featurizer = featurizer_from_config(self.cfg, self.device)
        self.image_size = int(self.cfg.get("image_size", 224) or 224)
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
        )
        self._resize = transforms.Resize(
            (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR
        )

    def image_tensor(self, pil: Any) -> torch.Tensor:
        """Square-resized, normalised ``(3, S, S)`` tensor (the manifest-scoring view)."""
        return self.normalize(self._resize(pil.convert("RGB")))

    @torch.inference_mode()
    def __call__(
        self,
        images: torch.Tensor,
        captions: str | Sequence[str],
        num_ar_blocks: int = -1,
        *,
        tta: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Model outputs for a ``(B, 3, S, S)`` batch.

        ``captions`` is one caption shared by every image or one per image; ``tta`` averages with the
        horizontal flip.
        """
        from utils.modeling.architecture import ar_conditioning_vector, normalize_num_ar_blocks

        x = images.to(self.device)
        b = x.shape[0]
        if isinstance(captions, str):
            txt = self.featurizer(captions).to(self.device).unsqueeze(0).expand(b, -1)
        else:
            txt = self.featurizer.embed_many(list(captions)).to(self.device)
        ar = None
        if bool(self.cfg.get("use_ar_conditioning", False)):
            nb = normalize_num_ar_blocks(num_ar_blocks)
            ar = ar_conditioning_vector(nb, device=self.device, dtype=txt.dtype).unsqueeze(0).expand(b, -1)
        out = self.model(x, txt, ar)
        if tta:
            flipped = self.model(torch.flip(x, dims=[3]), txt, ar)
            out = {k: (v + flipped[k]) / 2 for k, v in out.items()}
        return out


@lru_cache(maxsize=1)
def cached_vit_scorer(ckpt_path: str, device: str = "cuda") -> ViTScorer:
    """Process-wide scorer reused across calls with the same checkpoint and device."""
    return ViTScorer(ckpt_path, device)
