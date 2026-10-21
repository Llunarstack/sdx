"""Load ViT quality/adherence checkpoints; ``ViTScorer`` is the one inference path for every caller."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from vit_quality.model import ViTQualityAdherenceModel, build_vit_model

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

        from vit_quality.text_encoder import featurizer_from_config

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
        from utils.architecture.ar_block_conditioning import ar_conditioning_vector, normalize_num_ar_blocks

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
