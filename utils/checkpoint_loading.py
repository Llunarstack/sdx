from __future__ import annotations

from typing import Optional, Tuple

import torch

from config import get_dit_build_kwargs
from models import DiT_models_text
from models.rae_latent_bridge import RAELatentBridge


def load_dit_text_checkpoint(
    ckpt_path: str,
    device: str = "cuda",
    *,
    reject_enhanced: bool = False,
) -> Tuple[torch.nn.Module, object, Optional[RAELatentBridge], str, Optional[dict]]:
    """
    Load a DiT-Text checkpoint and optional RAE latent bridge.

    Returns: (model, config, rae_bridge_or_none, model_name, text_encoder_fusion_sd_or_none)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("Checkpoint must contain config")

    model_name = getattr(cfg, "model_name", "DiT-XL/2-Text")
    if reject_enhanced and str(model_name).startswith("EnhancedDiT"):
        raise ValueError(
            "This checkpoint is an EnhancedDiT model, which is not compatible with sample.py (DiT-Text/T5 sampler).\n"
            "Use sample_enhanced.py instead:\n"
            "  python sample_enhanced.py \"<prompt>\" --checkpoint <path_to_ckpt> --output out.png\n"
            "Or train a DiT-*-Text checkpoint with train.py to use sample.py."
        )

    model_fn = DiT_models_text.get(model_name) or DiT_models_text["DiT-XL/2-Text"]
    model = model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    state = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    rae_bridge = None
    sd_b = ckpt.get("rae_latent_bridge")
    if isinstance(sd_b, dict) and sd_b.get("to_dit.weight") is not None:
        rae_c = int(sd_b["to_dit.weight"].shape[1])
        rae_bridge = RAELatentBridge(rae_c, 4)
        rae_bridge.load_state_dict(sd_b, strict=True)
        rae_bridge = rae_bridge.to(device).eval()

    fusion_sd = ckpt.get("text_encoder_fusion")
    if not isinstance(fusion_sd, dict):
        fusion_sd = None
    return model, cfg, rae_bridge, str(model_name), fusion_sd
