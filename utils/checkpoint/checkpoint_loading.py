from __future__ import annotations

from typing import Optional, Tuple

import torch
from models import DiT_models_text
from models.rae_latent_bridge import RAELatentBridge

from config import get_dit_build_kwargs


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
    checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint_data.get("config")
    if config is None:
        raise ValueError("Checkpoint must contain config")

    model_name = getattr(config, "model_name", "DiT-XL/2-Text")
    if reject_enhanced and str(model_name).startswith("EnhancedDiT"):
        raise ValueError(
            "This checkpoint is an EnhancedDiT model, which is not compatible with sample.py (DiT-Text/T5 sampler).\n"
            "Use scripts/enhanced/sample_enhanced.py instead:\n"
            '  python scripts/enhanced/sample_enhanced.py "<prompt>" --checkpoint <path_to_ckpt> --output out.png\n'
            "Or train a DiT-*-Text checkpoint with train.py to use sample.py."
        )

    model_builder = DiT_models_text.get(model_name) or DiT_models_text["DiT-XL/2-Text"]
    model = model_builder(**get_dit_build_kwargs(config, class_dropout_prob=0.0))
    model_state_dict = checkpoint_data.get("ema") or checkpoint_data.get("model")
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device).eval()

    rae_bridge = None
    rae_bridge_state_dict = checkpoint_data.get("rae_latent_bridge")
    if isinstance(rae_bridge_state_dict, dict) and rae_bridge_state_dict.get("to_dit.weight") is not None:
        rae_channels = int(rae_bridge_state_dict["to_dit.weight"].shape[1])
        rae_bridge = RAELatentBridge(rae_channels, 4)
        rae_bridge.load_state_dict(rae_bridge_state_dict, strict=True)
        rae_bridge = rae_bridge.to(device).eval()

    fusion_state_dict = checkpoint_data.get("text_encoder_fusion")
    if not isinstance(fusion_state_dict, dict):
        fusion_state_dict = None
    return model, config, rae_bridge, str(model_name), fusion_state_dict
