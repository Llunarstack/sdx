"""Compete-mode train recipe soft defaults (REPA + PixAI-class stack).

Import and merge into TrainConfig / CLI when starting a quality train run:

    from config.defaults.compete_train import apply_compete_train_defaults
    apply_compete_train_defaults(cfg)
"""

from __future__ import annotations

from typing import Any

from utils.modeling.model_paths import default_repa_e_vae_path, default_t5_path


def apply_compete_train_defaults(cfg: Any, *, aggression: str = "max") -> dict[str, Any]:
    """
    Soft-fill train knobs that are coded but default-off.

    Does not overwrite explicit non-default values.
    """
    applied: list[str] = []

    def soft(name: str, value: Any, unset: tuple[Any, ...]) -> None:
        if not hasattr(cfg, name):
            return
        cur = getattr(cfg, name)
        if cur in unset:
            setattr(cfg, name, value)
            applied.append(name)

    soft("repa_weight", 0.5 if aggression == "max" else 0.25, (0.0, 0, None))
    soft("train_shortcomings_mitigation", "all", ("none", "", None))
    soft("train_anatomy_guidance", "strong", ("none", "auto", "", None))
    # PixAI-class train stack: flow + REPA-E VAE + best available T5 TE.
    soft("flow_matching_training", True, (False, 0, None))
    soft("vae_model", default_repa_e_vae_path(), ("", None, "stabilityai/sd-vae-ft-mse"))
    soft("text_encoder", default_t5_path(), ("", None, "google/t5-v1_1-xxl"))
    return {"applied": applied, "aggression": aggression}


__all__ = ["apply_compete_train_defaults"]
