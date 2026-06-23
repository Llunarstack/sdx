"""Optional enhanced training loop (separate from main `train.py` path)."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import of enhanced trainer components to avoid dependency chain."""
    if name == "EnhancedTrainer":
        from .enhanced_trainer import EnhancedTrainer

        return EnhancedTrainer
    elif name == "EnhancedTrainingBatch":
        from .enhanced_trainer import EnhancedTrainingBatch

        return EnhancedTrainingBatch
    elif name == "create_enhanced_trainer":
        from .enhanced_trainer import create_enhanced_trainer

        return create_enhanced_trainer
    elif name == "build_train_config_from_args":
        from .train_args import build_train_config_from_args

        return build_train_config_from_args
    elif name == "build_train_arg_parser":
        from .train_cli_parser import build_train_arg_parser

        return build_train_arg_parser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnhancedTrainer",
    "create_enhanced_trainer",
    "EnhancedTrainingBatch",
    "build_train_config_from_args",
    "build_train_arg_parser",
]
