"""Public config API: training dataclass, DiT build kwargs, shared prompt defaults."""

from .defaults.prompt_domains import DEFAULT_NEGATIVE_PROMPT
from .train_config import TrainConfig, get_dit_build_kwargs

__all__ = ["TrainConfig", "get_dit_build_kwargs", "DEFAULT_NEGATIVE_PROMPT"]
