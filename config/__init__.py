from .train_config import TrainConfig, get_dit_build_kwargs

# Optional: prompt domains and defaults (used by sample.py, data pipeline)
try:
    from .prompt_domains import DEFAULT_NEGATIVE_PROMPT
except ImportError:
    DEFAULT_NEGATIVE_PROMPT = " "

__all__ = ["TrainConfig", "get_dit_build_kwargs", "DEFAULT_NEGATIVE_PROMPT"]
