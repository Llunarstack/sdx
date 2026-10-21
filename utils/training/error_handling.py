"""Training logging helpers: the ``sdx`` logger, GPU memory lines and model size summaries."""

import logging
from pathlib import Path

import torch


def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """``sdx`` logger with a console handler and, when ``log_dir`` is given, ``<log_dir>/sdx.log``."""
    logger = logging.getLogger("sdx")
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    if log_dir:
        log_path = Path(log_dir) / "sdx.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


def log_gpu_memory(logger: logging.Logger, prefix: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def get_model_info(model: torch.nn.Module) -> dict:
    """Parameter counts, fp32 size in MB and device."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
        "device": next(model.parameters()).device if total_params > 0 else "unknown",
    }
