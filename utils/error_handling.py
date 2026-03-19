"""
Enhanced error handling and logging utilities for SDX.
"""
import logging
import traceback
import functools
from typing import Any, Callable, Optional
from pathlib import Path
import torch


class SDXError(Exception):
    """Base exception for SDX-specific errors."""
    pass


class ModelLoadError(SDXError):
    """Error loading model checkpoint."""
    pass


class DatasetError(SDXError):
    """Error with dataset loading or processing."""
    pass


class InferenceError(SDXError):
    """Error during inference."""
    pass


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup enhanced logging with file and console handlers."""
    logger = logging.getLogger("sdx")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if log_dir provided
    if log_dir:
        log_path = Path(log_dir) / "sdx.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def log_gpu_memory(logger: logging.Logger, prefix: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.info(f"{prefix}GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")


def safe_execute(func: Callable, *args, logger: Optional[logging.Logger] = None, **kwargs) -> Any:
    """Safely execute a function with error logging."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
        raise


def retry_on_cuda_oom(max_retries: int = 3, reduce_batch_size: bool = True):
    """Decorator to retry function on CUDA OOM with optional batch size reduction."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            original_batch_size = kwargs.get('batch_size', None)
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        last_exception = e
                        torch.cuda.empty_cache()
                        
                        if reduce_batch_size and original_batch_size and attempt < max_retries - 1:
                            new_batch_size = max(1, original_batch_size // (2 ** (attempt + 1)))
                            kwargs['batch_size'] = new_batch_size
                            print(f"CUDA OOM detected, reducing batch size to {new_batch_size} (attempt {attempt + 1})")
                        else:
                            break
                    else:
                        raise
            
            raise last_exception
        return wrapper
    return decorator


def validate_checkpoint(ckpt_path: str) -> bool:
    """Validate checkpoint file integrity."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required_keys = ["config", "model"]
        
        for key in required_keys:
            if key not in ckpt and "ema" not in ckpt:
                return False
        
        return True
    except Exception:
        return False


def get_model_info(model: torch.nn.Module) -> dict:
    """Get detailed model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "device": next(model.parameters()).device if total_params > 0 else "unknown"
    }