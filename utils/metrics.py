"""
Training metrics and progress tracking utilities.
"""
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import torch
import numpy as np


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    step: int
    epoch: int
    loss: float
    lr: float
    grad_norm: float
    time_per_step: float
    samples_per_second: float
    gpu_memory_gb: float
    
    # Optional validation metrics
    val_loss: Optional[float] = None
    val_steps: Optional[int] = None
    
    # Optional refinement metrics
    refinement_loss: Optional[float] = None
    refinement_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics_history: List[TrainingMetrics] = []
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = self.log_dir / "metrics.jsonl"
    
    def log_step(self, metrics: TrainingMetrics):
        """Log metrics for a training step."""
        self.metrics_history.append(metrics)
        
        # Update best losses
        if metrics.loss < self.best_loss:
            self.best_loss = metrics.loss
        
        if metrics.val_loss is not None and metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
        
        # Save to file
        if self.log_dir:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def get_recent_avg_loss(self, window: int = 100) -> float:
        """Get average loss over recent steps."""
        if not self.metrics_history:
            return float('inf')
        
        recent_metrics = self.metrics_history[-window:]
        return np.mean([m.loss for m in recent_metrics])
    
    def get_training_speed(self) -> Dict[str, float]:
        """Get training speed statistics."""
        if len(self.metrics_history) < 2:
            return {"samples_per_second": 0.0, "steps_per_minute": 0.0}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 steps
        avg_samples_per_sec = np.mean([m.samples_per_second for m in recent_metrics])
        avg_time_per_step = np.mean([m.time_per_step for m in recent_metrics])
        steps_per_minute = 60.0 / avg_time_per_step if avg_time_per_step > 0 else 0.0
        
        return {
            "samples_per_second": avg_samples_per_sec,
            "steps_per_minute": steps_per_minute
        }
    
    def estimate_time_remaining(self, current_step: int, total_steps: int) -> str:
        """Estimate remaining training time."""
        if len(self.metrics_history) < 5:
            return "Estimating..."
        
        recent_metrics = self.metrics_history[-10:]
        avg_time_per_step = np.mean([m.time_per_step for m in recent_metrics])
        
        remaining_steps = total_steps - current_step
        remaining_seconds = remaining_steps * avg_time_per_step
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.metrics_history:
            return {}
        
        total_time = time.time() - self.start_time
        total_steps = len(self.metrics_history)
        
        return {
            "total_steps": total_steps,
            "total_time_hours": total_time / 3600,
            "best_loss": self.best_loss,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float('inf') else None,
            "final_loss": self.metrics_history[-1].loss,
            "avg_samples_per_second": np.mean([m.samples_per_second for m in self.metrics_history]),
            "avg_gpu_memory_gb": np.mean([m.gpu_memory_gb for m in self.metrics_history])
        }
    
    def save_summary(self):
        """Save training summary to file."""
        if not self.log_dir:
            return
        
        summary = self.get_summary()
        summary_file = self.log_dir / "training_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


class ProgressBar:
    """Simple progress bar for training."""
    
    def __init__(self, total_steps: int, desc: str = "Training"):
        self.total_steps = total_steps
        self.desc = desc
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step: int, loss: float, lr: float, extra_info: Optional[Dict] = None):
        """Update progress bar."""
        self.current_step = step
        progress = step / self.total_steps
        bar_length = 40
        filled_length = int(bar_length * progress)
        
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        percent = progress * 100
        
        elapsed_time = time.time() - self.start_time
        if step > 0:
            eta = elapsed_time * (self.total_steps - step) / step
            eta_str = f"{int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}"
        else:
            eta_str = "--:--:--"
        
        info_str = f"loss={loss:.4f} lr={lr:.2e}"
        if extra_info:
            info_str += " " + " ".join([f"{k}={v}" for k, v in extra_info.items()])
        
        print(f'\r{self.desc}: |{bar}| {percent:.1f}% [{step}/{self.total_steps}] ETA: {eta_str} {info_str}', end='', flush=True)
    
    def close(self):
        """Close progress bar."""
        print()  # New line


def calculate_model_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """Estimate FLOPs for model forward pass (rough approximation)."""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Very rough FLOP estimation for transformer models
    # This is a simplified calculation and may not be accurate
    batch_size, seq_len, hidden_dim = input_shape
    
    # Attention: O(seq_len^2 * hidden_dim) per layer
    # FFN: O(seq_len * hidden_dim * ffn_dim) per layer
    # Rough estimate assuming ~24 layers and 4x FFN expansion
    
    attention_flops = seq_len * seq_len * hidden_dim * 24
    ffn_flops = seq_len * hidden_dim * (hidden_dim * 4) * 24
    
    total_flops = (attention_flops + ffn_flops) * batch_size
    
    return total_flops


def log_system_info():
    """Log system information for debugging."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}"] = {
                "name": gpu_props.name,
                "memory_gb": gpu_props.total_memory / (1024**3),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
            }
    
    return info