"""
Host **CUDA / CPU throughput** tuning for DiT-style training.

- **cuDNN benchmark**: trades kernel search time each shape for fastest conv (when not deterministic).
- **TF32**: faster matmul/Conv on Ampere+; tiny numeric drift vs FP32 baseline.
- **CPU threads**: cap BLAS-ish parallelism so ``DataLoader`` workers keep cores.

 Allocator env (fragmentation): set ``PYTORCH_CUDA_ALLOC_CONF`` **before**
``import torch`` in the process — use ``scripts/tools/training/train_with_expandable_segments.ps1`` wrapper.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DevicePerfHints",
    "configure_training_cuda_and_cpu",
    "log_cuda_quick_stats",
    "log_training_perf_hints",
    "parallel_train_torchrun_example",
    "training_perf_hints",
]


@dataclass(frozen=True, slots=True)
class DevicePerfHints:
    """Printed once at startup; actionable strings only."""

    ddp_note: str
    vram_fragmentation_note: str
    datloader_note: str


def parallel_train_torchrun_example(n_gpu: int = 2, extra: str = "YOUR_TRAIN_ARGS") -> str:
    """Minimal ``torchrun`` one-liner (Linux + Windows CUDA)."""
    n = max(1, int(n_gpu))
    return f"python -m torch.distributed.run --standalone --nproc_per_node={n} train.py {extra}".strip()


def configure_training_cuda_and_cpu(
    *,
    deterministic: bool,
    cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
    cpu_num_threads: int = 0,
    cpu_num_interop_threads: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Apply cudnn / TF32 flags and optional CPU thread caps (called after CUDA device binding)."""
    import torch

    if cpu_num_threads > 0:
        torch.set_num_threads(int(cpu_num_threads))
        if logger:
            logger.info("torch.set_num_threads(%s)", cpu_num_threads)
    if cpu_num_interop_threads > 0:
        torch.set_num_interop_threads(int(cpu_num_interop_threads))
        if logger:
            logger.info("torch.set_num_interop_threads(%s)", cpu_num_interop_threads)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if logger:
            logger.info("cuDNN deterministic mode (benchmark off). TF32 overrides skipped for reproducibility skew.")
        return

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if logger:
        logger.info("cuDNN benchmark=%s", cudnn_benchmark)

    if not torch.cuda.is_available():
        return

    matmul_prior = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    cudnn_tf32_prior = getattr(torch.backends.cudnn, "allow_tf32", None)

    if enable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception as ex:
            if logger:
                logger.warning("Could not enable TF32 / matmul precision: %s", ex)
        if logger:
            logger.info(
                "TF32 enabled (matmul_allow=%s cudnn_allow=%s); use --no-tf32 for stricter FP32.",
                getattr(torch.backends.cuda.matmul, "allow_tf32", matmul_prior),
                getattr(torch.backends.cudnn, "allow_tf32", cudnn_tf32_prior),
            )
    else:
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        if logger:
            logger.info("TF32 disabled (highest FP32 precision for matmul where supported).")


def log_cuda_quick_stats(logger: logging.Logger, *, prefix: str = "") -> None:
    """Log device name(s) and free memory if NVIDIA runtime is reachable."""
    import torch

    if not torch.cuda.is_available():
        logger.info("%sCUDA not visible to this PyTorch build.", prefix)
        return
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    free_b, total_b = torch.cuda.mem_get_info(idx)

    def _to_gib(b):
        return round(float(b) / (1024.0**3), 2)

    logger.info(
        "%sGPU[%s] %s — free %.2f / total %.2f GiB",
        prefix,
        idx,
        name,
        _to_gib(free_b),
        _to_gib(total_b),
    )


def training_perf_hints(num_workers: int, world_size: int) -> DevicePerfHints:
    """Short reference text for manifests / logs."""
    cpu = os.cpu_count() or 8
    # Rule of thumb: workers scale with cpus / (world*(1+tiny))); stay bounded.
    sug = max(2, min(16, math.ceil(cpu / max(1, world_size)) - 1))
    ddp_note = (
        f"world_size={world_size}: sync batch via global_batch_size divisible by {world_size}; "
        f"launcher: `{parallel_train_torchrun_example(world_size)}` (adjust GPUs)."
    )
    vram_frag = (
        "If you see CUDA OOM with plenty of idle VRAM, try restarting with "
        "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` "
        "(see scripts/tools/training/train_with_expandable_segments.ps1)."
    )
    dl = (
        f"num_workers={num_workers}: if GPU underfed, tune toward ~{sug} workers on this CPU count "
        f"({cpu} logical); pair with prefetch + pin_memory in DataLoader."
    )
    return DevicePerfHints(ddp_note=ddp_note, vram_fragmentation_note=vram_frag, datloader_note=dl)


def log_training_perf_hints(logger: logging.Logger, num_workers: int, world_size: int) -> None:
    hints = training_perf_hints(num_workers, world_size)
    logger.info("Perf hint — DDP: %s", hints.ddp_note)
    logger.info("Perf hint — VRAM: %s", hints.vram_fragmentation_note)
    logger.info("Perf hint — DataLoader: %s", hints.datloader_note)
