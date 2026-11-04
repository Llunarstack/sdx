"""Training throughput: CUDA/CPU backend tuning, fast data loading, latent caching and text-encode batching."""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


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


def configure_pytorch_attention_backends(*, logger: logging.Logger | None = None) -> None:
    """Enable fast SDPA kernels (Flash / memory-efficient) when PyTorch supports them."""
    import torch

    if not torch.cuda.is_available():
        return
    cuda_be = getattr(torch.backends, "cuda", None)
    if cuda_be is None:
        return
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        fn = getattr(cuda_be, name, None)
        if callable(fn):
            try:
                fn(True)
            except Exception:
                pass
    if logger:
        logger.info(
            "SDPA backends: flash=%s mem_efficient=%s math=%s",
            getattr(cuda_be, "flash_sdp_enabled", lambda: "?")(),
            getattr(cuda_be, "mem_efficient_sdp_enabled", lambda: "?")(),
            getattr(cuda_be, "math_sdp_enabled", lambda: "?")(),
        )


def configure_inference_cuda(
    *,
    cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """Fast defaults for sample.py / inference (TF32 + cuDNN benchmark + SDPA)."""
    configure_training_cuda_and_cpu(
        deterministic=False,
        cudnn_benchmark=cudnn_benchmark,
        enable_tf32=enable_tf32,
        cpu_num_threads=0,
        cpu_num_interop_threads=0,
        logger=logger,
    )
    configure_pytorch_attention_backends(logger=logger)


def configure_training_cuda_and_cpu(
    *,
    deterministic: bool,
    cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
    cpu_num_threads: int = 0,
    cpu_num_interop_threads: int = 0,
    logger: logging.Logger | None = None,
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

    configure_pytorch_attention_backends(logger=logger)


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
        "(or run: python -m scripts.tools trainers expandable-train <train.py args>)."
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


def encode_neg_and_style_embeddings(
    neg_caps: Sequence[str],
    styles: Sequence[str],
    *,
    style_embed_dim: int,
    encode_fn,
    batch_text_encode: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Encode negative and style caption lists; batched T5 when both are needed."""
    need_neg = any(n and str(n).strip() for n in neg_caps)
    need_style = bool(style_embed_dim) and any(s and str(s).strip() for s in styles)
    encoder_hidden_neg: torch.Tensor | None = None
    style_embedding: torch.Tensor | None = None
    if not need_neg and not need_style:
        return None, None

    if batch_text_encode and need_neg and need_style:
        neg_emb, style_emb = encode_text_multi_group(
            [[n or "" for n in neg_caps], [s or "" for s in styles]],
            encode_fn,
        )
        encoder_hidden_neg = neg_emb
        if style_emb is not None:
            style_embedding = style_emb.mean(dim=1)
    else:
        if need_neg:
            encoder_hidden_neg = encode_fn([n or "" for n in neg_caps])
        if need_style:
            style_embedding = encode_fn([s or "" for s in styles]).mean(dim=1)
    return encoder_hidden_neg, style_embedding


def to_train_device(
    x: torch.Tensor,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move only when needed (skips work when CUDA prefetch already placed tensors)."""
    if x.device != device:
        x = x.to(device, non_blocking=device.type == "cuda")
    if dtype is not None and x.dtype != dtype:
        x = x.to(dtype=dtype)
    return x


def create_adamw_optimizer(
    params,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    fused: bool = True,
) -> torch.optim.AdamW:
    """AdamW with fused CUDA kernel when supported (same numerics, faster step)."""
    kwargs = {"lr": lr, "weight_decay": weight_decay, "betas": betas}
    if fused and torch.cuda.is_available():
        kwargs["fused"] = True
        try:
            return torch.optim.AdamW(params, **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
    return torch.optim.AdamW(params, **kwargs)


def encode_text_multi_group(
    groups: Sequence[Sequence[str] | None],
    encode_fn,
) -> list[torch.Tensor | None]:
    """
    One ``encode_fn(captions)`` call for all non-empty caption groups.

    *encode_fn* must accept a list of strings and return ``(N, L, D)`` embeddings.
    Groups that are ``None`` or empty stay ``None`` in the output.
    """
    indexed: list[tuple[int, list[str]]] = []
    for i, g in enumerate(groups):
        if not g:
            continue
        caps = [c or "" for c in g]
        if caps:
            indexed.append((i, caps))

    out: list[torch.Tensor | None] = [None] * len(groups)
    if not indexed:
        return out

    flat: list[str] = []
    spans: list[tuple[int, int, int]] = []
    for i, caps in indexed:
        start = len(flat)
        flat.extend(caps)
        spans.append((i, start, len(flat)))

    emb = encode_fn(flat)
    for i, start, end in spans:
        out[i] = emb[start:end]
    return out


# ---------------------------------------------------------------------------
# Optimal worker count heuristic
# ---------------------------------------------------------------------------


def optimal_num_workers(
    dataset_size: int,
    batch_size: int,
    *,
    max_workers: int = 16,
    min_workers: int = 2,
) -> int:
    """
    Heuristic for num_workers based on CPU count and dataset size.

    Rules:
    - Never more than cpu_count - 1 (leave one for the main process)
    - Never more than max_workers (diminishing returns + memory pressure)
    - For small datasets (<1000 samples), fewer workers avoids overhead
    - For large datasets, scale with CPU count
    """
    cpu_count = os.cpu_count() or 4
    # Leave 1-2 CPUs for the main process and OS
    available = max(1, cpu_count - 2)
    # Scale: more workers for larger datasets
    if dataset_size < 500:
        suggested = min(2, available)
    elif dataset_size < 5000:
        suggested = min(4, available)
    else:
        suggested = min(available, max_workers)
    return max(min_workers, min(suggested, max_workers))


# ---------------------------------------------------------------------------
# PrefetchDataLoader: overlaps H2D transfer with GPU compute
# ---------------------------------------------------------------------------


class PrefetchDataLoader:
    """
    Wraps a DataLoader and prefetches the next batch to GPU using a CUDA stream.

    The key insight: while the GPU is processing batch N, we start the H2D
    transfer for batch N+1 on a separate CUDA stream. By the time the GPU
    finishes batch N, batch N+1 is already in VRAM.

    This eliminates H2D transfer from the critical path entirely.

    Usage:
        loader = PrefetchDataLoader(dataloader, device=device)
        for batch in loader:
            # batch is already on GPU
            loss = model(batch['pixel_values'])

    Note: tensors in the batch are moved to GPU. Non-tensor items (strings,
    lists) are left as-is.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        *,
        non_blocking: bool = True,
    ) -> None:
        self.dataloader = dataloader
        self.device = device
        self.non_blocking = non_blocking

    def _to_device(self, x: Any, *, _key: str | None = None) -> Any:
        if isinstance(x, torch.Tensor):
            if _key == "grounding_mask_valid":
                return x
            return x.to(self.device, non_blocking=self.non_blocking)
        if isinstance(x, dict):
            return {k: self._to_device(v, _key=k) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            converted = [self._to_device(v) for v in x]
            return type(x)(converted)
        return x

    def __iter__(self) -> Iterator[Any]:
        if not torch.cuda.is_available() or self.device.type != "cuda":
            # CPU fallback: just yield batches as-is
            for batch in self.dataloader:
                yield self._to_device(batch)
            return

        stream = torch.cuda.Stream(device=self.device)
        loader_iter = iter(self.dataloader)

        # Prefetch first batch
        try:
            next_batch = next(loader_iter)
        except StopIteration:
            return

        with torch.cuda.stream(stream):
            next_batch_gpu = self._to_device(next_batch)

        while True:
            # Wait for prefetch to complete
            torch.cuda.current_stream().wait_stream(stream)
            batch = next_batch_gpu

            # Start prefetching the next batch
            try:
                next_batch = next(loader_iter)
                with torch.cuda.stream(stream):
                    next_batch_gpu = self._to_device(next_batch)
            except StopIteration:
                next_batch_gpu = None

            yield batch

            if next_batch_gpu is None:
                break

    def __len__(self) -> int:
        return len(self.dataloader)


# ---------------------------------------------------------------------------
# build_fast_dataloader: factory with optimal defaults
# ---------------------------------------------------------------------------


def build_fast_dataloader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    *,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    collate_fn: Callable | None = None,
    sampler: Any = None,
    seed: int = 42,
) -> PrefetchDataLoader:
    """
    Build a high-throughput DataLoader with all performance optimizations enabled.

    Key settings:
    - persistent_workers=True: workers stay alive between epochs (no fork overhead)
    - pin_memory=True: tensors allocated in pinned memory for faster H2D transfer
    - prefetch_factor=2: each worker prefetches 2 batches ahead
    - PrefetchDataLoader: overlaps H2D with GPU compute via CUDA streams

    Args:
        dataset: Any PyTorch Dataset.
        batch_size: Samples per batch.
        device: Target device (used for prefetch stream).
        num_workers: If None, auto-detected from CPU count and dataset size.
        prefetch_factor: Batches to prefetch per worker (2-4 typical).
        persistent_workers: Keep workers alive between epochs.
        pin_memory: Use pinned memory for faster H2D transfer.
        shuffle: Shuffle dataset each epoch.
        drop_last: Drop incomplete last batch.
        collate_fn: Custom collate function.
        sampler: Custom sampler (overrides shuffle).
        seed: Worker seed for reproducibility.

    Returns:
        PrefetchDataLoader wrapping the configured DataLoader.
    """
    n_workers = num_workers
    if n_workers is None:
        n_workers = optimal_num_workers(len(dataset), batch_size)

    # pin_memory only makes sense on CUDA
    use_pin = pin_memory and torch.cuda.is_available() and device.type == "cuda"

    # persistent_workers requires num_workers > 0
    use_persistent = persistent_workers and n_workers > 0

    # prefetch_factor requires num_workers > 0
    pf = prefetch_factor if n_workers > 0 else None

    def worker_init_fn(worker_id: int) -> None:
        import random

        import numpy as np

        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=use_pin,
        persistent_workers=use_persistent,
        prefetch_factor=pf,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    return PrefetchDataLoader(loader, device)


# ---------------------------------------------------------------------------
# LatentCacheBuilder: precompute VAE latents to disk
# ---------------------------------------------------------------------------


class LatentCacheBuilder:
    """
    Precompute and cache VAE latents to disk so training never touches the VAE.

    For a dataset of N images, this runs the VAE encoder once and saves
    each latent as a .pt file. During training, the dataset loads the .pt
    file instead of decoding the image and running the VAE.

    Speedup: 3-10x depending on VAE size and image resolution.

    Usage:
        builder = LatentCacheBuilder(vae, cache_dir="./latent_cache", device=device)
        builder.build(dataset, batch_size=16, show_progress=True)
        # Then set dataset.latent_cache_dir = "./latent_cache"
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        cache_dir: str,
        device: torch.device,
        *,
        latent_scale: float = 0.18215,
        ae_type: str = "kl",
    ) -> None:
        self.vae = vae
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.latent_scale = float(latent_scale)
        self.ae_type = str(ae_type)

    def _latent_path(self, image_path: str) -> Path:
        stem = Path(image_path).stem
        return self.cache_dir / f"{stem}.pt"

    def _encode_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of pixel values to latents."""
        with torch.inference_mode():
            nbc = self.device.type == "cuda"
            pv = pixel_values.to(self.device, dtype=torch.float32, non_blocking=nbc)
            enc = self.vae.encode(pv)
            if hasattr(enc, "latent_dist"):
                z = enc.latent_dist.sample() * self.latent_scale
            else:
                z = enc.latent * self.latent_scale
        return z.cpu()

    def build(
        self,
        dataset: Any,
        batch_size: int = 16,
        show_progress: bool = True,
        skip_existing: bool = True,
        num_workers: int = 4,
    ) -> dict[str, int]:
        """
        Build the latent cache for all samples in the dataset.

        Args:
            dataset: Dataset with samples having 'path' and 'pixel_values' keys.
            batch_size: Encoding batch size.
            show_progress: Print progress.
            skip_existing: Skip samples that already have a cached latent.
            num_workers: DataLoader workers for image loading.

        Returns:
            Dict with 'cached', 'skipped', 'failed' counts.
        """
        from torch.utils.data import DataLoader as _DL

        stats = {"cached": 0, "skipped": 0, "failed": 0}

        loader = _DL(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

        total = len(dataset)
        processed = 0

        for batch in loader:
            paths = batch.get("path", [])
            pixel_values = batch.get("pixel_values")

            if pixel_values is None:
                stats["failed"] += len(paths)
                continue

            # Check which need encoding
            needs_encode = []
            encode_indices = []
            for i, path in enumerate(paths):
                lp = self._latent_path(str(path))
                if skip_existing and lp.exists():
                    stats["skipped"] += 1
                else:
                    needs_encode.append(str(path))
                    encode_indices.append(i)

            if encode_indices:
                try:
                    pv_subset = pixel_values[encode_indices]
                    latents = self._encode_batch(pv_subset)
                    for j, path in enumerate(needs_encode):
                        lp = self._latent_path(path)
                        torch.save(latents[j], lp)
                        stats["cached"] += 1
                except Exception as e:
                    stats["failed"] += len(encode_indices)
                    if show_progress:
                        print(f"  Cache build error: {e}")

            processed += len(paths)
            if show_progress and processed % (batch_size * 10) == 0:
                pct = 100 * processed / max(total, 1)
                print(
                    f"  Latent cache: {processed}/{total} ({pct:.1f}%) "
                    f"cached={stats['cached']} skipped={stats['skipped']} failed={stats['failed']}"
                )

        if show_progress:
            print(
                f"Latent cache complete: cached={stats['cached']} skipped={stats['skipped']} failed={stats['failed']}"
            )

        return stats


# ---------------------------------------------------------------------------
# AsyncCaptionPrefetcher: background T5 encoding
# ---------------------------------------------------------------------------


class AsyncCaptionPrefetcher:
    """
    Prefetch T5 text encodings in a background thread.

    T5 encoding is CPU-bound (or GPU-bound on a separate stream) and can
    overlap with the GPU's forward/backward pass. This class runs encoding
    in a background thread and caches results so the training loop never
    waits for text encoding.

    Usage:
        prefetcher = AsyncCaptionPrefetcher(encode_fn, queue_size=4)
        prefetcher.start()

        for batch in loader:
            captions = batch['captions']
            # Submit encoding job (non-blocking)
            prefetcher.submit(captions, batch_id=step)
            # ... do GPU work ...
            # Get result (blocks only if not ready yet)
            text_emb = prefetcher.get(batch_id=step)
    """

    def __init__(
        self,
        encode_fn: Callable[[list[str]], torch.Tensor],
        queue_size: int = 4,
    ) -> None:
        self.encode_fn = encode_fn
        self._input_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._output_cache: dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the background encoding thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread."""
        self._running = False
        try:
            self._input_queue.put_nowait(None)  # sentinel
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def submit(self, captions: list[str], batch_id: int) -> None:
        """Submit a list of captions for background encoding."""
        self._input_queue.put((batch_id, captions))

    def get(self, batch_id: int, timeout: float = 30.0) -> torch.Tensor | None:
        """
        Get the encoded tensor for batch_id.
        Blocks until encoding is complete or timeout is reached.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if batch_id in self._output_cache:
                    return self._output_cache.pop(batch_id)
            time.sleep(0.001)
        return None

    def _worker(self) -> None:
        while self._running:
            try:
                item = self._input_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            batch_id, captions = item
            try:
                result = self.encode_fn(captions)
                with self._lock:
                    self._output_cache[batch_id] = result
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "AsyncCaptionPrefetcher: encoding failed for batch_id=%s: %s",
                    batch_id,
                    exc,
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# DataLoaderProfiler: measure actual throughput
# ---------------------------------------------------------------------------


class DataLoaderProfiler:
    """
    Measure DataLoader throughput to empirically tune num_workers and prefetch_factor.

    Usage:
        profiler = DataLoaderProfiler(loader, device=device)
        stats = profiler.profile(num_batches=50)
        print(f"Throughput: {stats['samples_per_sec']:.1f} samples/sec")
        print(f"GPU idle: {stats['gpu_idle_pct']:.1f}%")
    """

    def __init__(self, loader: Any, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def profile(
        self,
        num_batches: int = 50,
        warmup_batches: int = 5,
    ) -> dict[str, float]:
        """
        Profile the loader for num_batches batches.

        Returns dict with:
        - samples_per_sec: actual throughput
        - batch_load_ms: average time to load one batch (CPU side)
        - gpu_idle_pct: estimated % of time GPU was waiting for data
        - total_samples: total samples processed
        """
        load_times: list[float] = []
        total_samples = 0
        batch_size = None

        loader_iter = iter(self.loader)

        # Warmup
        for _ in range(warmup_batches):
            try:
                next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)

        # Profile
        t_start = time.perf_counter()
        for _ in range(num_batches):
            t0 = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                batch = next(loader_iter)
            t1 = time.perf_counter()

            load_times.append((t1 - t0) * 1000)  # ms

            # Count samples — handle dict, plain tensor, list/tuple of tensors
            if isinstance(batch, dict):
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        if batch_size is None:
                            batch_size = v.shape[0]
                        total_samples += v.shape[0]
                        break
            elif isinstance(batch, torch.Tensor):
                if batch_size is None:
                    batch_size = batch.shape[0]
                total_samples += batch.shape[0]
            elif isinstance(batch, (list, tuple)):
                for v in batch:
                    if isinstance(v, torch.Tensor):
                        if batch_size is None:
                            batch_size = v.shape[0]
                        total_samples += v.shape[0]
                        break

        t_end = time.perf_counter()
        total_time = t_end - t_start

        avg_load_ms = sum(load_times) / len(load_times) if load_times else 0.0
        samples_per_sec = total_samples / total_time if total_time > 0 else 0.0

        # Estimate GPU idle: if load time > 5ms, GPU was likely waiting
        # (rough heuristic — real measurement needs CUDA events)
        gpu_idle_pct = min(100.0, avg_load_ms / max(1.0, total_time * 1000 / num_batches) * 100)

        return {
            "samples_per_sec": samples_per_sec,
            "batch_load_ms": avg_load_ms,
            "gpu_idle_pct": gpu_idle_pct,
            "total_samples": float(total_samples),
            "total_time_sec": total_time,
            "num_batches": float(num_batches),
            "batch_size": float(batch_size or 0),
        }

    def suggest_config(self, stats: dict[str, float]) -> str:
        """Return a human-readable suggestion based on profiling results."""
        lines = [
            f"Throughput: {stats['samples_per_sec']:.1f} samples/sec",
            f"Batch load: {stats['batch_load_ms']:.1f}ms avg",
            f"GPU idle est: {stats['gpu_idle_pct']:.1f}%",
        ]
        if stats["gpu_idle_pct"] > 20:
            lines.append("-> GPU is waiting for data. Increase num_workers or prefetch_factor.")
        elif stats["gpu_idle_pct"] < 5:
            lines.append("-> Data loading is not the bottleneck. GPU is well-fed.")
        if stats["batch_load_ms"] > 50:
            lines.append("-> High batch load time. Consider latent caching (LatentCacheBuilder).")
        return "\n".join(lines)


def dataloader_perf_kwargs(
    *,
    num_workers: int,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    worker_init_fn: Callable | None = None,
) -> dict[str, Any]:
    """
    Shared DataLoader kwargs for train.py (pin_memory, prefetch, persistent workers).

    ``prefetch_factor`` is omitted when ``num_workers == 0``.
    """
    use_pin = pin_memory and torch.cuda.is_available()
    use_persistent = persistent_workers and num_workers > 0
    kw: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": use_pin,
        "persistent_workers": use_persistent,
    }
    if worker_init_fn is not None:
        kw["worker_init_fn"] = worker_init_fn
    if num_workers > 0 and prefetch_factor > 0:
        kw["prefetch_factor"] = int(prefetch_factor)
    return kw


def maybe_cuda_prefetch_dataloader(
    loader: DataLoader,
    device: torch.device,
    *,
    enabled: bool = True,
) -> PrefetchDataLoader | DataLoader:
    """Overlap H2D copies with GPU compute when training on CUDA."""
    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        return loader
    return PrefetchDataLoader(loader, device)


def resolve_training_num_workers(
    num_workers: int,
    dataset_size: int,
    batch_size: int,
    *,
    auto: bool = False,
) -> int:
    """Return effective worker count (``auto`` or negative values pick a heuristic)."""
    if auto or num_workers < 0:
        return optimal_num_workers(dataset_size, batch_size)
    return max(0, int(num_workers))
