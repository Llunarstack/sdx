# Fast training and sampling (no quality trade-offs)

These settings improve **throughput** without changing loss math, schedulers, or default precision.

## Training (`train.py`)

Defaults are tuned for speed on CUDA (see `config/train_config.py`):

| Setting | Default | What it does |
|---------|---------|----------------|
| `auto_num_workers` | on | Picks DataLoader workers from CPU + dataset size |
| `prefetch_factor` | 4 | Workers prefetch batches ahead |
| `cuda_stream_prefetch` | on | Overlaps H2D copy with GPU step |
| `use_compile` | on | `torch.compile` DiT (`reduce-overhead`) |
| `use_bf16` | on | AMP bfloat16 (same quality as full FP32 for DiT) |
| `fused_adamw` | on | Fused CUDA AdamW optimizer step |
| `batch_text_encode` | on | One T5 forward when both neg + style are used |
| `enable_tf32` | on | Faster matmul on Ampere+ |

**Largest win when I/O-bound:** precompute latents then train from cache:

```bash
python -m scripts.training.precompute_latents --data-path datasets/train --out-dir ./latent_cache
python train.py --data-path datasets/train --latent-cache-dir ./latent_cache ...
```

**Disable a optimization (debug / repro):**

```bash
python train.py --no-compile --no-cuda-stream-prefetch --no-fused-adamw --no-batch-text-encode ...
```

**More compile time, often faster steady-state:**

```bash
python train.py --compile-mode max-autotune ...
```

## Sampling (`sample.py`)

```bash
python sample.py --ckpt results/.../best.pt --prompt "..." --out out.png --compile-inference
```

Also enabled automatically on CUDA: TF32, cuDNN benchmark, fast SDPA backends (`configure_inference_cuda`).

T5 embeddings are cached for repeated prompts (`_t5_cache` in `sample.py`). Cond+uncond are encoded in **one** T5 forward when not using segmented layout.

**Inference CFG:** `gaussian_diffusion` and `speculative_denoise` use batched cond/uncond (`utils/generation/cfg_batched.py`) — ~2× fewer DiT forwards per step when CFG is on.

**CLIP guard/monitor:** `clip_alignment.py` caches the CLIP model per process (no reload every step).

## Environment

- Install matching **xformers** for your torch build (attention falls back to SDPA if mismatched).
- On Windows CUDA OOM with free VRAM: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see `scripts/tools/training/train_with_expandable_segments.ps1`).

## See also

- [local_ci_mirror.md](local_ci_mirror.md) — verify changes locally
- [CODEBASE_GUIDE.md](../CODEBASE_GUIDE.md) — module map
