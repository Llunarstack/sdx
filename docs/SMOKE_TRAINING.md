# Smoke training (prove the pipeline works)

Use this when you want a **real** `train.py` run on a **tiny** dataset and **small** DiT—without claiming production quality.

## 1. Create a minimal dataset

From the repo root:

```bash
python scripts/tools/make_smoke_dataset.py --out data/smoke_tiny
```

This writes a few synthetic PNGs and `.txt` captions under **`data/smoke_tiny/train/`** (folder layout required by `Text2ImageDataset`).

## 2. One training step (fastest check)

Requires **CUDA** (see `train.py`). First run may download T5 + VAE from Hugging Face—ensure disk space and network.

```bash
python train.py ^
  --data-path data/smoke_tiny ^
  --results-dir results/smoke_run ^
  --model DiT-B/2-Text ^
  --image-size 256 ^
  --global-batch-size 1 ^
  --no-compile ^
  --num-workers 0 ^
  --dry-run
```

`--dry-run` forces **one optimizer step** then exit (see `train.py`).

Linux / macOS: use `\` instead of `^` for line continuation, or put the command on one line.

## 3. A few steps (slightly stronger test)

```bash
python train.py ^
  --data-path data/smoke_tiny ^
  --results-dir results/smoke_run ^
  --model DiT-B/2-Text ^
  --image-size 256 ^
  --global-batch-size 1 ^
  --max-steps 5 ^
  --no-compile ^
  --num-workers 0 ^
  --log-every 1
```

## Tips for low VRAM (e.g. ~16 GB)

- Prefer **`DiT-B/2-Text`** over XL/Supreme for smoke tests.
- Keep **`--image-size 256`** (or lower if your pipeline allows; align with docs).
- Use **`--no-compile`** if compile causes issues on your setup.
- **`--global-batch-size 1`** is fine for smoke; increase later when things work.
- If you still OOM, see [HARDWARE.md](HARDWARE.md) and turn off optional extras (MoE, REPA, large AR blocks) in config.

## What “success” means

- Process starts, loads encoders/VAE, runs forward + backward at least once, exits without crash.
- You are **not** expecting good images from synthetic data—only that the **stack is wired**.

For forward-only checks without data, use `python scripts/tools/dev/quick_test.py`.
