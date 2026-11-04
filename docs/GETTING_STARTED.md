# Getting Started with SDX

New to SDX? This guide explains what it does and gets you running in minutes.

---

## What Does SDX Do?

SDX is a tool for training custom image generation models on your own data.

**Basic workflow:**
- Input: Your images + text descriptions
- Process: Train a diffusion transformer model
- Output: A model that generates images from text

**Advanced capabilities:**
- Precise per-step generation control (Holy Grail scheduling)
- Create novel artistic styles beyond artist imitation (Style Genome)
- Automatic quality filtering for difficult prompts (TCIS)
- Modern training techniques (flow matching, preference learning, distillation)

---

## Quick Start (30 seconds)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate your first image (uses pretrained weights)
python demo.py

# Output is saved to out.png
```

---

## Core Concepts

### Training (train.py)

Train a model on your image dataset.

Input:
- Image folder with captions (or JSONL manifest)
- Configuration (batch size, learning rate, epochs)

Output:
- Trained checkpoint file (best.pt)

```bash
python train.py --data-path my_images/ --results-dir outputs/
```

Features:
- Flow matching (faster, newer approach)
- DPO training (learn from preference feedback)
- LoRA adapters (efficient training)
- Multi-GPU support (distributed training)

### Sampling (sample.py)

Generate images from a trained model.

Input:
- Text prompt
- Checkpoint file

Output:
- Generated image

```bash
python sample.py --ckpt outputs/best.pt --prompt "a cat wearing sunglasses" --out result.png
```

Features:
- Holy Grail scheduling (per-step adaptation)
- TCIS quality filtering (hard prompt reliability)
- Style Genome (novel aesthetic generation)
- LoRA blending (combine multiple models)

### Style Genome

Create original visual aesthetics instead of artist imitation.

```bash
# Generate 6 novel styles for a prompt
python -m scripts.tools explore_styles \
  --prompt "samurai warrior" --genomes 6 --mode apocalypse
```

Five style axes:
- Palette: colors and dominant hues
- Line: stroke mechanics and contours
- Surface: texture and material properties
- Camera: composition and perspective
- Signature: recurring visual motifs

---

## First Training Run

### Step 1: Prepare Data

Create a folder with images and captions:

```
my_dataset/
  ├── image1.png
  ├── image1.txt          # caption: "a red car"
  ├── image2.png
  ├── image2.txt          # caption: "a blue bicycle"
  └── ...
```

Or create a manifest.jsonl:

```json
{"image_path": "image1.png", "caption": "a red car"}
{"image_path": "image2.png", "caption": "a blue bicycle"}
```

### Step 2: Verify GPU Setup

```bash
python -m toolkit.env_health
```

Checks:
- CUDA availability
- VRAM (minimum 16GB recommended)
- Library installation

### Step 3: Start Training

```bash
python train.py \
  --data-path my_dataset/ \
  --results-dir results/ \
  --flow-matching-training \
  --epochs 20
```

Training process:
1. Images loaded and encoded to latent space
2. Model initialized
3. Training loop executes (loss decreases over time)
4. Checkpoints saved every 100 steps
5. Best checkpoint saved to results/best.pt

Timeline:
- 100 images on RTX 3090: ~20 hours
- 100 images on RTX 4090: ~5 hours
- Scales with number of images and GPU capability

### Step 4: Generate Images

```bash
python sample.py \
  --ckpt results/best.pt \
  --prompt "your description here" \
  --out result.png
```

Sample configurations:

```bash
# High quality (slower)
python sample.py --ckpt results/best.pt \
  --prompt "a red car, professional photo" \
  --steps 50 --cfg-scale 8.0 --out car.png

# Fast generation
python sample.py --ckpt results/best.pt \
  --prompt "a red car" \
  --steps 20 --cfg-scale 5.0 --out car_fast.png

# With style variation
python sample.py --ckpt results/best.pt \
  --prompt "a red car" \
  --invent-styles 1 --style-chaos-level 0.8 --out car_styled.png
```

---

## Configuration Parameters

Key training flags (full list: `python train.py --help`):

| Parameter | Purpose | Example |
|-----------|---------|---------|
| --data-path | Image dataset location | my_dataset/ |
| --results-dir | Checkpoint output directory | results/ |
| --epochs | Training passes through data | 20 |
| --global-batch-size | Images per training step | 4 |
| --lr | Training speed | 5e-5 |
| --flow-matching-training | Use flow matching objective | (recommended) |
| --no-bf16 | Turn off bf16 (on by default: faster, less memory) | — |
| --no-grad-checkpoint | Turn off gradient checkpointing (on by default) | — |

---

## Sampling Parameters

| Parameter | Purpose | Range | Notes |
|-----------|---------|-------|-------|
| --steps | Number of generation steps | 20-60 | Higher = better quality, slower |
| --cfg-scale | Prompt adherence strength | 5.0-15.0 | Higher = follow prompt strictly |
| --holy-grail-preset | Adaptive scheduling mode | auto, high_quality | auto recommended |
| --num | Batch generate N images | 1-10 | Multi-image generation |
| --pick-best | Auto quality filtering | auto | Uses ViT model to score |

---

## Common Use Cases

### Fine-tune Pretrained Model

Train on your data using a pretrained checkpoint as initialization:

```bash
python train.py \
  --data-path my_dataset/ \
  --init-from pretrained.pt \
  --results-dir results/
```

Training converges faster with pretrained weights.

### Blend Multiple Trained Models

Train separate models, then mix them:

```bash
# Train style-specific models
python train.py --data-path anime_images/ --results-dir anime_model/
python train.py --data-path photo_images/ --results-dir photo_model/

# Generate with both models blended
python sample.py \
  --ckpt anime_model/best.pt \
  --lora photo_model/best.pt:0.5:style \
  --prompt "a character"
```

### Hard Prompt Generation (Text, Layouts)

For difficult prompts (text in images, exact layouts), use quality filtering:

```bash
python -m scripts.tools hybrid_dit_vit_generate \
  --ckpt results/best.pt \
  --vit-ckpt vit_quality_model.pt \
  --prompt "poster with text HELLO WORLD" \
  --num 10 --pick-best auto
```

---

## Troubleshooting

**CUDA out of memory**
- Reduce --global-batch-size to 1 or 2
- Use --image-size 256 (smaller images)
- Grad checkpointing is on by default (reduces memory)

**"No module named X"**
- Reinstall: pip install -r requirements.txt

**Training too slow**
- Enable --flow-matching-training (faster)
- Keep bf16 on (default; don't pass --no-bf16)
- Reduce --epochs
- Use a larger batch size if VRAM allows

**Generated images are low quality**
- Train longer: increase --epochs (50+ recommended)
- Use more data (minimum 100 images)
- Write detailed captions

---

## Advanced Topics

For deeper exploration:

- [Training Techniques](GUIDES.md) - Flow matching, DPO, distillation
- [Sampling Guide](HOW_GENERATION_WORKS.md#holy-grail-adaptive-sampling-how-it-fits-together) - Holy Grail scheduling, TCIS
- [Style Genome](../README.md#style-genome-invent-original-looks) - Custom aesthetic creation
- [Codebase Architecture](CODEBASE.md) - How SDX is structured

Advanced sampling options:
- --style-inventor-mode insane for style variation
- --holy-grail-preset high_quality for maximum quality
- --flow-matching-sample for flow-trained checkpoints

---

## Frequently Asked Questions

**How much training data do I need?**

Minimum: 50-100 images. Recommended: 500-5000. Model quality scales with data quantity and quality.

**How long is training?**

Depends on GPU and data:
- RTX 3090 with 100 images: approximately 20 hours
- RTX 4090 with 100 images: approximately 5 hours

**Do I need a GPU?**

Training requires GPU (NVIDIA with CUDA). Sampling is 10-100x faster on GPU but possible on CPU.

**Can I use data from other artists?**

Only use images you have rights to. Respect copyright and intellectual property.

**Flow matching vs regular training?**

Flow matching is the newer approach. Use it for new projects unless you have specific requirements otherwise.

**Commercial use?**

SDX is Apache 2.0 licensed. Yes, commercial use is permitted. See LICENSE file.

---

## Resources

- [Documentation Index](README.md) - Complete reference
- [Reproducibility](#reproducibility-improvements-53) - Same seed / deterministic runs
- [Main README](../README.md) - Feature overview and architecture
- [GitHub Repository](https://github.com/Llunarstack/sdx) - Source code and issues

---

## Summary

You now understand:
- What SDX does (training and sampling text-to-image models)
- How to prepare data
- Basic training and sampling workflows
- Advanced features available

Start with the quick demo, then train on your own data. The codebase is designed to be readable—review train.py and sample.py to understand the full pipeline.

---

## PC specs for training SDX (consumer hardware, no datacenter GPUs)

Use this as a shopping list. All tiers assume **NVIDIA** GPUs (CUDA + xformers). AMD is possible with ROCm but the diffusion stack is most tested on NVIDIA.

---

### VRAM guide (single GPU)

| GPU VRAM | DiT-XL/2-Text (256px)      | DiT-P/2-Text (256px)       | Notes |
|----------|----------------------------|-----------------------------|--------|
| **12 GB** | Batch 1, grad checkpoint   | Too tight                   | Minimum to train at all |
| **16 GB** | Batch 2–4, grad checkpoint | Batch 1–2, grad checkpoint  | Usable |
| **24 GB** | Batch 6–12                 | Batch 4–8                   | **Sweet spot** |
| **32 GB+**| Batch 12+ or 512px         | Batch 8+ or 512px           | Future-proof |

T5-XXL and VAE are frozen; the DiT is what you train. Gradient checkpointing is on by default and saves a lot of VRAM.

---

### Tier 1: Minimum (train DiT-XL, batch 1–2)

- **GPU**: NVIDIA with **12–16 GB VRAM**  
  - e.g. RTX 4070 (12 GB), RTX 4060 Ti 16 GB, RTX 3080 (10/12 GB), RTX 4080 (16 GB)
- **CPU**: 6-core+ (e.g. Ryzen 5 5600, i5-12400)
- **RAM**: **32 GB** system RAM
- **Storage**: **500 GB NVMe** SSD (dataset + checkpoints)
- **PSU**: 650–750 W (match your GPU TDP)
- **Config**: `--global-batch-size 4` to 8, gradient checkpointing (on by default), keep `image_size=256`. Use `DiT-L/2-Text` or `DiT-B/2-Text` if you need a larger effective batch.

---

### Tier 2: Recommended (comfortable training, good throughput)

- **GPU**: **NVIDIA RTX 4090 (24 GB)** or **RTX 3090 / 3090 Ti (24 GB)**  
  - Best consumer option for this codebase; 24 GB lets you run batch 6–12 (DiT-XL) or 4–8 (DiT-P).
- **CPU**: 8-core+ (e.g. Ryzen 7 7700, i7-13700)
- **RAM**: **64 GB** (helps with large datasets and many workers)
- **Storage**: **1–2 TB NVMe** (Gen4 if possible)
- **PSU**: **850–1000 W** (4090 can spike 450 W+; headroom for CPU and rest)
- **Cooling**: Good case airflow or AIO; 4090 runs hot under long training.
- **Config**: `--global-batch-size 32` to 64 (single 4090), `--num-workers 8`, `image_size=256`. You can try 512px with a smaller batch.

---

### Tier 3: High-end (faster runs, DiT-P, or 512px)

- **GPU**: **2× RTX 4090 (24 GB each)** with NVLink (if you want multi-GPU)  
  - Or a single **RTX 4090** with the Tier 2 setup is already strong; second GPU doubles throughput with DDP.
- **CPU**: 12-core+ (e.g. Ryzen 9 7900X, i7-14700K) so data loading doesn’t bottleneck 2 GPUs.
- **RAM**: **128 GB** (large datasets, many workers, no swap)
- **Storage**: **2 TB NVMe** (Gen4); optional second SSD for datasets only.
- **PSU**: **1200–1500 W** for 2× 4090 + high-end CPU.
- **Motherboard**: PCIe 4.0/5.0, two x16 slots if going dual-GPU.
- **Config**: `torchrun --nproc_per_node=2 train.py ... --global-batch-size 64` to 128, or train DiT-P/2-Text at 256px with batch 8+ per GPU.

---

### What to buy (single-GPU “recommended” build)

| Part      | Example (buy current equivalent if newer) |
|-----------|-------------------------------------------|
| **GPU**   | NVIDIA RTX 4090 24 GB                     |
| **CPU**   | AMD Ryzen 7 7700 or Intel i7-13700         |
| **MB**    | B650 (AM5) or B760 (LGA1700), 2× M.2       |
| **RAM**   | 64 GB DDR5 (2×32 GB), 5600+ MT/s            |
| **SSD**   | 1–2 TB NVMe Gen4 (e.g. Samsung 990, WD SN850) |
| **PSU**   | 850–1000 W 80+ Gold (reputable brand)       |
| **Case**  | Mid-tower with good airflow                |
| **Cooling** | Good tower cooler or 240mm AIO for CPU; GPU stock is OK |

This gets you **24 GB VRAM**, so you can train DiT-XL and DiT-P at 256px with healthy batch sizes and no datacenter hardware.

---

### Optional tweaks to fit your GPU

- **12 GB VRAM**: Use `DiT-B/2-Text` or `DiT-L/2-Text`, `--global-batch-size 4`, `--no-compile` if it OOMs, keep gradient checkpointing on. Consider `--image-size 256` and fewer workers.
- **16 GB VRAM**: `DiT-XL/2-Text` with `--global-batch-size 8` to 16, gradient checkpointing on. Avoid `DiT-P/2-Text` at large batch or use batch 1–2.
- **24 GB VRAM**: Defaults are fine; you can raise `--global-batch-size` until you’re just under VRAM limit (watch for CUDA OOM).

All of the above assume **no cloud/datacenter** (no A100/H100); these specs are for a **single desktop or dual-GPU desktop** you can go buy and build.

---

### Storage for huge datasets (Rule34, Danbooru, e621, Gelbooru, + GIF/video frames)

When you’re pulling a **shit ton** of images from boorus (Rule34, Danbooru, e621, Rule34.xyz, Gelbooru), filtering to non–AI-generated/non–AI-assisted, and **extracting frames from GIFs and videos**, plan storage like this.

#### Per-item rough sizes

| Content | Typical size (approx) |
|--------|------------------------|
| **Booru image** (JPEG, web resolution) | 200 KB–800 KB each |
| **Booru image** (PNG or high-res) | 1–4 MB each |
| **Caption** (.txt or JSONL line) | 1–5 KB per image |
| **One extracted frame** (JPEG, 512–1024px) | 100–400 KB |
| **Precomputed latent** (one image, 256px → 32×32×4 float32) | ~16 KB |
| **Checkpoint** (DiT-XL + EMA, bf16) | ~5–6 GB each |

Use **~500 KB per image** as a round number for mixed booru JPEGs; **~2 MB** if you keep a lot of PNG/high-res. Use **~200 KB per frame** for extracted video/GIF frames at 512–1024px JPEG.

#### By dataset size (images + frames only)

| Scale | Image count (approx) | Raw images (500 KB avg) | + latent cache | + 20% headroom |
|-------|----------------------|--------------------------|----------------|----------------|
| **100K** | 100,000 | ~50 GB | +~1.6 GB | **~65 GB** |
| **500K** | 500,000 | ~250 GB | +~8 GB | **~320 GB** |
| **1M** | 1,000,000 | ~500 GB | +~16 GB | **~620 GB** |
| **2M** | 2,000,000 | ~1 TB | +~32 GB | **~1.3 TB** |
| **5M** | 5,000,000 | ~2.5 TB | +~80 GB | **~3.2 TB** |

If a large share is PNG/high-res, multiply the “raw” column by about **2–3×**. If you use mostly web-sized JPEGs, the table is in the right ballpark.

#### Video / GIF frame extraction

- **Frames per minute** (e.g. 1 fps): 60 frames × ~200 KB ≈ **12 MB/min** of video.
- **1 hour of video** at 1 fps: 60 × 12 MB ≈ **720 MB**.
- **10 hours**: ~**7 GB**; **100 hours**: ~**72 GB**.

So: **video/GIF storage = (hours of video) × ~0.7–1 GB** at 1 fps, 512–1024px JPEG frames. Higher fps or resolution → more.

#### Total storage recommendation

| Your situation | Suggested storage |
|----------------|--------------------|
| **100K–300K images**, little video | **500 GB – 1 TB** NVMe (dataset + latents + checkpoints + OS) |
| **500K–1M images**, or lots of video frames | **1.5 – 2 TB** NVMe (or 1 TB SSD + 2 TB HDD for cold images) |
| **1M–2M+ images**, heavy frame extraction | **2 – 4 TB** (NVMe for active training, HDD or second SSD for rest) |
| **Multi-million + hundreds of hours of video** | **4 – 8 TB** (2× 2 TB NVMe or  4 TB SSD + HDD for archive) |

Leave **20–30% free** on the drive you train from (so the latent cache and DataLoader don’t sit on a full disk). Precomputing latents (`python -m scripts.tools precompute_latents`) and training with `--latent-cache-dir` cuts I/O and lets you keep **only latents + captions** on the fast SSD and **raw images** on a bigger/cheaper drive if you want.

#### Full scrape: every non-AI image and video from all those sites

If you’re downloading **every** image and video that’s non-AI / non–AI-assisted from **Rule34, Danbooru, e621, Rule34.xyz, Gelbooru** (and similar), you’re building a full archive. Scale is much larger than a curated training subset.

| What you’re storing | Rough scale | Storage (approx) |
|----------------------|-------------|-------------------|
| **All images** (all sites, non-AI only) | 10M–30M+ images (sites have millions each; dedup across sites) | **5–15 TB** at 500 KB/img; **15–50 TB** if keeping full-res/PNG |
| **All videos** (raw files) | 100K–500K+ videos at 50–500 MB each | **10–50+ TB** |
| **Extracted frames only** (no raw video) | e.g. 50K–200K hours at ~1 GB/hr | **50–200 TB** if you keep every frame; **5–20 TB** at 1 fps subsample |
| **Captions / JSONL** | Same count as images + frames | Tens of GB (negligible next to media) |

So for **everything**:

- **Images only** (every non-AI image, web or full-res): **~10–50 TB** depending on resolution and dedup.
- **Images + raw videos**: **~20–60 TB** (videos dominate).
- **Images + extracted frames** (no raw video, 1 fps): add **~5–20 TB** depending on hours of video.
- **Images + raw videos + extracted frames**: **~30–100+ TB**.

**So yes: for a full archive of every non-AI image and video from all those sites, planning for 50–100 TB is reasonable.** Get there with a NAS or multi-drive array (e.g. 8× 12 TB or 6× 18 TB HDDs in RAID or JBOD). Keep a smaller fast SSD (2–4 TB) for active training (latent cache + current subset or symlinks) and point the loader at the big archive. Precomputing latents and training with `--latent-cache-dir` lets you train from the cache on the fast disk while the full image set lives on the large pool.

If you only care about **training** (not keeping a permanent copy of every asset), you can subset by score/tags/date and stay in the **4–8 TB** range; 100 TB is for “I want the full archive stored.”

---

## Smoke training (prove the pipeline works)

Use this when you want a **real** `train.py` run on a **tiny** dataset and **small** DiT—without claiming production quality.

### 1. Create a minimal dataset

From the repo root:

```bash
python -m scripts.tools make_smoke_dataset --out data/smoke_tiny
```

This writes a few synthetic PNGs and `.txt` captions under **`data/smoke_tiny/train/`** (folder layout required by `Text2ImageDataset`).

### 2. One training step (fastest check)

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

### 3. A few steps (slightly stronger test)

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

### Tips for low VRAM (e.g. ~16 GB)

- Prefer **`DiT-B/2-Text`** over XL/Supreme for smoke tests.
- Keep **`--image-size 256`** (or lower if your pipeline allows; align with docs).
- Use **`--no-compile`** if compile causes issues on your setup.
- **`--global-batch-size 1`** is fine for smoke; increase later when things work.
- If you still OOM, see [HARDWARE.md](#pc-specs-for-training-sdx-consumer-hardware-no-datacenter-gpus) and turn off optional extras (MoE, REPA, large AR blocks) in config.

### What “success” means

- Process starts, loads encoders/VAE, runs forward + backward at least once, exits without crash.
- You are **not** expecting good images from synthetic data—only that the **stack is wired**.

For forward-only checks without data, use `python -m scripts.tools quick_test`.

---

## Reproducibility (IMPROVEMENTS 5.3)

How to get **same command → same results** when possible.

---

### Sampling (`sample.py`)

- **`--seed N`** — Fixes PyTorch RNG so the same noise is used for the initial latent.
- **`--deterministic`** — Enables:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - `torch.use_deterministic_algorithms(True, warn_only=True)` when supported

Use both for reproducible images:

```bash
python sample.py --ckpt .../best.pt --prompt "a cat" --seed 42 --deterministic --out out.png
```

**Limitations:** Some CUDA ops (e.g. scatter, certain reductions) can remain non-deterministic. If you see a runtime warning about a non-deterministic op, the same seed may still give slightly different results. For full bit-identical runs, use a single GPU and the same CUDA/cuDNN version.

---

### Training (`train.py`)

- **`--seed N`** — Sets `global_seed`; used for dataset split, dataloader, and initial model init.
- **`--deterministic`** — Enables:
  - Same as above (cudnn, CUBLAS, `use_deterministic_algorithms`)
  - Worker seeds in the DataLoader when `worker_init_fn` is set

```bash
python train.py --data-path ... --passes 1 --deterministic --seed 42
```

**Limitations:** Multi-GPU (DDP) can introduce non-determinism (order of gradients). For closest reproducibility, train with a single GPU and `--deterministic`.

---

### Checklist

| Goal | Sampling | Training |
|------|-----------|----------|
| Same image per run | `--seed N --deterministic` | — |
| Same training curve | — | `--seed N --deterministic`, single GPU |

For full bit-identical reproducibility, document your env: Python version, PyTorch/CUDA/cuDNN versions, and any `PYTHONHASHSEED`.

---

## Fast training and sampling (no quality trade-offs)

These settings improve **throughput** without changing loss math, schedulers, or default precision.

### Training (`train.py`)

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

### Sampling (`sample.py`)

```bash
python sample.py --ckpt results/.../best.pt --prompt "..." --out out.png --compile-inference
```

Also enabled automatically on CUDA: TF32, cuDNN benchmark, fast SDPA backends (`configure_inference_cuda`).

T5 embeddings are cached for repeated prompts (`_t5_cache` in `sample.py`). Cond+uncond are encoded in **one** T5 forward when not using segmented layout.

**Inference CFG:** `gaussian_diffusion` uses batched cond/uncond (`utils/generation/guidance.py`) — ~2× fewer DiT forwards per step when CFG is on.

**CLIP guard/monitor:** `clip_alignment.py` caches the CLIP model per process (no reload every step).

### Environment

- Install matching **xformers** for your torch build (attention falls back to SDPA if mismatched).
- On Windows CUDA OOM with free VRAM: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see `python -m scripts.tools train_with_expandable_segments`).

### See also

- [local_ci_mirror.md](#mirror-github-ci-locally) — verify changes locally
- [CODEBASE.md](CODEBASE.md) — module map

---

## Mirror GitHub CI locally

Same checks as [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (minus `actions/checkout`).

### Setup

```bash
cd /path/to/sdx
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install ruff pytest basedpyright
pip install -r requirements.txt
```

### Commands

```bash
python -m ruff format --check .
python -m ruff check .

OUT=/tmp/basedpyright.json
python -m basedpyright --outputjson \
  native/_experimental/python/sdx_native/native_tools.py \
  diffusion/schedules.py \
  examples/run_baseline_eval.py \
  utils/runtime.py \
  training/cli.py \
  utils/generation/sample_cli_parser.py \
  utils/generation/sample_core.py \
  > "$OUT" || true
python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8-sig')); s=d.get('summary') or {}; ec=int(s.get('errorCount',0)); wc=int(s.get('warningCount',0)); print(f'basedpyright: {ec} errors, {wc} warnings'); sys.exit(1 if ec else 0)" "$OUT"

python -m scripts.tools smoke_imports
python -m scripts.tools verify_doc_links

python -m pytest tests/ -q --tb=short
```

CI uses the same logic so **warnings do not fail the job**; only `errorCount > 0` fails. The `|| true` after the redirect is required: **basedpyright** can exit non-zero when only warnings exist, and **bash `-e`** (as in GitHub Actions) would otherwise stop before the JSON gate runs.

### Optional: pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Uses [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) (Ruff check + format).
