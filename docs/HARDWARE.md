# PC specs for training SDX (consumer hardware, no datacenter GPUs)

Use this as a shopping list. All tiers assume **NVIDIA** GPUs (CUDA + xformers). AMD is possible with ROCm but the diffusion stack is most tested on NVIDIA.

---

## VRAM guide (single GPU)

| GPU VRAM | DiT-XL/2-Text (256px)      | DiT-P/2-Text (256px)       | Notes |
|----------|----------------------------|-----------------------------|--------|
| **12 GB** | Batch 1, grad checkpoint   | Too tight                   | Minimum to train at all |
| **16 GB** | Batch 2–4, grad checkpoint | Batch 1–2, grad checkpoint  | Usable |
| **24 GB** | Batch 6–12                 | Batch 4–8                   | **Sweet spot** |
| **32 GB+**| Batch 12+ or 512px         | Batch 8+ or 512px           | Future-proof |

T5-XXL and VAE are frozen; the DiT is what you train. Gradient checkpointing is on by default and saves a lot of VRAM.

---

## Tier 1: Minimum (train DiT-XL, batch 1–2)

- **GPU**: NVIDIA with **12–16 GB VRAM**  
  - e.g. RTX 4070 (12 GB), RTX 4060 Ti 16 GB, RTX 3080 (10/12 GB), RTX 4080 (16 GB)
- **CPU**: 6-core+ (e.g. Ryzen 5 5600, i5-12400)
- **RAM**: **32 GB** system RAM
- **Storage**: **500 GB NVMe** SSD (dataset + checkpoints)
- **PSU**: 650–750 W (match your GPU TDP)
- **Config**: `--global-batch-size 4` to 8, `--grad-checkpointing`, keep `image_size=256`. Use `DiT-L/2-Text` or `DiT-B/2-Text` if you need a larger effective batch.

---

## Tier 2: Recommended (comfortable training, good throughput)

- **GPU**: **NVIDIA RTX 4090 (24 GB)** or **RTX 3090 / 3090 Ti (24 GB)**  
  - Best consumer option for this codebase; 24 GB lets you run batch 6–12 (DiT-XL) or 4–8 (DiT-P).
- **CPU**: 8-core+ (e.g. Ryzen 7 7700, i7-13700)
- **RAM**: **64 GB** (helps with large datasets and many workers)
- **Storage**: **1–2 TB NVMe** (Gen4 if possible)
- **PSU**: **850–1000 W** (4090 can spike 450 W+; headroom for CPU and rest)
- **Cooling**: Good case airflow or AIO; 4090 runs hot under long training.
- **Config**: `--global-batch-size 32` to 64 (single 4090), `--num-workers 8`, `image_size=256`. You can try 512px with a smaller batch.

---

## Tier 3: High-end (faster runs, DiT-P, or 512px)

- **GPU**: **2× RTX 4090 (24 GB each)** with NVLink (if you want multi-GPU)  
  - Or a single **RTX 4090** with the Tier 2 setup is already strong; second GPU doubles throughput with DDP.
- **CPU**: 12-core+ (e.g. Ryzen 9 7900X, i7-14700K) so data loading doesn’t bottleneck 2 GPUs.
- **RAM**: **128 GB** (large datasets, many workers, no swap)
- **Storage**: **2 TB NVMe** (Gen4); optional second SSD for datasets only.
- **PSU**: **1200–1500 W** for 2× 4090 + high-end CPU.
- **Motherboard**: PCIe 4.0/5.0, two x16 slots if going dual-GPU.
- **Config**: `torchrun --nproc_per_node=2 train.py ... --global-batch-size 64` to 128, or train DiT-P/2-Text at 256px with batch 8+ per GPU.

---

## What to buy (single-GPU “recommended” build)

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

## Optional tweaks to fit your GPU

- **12 GB VRAM**: Use `DiT-B/2-Text` or `DiT-L/2-Text`, `--global-batch-size 4`, `--no-compile` if it OOMs, keep gradient checkpointing on. Consider `--image-size 256` and fewer workers.
- **16 GB VRAM**: `DiT-XL/2-Text` with `--global-batch-size 8` to 16, gradient checkpointing on. Avoid `DiT-P/2-Text` at large batch or use batch 1–2.
- **24 GB VRAM**: Defaults are fine; you can raise `--global-batch-size` until you’re just under VRAM limit (watch for CUDA OOM).

All of the above assume **no cloud/datacenter** (no A100/H100); these specs are for a **single desktop or dual-GPU desktop** you can go buy and build.

---

## Storage for huge datasets (Rule34, Danbooru, e621, Gelbooru, + GIF/video frames)

When you’re pulling a **shit ton** of images from boorus (Rule34, Danbooru, e621, Rule34.xyz, Gelbooru), filtering to non–AI-generated/non–AI-assisted, and **extracting frames from GIFs and videos**, plan storage like this.

### Per-item rough sizes

| Content | Typical size (approx) |
|--------|------------------------|
| **Booru image** (JPEG, web resolution) | 200 KB–800 KB each |
| **Booru image** (PNG or high-res) | 1–4 MB each |
| **Caption** (.txt or JSONL line) | 1–5 KB per image |
| **One extracted frame** (JPEG, 512–1024px) | 100–400 KB |
| **Precomputed latent** (one image, 256px → 32×32×4 float32) | ~16 KB |
| **Checkpoint** (DiT-XL + EMA, bf16) | ~5–6 GB each |

Use **~500 KB per image** as a round number for mixed booru JPEGs; **~2 MB** if you keep a lot of PNG/high-res. Use **~200 KB per frame** for extracted video/GIF frames at 512–1024px JPEG.

### By dataset size (images + frames only)

| Scale | Image count (approx) | Raw images (500 KB avg) | + latent cache | + 20% headroom |
|-------|----------------------|--------------------------|----------------|----------------|
| **100K** | 100,000 | ~50 GB | +~1.6 GB | **~65 GB** |
| **500K** | 500,000 | ~250 GB | +~8 GB | **~320 GB** |
| **1M** | 1,000,000 | ~500 GB | +~16 GB | **~620 GB** |
| **2M** | 2,000,000 | ~1 TB | +~32 GB | **~1.3 TB** |
| **5M** | 5,000,000 | ~2.5 TB | +~80 GB | **~3.2 TB** |

If a large share is PNG/high-res, multiply the “raw” column by about **2–3×**. If you use mostly web-sized JPEGs, the table is in the right ballpark.

### Video / GIF frame extraction

- **Frames per minute** (e.g. 1 fps): 60 frames × ~200 KB ≈ **12 MB/min** of video.
- **1 hour of video** at 1 fps: 60 × 12 MB ≈ **720 MB**.
- **10 hours**: ~**7 GB**; **100 hours**: ~**72 GB**.

So: **video/GIF storage = (hours of video) × ~0.7–1 GB** at 1 fps, 512–1024px JPEG frames. Higher fps or resolution → more.

### Total storage recommendation

| Your situation | Suggested storage |
|----------------|--------------------|
| **100K–300K images**, little video | **500 GB – 1 TB** NVMe (dataset + latents + checkpoints + OS) |
| **500K–1M images**, or lots of video frames | **1.5 – 2 TB** NVMe (or 1 TB SSD + 2 TB HDD for cold images) |
| **1M–2M+ images**, heavy frame extraction | **2 – 4 TB** (NVMe for active training, HDD or second SSD for rest) |
| **Multi-million + hundreds of hours of video** | **4 – 8 TB** (2× 2 TB NVMe or  4 TB SSD + HDD for archive) |

Leave **20–30% free** on the drive you train from (so the latent cache and DataLoader don’t sit on a full disk). Precomputing latents (`scripts/training/precompute_latents.py`) and training with `--latent-cache-dir` cuts I/O and lets you keep **only latents + captions** on the fast SSD and **raw images** on a bigger/cheaper drive if you want.

### Full scrape: every non-AI image and video from all those sites

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
