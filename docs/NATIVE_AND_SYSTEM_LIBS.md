# Native & lower-level libraries — quality, training, prompt adherence

This doc ties **systems-level code** (Rust, Zig, C/C++, Go, and common native deps) to SDX goals: **output quality**, **training throughput/reliability**, and **prompt adherence**. For tools already in `native/`, see [../native/README.md](../native/README.md).

---

## How to use this doc

| Your goal | Jump to |
|-----------|---------|
| Faster / safer dataset QA, manifests, captions | [In-repo native tools](#in-repo-native-tools-sdx) |
| Sharper training data (decode, resize, color) | [Image I/O & preprocessing](#image-io--preprocessing) |
| Stricter captions before T5 | [Text, tokenization & adherence](#text-tokenization--prompt-adherence) |
| Cheaper or portable inference | [Export & runtime](#export--inference-runtimes) |
| Don’t fight PyTorch | [What stays in PyTorch](#what-stays-in-pytorch) |

---

## In-repo native tools (SDX)

These are **optional**; Python paths work without them. They help **data quality** and **operational hygiene**, which indirectly improve trained models and adherence (clean captions, no broken paths, deduped manifests).

| Tool | Lang | Helps with |
|------|------|------------|
| `sdx-jsonl-tools` | Rust | **stats**, **validate**, **prompt-lint**, **image-paths**, **dup-image-paths**, **`file-md5`** (streaming MD5 = Python `hashlib`; used by `data_quality.py` dedup), **file-fnv** — caption length, structure, duplicates |
| `sdx-noise-schedule` | Rust | **linear** / **cosine** VP-DDPM schedule CSV (analysis vs Python `diffusion/` schedules) |
| `sdx-linecrc` | Zig | Streaming **FNV-1a** over manifest bytes — cache keys, “did the dataset change?” |
| `sdx-pathstat` | Zig | Fast **path → size / missing** for image lists from Rust **image-paths** |
| `libsdx_latent` | C++ | **Latent grid / patch-token math** (C ABI) — consistent geometry with `dit_variant_compare`, ctypes |
| `sdx_line_stats` | C++ | **Byte + newline count** for huge manifests (no JSON parse); `sdx_native.line_stats_native` |
| `sdx_cuda_hwc_to_chw` | CUDA/C++ | Optional HWC→NCHW float kernel (`-DSDX_BUILD_CUDA=ON`); `sdx_native.cuda_hwc_to_chw` |
| `sdx_cuda_flow_matching` | CUDA/C++ | Optional elementwise flow target `v = ε − x₀`; `sdx_native.flow_matching_velocity_native` |
| `sdx_cuda_nf4` | CUDA/C++ | NF4 block dequant (pairs with `utils/quantization/nf4_codec.py`); `sdx_native.nf4_dequant_native` |
| `sdx_cuda_sdpa_online` | CUDA/C++ | Online-softmax SDPA, head_dim **64** (research kernel); `sdx_native.sdpa_online_native` |
| Mojo `native/mojo` | Mojo + Python | Optional **Modular** kernels / `mojopy` CLI launcher |
| `sdx-manifest` | Go | **JSONL merge** + dedupe by image key |
| `sdx_native.jsonl_manifest_pure` | Python | Zero-build manifest **stats** + **prompt-lint** (replaces former Node `*.mjs`) |

**Python bridge:** `native/python/sdx_native/` → shims under `utils/native/` ([native/python/README.md](../native/python/README.md)).

**Wiring:** `scripts/tools/data/data_quality.py` (MD5 dedup tries `sdx_native.native_tools.maybe_rust_file_md5_hex` first; `--no-native-md5` forces Python-only), `manifest_paths.py`, `jsonl_merge.py`, `op_preflight.py`, etc. ([native/README.md § Python integration](../native/README.md#python-integration-repo-root-on-pythonpath)).

---

## Image I/O & preprocessing

**Why it matters for quality:** bad decode, wrong gamma, aggressive JPEG, or inconsistent resize/crop **teaches the wrong image–text alignment**. Training speed also depends on decode + resize being cheap.

| Library / stack | Typical use | Lang / binding |
|-----------------|-------------|----------------|
| **libvips** | Fast resize, thumbnail pipelines, many formats | C with `pyvips` |
| **OpenCV** | Augmentations, color transforms, landmark checks | C++ → `cv2` |
| **libjpeg-turbo** / **MozJPEG** | Correct, fast JPEG encode/decode | C |
| **stb_image** / **image** (Rust) | Embedded decode in custom loaders | C / Rust |
| **WebP / AVIF** stacks | Modern lossy formats for storage | Various |

**Prompt adherence angle:** crop/aspect policies should match what you describe in captions (e.g. “full body” vs tight crop). Keep **one policy** in code and document it next to `data/t2i_dataset.py` crop modes.

---

## Text, tokenization & prompt adherence

**Why it matters:** adherence starts **before** DiT — consistent **normalization**, **caption structure**, and **lint** reduce “model ignores tag #17” effects.

| Library / stack | Typical use | Notes |
|-----------------|-------------|--------|
| **Hugging Face `tokenizers`** | Fast BPE/SentencePiece in Rust | Same family as T5/CLIP tokenizers in Python |
| **sentencepiece** | Unigram/BPE training & encode | C++ core; used across many encoders |
| **ICU** | Unicode normalize, casefold, locale-safe text | C/C++/Java; useful for **multilingual** caption cleanup |
| **RE2** / **Rust `regex`** | Bounded-time patterns for tag extraction | Safer than naive regex on untrusted strings |
| **Aho–Corasick / `aho-corasick` (Rust)** | Many keyword triggers (like `infer_content_controls` ideas) | Fast offline lint of huge JSONL |
| **xxHash / BLAKE3** | Per-caption or per-row fingerprints | Dedup near-dupe captions |

**SDX Python side:** `data/caption_utils.py`, `data/vector_index_sampler.py` (in-memory / optional **Qdrant** similarity sampling), `utils/prompt/content_controls.py`, `utils/prompt/neg_filter.py`, `utils/training/ladd_distillation.py` (LADD-style distillation + latent D stubs), `utils/quantization/nf4_codec.py` (NF4 quant/dequant core), `scripts/tools/preview_generation_prompt.py` — native tools complement these with **bulk** JSONL checks, not replacement for model conditioning.

**Implemented in-repo (Python, stdlib + optional xxhash):**

| Module / tool | Role |
|----------------|------|
| [`native/python/sdx_native/text_hygiene.py`](../native/python/sdx_native/text_hygiene.py) | **NFKC** normalization, zero-width strip, comma-segment trim, **SHA256** or **xxhash** caption fingerprints, pos/neg token overlap helper |
| [`native/python/sdx_native/text_hygiene.py`](../native/python/sdx_native/text_hygiene.py) | Canonical Python bridge for `sdx_native.text_hygiene` |
| [`scripts/tools/data/caption_hygiene.py`](../scripts/tools/data/caption_hygiene.py) | JSONL CLI: `--report-dups`, `--report-overlap`, `--normalize-samples` |
| Training | `train.py --caption-unicode-normalize` → `TrainConfig.caption_unicode_normalize` → `Text2ImageDataset(caption_unicode_normalize=True)` |

---

## Hashing, dedup & storage

| Library | Use |
|---------|-----|
| **Zstandard (`zstd`)** | Compress JSONL shards; fast train-side streaming |
| **SQLite / DuckDB** | Index manifests (path → caption hash, labels) for QA dashboards |
| **FNV / xxHash** (see in-repo Zig/Rust) | Cheap change detection |

---

## Export & inference runtimes

| Runtime | Use |
|---------|-----|
| **ONNX Runtime** | Portable inference for **VAE**, small nets, or exported helpers |
| **TensorRT / OpenVINO** | GPU/CPU-optimized inference (ops-specific) |

These do **not** replace the main PyTorch training loop in SDX today; they are options if you export subgraphs for **edge deployment** or **pre/post pipelines** (e.g. fixed resize graph).

---

## What stays in PyTorch

Keep **conv / attention / diffusion steps** in PyTorch (and vendor kernels it already uses: **cuDNN**, **cuBLAS**, **FlashAttention**-class ops where installed). Rewriting the DiT forward pass in raw C for “speed” is rarely the first win versus:

- better **data** (native QA + image stack above),
- **batching**, **compile** (`torch.compile`), **mixed precision**,
- **latent cache** (`scripts/training/precompute_latents.py`),
- **caption hygiene** (Rust/Node lint + Python controls).

---

## Integration patterns (recommended)

1. **CLI / subprocess** — Rust/Zig/Go binaries called from Python (already used for JSONL tools). Simplest ops story.
2. **C ABI + `ctypes` / `cffi`** — `libsdx_latent` pattern for small numeric APIs without GIL-heavy Python loops.
3. **PyO3 / maturin** (future) — If a hot path must live in Rust **and** be imported as a module; higher maintenance than CLI.

---

## See also

- [native/README.md](../native/README.md) — build commands and examples  
- [PROMPT_STACK.md](PROMPT_STACK.md) — inference text pipeline  
- [CODEBASE.md](CODEBASE.md) — `data/`, `utils/prompt/`, training entrypoints  
- [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md) — what quality problems look like at the model level  
- [HARDWARE.md](HARDWARE.md) — when I/O and CPU preprocessing matter  
