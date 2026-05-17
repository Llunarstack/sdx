# SDX native helpers (Rust, Zig, C, C++, CUDA, Go, Mojo)

**Broader map:** [docs/NATIVE_AND_SYSTEM_LIBS.md](../docs/NATIVE_AND_SYSTEM_LIBS.md).

Optional high-throughput utilities around the Python stack. `train.py` / `sample.py` do not require them.

## Layout (one place per language)

```
native/
  rust/          # CLI tools + cdylibs (prompt-ops, jsonl-tools, …)
  zig/           # linecrc, pathstat
  c/             # small C-only helpers (buffer stats, image metrics)
  cpp/           # main C ABI libraries (latent, timesteps, …)
    cuda/        # ALL .cu kernels live here (not a second top-level cuda/)
    include/sdx/ # C/C++ headers
    src/         # C++ sources
    build/       # gitignored — local CMake output only
  go/sdx-manifest/
  mojo/src/      # optional Mojo experiments
  python/sdx_native/  # ctypes + Python fallbacks
```

**CUDA:** only under `cpp/cuda/`. See [cpp/cuda/README.md](cpp/cuda/README.md).  
**Clean local builds:** `.\scripts\tools\native\clean_native_builds.ps1`

| Component | Role |
|-----------|------|
| **Rust** `rust/sdx-jsonl-tools` | JSONL: **stats**, **validate**, **prompt-lint**, **`image-paths`**, **`dup-image-paths`**, **`file-fnv`**, **`caption-len-buckets`**. |
| **Rust** `rust/sdx-noise-schedule` | CLI: **linear** / **cosine** VP-DDPM schedule tables (CSV) for analysis vs `diffusion/`. |
| **Rust** `rust/sdx-image-metrics` | CLI: **stats** (mean luma, clip ratio, Laplacian variance) and **count-blobs** connected components for quick count heuristics. |
| **Rust** `rust/sdx-prompt-ops` | cdylib: **caption merge/dedupe**, **pos/neg filter**, **style Jaccard**, **FNV fingerprint**, **multi-axis merge** (`prompt_ops_native`, `style_ops_native`). |
| **Rust** `rust/sdx-diffusion-math` | cdylib: **alpha_cumprod**, **SNR**, beta schedules (`sdx_native.diffusion_math_native`). |
| **Zig** `zig/sdx-linecrc` | Streaming **FNV-1a 64-bit** fingerprint over file bytes (manifest change detection). |
| **Zig** `zig/sdx-pathstat` | Given a **newline-separated path list**, print `path<TAB>size_bytes<TAB>ok|missing` (fast file-exists + size audit; pair with Rust `image-paths`). |
| **C** `c/` | Lightweight **C ABI** buffer stats + image metrics (no C++). |
| **C++** `cpp/` | `libsdx_latent`, line stats, FNV, image metrics, inference/beta — **C ABI** for ctypes. |
| **CUDA** `cpp/cuda/` | Optional GPU libs; [cpp/cuda/README.md](cpp/cuda/README.md); `-DSDX_BUILD_CUDA=ON`. |
| **Go** `go/sdx-manifest` | JSONL **merge**, **explore-stats**, **explore-dedupe** (style manifests). |
| **Mojo** `mojo/` | [mojo/README.md](mojo/README.md) — `sdx_style_tokens.mojo`, Pixi env. |
| **Python** `python/sdx_native/` | ctypes wrappers + pure-Python fallbacks. |

## Build (quick)

### Rust
```bash
cd native/rust/sdx-jsonl-tools
cargo build --release
# Windows: target/release/sdx-jsonl-tools.exe

cd native/rust/sdx-noise-schedule
cargo build --release
# target/release/sdx-noise-schedule linear --steps 1000

cd native/rust/sdx-image-metrics
cargo build --release
# target/release/sdx-image-metrics stats --image sample.png

cd native/rust/sdx-prompt-ops
cargo build --release
# Python: sdx_native.prompt_ops_native.get_prompt_ops_lib().available

cd native/rust/sdx-diffusion-math
cargo build --release
```

### Zig (0.13+)
```bash
cd native/zig/sdx-linecrc
zig build -Doptimize=ReleaseFast
# zig-out/bin/sdx-linecrc (or .exe)

cd native/zig/sdx-pathstat
zig build -Doptimize=ReleaseFast
# zig-out/bin/sdx-pathstat — stat paths from a file list
```

### C++
```bash
cd native/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# shared library: build/libsdx_latent.so or sdx_latent.dll + import lib on MSVC
# optional CUDA kernel (requires nvcc):
# cmake -S . -B build -DSDX_BUILD_CUDA=ON
# cmake --build build --config Release
```

### Go
```bash
cd native/go/sdx-manifest
go build -o sdx-manifest .
```

### Python JSONL stat / prompt-lint (no install)
```bash
python -m sdx_native.jsonl_manifest_pure stat data/manifest.jsonl
python -m sdx_native.jsonl_manifest_pure promptlint data/manifest.jsonl
```

### One-shot native build (C++ + optional CUDA + Rust)
```powershell
.\scripts\tools\native\build_native.ps1
# CPU-only C++:
#   $env:SDX_BUILD_CUDA='0'; .\scripts\tools\native\build_native.ps1
```
The script passes ``-DSDX_BUILD_CUDA=ON|OFF`` to CMake with **quoted** arguments so the cache never stores a bogus literal. Invalid or mismatched ``SDX_BUILD_CUDA`` lines in ``CMakeCache.txt`` trigger an automatic ``build/`` delete.
```bash
bash scripts/tools/native/build_native.sh
```

### Mojo (Pixi, linux-64 / WSL2)
See [mojo/README.md](mojo/README.md) and `native/mojo/pixi.toml`. On Windows use `install_mojo_wsl.ps1` or WSL `pixi install` under `native/mojo`.

## Use with SDX manifests

Manifest lines should be JSON objects with at least:
- `image_path` (or `path` / `image`)
- `caption` (or `text`)

Same conventions as `data/t2i_dataset.py` and `scripts/tools/data/data_quality.py`.

### Example: Rust validate + stats
```bash
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools stats data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools validate --min-caption-len 5 data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools prompt-lint --max-caption-tokens 250 data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools image-paths --sort data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools dup-image-paths data/manifest.jsonl
```

### Example: Zig fingerprint (pipe or file)
```bash
native/zig/sdx-linecrc/zig-out/bin/sdx-linecrc --file data/manifest.jsonl
type data\manifest.jsonl | native\zig\sdx-linecrc\zig-out\bin\sdx-linecrc.exe
```

### Example: path list → sizes (Zig `sdx-pathstat`)
```bash
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools image-paths data/manifest.jsonl > paths.txt
native/zig/sdx-pathstat/zig-out/bin/sdx-pathstat --file paths.txt
```

### Example: C ABI from Python (`ctypes`)
```python
import ctypes
from pathlib import Path

dll = ctypes.CDLL(str(Path("native/cpp/build/Release/sdx_latent.dll")))  # adjust for your platform
dll.sdx_num_patch_tokens.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
dll.sdx_num_patch_tokens.restype = ctypes.c_int
n = dll.sdx_num_patch_tokens(256, 8, 2)  # 32x32 latent -> 256 patch tokens for patch 2
print(n)
dll.sdx_latent_numel.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
dll.sdx_latent_numel.restype = ctypes.c_int
print(dll.sdx_latent_numel(4, 32, 32))  # 4 * 32 * 32 latent elements
```

### Example: merge JSONL (Go)
```bash
./sdx-manifest merge a.jsonl b.jsonl -o merged.jsonl --dedupe-key image_path
```

### Example: style explore manifest (Go + Python)
```bash
python -m scripts.tools explore_styles --prompt "samurai at dusk" --insane --native-stats
go build -o sdx-manifest ./native/go/sdx-manifest
./sdx-manifest explore-stats data/style_genomes/explore_manifest.jsonl
./sdx-manifest explore-dedupe -o data/style_genomes/train_styles.jsonl --key style_genome_id data/style_genomes/explore_manifest.jsonl
python -m scripts.tools explore_styles --show-native
```

### Style native Python API
```python
from utils.prompt.style_native import native_stack_status, pick_best_embedding_index
```

## Python integration (repo root on `PYTHONPATH`)

| Location | Role |
|----------|------|
| **`native/python/sdx_native/`** | **Source of truth:** `latent_geometry.py`, `text_hygiene.py`, `native_tools.py` (ctypes, CLI discovery, FNV, merge). |
| **`utils/native/__init__.py`** | Unified shim (adds `native/python` to `sys.path`) and re-exports `sdx_native` helpers for stable `from utils.nt import …` imports. |
| **`pyproject.toml`** | Pytest **`pythonpath`** includes `native/python` so `import sdx_native` works in tests. |

**Wired scripts**

- **`scripts/tools/data/data_quality.py`** — `--native-preflight` (Rust `stats` before filter), `--native-stats` (stats only), `--native-validate` (strict Rust validate).
- **`scripts/tools/data/caption_hygiene.py`** — JSONL **NFKC** samples, **caption fingerprint** dup report, pos/neg overlap (`sdx_native.text_hygiene`); pairs with `train.py --caption-unicode-normalize`.
- **`scripts/tools/ops/op_preflight.py`** — `--native-manifest-check` (Rust `stats` to stderr before coverage scan).
- **`scripts/tools/dit_variant_compare.py`** — prints **patch token count** via `libsdx_latent` or Python math (`--vae-scale`).
- **`scripts/tools/data/jsonl_merge.py`** — merge manifests; prefers Go **`sdx-manifest`** if built.
- **`scripts/tools/data/manifest_paths.py`** — **`image-paths`** / **`dup-image-paths`** (Rust); pipe to Zig **`sdx-pathstat`** for byte sizes.
- **`scripts/tools/dev/quick_test.py`** — `--show-native` prints discovery JSON (paths empty until you build tools).

## Why these languages?

- **Rust**: safe parallel I/O–friendly CLI tooling, easy distribution as one binary.
- **Zig**: tiny, fast, predictable builds; good for checksum/fingerprint pipes.
- **C++**: stable **C ABI** shared library for grid math or future SIMD without pulling the full PyTorch stack.
- **Go**: simple concurrent merges and ops on huge text files with minimal dependencies.
