# SDX native helpers

Optional accelerators around the Python stack. Nothing here is required: `train.py`, `sample.py` and the tests
run without any of it, and every native path has a NumPy / PyTorch / pure-Python fallback.
Broader map of native libraries: [docs/MODEL_STACK.md](../docs/MODEL_STACK.md#native--lower-level-libraries--quality-training-prompt-adherence).

```
native/
  rust/                         one crate: cdylib sdx_rs + CLIs sdx-jsonl-tools, sdx-noise-schedule
  cpp/                          C ABI: sdx_cpu (C++) and sdx_cuda (CUDA, optional); header include/sdx.h
  go/sdx-manifest/              JSONL merge / explore-stats / explore-dedupe
  _experimental/zig/            sdx-linecrc (manifest fingerprint), sdx-pathstat (path -> size)
  _experimental/python/sdx_native/
    native_tools.py             find built tools/libraries, run the Rust CLIs, FNV fingerprint, latent math, merge
    ffi.py                      ctypes bindings for sdx_cpu / sdx_cuda / sdx_rs (None when not built)
    text_hygiene.py             caption NFKC + zero-width strip, fingerprints, JSONL caption hygiene
```

Repo code imports `from utils.native import ...` (adds `native/_experimental/python` to `sys.path` lazily);
pytest gets the same path from `pyproject.toml` (`pythonpath`).

## What uses what

| Library / tool | Used by |
|----------------|---------|
| `sdx_rs` (Rust cdylib) | `diffusion/schedules.py` (alpha_cumprod, SNR, beta schedules), style token Jaccard (`utils/prompt/style.py`), highlight-clip fraction (`utils/native.quality_highlight_frac`), blob counting (eval count-adherence scorer), image kernels for `utils/ideas` (Floyd-Steinberg error diffusion, void-and-cluster blue noise, bilateral, 3x3 median, Kuwahara) |
| `sdx-jsonl-tools` | `stats`, `validate`, `prompt-lint`, `image-paths`, `dup-image-paths`, `file-fnv`, `file-md5`, `caption-len-buckets`; data QA tools, op preflight, book pipeline, image dedup MD5 |
| `sdx-noise-schedule` | linear / cosine VP-DDPM schedule CSV (noise-schedule export tool) |
| `sdx_cpu` | latent / patch-token geometry (`dit_variant_compare`), inference timestep finalization, squared-cosine betas, image metrics + component count (test-time pick), mask -> patch weights (part-aware training) |
| `sdx_cuda` | latent Gaussian blur (sampler), luma stats (test-time pick), style-embedding cosine pick |
| `sdx-manifest` (Go) | JSONL merge (`utils.native.merge_jsonl_files`), style explore stats / dedupe |
| `sdx-linecrc` (Zig) | manifest fingerprint (same output as `sdx-jsonl-tools file-fnv` and the Python fallback) |
| `sdx-pathstat` (Zig) | `image-paths` output -> `path<TAB>size<TAB>ok|missing` |

`python -m scripts.tools.dev.quick_test --show-native` (or `utils.native.native_stack_status()`) prints what is built.

## Build

One shot (C++, CUDA when `nvcc` is found, Rust):

```powershell
.\scripts\tools\native\build_native.ps1        # SDX_BUILD_CUDA=0|1 to force
bash scripts/tools/native/build_native.sh
.\scripts\tools\native\clean_native_builds.ps1 # remove build outputs
```

By hand:

```bash
cd native/rust && cargo build --release        # target/release/{sdx_rs.dll|libsdx_rs.so, sdx-jsonl-tools, sdx-noise-schedule}
cmake -S native/cpp -B native/cpp/build -DCMAKE_BUILD_TYPE=Release [-DSDX_BUILD_CUDA=ON]
cmake --build native/cpp/build --config Release
cd native/go/sdx-manifest && go build -o sdx-manifest .
cd native/_experimental/zig/sdx-linecrc && zig build -Doptimize=ReleaseFast   # same for sdx-pathstat
```

Outputs stay where the loaders look: `native/rust/target/{release,debug}`, `native/cpp/build[/Release|/Debug]`,
the Go binary next to its source, Zig in `zig-out/bin`. All are gitignored.

### Image kernels vs the NumPy fallbacks

`utils/ideas/kernels.py` calls these when `sdx_rs` is built and falls back to NumPy otherwise (same results;
`tests/test_ideas.py` checks parity, and CI builds the crate so those tests run). One 1024x1024 RGB float image,
four threads, Windows desktop CPU (2026-11-22):

| Kernel | Rust | NumPy | Speed-up |
|--------|------|-------|----------|
| Bilateral, radius 3 | 113 ms | 1319 ms | 12x |
| 3x3 median | 41 ms | 365 ms | 9x |
| Kuwahara, radius 3 | 20 ms | 704 ms | 36x |
| Floyd-Steinberg to 8 bit | 16 ms | ~7.1 s (pure Python) | ~460x |
| Blue-noise mask 64x64 (cached once) | 35 ms | 62 ms | 2x |

Floyd-Steinberg has no NumPy form (each pixel depends on the last), so without the build `img_error_diffusion`
uses the blue-noise dither instead.

## Manifest conventions

JSONL rows are objects with `image_path` (or `path` / `image`) and `caption` (or `text`), as in
`data/t2i_dataset.py`.

```bash
T=native/rust/target/release/sdx-jsonl-tools
$T stats data/manifest.jsonl
$T validate --min-caption-len 5 data/manifest.jsonl
$T prompt-lint --max-caption-tokens 250 data/manifest.jsonl
$T image-paths --sort data/manifest.jsonl | native/_experimental/zig/sdx-pathstat/zig-out/bin/sdx-pathstat
native/go/sdx-manifest/sdx-manifest merge -o merged.jsonl --dedupe-key image_path a.jsonl b.jsonl
```
