# CUDA notes (`native/cuda`)

The **compiled** optional kernels live under `native/cpp/cuda/`:

| Library | Role |
|--------|------|
| `sdx_cuda_hwc_to_chw` | `hwc_to_chw.cu` — uint8 HWC → float NCHW |
| `sdx_cuda_ml` | `l2_normalize_rows.cu` — L2-normalize matrix rows (embeddings) |

- **Headers:** `native/cpp/include/sdx/hwc_to_chw.h`, `native/cpp/include/sdx/l2_normalize_rows.h`
- **Enable:** configure with `-DSDX_BUILD_CUDA=ON` (requires **CUDA Toolkit** / `nvcc` on PATH)

```bash
cd native/cpp
cmake -S . -B build -DSDX_BUILD_CUDA=ON
cmake --build build --config Release
```

**Python:** `sdx_native.cuda_hwc_to_chw`, `sdx_native.cuda_l2_normalize` (`maybe_l2_normalize_rows_cuda`, NumPy reference).

For **training**, keep using **PyTorch** (`tensor.cuda()`, `permute`, `to(torch.float32)`) — this kernel is for experiments, pre-PyTorch pipelines, or timing layout transforms without the autograd stack.

**Going faster in production:** `torch.compile`, **Triton**, or **cuDNN**-backed ops inside the dataloader usually beat a standalone memcpy + kernel for batched images.
