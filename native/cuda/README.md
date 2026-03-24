# CUDA notes (`native/cuda`)

The **compiled** optional kernel lives next to the C++ tree so one CMake project can emit both CPU and GPU DLLs:

- **Source:** `native/cpp/cuda/hwc_to_chw.cu`
- **Header:** `native/cpp/include/sdx/hwc_to_chw.h`
- **Enable:** configure with `-DSDX_BUILD_CUDA=ON` (requires **CUDA Toolkit** / `nvcc` on PATH)

```bash
cd native/cpp
cmake -S . -B build -DSDX_BUILD_CUDA=ON
cmake --build build --config Release
```

**Python:** `from sdx_native.cuda_hwc_to_chw import maybe_u8_hwc_to_chw_f32_cuda, u8_hwc_to_chw_f32_numpy`

For **training**, keep using **PyTorch** (`tensor.cuda()`, `permute`, `to(torch.float32)`) — this kernel is for experiments, pre-PyTorch pipelines, or timing layout transforms without the autograd stack.

**Going faster in production:** `torch.compile`, **Triton**, or **cuDNN**-backed ops inside the dataloader usually beat a standalone memcpy + kernel for batched images.
