# CUDA notes (`native/cuda`)

The **compiled** optional kernels live under `native/cpp/cuda/`:

| Library | Role |
|--------|------|
| `sdx_cuda_hwc_to_chw` | `hwc_to_chw.cu` — uint8 HWC → float NCHW |
| `sdx_cuda_ml` | `l2_normalize_rows.cu` — L2-normalize matrix rows (embeddings) |
| `sdx_cuda_flow_matching` | `flow_matching_velocity.cu` — elementwise `v = ε - x₀` (rectified-flow target) |
| `sdx_cuda_nf4` | `nf4_dequant.cu` — NF4 block **dequant** (table matches `utils/quantization/nf4_codec.py`) |
| `sdx_cuda_sdpa_online` | `sdpa_online_softmax.cu` — multi-head SDPA, **head_dim=64**, online softmax (Flash-style tiling; not FA3) |
| `sdx_cuda_rmsnorm` | `rmsnorm_rows.cu` — row-wise RMSNorm on float32 host matrices |
| `sdx_cuda_rope` | `rope_apply.cu` — in-place interleaved RoPE apply for Q/K host buffers |
| `sdx_cuda_silu_gate` | `silu_gate.cu` — fused `silu(x) * gate` elementwise |

- **Headers:** `hwc_to_chw.h`, `l2_normalize_rows.h`, `flow_matching_velocity.h`, `nf4_dequant.h`, `sdpa_online_softmax.h`, `rmsnorm_rows.h`, `rope_apply.h`, `silu_gate.h` under `native/cpp/include/sdx/`
- **Enable:** configure with `-DSDX_BUILD_CUDA=ON` (requires **CUDA Toolkit** / `nvcc` on PATH)

```bash
cd native/cpp
cmake -S . -B build -DSDX_BUILD_CUDA=ON
cmake --build build --config Release
```

**Python:** `sdx_native.cuda_hwc_to_chw`, `sdx_native.cuda_l2_normalize`, `sdx_native.flow_matching_velocity_native`, `sdx_native.nf4_dequant_native`, `sdx_native.sdpa_online_native`, `sdx_native.rmsnorm_native`, `sdx_native.rope_apply_native`, `sdx_native.silu_gate_native`.

For **training**, keep using **PyTorch** (`tensor.cuda()`, `permute`, `to(torch.float32)`) — this kernel is for experiments, pre-PyTorch pipelines, or timing layout transforms without the autograd stack.

**Going faster in production:** `torch.compile`, **Triton**, or **cuDNN**-backed ops inside the dataloader usually beat a standalone memcpy + kernel for batched images.
