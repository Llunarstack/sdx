# CUDA kernels (`native/cpp/cuda`)

All optional GPU code lives **here**, built by CMake from `native/cpp` with `-DSDX_BUILD_CUDA=ON`.

| Library | Source | Role |
|--------|--------|------|
| `sdx_cuda_hwc_to_chw` | `hwc_to_chw.cu` | uint8 HWC → float NCHW |
| `sdx_cuda_ml` | `l2_normalize_rows.cu`, `style_pick_best.cu` | L2-normalize rows; pick best cosine vs query embedding |
| `sdx_cuda_flow_matching` | `flow_matching_velocity.cu` | `v = ε - x₀` (rectified-flow target) |
| `sdx_cuda_nf4` | `nf4_dequant.cu` | NF4 block dequant (matches `utils/quantization/nf4_codec.py`) |
| `sdx_cuda_sdpa_online` | `sdpa_online_softmax.cu` | Multi-head SDPA, head_dim=64, online softmax |
| `sdx_cuda_rmsnorm` | `rmsnorm_rows.cu` | Row-wise RMSNorm (float32) |
| `sdx_cuda_rope` | `rope_apply.cu` | In-place interleaved RoPE on Q/K |
| `sdx_cuda_silu_gate` | `silu_gate.cu` | Fused `silu(x) * gate` |
| `sdx_cuda_image_metrics` | `image_metrics.cu` | Image metric helpers |
| `sdx_cuda_gaussian_blur` | `gaussian_blur_latent.cu` | Latent Gaussian blur |
| `sdx_cuda_percentile_clamp` | `percentile_clamp.cu` | Percentile clamp |

Headers: `native/cpp/include/sdx/*.h` (e.g. `l2_normalize_rows.h`, `hwc_to_chw.h`).

## Build

```bash
cd native/cpp
cmake -S . -B build -DSDX_BUILD_CUDA=ON
cmake --build build --config Release
```

Or from repo root: `.\scripts\tools\native\build_native.ps1`

## Python

`sdx_native.cuda_hwc_to_chw`, `cuda_l2_normalize`, `cuda_style_pick_native`, `flow_matching_velocity_native`, `nf4_dequant_native`, `sdpa_online_native`, `rmsnorm_native`, `rope_apply_native`, `silu_gate_native`, etc.

Training should stay on **PyTorch**; these are optional fast paths and experiments.
