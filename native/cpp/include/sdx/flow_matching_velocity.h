/**
 * Optional CUDA: flow-matching velocity residual v* = epsilon - x0 (elementwise, float32).
 * Matches the target in ``diffusion/flow_matching.py`` (rectified-flow-style path).
 */
#ifndef SDX_FLOW_MATCHING_VELOCITY_H
#define SDX_FLOW_MATCHING_VELOCITY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(_WIN32) && defined(SDX_CUDA_FM_BUILD)
#define SDX_CUDA_FM_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_FM_API __declspec(dllimport)
#else
#define SDX_CUDA_FM_API
#endif

/**
 * Host pointers: writes out[i] = eps[i] - x0[i] for i in [0, n).
 * Allocates device memory internally (demo / small tensors; for training use PyTorch).
 * @return 0 on success, non-zero on CUDA error.
 */
SDX_CUDA_FM_API int sdx_cuda_flow_velocity_residual_f32_host(const float *x0, const float *eps, float *out, int64_t n);

#ifdef __cplusplus
}
#endif

#endif /* SDX_FLOW_MATCHING_VELOCITY_H */
