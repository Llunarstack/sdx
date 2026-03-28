/**
 * Optional CUDA: fused SiLU + elementwise gate.
 * Computes ``out[i] = silu(x[i]) * gate[i]`` for contiguous float32 arrays.
 */
#ifndef SDX_SILU_GATE_H
#define SDX_SILU_GATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_SILU_GATE_BUILD)
#define SDX_CUDA_SILU_GATE_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_SILU_GATE_API __declspec(dllimport)
#else
#define SDX_CUDA_SILU_GATE_API
#endif

SDX_CUDA_SILU_GATE_API int sdx_cuda_silu_gate_f32_host(
    const float *x_host,
    const float *gate_host,
    float *out_host,
    int64_t n);

#ifdef __cplusplus
}
#endif

#endif
