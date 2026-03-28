/**
 * Optional CUDA: apply interleaved RoPE to Q/K host buffers in-place.
 * Layout is contiguous ``(n_tokens, dim)`` float32 with even ``dim``.
 */
#ifndef SDX_ROPE_APPLY_H
#define SDX_ROPE_APPLY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_ROPE_BUILD)
#define SDX_CUDA_ROPE_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_ROPE_API __declspec(dllimport)
#else
#define SDX_CUDA_ROPE_API
#endif

/**
 * q_host/k_host: contiguous float32 arrays of size n_tokens*dim.
 * theta_base: RoPE base (e.g. 10000.0f).
 */
SDX_CUDA_ROPE_API int sdx_cuda_apply_rope_interleaved_f32_host(
    float *q_host,
    float *k_host,
    int n_tokens,
    int dim,
    float theta_base);

#ifdef __cplusplus
}
#endif

#endif
