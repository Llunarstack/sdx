/**
 * Tiled SDPA forward with **online softmax** (FlashAttention-style memory pattern; not a full FA3 kernel).
 *
 * Layout: ``Q``, ``K``, ``V``, ``O`` are contiguous float32, shape (1, num_heads, seq_len, head_dim).
 * Constraints: head_dim == 64, seq_len <= 2048, num_heads <= 64, batch == 1.
 */
#ifndef SDX_SDPA_ONLINE_SOFTMAX_H
#define SDX_SDPA_ONLINE_SOFTMAX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(_WIN32) && defined(SDX_CUDA_SDPA_BUILD)
#define SDX_CUDA_SDPA_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_SDPA_API __declspec(dllimport)
#else
#define SDX_CUDA_SDPA_API
#endif

SDX_CUDA_SDPA_API int sdx_cuda_sdpa_online_f32_host(const float *Q, const float *K, const float *V, float *O, int32_t num_heads,
                                                     int32_t seq_len, float scale);

#ifdef __cplusplus
}
#endif

#endif
