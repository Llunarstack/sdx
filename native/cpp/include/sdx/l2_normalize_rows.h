/**
 * Optional CUDA: L2-normalize each row of a row-major float32 matrix ``(n_rows, dim)``.
 * Common for embedding cosine similarity pipelines outside PyTorch.
 */
#ifndef SDX_L2_NORMALIZE_ROWS_H
#define SDX_L2_NORMALIZE_ROWS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_L2_BUILD)
#define SDX_CUDA_L2_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_L2_API __declspec(dllimport)
#else
#define SDX_CUDA_L2_API
#endif

/** In-place; rows are contiguous blocks of ``dim`` floats. ``eps`` avoids div-by-zero. */
SDX_CUDA_L2_API int sdx_cuda_l2_normalize_rows_f32_host(float *host_data, int n_rows, int dim, float eps);

/**
 * Pick candidate row with max dot product vs ``query`` (rows should be L2-normalized).
 * ``candidates`` is row-major ``(n_cand, dim)``.
 */
SDX_CUDA_L2_API int sdx_cuda_style_pick_best_f32_host(const float *query, const float *candidates, int n_cand,
                                                      int dim, int *out_index, float *out_score);

#ifdef __cplusplus
}
#endif

#endif
