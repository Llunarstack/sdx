/**
 * Optional CUDA: RMSNorm each row of a row-major float32 matrix ``(n_rows, dim)``.
 * Useful for transformer token/feature normalization in preprocessing experiments.
 */
#ifndef SDX_RMSNORM_ROWS_H
#define SDX_RMSNORM_ROWS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_RMSNORM_BUILD)
#define SDX_CUDA_RMSNORM_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_RMSNORM_API __declspec(dllimport)
#else
#define SDX_CUDA_RMSNORM_API
#endif

/** In-place RMSNorm on host float32 matrix copied to CUDA and back. */
SDX_CUDA_RMSNORM_API int sdx_cuda_rmsnorm_rows_f32_host(float *host_data, int n_rows, int dim, float eps);

#ifdef __cplusplus
}
#endif

#endif
