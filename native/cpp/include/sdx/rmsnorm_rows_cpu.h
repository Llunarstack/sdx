/**
 * Fast CPU RMSNorm rows (float32).
 */
#ifndef SDX_RMSNORM_ROWS_CPU_H
#define SDX_RMSNORM_ROWS_CPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_RMSNORM_ROWS_CPU_BUILD)
#define SDX_RMSNORM_ROWS_CPU_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_RMSNORM_ROWS_CPU_API __declspec(dllimport)
#else
#define SDX_RMSNORM_ROWS_CPU_API
#endif

/** In-place row-wise RMSNorm on host float32 matrix (n_rows, dim). */
SDX_RMSNORM_ROWS_CPU_API int sdx_rmsnorm_rows_f32_host(float *host_data, int n_rows, int dim, float eps);

#ifdef __cplusplus
}
#endif

#endif
