/**
 * Per-sample percentile clamp on a float32 tensor (B, row_len).
 * Matches dynamic_percentile_clamp() in diffusion/sampling_extras/latent_refiner.py.
 */
#ifndef SDX_PERCENTILE_CLAMP_H
#define SDX_PERCENTILE_CLAMP_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_PCLAMP_BUILD)
#  define SDX_CUDA_PCLAMP_API __declspec(dllexport)
#elif defined(_WIN32)
#  define SDX_CUDA_PCLAMP_API __declspec(dllimport)
#else
#  define SDX_CUDA_PCLAMP_API
#endif

/**
 * In-place: for each row b, compute bound = quantile(|row|, q) clamped to
 * floor_val, then row[i] = clamp(row[i], -bound, bound) / bound.
 *
 * @param data_host  (B, row_len) float32 row-major, modified in-place.
 * @param quantile   e.g. 0.995
 * @param floor_val  minimum bound (e.g. 1.0)
 * @return 0 on success, -1 on alloc/CUDA error, -2 on bad arguments.
 */
SDX_CUDA_PCLAMP_API int sdx_cuda_percentile_clamp_f32(
    float *data_host,
    int    B,
    int    row_len,
    float  quantile,
    float  floor_val);

#ifdef __cplusplus
}
#endif

#endif /* SDX_PERCENTILE_CLAMP_H */
