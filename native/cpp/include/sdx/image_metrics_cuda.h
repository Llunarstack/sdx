/**
 * Optional CUDA image metrics for uint8 HWC buffers.
 */
#ifndef SDX_IMAGE_METRICS_CUDA_H
#define SDX_IMAGE_METRICS_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_IMAGE_METRICS_BUILD)
#define SDX_CUDA_IMAGE_METRICS_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_IMAGE_METRICS_API __declspec(dllimport)
#else
#define SDX_CUDA_IMAGE_METRICS_API
#endif

/**
 * Compute mean luma and clipped-ratio on host uint8 HWC image via CUDA.
 * clip_low/high apply to luma in [0,255].
 * Returns 0 on success, -1 on CUDA failure, -2 on invalid args.
 */
SDX_CUDA_IMAGE_METRICS_API int sdx_cuda_luma_stats_u8_host(
    const uint8_t *host_src,
    int height,
    int width,
    int channels,
    uint8_t clip_low,
    uint8_t clip_high,
    float *out_mean_luma,
    float *out_clip_ratio
);

#ifdef __cplusplus
}
#endif

#endif /* SDX_IMAGE_METRICS_CUDA_H */
