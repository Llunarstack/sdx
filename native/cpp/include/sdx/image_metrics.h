/**
 * Fast CPU image metrics for uint8 HWC buffers (C ABI for ctypes).
 */
#ifndef SDX_IMAGE_METRICS_H
#define SDX_IMAGE_METRICS_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_IMAGE_METRICS_BUILD)
#define SDX_IMAGE_METRICS_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_IMAGE_METRICS_API __declspec(dllimport)
#else
#define SDX_IMAGE_METRICS_API
#endif

/**
 * Mean luma (0..255) with Rec.601 integer approximation.
 * channels may be 1 (grayscale) or >=3 (RGB in first 3 channels).
 * Returns 0 on success, -1 on invalid args.
 */
SDX_IMAGE_METRICS_API int sdx_image_mean_luma_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    double *out_mean_luma
);

/**
 * Ratio of clipped luma pixels: y <= clip_low OR y >= clip_high.
 * Returns 0 on success, -1 on invalid args.
 */
SDX_IMAGE_METRICS_API int sdx_image_clip_ratio_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    unsigned char clip_low,
    unsigned char clip_high,
    double *out_ratio
);

/**
 * Laplacian variance sharpness heuristic over luma.
 * Returns 0 on success, -1 on invalid args.
 */
SDX_IMAGE_METRICS_API int sdx_image_laplacian_var_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    double *out_lap_var
);

/**
 * Connected-component count over thresholded luma map.
 * Foreground is y <= threshold. Keeps components with area in [min_area, max_area] (max<=0 disables upper bound).
 * Returns count on success, negative on error.
 */
SDX_IMAGE_METRICS_API int sdx_image_count_components_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    unsigned char threshold,
    int min_area,
    int max_area
);

#ifdef __cplusplus
}
#endif

#endif /* SDX_IMAGE_METRICS_H */
