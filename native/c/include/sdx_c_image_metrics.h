/*
 * Pure-C image metrics helpers for uint8 HWC buffers.
 * Minimal dependencies so this can be embedded in tiny native toolchains.
 */
#ifndef SDX_C_IMAGE_METRICS_H
#define SDX_C_IMAGE_METRICS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Mean luma (Rec.601-like integer weights) for HWC uint8 image.
 * channels may be 1 (grayscale) or >=3 (RGB expected in first 3 channels).
 * Returns 0 on success, -1 on invalid args.
 */
int sdx_c_mean_luma_u8(
    const uint8_t *hwc,
    int height,
    int width,
    int channels,
    double *out_mean_luma
);

/*
 * Fraction of pixels that are clipped near black/white:
 * luma <= clip_low OR luma >= clip_high.
 * Returns 0 on success, -1 on invalid args.
 */
int sdx_c_clip_ratio_u8(
    const uint8_t *hwc,
    int height,
    int width,
    int channels,
    uint8_t clip_low,
    uint8_t clip_high,
    double *out_ratio
);

/*
 * Naive connected-components count on thresholded luma map.
 * Foreground pixels are those with luma <= threshold.
 * min_area/max_area are inclusive filters in pixel counts; set max_area <= 0 to disable upper bound.
 * Returns component count on success, -1 on invalid args, -2 on allocation failure.
 */
int sdx_c_count_components_u8(
    const uint8_t *hwc,
    int height,
    int width,
    int channels,
    uint8_t threshold,
    int min_area,
    int max_area
);

#ifdef __cplusplus
}
#endif

#endif /* SDX_C_IMAGE_METRICS_H */
