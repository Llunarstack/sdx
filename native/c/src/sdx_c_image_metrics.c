#include "../include/sdx_c_image_metrics.h"

#include <stdlib.h>
#include <string.h>

static inline uint8_t sdx_c_luma_u8(const uint8_t *px, int channels) {
    if (channels <= 1) {
        return px[0];
    }
    const uint32_t r = (uint32_t)px[0];
    const uint32_t g = (uint32_t)px[1];
    const uint32_t b = (uint32_t)px[2];
    return (uint8_t)((77u * r + 150u * g + 29u * b + 128u) >> 8);
}

int sdx_c_mean_luma_u8(
    const uint8_t *hwc,
    int height,
    int width,
    int channels,
    double *out_mean_luma
) {
    if (!hwc || !out_mean_luma || height <= 0 || width <= 0 || channels <= 0) {
        return -1;
    }
    const size_t n = (size_t)height * (size_t)width;
    uint64_t acc = 0;
    const uint8_t *p = hwc;
    for (size_t i = 0; i < n; ++i) {
        acc += (uint64_t)sdx_c_luma_u8(p, channels);
        p += channels;
    }
    *out_mean_luma = (double)acc / (double)n;
    return 0;
}

int sdx_c_clip_ratio_u8(
    const uint8_t *hwc,
    int height,
    int width,
    int channels,
    uint8_t clip_low,
    uint8_t clip_high,
    double *out_ratio
) {
    if (!hwc || !out_ratio || height <= 0 || width <= 0 || channels <= 0) {
        return -1;
    }
    const size_t n = (size_t)height * (size_t)width;
    uint64_t clipped = 0;
    const uint8_t *p = hwc;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t y = sdx_c_luma_u8(p, channels);
        if (y <= clip_low || y >= clip_high) {
            clipped += 1;
        }
        p += channels;
    }
    *out_ratio = (double)clipped / (double)n;
    return 0;
}

int sdx_c_count_components_u8(
    const uint8_t *hwc,
    int height,
    int width,
    int channels,
    uint8_t threshold,
    int min_area,
    int max_area
) {
    if (!hwc || height <= 0 || width <= 0 || channels <= 0) {
        return -1;
    }
    const size_t n = (size_t)height * (size_t)width;
    uint8_t *fg = (uint8_t *)malloc(n);
    uint8_t *vis = (uint8_t *)malloc(n);
    int *queue = (int *)malloc(n * sizeof(int));
    if (!fg || !vis || !queue) {
        free(fg);
        free(vis);
        free(queue);
        return -2;
    }
    memset(vis, 0, n);

    const uint8_t *p = hwc;
    for (size_t i = 0; i < n; ++i) {
        fg[i] = (uint8_t)(sdx_c_luma_u8(p, channels) <= threshold ? 1 : 0);
        p += channels;
    }

    const int area_min = min_area > 0 ? min_area : 1;
    const int area_max = max_area > 0 ? max_area : 0;
    int count = 0;
    static const int nb_dy[4] = {-1, 1, 0, 0};
    static const int nb_dx[4] = {0, 0, -1, 1};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;
            if (!fg[idx] || vis[idx]) {
                continue;
            }
            int qh = 0;
            int qt = 0;
            int area = 0;
            vis[idx] = 1;
            queue[qt++] = idx;
            while (qh < qt) {
                const int cur = queue[qh++];
                const int cy = cur / width;
                const int cx = cur - cy * width;
                area += 1;
                for (int k = 0; k < 4; ++k) {
                    const int ny = cy + nb_dy[k];
                    const int nx = cx + nb_dx[k];
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                        continue;
                    }
                    const int ni = ny * width + nx;
                    if (!fg[ni] || vis[ni]) {
                        continue;
                    }
                    vis[ni] = 1;
                    queue[qt++] = ni;
                }
            }
            if (area >= area_min && (area_max <= 0 || area <= area_max)) {
                count += 1;
            }
        }
    }

    free(fg);
    free(vis);
    free(queue);
    return count;
}
