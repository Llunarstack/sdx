#include "sdx/image_metrics.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {
inline uint8_t luma_u8(const unsigned char *px, int channels) {
    if (channels <= 1) {
        return static_cast<uint8_t>(px[0]);
    }
    const uint32_t r = static_cast<uint32_t>(px[0]);
    const uint32_t g = static_cast<uint32_t>(px[1]);
    const uint32_t b = static_cast<uint32_t>(px[2]);
    return static_cast<uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
}
}  // namespace

int sdx_image_mean_luma_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    double *out_mean_luma
) {
    if (!hwc || !out_mean_luma || height <= 0 || width <= 0 || channels <= 0) {
        return -1;
    }
    const std::size_t n = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::uint64_t acc = 0;
    const unsigned char *p = hwc;
    for (std::size_t i = 0; i < n; ++i) {
        acc += static_cast<std::uint64_t>(luma_u8(p, channels));
        p += channels;
    }
    *out_mean_luma = static_cast<double>(acc) / static_cast<double>(n);
    return 0;
}

int sdx_image_clip_ratio_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    unsigned char clip_low,
    unsigned char clip_high,
    double *out_ratio
) {
    if (!hwc || !out_ratio || height <= 0 || width <= 0 || channels <= 0) {
        return -1;
    }
    const std::size_t n = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::size_t clipped = 0;
    const unsigned char *p = hwc;
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t y = luma_u8(p, channels);
        if (y <= clip_low || y >= clip_high) {
            ++clipped;
        }
        p += channels;
    }
    *out_ratio = static_cast<double>(clipped) / static_cast<double>(n);
    return 0;
}

int sdx_image_laplacian_var_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    double *out_lap_var
) {
    if (!hwc || !out_lap_var || height < 3 || width < 3 || channels <= 0) {
        return -1;
    }
    const int h = height;
    const int w = width;
    double sum = 0.0;
    double sumsq = 0.0;
    std::size_t n = 0;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const unsigned char *c = hwc + (static_cast<std::size_t>(y) * w + x) * channels;
            const unsigned char *l = c - channels;
            const unsigned char *r = c + channels;
            const unsigned char *u = c - static_cast<std::size_t>(w) * channels;
            const unsigned char *d = c + static_cast<std::size_t>(w) * channels;
            const double lap = -4.0 * static_cast<double>(luma_u8(c, channels))
                             + static_cast<double>(luma_u8(l, channels))
                             + static_cast<double>(luma_u8(r, channels))
                             + static_cast<double>(luma_u8(u, channels))
                             + static_cast<double>(luma_u8(d, channels));
            sum += lap;
            sumsq += lap * lap;
            ++n;
        }
    }
    if (n == 0) {
        return -1;
    }
    const double mean = sum / static_cast<double>(n);
    *out_lap_var = std::max(0.0, sumsq / static_cast<double>(n) - mean * mean);
    return 0;
}

int sdx_image_count_components_u8(
    const unsigned char *hwc,
    int height,
    int width,
    int channels,
    unsigned char threshold,
    int min_area,
    int max_area
) {
    if (!hwc || height <= 0 || width <= 0 || channels <= 0) {
        return -1;
    }
    const std::size_t n = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::vector<unsigned char> fg(n, 0);
    std::vector<unsigned char> vis(n, 0);
    std::vector<int> queue;
    queue.reserve(n);

    const unsigned char *p = hwc;
    for (std::size_t i = 0; i < n; ++i) {
        fg[i] = static_cast<unsigned char>(luma_u8(p, channels) <= threshold ? 1 : 0);
        p += channels;
    }

    const int area_min = min_area > 0 ? min_area : 1;
    const int area_max = max_area > 0 ? max_area : 0;
    int count = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;
            if (!fg[static_cast<std::size_t>(idx)] || vis[static_cast<std::size_t>(idx)]) {
                continue;
            }
            queue.clear();
            queue.push_back(idx);
            vis[static_cast<std::size_t>(idx)] = 1;
            int area = 0;
            for (std::size_t qh = 0; qh < queue.size(); ++qh) {
                const int cur = queue[qh];
                const int cy = cur / width;
                const int cx = cur - cy * width;
                ++area;
                const int nb[4][2] = {{cy - 1, cx}, {cy + 1, cx}, {cy, cx - 1}, {cy, cx + 1}};
                for (int k = 0; k < 4; ++k) {
                    const int ny = nb[k][0];
                    const int nx = nb[k][1];
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                        continue;
                    }
                    const int ni = ny * width + nx;
                    if (!fg[static_cast<std::size_t>(ni)] || vis[static_cast<std::size_t>(ni)]) {
                        continue;
                    }
                    vis[static_cast<std::size_t>(ni)] = 1;
                    queue.push_back(ni);
                }
            }
            if (area >= area_min && (area_max <= 0 || area <= area_max)) {
                ++count;
            }
        }
    }
    return count;
}
