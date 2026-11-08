/* CPU half of the SDX C ABI (see include/sdx.h). */
#include "sdx.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

/* ============================== latent geometry ============================== */

static int exact_div(int a, int b) { return (a <= 0 || b <= 0 || a % b != 0) ? 0 : a / b; }

extern "C" int sdx_latent_spatial_size(int image_hw, int vae_scale) { return exact_div(image_hw, vae_scale); }

extern "C" int sdx_patch_grid_dim(int latent_hw, int patch_size) { return exact_div(latent_hw, patch_size); }

extern "C" int sdx_num_patch_tokens(int image_hw, int vae_scale, int patch_size) {
    const int g = exact_div(exact_div(image_hw, vae_scale), patch_size);
    return g * g;
}

extern "C" int sdx_latent_numel(int channels, int latent_h, int latent_w) {
    if (channels <= 0 || latent_h <= 0 || latent_w <= 0) return 0;
    const int64_t n = static_cast<int64_t>(channels) * latent_h * latent_w;
    return n > INT32_MAX ? 0 : static_cast<int>(n);
}

/* ======================= inference timestep finalization ======================= */

/** One sample of ``numpy.linspace(a, b, n)`` (endpoint inclusive, ``n >= 1``). */
static double numpy_linspace_point(double a, double b, int n, int i) {
    if (n <= 1) return a;
    if (i == n - 1) return b;
    return a + static_cast<double>(i) * ((b - a) / static_cast<double>(n - 1));
}

/** ``numpy.round`` on finite reals (ties to even). */
static int64_t numpy_round_dto_i64(double x) {
    if (!std::isfinite(x)) return 0;
    double ip = 0.0;
    const double frac = std::modf(std::fabs(x), &ip);
    const int64_t bi = static_cast<int64_t>(ip);
    const int64_t out = frac < 0.5 ? bi : frac > 0.5 ? bi + 1 : (bi % 2 == 0 ? bi : bi + 1);
    return x < 0.0 ? -out : out;
}

/* NumPy compiled_base.c interp (1-D monotonic xp): same search + evaluation order for bit parity. */
static int64_t linear_search_guess(double key, const double *arr, int64_t len, int64_t i0) {
    int64_t i = i0;
    while (i < len && key >= arr[i]) ++i;
    return i - 1;
}

static constexpr int kLikelyInCacheSize = 8;

static int64_t binary_search_with_guess(double key, const double *arr, int64_t len, int64_t guess) {
    int64_t imin = 0;
    int64_t imax = len;
    if (key > arr[len - 1]) return len;
    if (key < arr[0]) return -1;
    if (len <= 4) return linear_search_guess(key, arr, len, 1);
    guess = std::max<int64_t>(1, std::min<int64_t>(guess, len - 3));

    if (key < arr[guess]) {
        if (key >= arr[guess - 1]) return guess - 1;
        imax = guess - 1;
        if (guess > kLikelyInCacheSize && key >= arr[guess - kLikelyInCacheSize]) imin = guess - kLikelyInCacheSize;
    } else {
        if (key < arr[guess + 1]) return guess;
        if (key < arr[guess + 2]) return guess + 1;
        imin = guess + 2;
        if (guess < len - kLikelyInCacheSize - 1 && key < arr[guess + kLikelyInCacheSize]) {
            imax = guess + kLikelyInCacheSize;
        }
    }
    while (imin < imax) {
        const int64_t imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        } else {
            imax = imid;
        }
    }
    return imin - 1;
}

static void enforce_strict_descending(const int64_t *raw, int64_t raw_n, int num_train, std::vector<int64_t> *out) {
    out->clear();
    const int64_t hi = static_cast<int64_t>(num_train) - 1;
    /* Python: empty idx -> [num_train - 1, 0] (two elements, even when T == 1). */
    if (raw_n <= 0) {
        out->push_back(std::max<int64_t>(0, hi));
        out->push_back(0);
        return;
    }
    int64_t prev = std::clamp<int64_t>(raw[0], 0, hi);
    out->push_back(prev);
    for (int64_t j = 1; j < raw_n; ++j) {
        int64_t v = std::clamp<int64_t>(raw[j], 0, hi);
        if (v >= prev) v = prev - 1;
        prev = std::max<int64_t>(0, v);
        if (prev < out->back()) out->push_back(prev);
    }
}

static void subsample_evenly(const std::vector<int64_t> &d, int target_len, int num_train, std::vector<int64_t> *tmp) {
    tmp->clear();
    const int64_t dn = static_cast<int64_t>(d.size());
    if (target_len <= 0 || dn <= 0) return;
    tmp->reserve(static_cast<size_t>(target_len));
    if (dn >= target_len) {
        if (dn == target_len) {
            *tmp = d;
            return;
        }
        /* np.linspace(0, dn-1, target_len), rounded. */
        for (int i = 0; i < target_len; ++i) {
            const int64_t idx = numpy_round_dto_i64(numpy_linspace_point(0.0, static_cast<double>(dn - 1), target_len, i));
            tmp->push_back(d[std::clamp<int64_t>(idx, 0, dn - 1)]);
        }
        return;
    }
    if (dn == 1) {
        const int64_t hi = static_cast<int64_t>(num_train) - 1;
        if (target_len == 1) {
            tmp->push_back(std::max<int64_t>(0, hi));
            return;
        }
        /* np.linspace(hi, 0, target_len, dtype=np.int64): float ramp truncated to int. */
        for (int i = 0; i < target_len; ++i) {
            tmp->push_back(static_cast<int64_t>(numpy_linspace_point(static_cast<double>(hi), 0.0, target_len, i)));
        }
        return;
    }
    /* np.interp(linspace(0, 1, target_len), linspace(0, 1, dn), d), rounded. */
    std::vector<double> xp(static_cast<size_t>(dn));
    std::vector<double> fp(static_cast<size_t>(dn));
    for (int64_t k = 0; k < dn; ++k) {
        xp[k] = numpy_linspace_point(0.0, 1.0, static_cast<int>(dn), static_cast<int>(k));
        fp[k] = static_cast<double>(d[k]);
    }
    std::vector<double> slopes;
    if (dn <= target_len) {
        slopes.resize(static_cast<size_t>(dn - 1));
        for (int64_t si = 0; si < dn - 1; ++si) slopes[si] = (fp[si + 1] - fp[si]) / (xp[si + 1] - xp[si]);
    }
    int64_t j_guess = 0;
    for (int i = 0; i < target_len; ++i) {
        const double x = numpy_linspace_point(0.0, 1.0, target_len, i);
        const int64_t j = binary_search_with_guess(x, xp.data(), dn, j_guess);
        double v;
        if (j == -1) {
            v = fp[0];
        } else if (j >= dn - 1) {
            v = fp[dn - 1];
        } else if (xp[j] == x) {
            v = fp[j];
        } else {
            const double slope = slopes.empty() ? (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) : slopes[j];
            v = slope * (x - xp[j]) + fp[j];
        }
        j_guess = j;
        tmp->push_back(numpy_round_dto_i64(v));
    }
}

extern "C" int64_t sdx_it_finalize_path(const int64_t *raw, int64_t raw_n, int target_len, int num_train, int64_t *out,
                                        int64_t out_cap) {
    if (out == nullptr || out_cap < 1 || num_train < 1) return -1;
    if (target_len <= 0) {
        out[0] = 0;
        return 1;
    }
    std::vector<int64_t> d;
    enforce_strict_descending(raw, raw_n, num_train, &d);
    std::vector<int64_t> tmp;
    subsample_evenly(d, target_len, num_train, &tmp);
    /* Python returns the linspace directly when d has one element (duplicates allowed). */
    const std::vector<int64_t> *res = &tmp;
    if (d.size() != 1) {
        enforce_strict_descending(tmp.data(), static_cast<int64_t>(tmp.size()), num_train, &d);
        res = &d;
    }
    if (static_cast<int64_t>(res->size()) > out_cap) return -1;
    std::copy(res->begin(), res->end(), out);
    return static_cast<int64_t>(res->size());
}

/* ============================== beta schedule ============================== */

extern "C" int64_t sdx_squared_cosine_betas_v2(int n, double max_beta, double *out, int64_t cap) {
    if (n < 1 || out == nullptr || cap < n || max_beta <= 0.0) return -1;
    const double pi_h = 3.14159265358979323846 * 0.5;
    for (int i = 0; i < n; ++i) {
        const double c1 = std::cos((static_cast<double>(i) / n + 0.008) / 1.008 * pi_h);
        const double c2 = std::cos((static_cast<double>(i + 1) / n + 0.008) / 1.008 * pi_h);
        const double b = std::min(1.0 - (c2 * c2) / std::max(c1 * c1, 1e-12), max_beta);
        out[i] = std::clamp(b, 1e-4, 0.999);
    }
    return n;
}

/* ============================== image metrics ============================== */

static inline uint8_t luma_u8(const unsigned char *px, int channels) {
    if (channels <= 1) return px[0];
    return static_cast<uint8_t>((77u * px[0] + 150u * px[1] + 29u * px[2] + 128u) >> 8);
}

static bool bad_hwc(const unsigned char *hwc, int height, int width, int channels) {
    return !hwc || height <= 0 || width <= 0 || channels <= 0;
}

extern "C" int sdx_image_mean_luma_u8(const unsigned char *hwc, int height, int width, int channels,
                                      double *out_mean_luma) {
    if (bad_hwc(hwc, height, width, channels) || !out_mean_luma) return -1;
    const size_t n = static_cast<size_t>(height) * width;
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc += luma_u8(hwc + i * channels, channels);
    *out_mean_luma = static_cast<double>(acc) / static_cast<double>(n);
    return 0;
}

extern "C" int sdx_image_clip_ratio_u8(const unsigned char *hwc, int height, int width, int channels,
                                       unsigned char clip_low, unsigned char clip_high, double *out_ratio) {
    if (bad_hwc(hwc, height, width, channels) || !out_ratio) return -1;
    const size_t n = static_cast<size_t>(height) * width;
    size_t clipped = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t y = luma_u8(hwc + i * channels, channels);
        clipped += (y <= clip_low || y >= clip_high);
    }
    *out_ratio = static_cast<double>(clipped) / static_cast<double>(n);
    return 0;
}

extern "C" int sdx_image_laplacian_var_u8(const unsigned char *hwc, int height, int width, int channels,
                                          double *out_lap_var) {
    if (bad_hwc(hwc, height, width, channels) || !out_lap_var || height < 3 || width < 3) return -1;
    const size_t row = static_cast<size_t>(width) * channels;
    double sum = 0.0;
    double sumsq = 0.0;
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const unsigned char *c = hwc + y * row + static_cast<size_t>(x) * channels;
            const double lap = -4.0 * luma_u8(c, channels) + luma_u8(c - channels, channels) +
                               luma_u8(c + channels, channels) + luma_u8(c - row, channels) +
                               luma_u8(c + row, channels);
            sum += lap;
            sumsq += lap * lap;
        }
    }
    const double n = static_cast<double>(height - 2) * (width - 2);
    const double mean = sum / n;
    *out_lap_var = std::max(0.0, sumsq / n - mean * mean);
    return 0;
}

extern "C" int sdx_image_count_components_u8(const unsigned char *hwc, int height, int width, int channels,
                                             unsigned char threshold, int min_area, int max_area) {
    if (bad_hwc(hwc, height, width, channels)) return -1;
    const size_t n = static_cast<size_t>(height) * width;
    /* open[i]: foreground pixel not yet visited. */
    std::vector<unsigned char> open(n);
    for (size_t i = 0; i < n; ++i) open[i] = luma_u8(hwc + i * channels, channels) <= threshold;
    const int area_min = min_area > 0 ? min_area : 1;
    std::vector<int> queue;
    int count = 0;
    for (size_t start = 0; start < n; ++start) {
        if (!open[start]) continue;
        open[start] = 0;
        queue.assign(1, static_cast<int>(start));
        for (size_t qh = 0; qh < queue.size(); ++qh) {
            const int cur = queue[qh];
            const int cy = cur / width;
            const int cx = cur - cy * width;
            const auto visit = [&](int ni) {
                if (open[ni]) {
                    open[ni] = 0;
                    queue.push_back(ni);
                }
            };
            if (cy > 0) visit(cur - width);
            if (cy + 1 < height) visit(cur + width);
            if (cx > 0) visit(cur - 1);
            if (cx + 1 < width) visit(cur + 1);
        }
        const int area = static_cast<int>(queue.size());
        if (area >= area_min && (max_area <= 0 || area <= max_area)) ++count;
    }
    return count;
}

/* ============================== mask pooling ============================== */

extern "C" int sdx_mask_to_patch_weights_f32(const float *mask, float *out, int B, int H, int W, int ph, int pw) {
    if (!mask || !out || B <= 0 || H <= 0 || W <= 0 || ph <= 0 || pw <= 0) return -2;
    if (H % ph != 0 || W % pw != 0) return -3;
    const int cell_h = H / ph;
    const int cell_w = W / pw;
    const float inv_cell = 1.0f / static_cast<float>(cell_h * cell_w);
    for (int b = 0; b < B; ++b) {
        const float *plane = mask + static_cast<size_t>(b) * H * W;
        float *row = out + static_cast<size_t>(b) * ph * pw;
        for (int py = 0; py < ph; ++py) {
            for (int px = 0; px < pw; ++px) {
                float sum = 0.0f;
                for (int dy = 0; dy < cell_h; ++dy) {
                    const float *src = plane + static_cast<size_t>(py * cell_h + dy) * W + px * cell_w;
                    for (int dx = 0; dx < cell_w; ++dx) sum += src[dx];
                }
                row[py * pw + px] = sum * inv_cell;
            }
        }
    }
    return 0;
}
