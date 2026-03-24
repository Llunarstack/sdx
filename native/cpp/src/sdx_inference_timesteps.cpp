/**
 * Inference timestep path finalization - mirrors diffusion/inference_timesteps.py helpers.
 */
#include "sdx/inference_timesteps.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

/** One sample of ``numpy.linspace(a, b, n)`` (endpoint inclusive, ``n >= 1``). */
static double numpy_linspace_point(double a, double b, int n, int i) {
    if (n <= 1) {
        return a;
    }
    if (i == n - 1) {
        return b;
    }
    const double step = (b - a) / static_cast<double>(n - 1);
    return a + static_cast<double>(i) * step;
}

/** Match ``numpy.round`` on finite reals (ties to nearest even). */
static int64_t numpy_round_dto_i64(double x) {
    if (!std::isfinite(x)) {
        return 0;
    }
    const double ax = std::fabs(x);
    double ip = 0.0;
    const double frac = std::modf(ax, &ip);
    const int64_t bi = static_cast<int64_t>(ip);
    int64_t out = 0;
    if (frac < 0.5) {
        out = bi;
    } else if (frac > 0.5) {
        out = bi + 1;
    } else {
        out = (bi % 2 == 0) ? bi : (bi + 1);
    }
    return (x < 0.0) ? -out : out;
}

/* NumPy compiled_base.c interp (1-D monotonic xp): search + eval order (parity). */
static int64_t linear_search_guess(double key, const double *arr, int64_t len, int64_t i0) {
    int64_t i = i0;
    for (; i < len && key >= arr[static_cast<size_t>(i)]; ++i) {
    }
    return i - 1;
}

static constexpr int kLikelyInCacheSize = 8;

static int64_t binary_search_with_guess(double key, const double *arr, int64_t len, int64_t guess) {
    int64_t imin = 0;
    int64_t imax = len;

    if (key > arr[static_cast<size_t>(len - 1)]) {
        return len;
    }
    if (key < arr[0]) {
        return -1;
    }

    if (len <= 4) {
        return linear_search_guess(key, arr, len, 1);
    }

    if (guess > len - 3) {
        guess = len - 3;
    }
    if (guess < 1) {
        guess = 1;
    }

    if (key < arr[static_cast<size_t>(guess)]) {
        if (key < arr[static_cast<size_t>(guess - 1)]) {
            imax = guess - 1;
            if (guess > kLikelyInCacheSize && key >= arr[static_cast<size_t>(guess - kLikelyInCacheSize)]) {
                imin = guess - kLikelyInCacheSize;
            }
        } else {
            return guess - 1;
        }
    } else {
        if (key < arr[static_cast<size_t>(guess + 1)]) {
            return guess;
        }
        if (key < arr[static_cast<size_t>(guess + 2)]) {
            return guess + 1;
        }
        imin = guess + 2;
        if (guess < len - kLikelyInCacheSize - 1 && key < arr[static_cast<size_t>(guess + kLikelyInCacheSize)]) {
            imax = guess + kLikelyInCacheSize;
        }
    }

    while (imin < imax) {
        const int64_t imid = imin + ((imax - imin) >> 1);
        if (key >= arr[static_cast<size_t>(imid)]) {
            imin = imid + 1;
        } else {
            imax = imid;
        }
    }
    return imin - 1;
}

static int64_t clamp_i64(int64_t x, int64_t lo, int64_t hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static void enforce_strict_descending(const int64_t *raw, int64_t raw_n, int num_train, std::vector<int64_t> *out) {
    out->clear();
    const int64_t hi = static_cast<int64_t>(num_train) - 1;
    /* Match Python: empty idx -> np.array([num_train - 1, 0]) (two elements, even when T==1). */
    if (raw_n <= 0) {
        out->push_back(std::max<int64_t>(0, hi));
        out->push_back(0);
        return;
    }
    std::vector<int64_t> chain;
    chain.reserve(static_cast<size_t>(raw_n));
    chain.push_back(clamp_i64(raw[0], 0, hi));
    for (int64_t j = 1; j < raw_n; ++j) {
        int64_t v = clamp_i64(raw[j], 0, hi);
        if (v >= chain.back()) {
            v = chain.back() - 1;
        }
        chain.push_back(std::max<int64_t>(0, v));
    }
    out->push_back(chain[0]);
    for (size_t k = 1; k < chain.size(); ++k) {
        if (chain[k] < out->back()) {
            out->push_back(chain[k]);
        }
    }
}

static void subsample_evenly(const std::vector<int64_t> &d, int target_len, int num_train, std::vector<int64_t> *tmp) {
    tmp->clear();
    const int64_t dn = static_cast<int64_t>(d.size());
    if (target_len <= 0 || dn <= 0) {
        return;
    }
    if (dn >= target_len) {
        if (dn == target_len) {
            *tmp = d;
            return;
        }
        tmp->reserve(static_cast<size_t>(target_len));
        /* Match ``np.linspace(0, dn-1, target_len)``: i * step, last point exact. */
        for (int i = 0; i < target_len; ++i) {
            const double pos = numpy_linspace_point(0.0, static_cast<double>(dn - 1), target_len, i);
            int64_t idx = numpy_round_dto_i64(pos);
            idx = std::max<int64_t>(0, std::min(idx, dn - 1));
            tmp->push_back(d[static_cast<size_t>(idx)]);
        }
        return;
    }
    if (dn == 1) {
        tmp->reserve(static_cast<size_t>(target_len));
        const int64_t hi = static_cast<int64_t>(num_train) - 1;
        if (target_len == 1) {
            tmp->push_back(std::max<int64_t>(0, hi));
            return;
        }
        /* Match ``np.linspace(hi, 0, target_len, dtype=np.int64)`` (float ramp, truncate to int). */
        for (int i = 0; i < target_len; ++i) {
            const double v = numpy_linspace_point(static_cast<double>(hi), 0.0, target_len, i);
            tmp->push_back(static_cast<int64_t>(v));
        }
        return;
    }
    tmp->reserve(static_cast<size_t>(target_len));
    std::vector<double> xp(static_cast<size_t>(dn));
    std::vector<double> fp(static_cast<size_t>(dn));
    for (int k = 0; k < dn; ++k) {
        xp[static_cast<size_t>(k)] = numpy_linspace_point(0.0, 1.0, static_cast<int>(dn), k);
        fp[static_cast<size_t>(k)] = static_cast<double>(d[static_cast<size_t>(k)]);
    }
    std::vector<double> slopes;
    const int64_t lenxp = dn;
    const int64_t lenx = static_cast<int64_t>(target_len);
    if (lenxp <= lenx && lenxp > 1) {
        slopes.resize(static_cast<size_t>(lenxp - 1));
        for (int64_t si = 0; si < lenxp - 1; ++si) {
            slopes[static_cast<size_t>(si)] =
                (fp[static_cast<size_t>(si + 1)] - fp[static_cast<size_t>(si)]) /
                (xp[static_cast<size_t>(si + 1)] - xp[static_cast<size_t>(si)]);
        }
    }

    int64_t j_guess = 0;
    for (int i = 0; i < target_len; ++i) {
        const double x_val = numpy_linspace_point(0.0, 1.0, target_len, i);
        const int64_t j = binary_search_with_guess(x_val, xp.data(), lenxp, j_guess);
        double dres = 0.0;
        if (j == -1) {
            dres = fp[0];
        } else if (j == lenxp) {
            dres = fp[static_cast<size_t>(lenxp - 1)];
        } else if (j == lenxp - 1) {
            dres = fp[static_cast<size_t>(j)];
        } else if (xp[static_cast<size_t>(j)] == x_val) {
            dres = fp[static_cast<size_t>(j)];
        } else {
            const double slope = slopes.empty()
                ? ((fp[static_cast<size_t>(j + 1)] - fp[static_cast<size_t>(j)]) /
                   (xp[static_cast<size_t>(j + 1)] - xp[static_cast<size_t>(j)]))
                : slopes[static_cast<size_t>(j)];
            dres = slope * (x_val - xp[static_cast<size_t>(j)]) + fp[static_cast<size_t>(j)];
        }
        j_guess = j;
        tmp->push_back(numpy_round_dto_i64(dres));
    }
}

extern "C" int64_t sdx_it_finalize_path(const int64_t *raw, int64_t raw_n, int target_len, int num_train, int64_t *out,
                                        int64_t out_cap) {
    if (out == nullptr || out_cap < 1) {
        return -1;
    }
    if (num_train < 1) {
        return -1;
    }
    if (target_len <= 0) {
        out[0] = 0;
        return 1;
    }

    std::vector<int64_t> d;
    enforce_strict_descending(raw, raw_n, num_train, &d);

    std::vector<int64_t> tmp;
    subsample_evenly(d, target_len, num_train, &tmp);

    /*
     * Python ``_resample_length_numpy``: if ``d.size == 1``, return ``linspace`` directly
     * (duplicates allowed). Otherwise final ``_enforce_strict_descending``.
     */
    if (d.size() == 1) {
        if (static_cast<int64_t>(tmp.size()) > out_cap) {
            return -1;
        }
        for (size_t i = 0; i < tmp.size(); ++i) {
            out[i] = tmp[i];
        }
        return static_cast<int64_t>(tmp.size());
    }

    enforce_strict_descending(tmp.data(), static_cast<int64_t>(tmp.size()), num_train, &d);

    if (static_cast<int64_t>(d.size()) > out_cap) {
        return -1;
    }
    for (size_t i = 0; i < d.size(); ++i) {
        out[i] = d[i];
    }
    return static_cast<int64_t>(d.size());
}
