#include "sdx/score_ops.h"

#include <algorithm>

int sdx_score_minmax_norm_f32(const float* in_scores, int n, float* out_scores) {
    if (!in_scores || !out_scores || n <= 0) {
        return 1;
    }
    float lo = in_scores[0];
    float hi = in_scores[0];
    for (int i = 1; i < n; ++i) {
        lo = std::min(lo, in_scores[i]);
        hi = std::max(hi, in_scores[i]);
    }
    const float span = hi - lo;
    if (span <= 1e-8f) {
        for (int i = 0; i < n; ++i) {
            out_scores[i] = 0.5f;
        }
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        out_scores[i] = (in_scores[i] - lo) / span;
    }
    return 0;
}

int sdx_score_weighted_sum_f32(
    const float* score_matrix,
    int rows,
    int cols,
    const float* weights,
    float* out_scores) {
    if (!score_matrix || !weights || !out_scores || rows <= 0 || cols <= 0) {
        return 1;
    }
    for (int c = 0; c < cols; ++c) {
        float s = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float v = score_matrix[r * cols + c];
            s += (weights[r] * v);
        }
        out_scores[c] = s;
    }
    return 0;
}
