#pragma once

#if defined(_WIN32) && defined(SDX_SCORE_OPS_BUILD)
#define SDX_SCORE_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_SCORE_API __declspec(dllimport)
#else
#define SDX_SCORE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Min-max normalize float32 scores to [0,1].
// Returns 0 on success, non-zero on invalid args.
SDX_SCORE_API int sdx_score_minmax_norm_f32(const float* in_scores, int n, float* out_scores);

// Weighted sum over a row-major matrix of shape (rows, cols),
// where rows = number of score components, cols = number of candidates.
// out_scores has length cols.
// Returns 0 on success, non-zero on invalid args.
SDX_SCORE_API int sdx_score_weighted_sum_f32(
    const float* score_matrix,
    int rows,
    int cols,
    const float* weights,
    float* out_scores);

#ifdef __cplusplus
}
#endif
