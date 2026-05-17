#include "sdx/l2_normalize_rows.h"

#include <cmath>
#include <cuda_runtime.h>

/** Dot product of two length-`dim` vectors (assumed L2-normalized => cosine). */
__device__ float sdx_dot_row(const float *a, const float *b, int dim) {
    float s = 0.f;
    for (int i = 0; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

__global__ void sdx_k_style_pick_best(const float *__restrict__ query, const float *__restrict__ candidates,
                                      int n_cand, int dim, int *__restrict__ out_index, float *__restrict__ out_score) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float best = -2.f;
    int best_i = 0;
    for (int i = 0; i < n_cand; ++i) {
        const float *row = candidates + static_cast<size_t>(i) * dim;
        const float d = sdx_dot_row(query, row, dim);
        if (d > best) {
            best = d;
            best_i = i;
        }
    }
    if (out_index) {
        *out_index = best_i;
    }
    if (out_score) {
        *out_score = best;
    }
}

extern "C" int sdx_cuda_style_pick_best_f32_host(const float *query, const float *candidates, int n_cand, int dim,
                                                 int *out_index, float *out_score) {
    if (!query || !candidates || n_cand <= 0 || dim <= 0 || !out_index || !out_score) {
        return -2;
    }
    const size_t qbytes = static_cast<size_t>(dim) * sizeof(float);
    const size_t cbytes = static_cast<size_t>(n_cand) * static_cast<size_t>(dim) * sizeof(float);
    float *d_q = nullptr;
    float *d_c = nullptr;
    int *d_idx = nullptr;
    float *d_score = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&d_q), qbytes);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&d_c), cbytes);
    if (e != cudaSuccess) {
        cudaFree(d_q);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&d_idx), sizeof(int));
    if (e != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_c);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&d_score), sizeof(float));
    if (e != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_c);
        cudaFree(d_idx);
        return -1;
    }
    e = cudaMemcpy(d_q, query, qbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(d_c, candidates, cbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    sdx_k_style_pick_best<<<1, 1>>>(d_q, d_c, n_cand, dim, d_idx, d_score);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(out_index, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(out_score, d_score, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_c);
    cudaFree(d_idx);
    cudaFree(d_score);
    return e == cudaSuccess ? 0 : -1;
fail:
    cudaFree(d_q);
    cudaFree(d_c);
    cudaFree(d_idx);
    cudaFree(d_score);
    return -1;
}
