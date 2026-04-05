/**
 * Per-sample percentile clamp on a float32 tensor (B, *).
 *
 * For each sample b: bound = quantile(|x[b]|, q), then x[b] = clamp(x[b], -bound, bound) / bound.
 * Matches dynamic_percentile_clamp() in diffusion/holy_grail/latent_refiner.py.
 *
 * C ABI: sdx_cuda_percentile_clamp_f32
 */
#include "sdx/percentile_clamp.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Comparison function for qsort (ascending absolute value). */
static int _cmp_float_abs(const void *a, const void *b) {
    const float fa = fabsf(*(const float *)a);
    const float fb = fabsf(*(const float *)b);
    return (fa > fb) - (fa < fb);
}

/**
 * Each thread block handles one sample row.
 * We use a simple approach: copy row to host scratch, sort, pick quantile, clamp.
 * For latent sizes (B=1..8, C*H*W ~ 4*32*32=4096) this is fast enough.
 */
__global__ void sdx_k_percentile_clamp(
    float *__restrict__ data,
    const float *__restrict__ bounds, /* (B,) precomputed bounds */
    int B, int row_len)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float bound = bounds[b];
    if (bound <= 0.0f) return;

    float *row = data + (size_t)b * row_len;
    for (int i = 0; i < row_len; ++i) {
        float v = row[i];
        if (v >  bound) v =  bound;
        if (v < -bound) v = -bound;
        row[i] = v / bound;
    }
}

extern "C" int sdx_cuda_percentile_clamp_f32(
    float *data_host,   /* (B, row_len) row-major, modified in-place */
    int B,
    int row_len,
    float quantile,     /* e.g. 0.995 */
    float floor_val)    /* minimum bound value */
{
    if (!data_host || B <= 0 || row_len <= 0) return -2;
    if (quantile <= 0.0f || quantile >= 1.0f) return -2;

    /* Compute per-sample bounds on CPU (sort is not worth a CUDA kernel at latent sizes). */
    float *bounds = (float *)malloc((size_t)B * sizeof(float));
    if (!bounds) return -1;

    float *scratch = (float *)malloc((size_t)row_len * sizeof(float));
    if (!scratch) { free(bounds); return -1; }

    for (int b = 0; b < B; ++b) {
        const float *row = data_host + (size_t)b * row_len;
        /* Copy absolute values for sorting. */
        for (int i = 0; i < row_len; ++i) scratch[i] = fabsf(row[i]);
        qsort(scratch, (size_t)row_len, sizeof(float), _cmp_float_abs);
        /* Quantile index (lower bound). */
        int idx = (int)(quantile * (float)(row_len - 1));
        if (idx < 0) idx = 0;
        if (idx >= row_len) idx = row_len - 1;
        float b_val = scratch[idx];
        if (b_val < floor_val) b_val = floor_val;
        bounds[b] = b_val;
    }
    free(scratch);

    /* Apply clamp on GPU. */
    const size_t data_bytes   = (size_t)B * row_len * sizeof(float);
    const size_t bounds_bytes = (size_t)B * sizeof(float);

    float *d_data = nullptr, *d_bounds = nullptr;
    cudaError_t e;

    e = cudaMalloc((void **)&d_data,   data_bytes);   if (e != cudaSuccess) { free(bounds); return -1; }
    e = cudaMalloc((void **)&d_bounds, bounds_bytes);
    if (e != cudaSuccess) { cudaFree(d_data); free(bounds); return -1; }

    e = cudaMemcpy(d_data,   data_host, data_bytes,   cudaMemcpyHostToDevice); if (e != cudaSuccess) goto fail;
    e = cudaMemcpy(d_bounds, bounds,    bounds_bytes,  cudaMemcpyHostToDevice); if (e != cudaSuccess) goto fail;

    {
        const int threads = 256;
        const int blocks  = (B + threads - 1) / threads;
        sdx_k_percentile_clamp<<<blocks, threads>>>(d_data, d_bounds, B, row_len);
    }

    e = cudaGetLastError();      if (e != cudaSuccess) goto fail;
    e = cudaDeviceSynchronize(); if (e != cudaSuccess) goto fail;
    e = cudaMemcpy(data_host, d_data, data_bytes, cudaMemcpyDeviceToHost);

fail:
    cudaFree(d_data);
    cudaFree(d_bounds);
    free(bounds);
    return e == cudaSuccess ? 0 : -1;
}
