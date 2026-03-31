/**
 * Fused depthwise Gaussian blur on a float32 latent tensor (B, C, H, W).
 *
 * Avoids the Python overhead of rebuilding the kernel tensor every call.
 * The kernel is computed on-device from sigma; radius is clamped to [1, 7].
 *
 * C ABI: sdx_cuda_gaussian_blur_latent_f32
 */
#include "sdx/gaussian_blur_latent.h"

#include <cuda_runtime.h>
#include <math.h>

/* Maximum kernel radius supported (kernel size = 2*MAX_R+1 = 15). */
#define MAX_R 7

/**
 * Build a 1-D Gaussian kernel of length (2*r+1) into g[].
 * Normalised so sum == 1.
 */
__device__ static void build_gaussian_1d(float *g, int r, float sigma) {
    const float s2 = 2.0f * sigma * sigma + 1e-8f;
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float v = expf(-(float)(i * i) / s2);
        g[i + r] = v;
        sum += v;
    }
    for (int i = 0; i < 2 * r + 1; ++i) {
        g[i] /= sum;
    }
}

/**
 * Each thread handles one output pixel (b, c, oh, ow).
 * Separable 2-D convolution: horizontal pass stored in tmp, vertical pass to out.
 * For simplicity we do a single non-separable pass here (small kernels, latent sizes).
 */
__global__ void sdx_k_gaussian_blur_latent(
    const float *__restrict__ src,
    float *__restrict__ dst,
    int B, int C, int H, int W,
    float sigma, int r)
{
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z; /* b * C + c */

    if (ow >= W || oh >= H || bc >= B * C) return;

    /* Build 1-D kernel in registers (max 15 taps). */
    float g[2 * MAX_R + 1];
    build_gaussian_1d(g, r, sigma);

    const int k = 2 * r + 1;
    const float *plane = src + (size_t)bc * H * W;
    float acc = 0.0f;

    for (int ky = 0; ky < k; ++ky) {
        const int sy = oh + ky - r;
        if (sy < 0 || sy >= H) continue;
        for (int kx = 0; kx < k; ++kx) {
            const int sx = ow + kx - r;
            if (sx < 0 || sx >= W) continue;
            acc += g[ky] * g[kx] * plane[sy * W + sx];
        }
    }

    dst[(size_t)bc * H * W + oh * W + ow] = acc;
}

extern "C" int sdx_cuda_gaussian_blur_latent_f32(
    const float *src_host,
    float *dst_host,
    int B, int C, int H, int W,
    float sigma)
{
    if (!src_host || !dst_host || B <= 0 || C <= 0 || H <= 0 || W <= 0 || sigma <= 0.0f)
        return -2;

    const int r = (int)fminf((float)MAX_R, fmaxf(1.0f, roundf(sigma * 2.0f)));
    const size_t n = (size_t)B * C * H * W;
    const size_t nbytes = n * sizeof(float);

    float *d_src = nullptr, *d_dst = nullptr;
    cudaError_t e;

    e = cudaMalloc((void **)&d_src, nbytes); if (e != cudaSuccess) return -1;
    e = cudaMalloc((void **)&d_dst, nbytes); if (e != cudaSuccess) { cudaFree(d_src); return -1; }

    e = cudaMemcpy(d_src, src_host, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) goto fail;

    {
        dim3 threads(16, 16, 1);
        dim3 blocks(
            (W + 15) / 16,
            (H + 15) / 16,
            B * C
        );
        sdx_k_gaussian_blur_latent<<<blocks, threads>>>(d_src, d_dst, B, C, H, W, sigma, r);
    }

    e = cudaGetLastError();   if (e != cudaSuccess) goto fail;
    e = cudaDeviceSynchronize(); if (e != cudaSuccess) goto fail;
    e = cudaMemcpy(dst_host, d_dst, nbytes, cudaMemcpyDeviceToHost);

fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return e == cudaSuccess ? 0 : -1;
}
