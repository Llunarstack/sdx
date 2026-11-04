/* CUDA half of the SDX C ABI (see include/sdx.h). Every entry point takes host buffers. */
#include "sdx.h"

#include <cuda_runtime.h>
#include <math.h>

#include <vector>

namespace {

/* Device buffers freed on scope exit; any failed CUDA call short-circuits to -1. */
struct DeviceBuffers {
    std::vector<void *> ptrs;
    ~DeviceBuffers() {
        for (void *p : ptrs) cudaFree(p);
    }
    template <typename T>
    T *alloc(size_t bytes) {
        void *p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        ptrs.push_back(p);
        return static_cast<T *>(p);
    }
};

inline bool launched_ok() { return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess; }

constexpr int kMaxBlurRadius = 7;

__global__ void k_gaussian_blur_latent(const float *__restrict__ src, float *__restrict__ dst, int B, int C, int H,
                                       int W, float sigma, int r) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    if (ow >= W || oh >= H || bc >= B * C) return;

    float g[2 * kMaxBlurRadius + 1];
    const float s2 = 2.0f * sigma * sigma + 1e-8f;
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        g[i + r] = expf(-(float)(i * i) / s2);
        sum += g[i + r];
    }
    for (int i = 0; i < 2 * r + 1; ++i) g[i] /= sum;

    const float *plane = src + (size_t)bc * H * W;
    float acc = 0.0f;
    for (int ky = 0; ky < 2 * r + 1; ++ky) {
        const int sy = oh + ky - r;
        if (sy < 0 || sy >= H) continue;
        for (int kx = 0; kx < 2 * r + 1; ++kx) {
            const int sx = ow + kx - r;
            if (sx < 0 || sx >= W) continue;
            acc += g[ky] * g[kx] * plane[sy * W + sx];
        }
    }
    dst[(size_t)bc * H * W + oh * W + ow] = acc;
}

/* Grid-stride per-thread totals, warp-reduced, one atomic per warp (was one atomic per pixel). */
__global__ void k_luma_stats_u8(const uint8_t *__restrict__ src, int n_pixels, int channels, uint8_t clip_low,
                                uint8_t clip_high, unsigned long long *__restrict__ out_sum,
                                unsigned long long *__restrict__ out_clipped) {
    unsigned long long sum = 0, clipped = 0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_pixels; idx += gridDim.x * blockDim.x) {
        const uint8_t *px = src + (size_t)idx * channels;
        const uint8_t y = channels <= 1 ? px[0] : (uint8_t)((77u * px[0] + 150u * px[1] + 29u * px[2] + 128u) >> 8);
        sum += y;
        clipped += (y <= clip_low || y >= clip_high);
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        clipped += __shfl_down_sync(0xffffffffu, clipped, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(out_sum, sum);
        atomicAdd(out_clipped, clipped);
    }
}

__global__ void k_l2_normalize_rows(float *data, int n_rows, int dim, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float *r = data + (size_t)row * dim;
    float s = 0.f;
    for (int i = 0; i < dim; ++i) s += r[i] * r[i];
    const float inv = rsqrtf(fmaxf(s, eps * eps));
    for (int i = 0; i < dim; ++i) r[i] *= inv;
}

/* One thread per candidate (was a single thread looping over all candidates). */
__global__ void k_dot_rows(const float *__restrict__ query, const float *__restrict__ candidates, int n_cand, int dim,
                           float *__restrict__ scores) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cand) return;
    const float *row = candidates + (size_t)i * dim;
    float s = 0.f;
    for (int k = 0; k < dim; ++k) s += query[k] * row[k];
    scores[i] = s;
}

}  // namespace

extern "C" int sdx_cuda_gaussian_blur_latent_f32(const float *src_host, float *dst_host, int B, int C, int H, int W,
                                                 float sigma) {
    if (!src_host || !dst_host || B <= 0 || C <= 0 || H <= 0 || W <= 0 || sigma <= 0.0f) return -2;
    const int r = (int)fminf((float)kMaxBlurRadius, fmaxf(1.0f, roundf(sigma * 2.0f)));
    const size_t nbytes = (size_t)B * C * H * W * sizeof(float);
    DeviceBuffers buf;
    float *d_src = buf.alloc<float>(nbytes);
    float *d_dst = buf.alloc<float>(nbytes);
    if (!d_src || !d_dst || cudaMemcpy(d_src, src_host, nbytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    k_gaussian_blur_latent<<<dim3((W + 15) / 16, (H + 15) / 16, B * C), dim3(16, 16, 1)>>>(d_src, d_dst, B, C, H, W,
                                                                                          sigma, r);
    if (!launched_ok()) return -1;
    return cudaMemcpy(dst_host, d_dst, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

extern "C" int sdx_cuda_luma_stats_u8_host(const uint8_t *host_src, int height, int width, int channels,
                                           uint8_t clip_low, uint8_t clip_high, float *out_mean_luma,
                                           float *out_clip_ratio) {
    if (!host_src || !out_mean_luma || !out_clip_ratio || height <= 0 || width <= 0 || channels <= 0) return -2;
    const int n_pixels = height * width;
    const size_t src_bytes = (size_t)n_pixels * channels;
    DeviceBuffers buf;
    uint8_t *d_src = buf.alloc<uint8_t>(src_bytes);
    unsigned long long *d_acc = buf.alloc<unsigned long long>(2 * sizeof(unsigned long long));
    if (!d_src || !d_acc || cudaMemcpy(d_src, host_src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_acc, 0, 2 * sizeof(unsigned long long)) != cudaSuccess) {
        return -1;
    }
    const int threads = 256;
    const int blocks = (n_pixels + threads - 1) / threads < 1024 ? (n_pixels + threads - 1) / threads : 1024;
    k_luma_stats_u8<<<blocks, threads>>>(d_src, n_pixels, channels, clip_low, clip_high, d_acc, d_acc + 1);
    unsigned long long acc[2] = {0, 0};
    if (!launched_ok() || cudaMemcpy(acc, d_acc, sizeof(acc), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    *out_mean_luma = (float)((double)acc[0] / (double)n_pixels);
    *out_clip_ratio = (float)((double)acc[1] / (double)n_pixels);
    return 0;
}

extern "C" int sdx_cuda_l2_normalize_rows_f32_host(float *host_data, int n_rows, int dim, float eps) {
    if (!host_data || n_rows <= 0 || dim <= 0) return -2;
    const size_t nbytes = (size_t)n_rows * dim * sizeof(float);
    DeviceBuffers buf;
    float *d = buf.alloc<float>(nbytes);
    if (!d || cudaMemcpy(d, host_data, nbytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    k_l2_normalize_rows<<<(n_rows + 255) / 256, 256>>>(d, n_rows, dim, eps > 0.f ? eps : 1e-8f);
    if (!launched_ok()) return -1;
    return cudaMemcpy(host_data, d, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

extern "C" int sdx_cuda_style_pick_best_f32_host(const float *query, const float *candidates, int n_cand, int dim,
                                                 int *out_index, float *out_score) {
    if (!query || !candidates || n_cand <= 0 || dim <= 0 || !out_index || !out_score) return -2;
    const size_t qbytes = (size_t)dim * sizeof(float);
    const size_t cbytes = (size_t)n_cand * dim * sizeof(float);
    DeviceBuffers buf;
    float *d_q = buf.alloc<float>(qbytes);
    float *d_c = buf.alloc<float>(cbytes);
    float *d_s = buf.alloc<float>((size_t)n_cand * sizeof(float));
    if (!d_q || !d_c || !d_s || cudaMemcpy(d_q, query, qbytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_c, candidates, cbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    k_dot_rows<<<(n_cand + 255) / 256, 256>>>(d_q, d_c, n_cand, dim, d_s);
    std::vector<float> scores((size_t)n_cand);
    if (!launched_ok() ||
        cudaMemcpy(scores.data(), d_s, scores.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -1;
    }
    float best = -2.f;
    int best_i = 0;
    for (int i = 0; i < n_cand; ++i) {
        if (scores[i] > best) {
            best = scores[i];
            best_i = i;
        }
    }
    *out_index = best_i;
    *out_score = best;
    return 0;
}
