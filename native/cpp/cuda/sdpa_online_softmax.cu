#include "sdx/sdpa_online_softmax.h"

#include <cmath>
#include <cuda_runtime.h>

static constexpr int kHeadDim = 64;
static constexpr int kTile = 32;

/**
 * One CUDA block = one (query position, head). Online softmax over key positions in tiles of kTile.
 * Layout: (1, H, S, 64) row-major contiguous.
 */
__global__ void sdx_k_sdpa_online(const float *__restrict__ Q, const float *__restrict__ K,
                                  const float *__restrict__ V, float *__restrict__ O, int H, int S, float scale) {
    const int qi = blockIdx.x;
    const int h = blockIdx.y;
    if (qi >= S || h >= H) {
        return;
    }

    __shared__ float sm_q[kHeadDim];
    __shared__ float sm_k[kTile][kHeadDim];
    __shared__ float sm_v[kTile][kHeadDim];
    __shared__ float scores[kTile];
    __shared__ float acc[kHeadDim];
    __shared__ float m_stat;
    __shared__ float l_stat;

    const int q_off = (h * S + qi) * kHeadDim;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        sm_q[d] = Q[q_off + d];
    }
    if (threadIdx.x == 0) {
        m_stat = -1e30f;
        l_stat = 0.f;
        for (int d = 0; d < kHeadDim; ++d) {
            acc[d] = 0.f;
        }
    }
    __syncthreads();

    for (int j0 = 0; j0 < S; j0 += kTile) {
        const int valid = min(kTile, S - j0);
        for (int t = threadIdx.x; t < kTile * kHeadDim; t += blockDim.x) {
            const int kk = t / kHeadDim;
            const int d = t % kHeadDim;
            if (kk < valid) {
                const int j = j0 + kk;
                const int kv_off = (h * S + j) * kHeadDim;
                sm_k[kk][d] = K[kv_off + d];
                sm_v[kk][d] = V[kv_off + d];
            }
        }
        __syncthreads();

        if (threadIdx.x < kTile) {
            const int kk = threadIdx.x;
            if (kk < valid) {
                float s = 0.f;
#pragma unroll
                for (int d = 0; d < kHeadDim; ++d) {
                    s += sm_q[d] * sm_k[kk][d];
                }
                scores[kk] = s * scale;
            } else {
                scores[kk] = -1e30f;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float m_row = scores[0];
            for (int k = 1; k < valid; ++k) {
                m_row = fmaxf(m_row, scores[k]);
            }
            const float m_prev = m_stat;
            const float m_new = fmaxf(m_prev, m_row);
            const float alpha = expf(m_prev - m_new);
            float sum_p = 0.f;
            for (int k = 0; k < valid; ++k) {
                sum_p += expf(scores[k] - m_new);
            }
            const float l_new = alpha * l_stat + sum_p;
            for (int d = 0; d < kHeadDim; ++d) {
                float add = 0.f;
                for (int k = 0; k < valid; ++k) {
                    add += expf(scores[k] - m_new) * sm_v[k][d];
                }
                acc[d] = alpha * acc[d] + add;
            }
            m_stat = m_new;
            l_stat = l_new;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float invl = l_stat > 0.f ? (1.f / l_stat) : 0.f;
        const int o_off = (h * S + qi) * kHeadDim;
        for (int d = 0; d < kHeadDim; ++d) {
            O[o_off + d] = acc[d] * invl;
        }
    }
}

extern "C" int sdx_cuda_sdpa_online_f32_host(const float *Q, const float *K, const float *V, float *O, int32_t num_heads,
                                              int32_t seq_len, float scale) {
    if (!Q || !K || !V || !O || num_heads <= 0 || num_heads > 64 || seq_len <= 0 || seq_len > 2048) {
        return -2;
    }
    const size_t nfloat = static_cast<size_t>(num_heads) * static_cast<size_t>(seq_len) * 64u;
    const size_t nbytes = nfloat * sizeof(float);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&dQ), nbytes);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&dK), nbytes);
    if (e != cudaSuccess) {
        cudaFree(dQ);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&dV), nbytes);
    if (e != cudaSuccess) {
        cudaFree(dQ);
        cudaFree(dK);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&dO), nbytes);
    if (e != cudaSuccess) {
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        return -1;
    }
    e = cudaMemcpy(dQ, Q, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(dK, K, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(dV, V, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    {
        dim3 grid(seq_len, num_heads);
        sdx_k_sdpa_online<<<grid, 128>>>(dQ, dK, dV, dO, num_heads, seq_len, scale);
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(O, dO, nbytes, cudaMemcpyDeviceToHost);
fail:
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    return e == cudaSuccess ? 0 : -1;
}
