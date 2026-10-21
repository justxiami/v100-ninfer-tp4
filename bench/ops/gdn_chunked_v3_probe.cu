// Chunked GDN core probe, v3: chunk = 64 tokens (FLA's BC), not llama.cpp's 16. With 8192 tokens
// that is 128 sequential inter-chunk steps instead of 512, which is where the 16-token version
// spends its time. The state is still tiled along the value dimension (BV) and kept in FP32
// registers, and decay is still applied in log space.
//
// Three structural lessons taken from that implementation:
//   1. chunk = 64 tokens. The intra-chunk coupling matrix is 16x16, so the (I+L)^-1 solve is 16
//      sequential steps per output column, and there are 4x more chunks => 4x more blocks.
//   2. the recurrent state is tiled along the value dimension (BV) so a block only holds BK x BV
//      floats: it stays in FP32 registers, with an fp16 mirror in shared memory for the WMMA
//      operands. This is what removes the register spill that made v1 100x slow.
//   3. decay is applied in log space: exp(g_cum[t] - g_cum[s]), never P_t / P_s.
//
// kernel 1 (prepare, one block per (head, chunk)):
//    g_cum prefix sum -> L[t][s] = beta_t exp(g_cum_t - g_cum_s) (k_t . k_s) for s < t
//    forward substitution: k_cumdecay = (I+L)^-1 (beta exp(g_cum) k), v_corr = (I+L)^-1 (beta v)
//    qk_buf[t][s] = (q_t . k_s) exp(g_cum_t - g_cum_s) for s <= t
// kernel 2 (state + output, one block per (head, v-tile), sequential over chunks):
//    v_new = v_corr - k_cumdecay @ H ; o_inter = (q scale) @ H * exp(g_cum_t)
//    H = exp(g_last) H + Vnew^T (k * exp(g_last - g_cum)) ; o = o_inter + qk @ Vnew

#include "ninfer_bench_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace ninfer;
namespace wmma = nvcuda::wmma;

namespace {

constexpr int kD  = 128;   // head dim
constexpr int kCS = 32;    // chunk size
constexpr int kBV = 16;    // value-dim tile per block
constexpr int kNT = 64;   // threads in the state kernel

__device__ __forceinline__ __half to_h(float v) { return __float2half(v); }

// prepare: L, k_cumdecay, v_corr, qk_buf for one (head, chunk)
__global__ void prepare_kernel(const __half* __restrict__ k, const __half* __restrict__ q,
                               const __half* __restrict__ v, const float* __restrict__ g,
                               const float* __restrict__ beta, float* __restrict__ kcd,
                               float* __restrict__ vcorr, float* __restrict__ qk,
                               float* __restrict__ gcum_out, __half* __restrict__ kcd16,
                               __half* __restrict__ qs16, __half* __restrict__ ks16,
                               __half* __restrict__ qk16, int chunks, float scale) {
    extern __shared__ __align__(32) unsigned char smem_rawp[];
    float (*s_k)[kD] = reinterpret_cast<float (*)[kD]>(smem_rawp);
    float* s_l       = reinterpret_cast<float*>(s_k + kCS);
    __shared__ float s_g[kCS];
    __shared__ float s_b[kCS];

    const int head  = blockIdx.x;
    const int chunk = blockIdx.y;
    const int tid   = threadIdx.x;
    const std::int64_t t0 = static_cast<std::int64_t>(chunk) * kCS;

    for (int i = tid; i < kCS * kD; i += blockDim.x) {
        s_k[i / kD][i % kD] = __half2float(k[(t0 + i / kD) * kD + (i % kD)]);
    }
    if (tid < kCS) {
        const std::int64_t idx = static_cast<std::int64_t>(head) * chunks * kCS + t0 + tid;
        s_g[tid] = g[idx];
        s_b[tid] = beta[idx];
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < kCS; ++t) { acc += s_g[t]; s_g[t] = acc; }
    }
    __syncthreads();
    if (tid < kCS) { gcum_out[static_cast<std::int64_t>(head) * chunks * kCS + t0 + tid] = s_g[tid]; }
    __syncthreads();

    // Gram and qk are matmuls: do them on tensor cores instead of scalar dot loops.
    __half* s_buf = reinterpret_cast<__half*>(s_l + kCS * kCS);   // [CS][kD] fp16 operands
    __half* s_kh  = s_buf;                                        // K
    __half* s_qh  = s_buf + kCS * kD;                             // Q * scale
    for (int i = tid; i < kCS * kD; i += blockDim.x) {
        s_kh[i] = to_h(s_k[i / kD][i % kD]);
        s_qh[i] = to_h(__half2float(q[(t0 + i / kD) * kD + (i % kD)]) * scale);
    }
    __syncthreads();
    {
        const int warp = tid / 32;
        const int tiles = (kCS / 16) * (kCS / 16);
        for (int tile = warp; tile < tiles; tile += blockDim.x / 32) {
            const int ti = tile / (kCS / 16);
            const int tj = tile % (kCS / 16);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < kD; kk += 16) {
                wmma::load_matrix_sync(a, s_kh + ti * 16 * kD + kk, kD);
                wmma::load_matrix_sync(b, s_kh + tj * 16 * kD + kk, kD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(s_l + ti * 16 * kCS + tj * 16, acc, kCS,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = tid; i < kCS * kCS; i += blockDim.x) {
        const int t = i / kCS, s = i % kCS;
        s_l[i] = (s < t) ? s_b[t] * __expf(s_g[t] - s_g[s]) * s_l[i] : 0.0f;
    }
    __syncthreads();

    // k_cumdecay = (I+L)^-1 (beta * exp(g_cum) * k)
    // k_cumdecay and v_corr are independent solves of the same triangular system: run them on
    // two halves of the block instead of back to back.
    if (tid < kD) {
        float x[kCS];
#pragma unroll
        for (int t = 0; t < kCS; ++t) {
            float xt = s_b[t] * __expf(s_g[t]) * s_k[t][tid];
#pragma unroll
            for (int ss = 0; ss < t; ++ss) { xt -= s_l[t * kCS + ss] * x[ss]; }
            x[t] = xt;
        }
        const std::int64_t base = (static_cast<std::int64_t>(head) * chunks + chunk) * kCS * kD;
        for (int t = 0; t < kCS; ++t) {
            kcd[base + t * kD + tid]   = x[t];
            kcd16[base + t * kD + tid] = to_h(x[t]);
        }
    }
    if (tid >= kD && tid < 2 * kD) {
        const int col = tid - kD;
        float x[kCS];
#pragma unroll
        for (int t = 0; t < kCS; ++t) {
            float xt = s_b[t] * __half2float(v[(static_cast<std::int64_t>(head) * chunks * kCS + t0 + t) * kD + col]);
#pragma unroll
            for (int ss = 0; ss < t; ++ss) { xt -= s_l[t * kCS + ss] * x[ss]; }
            x[t] = xt;
        }
        for (int t = 0; t < kCS; ++t) {
            vcorr[(static_cast<std::int64_t>(head) * chunks + chunk) * kCS * kD + t * kD + col] = x[t];
        }
    }
    __syncthreads();

    // fp16 operands consumed by the state kernel straight from global memory
    {
        const std::int64_t base = (static_cast<std::int64_t>(head) * chunks + chunk) * kCS * kD;
        const float g_last      = s_g[kCS - 1];
        for (int i = tid; i < kCS * kD; i += blockDim.x) {
            const int t = i / kD, d = i % kD;
            qs16[base + i] = to_h(__half2float(q[(t0 + t) * kD + d]) * scale);
            ks16[base + i] = to_h(__half2float(k[(t0 + t) * kD + d]) * __expf(g_last - s_g[t]));
        }
    }
    __syncthreads();

    // qk_buf[t][s] = scale * (q_t . k_s) * exp(g_cum_t - g_cum_s), s <= t
    __syncthreads();
    for (int i = tid; i < kCS * kCS; i += blockDim.x) { s_l[i] = 0.0f; }
    __syncthreads();
    {
        const int warp = tid / 32;
        const int tiles = (kCS / 16) * (kCS / 16);
        for (int tile = warp; tile < tiles; tile += blockDim.x / 32) {
            const int ti = tile / (kCS / 16);
            const int tj = tile % (kCS / 16);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < kD; kk += 16) {
                wmma::load_matrix_sync(a, s_qh + ti * 16 * kD + kk, kD);
                wmma::load_matrix_sync(b, s_kh + tj * 16 * kD + kk, kD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(s_l + ti * 16 * kCS + tj * 16, acc, kCS,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = tid; i < kCS * kCS; i += blockDim.x) {
        const int t = i / kCS, s = i % kCS;
        const float val = (s <= t) ? s_l[i] * __expf(s_g[t] - s_g[s]) : 0.0f;
        const std::int64_t qbase = (static_cast<std::int64_t>(head) * chunks + chunk) * kCS * kCS;
        qk[qbase + i]   = val;
        qk16[qbase + i] = to_h(val);
    }
}

// state + output: one block per (head, v tile), sequential over chunks
__global__ void state_kernel(const __half* __restrict__ k, const __half* __restrict__ q,
                             const __half* __restrict__ kcd, const float* __restrict__ vcorr,
                             const __half* __restrict__ qk, const float* __restrict__ gcum,
                             const float* __restrict__ state_in, float* __restrict__ state_out,
                             __half* __restrict__ out, int chunks, float scale,
                             const __half* __restrict__ qs, const __half* __restrict__ ks) {
    extern __shared__ __align__(32) unsigned char smem_raw3[];
    __half* s_h   = reinterpret_cast<__half*>(smem_raw3);           // [v][k] fp16 mirror
    __half* s_vn  = s_h + kBV * kD;                                 // v_new fp16
    float*  s_res = reinterpret_cast<float*>(s_vn + kCS * kBV);     // CS x BV fp32
    float*  s_oi  = s_res + kCS * kBV;                              // CS x BV fp32 (o_inter)
    float*  s_g   = s_oi + kCS * kBV;                               // CS fp32
    float*  s_hd  = s_g + kCS;                                      // delta H, [k][v] fp32

    const int head  = blockIdx.x;
    const int vtile = blockIdx.y;
    const int v0    = vtile * kBV;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;

    constexpr int kEptH = (kD * kBV) / kNT;   // 32 floats of state per thread
    constexpr int kEpt  = (kCS * kBV) / kNT;  // 4 elements of a CS x BV tile per thread

    float h[kEptH];
    const std::int64_t sbase = static_cast<std::int64_t>(head) * kD * kD;
    for (int j = 0; j < kEptH; ++j) {
        const int idx = tid + j * kNT;                  // idx = v_local * kD + kk
        h[j] = state_in[sbase + static_cast<std::int64_t>(v0 + idx / kD) * kD + (idx % kD)];
        s_h[idx] = to_h(h[j]);
    }
    __syncthreads();

    for (int c = 0; c < chunks; ++c) {
        const std::int64_t cbase = (static_cast<std::int64_t>(head) * chunks + c) * kCS * kD;
        const std::int64_t qkbase = (static_cast<std::int64_t>(head) * chunks + c) * kCS * kCS;
        const std::int64_t gbase = static_cast<std::int64_t>(head) * chunks * kCS + c * kCS;
        const std::int64_t t0 = static_cast<std::int64_t>(c) * kCS;

        for (int i = tid; i < kCS; i += kNT) { s_g[i] = gcum[gbase + i]; }
        __syncthreads();

        // 1) v_new = v_corr - k_cumdecay @ H  and  o_inter = (q scale) @ H.
        // Both contractions use the same H tile. They are mapped to different warps rather than to
        // two accumulators in one warp: 8 warps stay busy instead of 4, and each warp loads one A
        // fragment plus the shared B per k-step.
        {
            const int tiles = (kCS / 16) * (kBV / 16);
            const int half  = kNT / 64;              // warps per output (>=1; see static_assert)
            static_assert(kNT >= 64, "need at least two warps: one per output");
            const bool o_inter = (warp >= half);
            const int w        = o_inter ? warp - half : warp;
            for (int tile = w; tile < tiles; tile += half) {
                const int ti = tile / (kBV / 16);
                const int tj = tile % (kBV / 16);
                const __half* src = o_inter ? (qs + cbase) : (kcd + cbase);
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int kk = 0; kk < kD; kk += 16) {
                    wmma::load_matrix_sync(a, src + ti * 16 * kD + kk, kD);
                    wmma::load_matrix_sync(b, s_h + tj * 16 * kD + kk, kD);
                    wmma::mma_sync(acc, a, b, acc);
                }
                float* dst = o_inter ? (s_oi + ti * 16 * kBV + tj * 16) : (s_res + ti * 16 * kBV + tj * 16);
                wmma::store_matrix_sync(dst, acc, kBV, wmma::mem_row_major);
            }
            __syncthreads();
        }
        for (int j = 0; j < kEpt; ++j) {
            const int idx  = tid + j * kNT;
            const int t    = idx / kBV;
            const int vl   = idx % kBV;
            const float vc = vcorr[cbase + t * kD + v0 + vl];
            s_vn[idx]      = to_h(vc - s_res[idx]);
        }
        float oi[kEpt];
        for (int j = 0; j < kEpt; ++j) {
            const int idx = tid + j * kNT;
            oi[j] = s_oi[idx] * __expf(s_g[idx / kBV]);
        }
        __syncthreads();

        // 3) decay-scaled k for the state update, and qk staged for the intra-chunk output
        const float g_last = s_g[kCS - 1];
        __syncthreads();

        // 4+5) fused: delta H = Kch^T @ Vnew and o_intra = qk @ Vnew. Both contract over the
        // chunk's tokens with Vnew as B, so the tile loop runs once for both outputs.
        {
            const int tiles_h = (kD / 16) * (kBV / 16);
            const int tiles_o = (kCS / 16) * (kBV / 16);
            const int tiles   = tiles_h + tiles_o;
            for (int tile = warp; tile < tiles; tile += kNT / 32) {
                if (tile < tiles_h) {
                    const int ti = tile / (kBV / 16);
                    const int tj = tile % (kBV / 16);
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                    wmma::fill_fragment(acc, 0.0f);
                    for (int tt = 0; tt < kCS; tt += 16) {
                        wmma::load_matrix_sync(a, ks + cbase + tt * kD + ti * 16, kD);
                        wmma::load_matrix_sync(b, s_vn + tt * kBV + tj * 16, kBV);
                        wmma::mma_sync(acc, a, b, acc);
                    }
                    wmma::store_matrix_sync(s_hd + ti * 16 * kBV + tj * 16, acc, kBV,
                                            wmma::mem_row_major);
                } else {
                    const int t2 = tile - tiles_h;
                    const int ti = t2 / (kBV / 16);
                    const int tj = t2 % (kBV / 16);
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                    wmma::fill_fragment(acc, 0.0f);
                    for (int tt = 0; tt < kCS; tt += 16) {
                        wmma::load_matrix_sync(a, qk + qkbase + ti * 16 * kCS + tt, kCS);
                        wmma::load_matrix_sync(b, s_vn + tt * kBV + tj * 16, kBV);
                        wmma::mma_sync(acc, a, b, acc);
                    }
                    wmma::store_matrix_sync(s_res + ti * 16 * kBV + tj * 16, acc, kBV,
                                            wmma::mem_row_major);
                }
            }
            __syncthreads();
        }
        const float exp_g = __expf(g_last);
        for (int j = 0; j < kEptH; ++j) {
            const int idx = tid + j * kNT;   // idx = v_local * kD + kk
            const int vl  = idx / kD;
            const int kk  = idx % kD;
            h[j] = exp_g * h[j] + s_hd[kk * kBV + vl];   // s_hd is [k][v], state is [v][k]
            s_h[idx] = to_h(h[j]);
        }
        for (int j = 0; j < kEpt; ++j) {
            const int idx = tid + j * kNT;
            const int t   = idx / kBV;
            const int vl  = idx % kBV;
            out[(static_cast<std::int64_t>(head) * chunks * kCS + t0 + t) * kD + v0 + vl] =
                to_h(s_res[idx] + oi[j]);
        }
        __syncthreads();
    }

    for (int j = 0; j < kEptH; ++j) {
        const int idx = tid + j * kNT;
        state_out[sbase + static_cast<std::int64_t>(v0 + idx / kD) * kD + (idx % kD)] = h[j];
    }
}

float frand(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

} // namespace

int main(int argc, char** argv) {
    const int hv     = argc > 1 ? std::atoi(argv[1]) : 12;
    const int tokens = argc > 2 ? std::atoi(argv[2]) : 512;
    const int chunks = tokens / kCS;
    const char* dump = argc > 3 ? argv[3] : "/tmp/gdn_chunk_v2.bin";

    std::vector<__half> hq(static_cast<std::size_t>(tokens) * kD);
    std::vector<__half> hk(hq.size());
    std::vector<__half> hver(static_cast<std::size_t>(hv) * tokens * kD);
    std::vector<float> hg(static_cast<std::size_t>(hv) * tokens);
    std::vector<float> hbeta(static_cast<std::size_t>(hv) * tokens);
    std::vector<float> hstate_in(static_cast<std::size_t>(hv) * kD * kD);

    std::uint32_t seed = 987u;
    for (std::size_t t = 0; t < static_cast<std::size_t>(tokens); ++t) {
        float nk = 0.0f, nq = 0.0f;
        for (int d = 0; d < kD; ++d) {
            const float a = frand(seed), b = frand(seed);
            hk[t * kD + d] = __float2half(a);
            hq[t * kD + d] = __float2half(b);
            nk += a * a;
            nq += b * b;
        }
        nk = 1.0f / std::sqrt(nk);
        nq = 1.0f / std::sqrt(nq);
        for (int d = 0; d < kD; ++d) {
            hk[t * kD + d] = __float2half(__half2float(hk[t * kD + d]) * nk);
            hq[t * kD + d] = __float2half(__half2float(hq[t * kD + d]) * nq);
        }
    }
    for (int h = 0; h < hv; ++h) {
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < kD; ++d) {
                hver[(static_cast<std::size_t>(h) * tokens + t) * kD + d] = __float2half(0.5f * frand(seed));
            }
            hg[static_cast<std::size_t>(h) * tokens + t] = -0.05f;
            hbeta[static_cast<std::size_t>(h) * tokens + t] = 0.6f;
        }
        for (int q = 0; q < kD; ++q) {
            for (int d = 0; d < kD; ++d) {
                hstate_in[(static_cast<std::size_t>(h) * kD + q) * kD + d] = 0.2f * frand(seed);
            }
        }
    }

    const std::size_t n_qkv = static_cast<std::size_t>(hv) * tokens * kD;
    ninfer::DeviceBuffer dq(tokens * kD * sizeof(__half));
    ninfer::DeviceBuffer dk(tokens * kD * sizeof(__half));
    ninfer::DeviceBuffer dv(n_qkv * sizeof(__half));
    ninfer::DeviceBuffer dg(static_cast<std::size_t>(hv) * tokens * sizeof(float));
    ninfer::DeviceBuffer dbeta(dg.bytes);
    ninfer::DeviceBuffer dstate_in(static_cast<std::size_t>(hv) * kD * kD * sizeof(float));
    ninfer::DeviceBuffer dstate_out(dstate_in.bytes);
    ninfer::DeviceBuffer dkcd(static_cast<std::size_t>(hv) * chunks * kCS * kD * sizeof(float));
    ninfer::DeviceBuffer dvcorr(dkcd.bytes);
    ninfer::DeviceBuffer dqk(static_cast<std::size_t>(hv) * chunks * kCS * kCS * sizeof(float));
    ninfer::DeviceBuffer dgcum(static_cast<std::size_t>(hv) * chunks * kCS * sizeof(float));
    ninfer::DeviceBuffer dout(n_qkv * sizeof(__half));
    const std::size_t n_cs = static_cast<std::size_t>(hv) * chunks * kCS * kD;
    const std::size_t n_qq = static_cast<std::size_t>(hv) * chunks * kCS * kCS;
    ninfer::DeviceBuffer dkcd16(n_cs * sizeof(__half));
    ninfer::DeviceBuffer dqs16(n_cs * sizeof(__half));
    ninfer::DeviceBuffer dks16(n_cs * sizeof(__half));
    ninfer::DeviceBuffer dqk16(n_qq * sizeof(__half));

    CUDA_CHECK(cudaMemcpy(dq.p, hq.data(), dq.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk.p, hk.data(), dk.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.p, hver.data(), dv.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.p, hg.data(), dg.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta.p, hbeta.data(), dbeta.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dstate_in.p, hstate_in.data(), dstate_in.bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));

    constexpr std::size_t kSmemPrep =
        (kCS * kD + kCS * kCS) * sizeof(float) + 2 * kCS * kD * sizeof(__half);
    CUDA_CHECK(cudaFuncSetAttribute(prepare_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(kSmemPrep)));
    constexpr std::size_t kSmemState = (kBV * kD + kCS * kBV) * sizeof(__half) +
                                       (2 * kCS * kBV + kCS + kD * kBV) * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(kSmemState)));

    const auto run = [&](cudaStream_t s) {
        const dim3 gp(hv, chunks);
        if (getenv("SKIP_PREP") == nullptr) prepare_kernel<<<gp, 256, kSmemPrep, s>>>(
            static_cast<const __half*>(dk.p), static_cast<const __half*>(dq.p),
            static_cast<const __half*>(dv.p), static_cast<const float*>(dg.p),
            static_cast<const float*>(dbeta.p), static_cast<float*>(dkcd.p),
            static_cast<float*>(dvcorr.p), static_cast<float*>(dqk.p),
            static_cast<float*>(dgcum.p), static_cast<__half*>(dkcd16.p),
            static_cast<__half*>(dqs16.p), static_cast<__half*>(dks16.p),
            static_cast<__half*>(dqk16.p), chunks, scale);
        const dim3 gs(hv, kD / kBV);
        if (getenv("SKIP_STATE") == nullptr) state_kernel<<<gs, kNT, kSmemState, s>>>(
            static_cast<const __half*>(dk.p), static_cast<const __half*>(dq.p),
            static_cast<const __half*>(dkcd16.p), static_cast<const float*>(dvcorr.p),
            static_cast<const __half*>(dqk16.p), static_cast<const float*>(dgcum.p),
            static_cast<const float*>(dstate_in.p), static_cast<float*>(dstate_out.p),
            static_cast<__half*>(dout.p), chunks, scale, static_cast<const __half*>(dqs16.p),
            static_cast<const __half*>(dks16.p));
    };

    run(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ninfer::DeviceBuffer flush(256ULL << 20);
    bench::TimedGraph graph;
    graph.capture(stream, run);
    const bench::ColdTiming timing = bench::measure_cold_graph(graph, flush, stream, 3, 20);

    std::vector<__half> oout(n_qkv);
    std::vector<float> ostate(static_cast<std::size_t>(hv) * kD * kD);
    CUDA_CHECK(cudaMemcpy(oout.data(), dout.p, dout.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ostate.data(), dstate_out.p, dstate_out.bytes, cudaMemcpyDeviceToHost));

    std::ofstream f(dump, std::ios::binary);
    const std::int32_t meta[3] = {hv, tokens, kCS};
    f.write(reinterpret_cast<const char*>(meta), sizeof(meta));
    f.write(reinterpret_cast<const char*>(hq.data()), hq.size() * sizeof(__half));
    f.write(reinterpret_cast<const char*>(hk.data()), hk.size() * sizeof(__half));
    f.write(reinterpret_cast<const char*>(hver.data()), hver.size() * sizeof(__half));
    f.write(reinterpret_cast<const char*>(hg.data()), hg.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(hbeta.data()), hbeta.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(hstate_in.data()), hstate_in.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(ostate.data()), ostate.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(oout.data()), oout.size() * sizeof(__half));
    f.close();

    std::printf("v2 core: Hv=%d T=%d chunks(CS=%d)=%d  WY+state+output median %.3f ms/layer\n", hv,
                tokens, kCS, chunks, timing.median_us / 1000.0);
    std::printf("  48-layer extrapolation: %.1f ms (sequential today: 333.3 ms)\n",
                timing.median_us / 1000.0 * 48.0);
    return 0;
}
