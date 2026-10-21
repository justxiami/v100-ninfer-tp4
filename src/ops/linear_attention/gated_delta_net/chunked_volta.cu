// Chunked GDN forward for the VOLTA (sm_70) build.
//
// Two kernels, mirroring the validated probe (bench/ops/gdn_chunked_v3_probe.cu):
//   prepare: per (value head, chunk) -- g_cum prefix sum, the strictly lower coupling matrix
//            L[t][s] = beta_t exp(g_cum_t - g_cum_s) (k_t . k_s), forward substitution for both
//            the beta*exp(g_cum)*k and beta*v right-hand sides, and the masked qk product.
//   state:   per (value head, v tile) -- chunk-sequential scan
//            v_new = v_corr - k_cumdecay @ H, o_inter = (q scale) @ H
//            H <- exp(g_cum_last) H + Kch^T v_new,  o = o_inter + qk @ v_new
// The FP32 state stays in registers, tiled along the value dimension, with an fp16 mirror in
// shared memory for the WMMA operands; decay is always applied in log space.
//
// Layout notes: q/k/v/out are [T][heads][D] (token-major), g/beta are [T][heads], and the
// recurrent state is [dv][qk] per head, matching the sequential kernel.

#include "ops/linear_attention/gated_delta_net/chunked_volta.cuh"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kD  = kStateDim;                 // 128
constexpr int kCS = kChunkedTokens;            // 32
constexpr int kBV = 16;                        // value-dim tile per state block
constexpr int kNT = 64;                        // threads in the state kernel
constexpr float kL2Eps = 1.0e-6f;

__device__ __forceinline__ __half to_h(float v) { return __float2half(v); }

struct Layout {
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
    std::int32_t chunks;
    float scale;
};

// ------------------------------------------------------------------ prepare
__global__ void chunked_prepare_kernel(const __nv_bfloat16* __restrict__ q,
                                       const __nv_bfloat16* __restrict__ k,
                                       const __nv_bfloat16* __restrict__ v,
                                       const float* __restrict__ g,
                                       const float* __restrict__ beta, Layout layout,
                                       bool normalize_qk, __half* __restrict__ kcd,
                                       __half* __restrict__ qs, __half* __restrict__ ks,
                                       __half* __restrict__ qk_buf, float* __restrict__ vcorr,
                                       float* __restrict__ gcum) {
    extern __shared__ __align__(32) unsigned char smem_rawp[];
    float (*s_k)[kD] = reinterpret_cast<float (*)[kD]>(smem_rawp);
    float* s_l       = reinterpret_cast<float*>(s_k + kCS);
    __half* s_buf    = reinterpret_cast<__half*>(s_l + kCS * kCS);
    __half* s_kh     = s_buf;
    __half* s_qh     = s_buf + kCS * kD;
    __shared__ float s_g[kCS];
    __shared__ float s_b[kCS];

    const int head  = blockIdx.x;
    const int chunk = blockIdx.y;
    const std::int64_t t0 = static_cast<std::int64_t>(chunk) * kCS;

    // GQA mapping matches head_map::qk_head: value heads are blocked, not interleaved.
    const int group = layout.value_heads / layout.qk_heads;
    const int h_qk = head / group;

    for (int i = threadIdx.x; i < kCS * kD; i += blockDim.x) {
        const int t = i / kD, d = i % kD;
        s_k[t][d] = __bfloat162float(k[(t0 + t) * layout.qk_heads * kD + h_qk * kD + d]);
    }
    if (threadIdx.x < kCS) {
        s_g[threadIdx.x] = g[(t0 + threadIdx.x) * layout.value_heads + head];
        s_b[threadIdx.x] = beta[(t0 + threadIdx.x) * layout.value_heads + head];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int t = 0; t < kCS; ++t) {
            acc += s_g[t];
            s_g[t] = acc;
        }
    }
    __syncthreads();
    if (threadIdx.x < kCS) {
        gcum[(static_cast<std::int64_t>(head) * layout.chunks + chunk) * kCS + threadIdx.x] =
            s_g[threadIdx.x];
    }
    __syncthreads();

    // L2-normalise q/k rows when the caller asked for it (the op passes raw BF16 rows); this
    // matches normalize_qk_lane's rsqrtf(sum + eps).
    if (normalize_qk) {
        for (int t = threadIdx.x; t < kCS; t += blockDim.x) {
            float nk = 0.0f;
            float nq = 0.0f;
            for (int d = 0; d < kD; ++d) {
                const float kv = s_k[t][d];
                const float qv = __bfloat162float(q[(t0 + t) * layout.qk_heads * kD + h_qk * kD + d]);
                nk += kv * kv;
                nq += qv * qv;
            }
            const float ik = rsqrtf(nk + kL2Eps);
            const float iq = rsqrtf(nq + kL2Eps);
            for (int d = 0; d < kD; ++d) {
                s_k[t][d] *= ik;
            }
            for (int d = 0; d < kD; ++d) {
                s_qh[t * kD + d] = to_h(
                    __bfloat162float(q[(t0 + t) * layout.qk_heads * kD + h_qk * kD + d]) * iq);
            }
        }
    } else {
        for (int i = threadIdx.x; i < kCS * kD; i += blockDim.x) {
            const int t = i / kD, d = i % kD;
            s_qh[i] = to_h(__bfloat162float(q[(t0 + t) * layout.qk_heads * kD + h_qk * kD + d]));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kCS * kD; i += blockDim.x) { s_kh[i] = to_h(s_k[i / kD][i % kD]); }
    __syncthreads();

    // Gram (tensor cores) then the decay/beta scaling and strict lower mask.
    {
        const int warp  = threadIdx.x / 32;
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
            wmma::store_matrix_sync(s_l + ti * 16 * kCS + tj * 16, acc, kCS, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kCS * kCS; i += blockDim.x) {
        const int t = i / kCS, s = i % kCS;
        s_l[i] = (s < t) ? s_b[t] * __expf(s_g[t] - s_g[s]) * s_l[i] : 0.0f;
    }
    __syncthreads();

    const std::int64_t base = (static_cast<std::int64_t>(head) * layout.chunks + chunk) * kCS * kD;
    const std::int64_t gbase = (static_cast<std::int64_t>(head) * layout.chunks + chunk) * kCS;
    // Two independent solves of the same triangular system on two halves of the block.
    if (threadIdx.x < kD) {
        const int col = threadIdx.x;
        float x[kCS];
#pragma unroll
        for (int t = 0; t < kCS; ++t) {
            float xt = s_b[t] * __expf(s_g[t]) * s_k[t][col];
#pragma unroll
            for (int ss = 0; ss < t; ++ss) { xt -= s_l[t * kCS + ss] * x[ss]; }
            x[t] = xt;
        }
        for (int t = 0; t < kCS; ++t) { kcd[base + t * kD + col] = to_h(x[t]); }
    } else if (threadIdx.x < 2 * kD) {
        const int col = threadIdx.x - kD;
        float x[kCS];
#pragma unroll
        for (int t = 0; t < kCS; ++t) {
            float xt = s_b[t] * __bfloat162float(v[(t0 + t) * layout.value_heads * kD + head * kD + col]);
#pragma unroll
            for (int ss = 0; ss < t; ++ss) { xt -= s_l[t * kCS + ss] * x[ss]; }
            x[t] = xt;
        }
        for (int t = 0; t < kCS; ++t) { vcorr[base + t * kD + col] = x[t]; }
    }
    __syncthreads();

    const float g_last = s_g[kCS - 1];
    // s_k already holds the (optionally normalised) k rows, so the decay-scaled copy is a
    // straight elementwise scale -- no second normalisation pass.
    for (int i = threadIdx.x; i < kCS * kD; i += blockDim.x) {
        const int t = i / kD;
        qs[base + i] = to_h(__half2float(s_qh[i]) * layout.scale);
        ks[base + i] = to_h(s_k[t][i % kD] * __expf(g_last - s_g[t]));
    }
    __syncthreads();

    // qk = scale * (q_t . k_s) * exp(g_cum_t - g_cum_s) for s <= t
    for (int i = threadIdx.x; i < kCS * kCS; i += blockDim.x) { s_l[i] = 0.0f; }
    __syncthreads();
    {
        const int warp  = threadIdx.x / 32;
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
            wmma::store_matrix_sync(s_l + ti * 16 * kCS + tj * 16, acc, kCS, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kCS * kCS; i += blockDim.x) {
        const int t = i / kCS, s = i % kCS;
        const float val =
            (s <= t) ? s_l[i] * layout.scale * __expf(s_g[t] - s_g[s]) : 0.0f;
        qk_buf[gbase * kCS + i] = to_h(val);
    }
}

// ------------------------------------------------------------------ state + output
__global__ void chunked_state_kernel(const __half* __restrict__ qk_buf,
                                     const __half* __restrict__ kcd,
                                     const __half* __restrict__ qs, const __half* __restrict__ ks,
                                     const float* __restrict__ vcorr,
                                     const float* __restrict__ gcum, Layout layout,
                                     const float* __restrict__ state_in,
                                     float* __restrict__ state_out,
                                     __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(32) unsigned char smem_raw3[];
    __half* s_h   = reinterpret_cast<__half*>(smem_raw3);        // [v][k] fp16 mirror
    __half* s_vn  = s_h + kBV * kD;                              // v_new fp16
    float* s_res  = reinterpret_cast<float*>(s_vn + kCS * kBV);  // CS x BV fp32
    float* s_oi   = s_res + kCS * kBV;
    float* s_g    = s_oi + kCS * kBV;
    float* s_hd   = s_g + kCS;                                   // delta H [k][v]
    float* s_part = s_hd + kD * kBV;

    const int head  = blockIdx.x;
    const int vtile = blockIdx.y;
    const int v0    = vtile * kBV;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;

    constexpr int kEptH = (kD * kBV) / kNT;
    constexpr int kEpt  = (kCS * kBV) / kNT;

    float h[kEptH];
    const std::int64_t sbase = static_cast<std::int64_t>(head) * kD * kD;
    for (int j = 0; j < kEptH; ++j) {
        const int idx = tid + j * kNT;
        h[j]          = state_in[sbase + static_cast<std::int64_t>(v0 + idx / kD) * kD + (idx % kD)];
        s_h[idx]      = to_h(h[j]);
    }
    __syncthreads();

    for (int c = 0; c < layout.chunks; ++c) {
        const std::int64_t cbase = (static_cast<std::int64_t>(head) * layout.chunks + c) * kCS * kD;
        const std::int64_t qbase = (static_cast<std::int64_t>(head) * layout.chunks + c) * kCS * kCS;
        const std::int64_t gbase = static_cast<std::int64_t>(head) * layout.chunks * kCS + c * kCS;
        const std::int64_t t0    = static_cast<std::int64_t>(c) * kCS;

        for (int i = tid; i < kCS; i += kNT) { s_g[i] = gcum[gbase + i]; }
        __syncthreads();

        // v_new and o_inter against the same H tiles, one warp group per output
        {
            const int half   = kNT / 64;
            const bool o_int = (warp >= half);
            const int w      = o_int ? warp - half : warp;
            const int tiles  = (kCS / 16) * (kBV / 16);
            for (int tile = w; tile < tiles; tile += half) {
                const int ti = tile / (kBV / 16);
                const int tj = tile % (kBV / 16);
                const __half* src = o_int ? (qs + cbase) : (kcd + cbase);
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int kk = 0; kk < kD; kk += 16) {
                    wmma::load_matrix_sync(a, src + ti * 16 * kD + kk, kD);
                    wmma::load_matrix_sync(b, s_h + tj * 16 * kD + kk, kD);
                    wmma::mma_sync(acc, a, b, acc);
                }
                float* dst = o_int ? (s_oi + ti * 16 * kBV + tj * 16)
                                   : (s_res + ti * 16 * kBV + tj * 16);
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
            oi[j]         = s_oi[idx] * __expf(s_g[idx / kBV]);
        }
        __syncthreads();

        const float g_last = s_g[kCS - 1];
        // delta H and o_intra: both contract over the chunk's tokens with v_new as B
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
                        wmma::load_matrix_sync(a, qk_buf + qbase + ti * 16 * kCS + tt, kCS);
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
            const int idx = tid + j * kNT;
            const int vl  = idx / kD;
            const int kk  = idx % kD;
            h[j]          = exp_g * h[j] + s_hd[kk * kBV + vl];
            s_h[idx]      = to_h(h[j]);
        }
        for (int j = 0; j < kEpt; ++j) {
            const int idx = tid + j * kNT;
            const int t   = idx / kBV;
            const int vl  = idx % kBV;
            out[(t0 + t) * layout.value_heads * kD + head * kD + v0 + vl] =
                __float2bfloat16(s_res[idx] + oi[j]);
        }
        __syncthreads();
    }

    for (int j = 0; j < kEptH; ++j) {
        const int idx = tid + j * kNT;
        state_out[sbase + static_cast<std::int64_t>(v0 + idx / kD) * kD + (idx % kD)] = h[j];
    }
}

struct Smem {
    static constexpr std::size_t prepare = (kCS * kD + kCS * kCS) * sizeof(float) +
                                           2 * kCS * kD * sizeof(__half);
    static constexpr std::size_t state =
        (kBV * kD + kCS * kBV) * sizeof(__half) +
        (2 * kCS * kBV + kCS + kD * kBV + (kNT / 32) * 256) * sizeof(float);
};

} // namespace

void launch_chunked_volta(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          cudaStream_t stream) {
    const std::int32_t tokens = q.ne[2];
    Layout layout{q.ne[1], v.ne[1], tokens, (tokens + kCS - 1) / kCS, scale};
    if (layout.value_heads % layout.qk_heads != 0) {
        throw std::invalid_argument("chunked gdn requires value_heads % qk_heads == 0");
    }
    if (tokens % kCS != 0) {
        throw std::invalid_argument("chunked gdn requires tokens % chunk == 0");
    }

    [[maybe_unused]] auto scope = ws.scope();
    const std::int64_t elems    = static_cast<std::int64_t>(layout.value_heads) * tokens * kD;
    const std::int64_t qk_elems = static_cast<std::int64_t>(layout.value_heads) * layout.chunks *
                                  kCS * kCS;
    __half* kcd   = static_cast<__half*>(ws.alloc_bytes(elems * sizeof(__half)).data);
    __half* qs    = static_cast<__half*>(ws.alloc_bytes(elems * sizeof(__half)).data);
    __half* ksb   = static_cast<__half*>(ws.alloc_bytes(elems * sizeof(__half)).data);
    float* vcorr  = static_cast<float*>(ws.alloc_bytes(elems * sizeof(float)).data);
    __half* qkb   = static_cast<__half*>(ws.alloc_bytes(qk_elems * sizeof(__half)).data);
    float* gcum   = static_cast<float*>(
        ws.alloc_bytes(static_cast<std::int64_t>(layout.value_heads) * tokens * sizeof(float))
            .data);

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(chunked_prepare_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(Smem::prepare));
        cudaFuncSetAttribute(chunked_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(Smem::state));
        configured = true;
    }

    const dim3 gp(static_cast<unsigned>(layout.value_heads), static_cast<unsigned>(layout.chunks));
    chunked_prepare_kernel<<<gp, 2 * kD, Smem::prepare, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), layout, normalize_qk, kcd, qs, ksb, qkb, vcorr,
        gcum);
    CUDA_CHECK(cudaGetLastError());
    if (std::getenv("NINFER_CHUNK_DEBUG") != nullptr) {
        cudaStreamSynchronize(stream);
        __half h[2] = {0, 0};
        float f[2]  = {0, 0};
        cudaMemcpy(h, kcd, sizeof(__half), cudaMemcpyDeviceToHost);
        float kcdf = __half2float(h[0]);
        cudaMemcpy(h, qs, sizeof(__half), cudaMemcpyDeviceToHost);
        float qsf = __half2float(h[0]);
        cudaMemcpy(f, vcorr, sizeof(float), cudaMemcpyDeviceToHost);
        float vcf = f[0];
        cudaMemcpy(h, qkb, sizeof(__half), cudaMemcpyDeviceToHost);
        float qkf = __half2float(h[0]);
        cudaMemcpy(f, gcum, sizeof(float), cudaMemcpyDeviceToHost);
        std::printf("[chunk debug] kcd=%.6f qs=%.6f vcorr=%.6f qk=%.6f gcum=%.6f\n", kcdf, qsf,
                    vcf, qkf, f[0]);
    }
    const dim3 gs(static_cast<unsigned>(layout.value_heads),
                  static_cast<unsigned>(kD / kBV));
    chunked_state_kernel<<<gs, kNT, Smem::state, stream>>>(
        qkb, kcd, qs, ksb, vcorr, gcum, layout, static_cast<const float*>(state_in.data),
        static_cast<float*>(state_out.data), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
    if (std::getenv("NINFER_CHUNK_DEBUG") != nullptr) {
        cudaStreamSynchronize(stream);
        __half oh = __float2half(0.0f);
        float  of = 0.0f;
        cudaMemcpy(&oh, out.data, sizeof(__half), cudaMemcpyDeviceToHost);
        cudaMemcpy(&of, state_out.data, sizeof(float), cudaMemcpyDeviceToHost);
        std::printf("[chunk debug] after state: out[0]=%.6f state_out[0]=%.6f (out ptr %p)\n",
                    __half2float(oh), of, out.data);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
