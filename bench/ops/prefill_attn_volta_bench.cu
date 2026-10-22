// ninfer::bench - V100 prefill attention sandbox (standalone, no ops/ deps).
//
// Why this file exists
// --------------------
// The vendored llama.cpp flash-attention kernel (third_party/llama_cpp_fattn) is the current
// V100 prefill route and measures 35.5 TFLOPS (28.4% of the 125 TFLOPS fp16 peak) on the
// production geometry: per rank 6 q heads, 1 kv head, head_dim 256, int8 KV, causal. Three
// configuration probes around it are all worse (ncols 64: attention +38%; occupancy 4 with a
// narrower combine buffer: -33% at 32k; 16 warps/SM does not even compile because nbatch_fa
// must divide np*8), so the next step is a kernel of our own.
//
// Iterating that kernel through the engine costs ~3 minutes per data point (stop, start,
// measure, restore). This sandbox gives the same measurement in seconds: it builds Q/K/V
// directly, runs a kernel, and reports TFLOPS for the exact production shape.
//
// Baseline to beat: 35.5 TFLOPS at avg_ctx > 32k. Gate for the rewrite: 42.6 TFLOPS (1.2x).
//
// What is here now
// ----------------
// A SIMT (no tensor core) reference kernel. Its job is to close the question "is the SIMT
// version good enough to skip the mma work?" -- it is not, and the number it prints is the
// floor to compare any future core against -- and to prove the harness (shapes, causal
// masking, online softmax, int8 KV with per-64-group scales, CPU cross-check) end to end.
//
// The mma core that replaces it is designed in
// ~/bench-results/2026-10-22/attn-ceiling/PHASE2-DESIGN.md: 8 warps, Br=32 (4 row-groups x 2
// dim-splits), Bc=32, K/V int8 double-buffered (2-stage ld.global -> smem, sm_70 has no
// cp.async), Q in smem, target 16 warps/SM. Its QK/PV fragment code should be templated from
// src/ops/kernel/gqa_attention_prefill_volta.cuh (the in-tree m8n8k4 kernel), which already
// solves the Volta register wall via the dim split.

#include "ops/common/volta_mma.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1);} } while (0)

namespace {

constexpr int kHeadDim   = 256;   // production head_dim (rotary 64 of 256 is partial, pre-attention)
constexpr int kQHeads     = 6;    // per rank at tp4 (24 / 4)
constexpr int kKvHeads    = 1;    // per rank at tp4 (4 / 4)
constexpr int kQuantGroup = 64;   // KV int8 scale group along the head dim
constexpr int kGroups     = kHeadDim / kQuantGroup;
constexpr int kBr         = 8;    // rows (q tokens) per CTA in the SIMT reference
constexpr int kBc         = 8;    // keys per KV tile (smem budget: 2 CTAs/SM)
constexpr int kThreads    = 256;  // 8 warps, one warp per row

// ---------------------------------------------------------------------------
// SIMT reference kernel: one warp per q row, one CTA per kBr rows, one q head per grid.y.
// Lane l owns head dims {l, l+32, ..., l+224} (8 values) for both the dot product and the PV
// accumulation, so no cross-lane dim reduction is ever needed -- only the per-key QK dot is
// warp-reduced, which is 5 shuffles.
// ---------------------------------------------------------------------------
__global__ void prefill_attn_simt(const __half* __restrict__ q,   // [kQHeads][T][kHeadDim]
                                  const std::int8_t* __restrict__ k,  // [P][kHeadDim] int8
                                  const __half* __restrict__ k_scale, // [P][kGroups]
                                  const std::int8_t* __restrict__ v,  // [P][kHeadDim] int8
                                  const __half* __restrict__ v_scale, // [P][kGroups]
                                  __half* __restrict__ out,           // [kQHeads][T][kHeadDim]
                                  int T, int P, int base, float scale, int causal) {
    __shared__ std::int8_t s_k[kBc][kHeadDim];
    __shared__ std::int8_t s_v[kBc][kHeadDim];
    __shared__ __half s_ks[kBc][kGroups];
    __shared__ __half s_vs[kBc][kGroups];

    const int row       = blockIdx.x * kBr + (threadIdx.x >> 5);   // one warp per row
    const int head      = blockIdx.y;
    const int lane      = threadIdx.x & 31;
    if (row >= T) { return; }

    // Causal bound must be UNIFORM across the CTA because the tile loop below carries a
    // __syncthreads(): the loop runs to the block's last row's bound and each key is masked
    // per warp instead. (A per-warp bound here makes warps disagree on the sync count.)
    const int q_pos       = base + row;
    int kv_end            = P;
    if (causal) { kv_end = base + blockIdx.x * kBr + kBr; if (kv_end > P) { kv_end = P; } }
    if (kv_end <= 0) { return; }

    // Load this row's 8 q values (dims lane + 32*i) and keep them in registers.
    float qr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qr[i] = __half2float(q[(static_cast<std::int64_t>(head) * T + row) * kHeadDim + lane + i * 32]);
    }

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = 0.0f; }
    float m_prev = -INFINITY, l_prev = 0.0f;

    for (int base = 0; base < kv_end; base += kBc) {
        const int remain = kv_end - base;
        const int bc     = remain < kBc ? remain : kBc;
        // Stage K/V + scales for this tile.
        for (int idx = threadIdx.x; idx < bc * kHeadDim; idx += kThreads) {
            const int c = idx / kHeadDim, d = idx - c * kHeadDim;
            s_k[c][d] = k[static_cast<std::int64_t>(base + c) * kHeadDim + d];
            s_v[c][d] = v[static_cast<std::int64_t>(base + c) * kHeadDim + d];
        }
        for (int idx = threadIdx.x; idx < bc * kGroups; idx += kThreads) {
            const int c = idx / kGroups, g = idx - c * kGroups;
            s_ks[c][g] = k_scale[static_cast<std::int64_t>(base + c) * kGroups + g];
            s_vs[c][g] = v_scale[static_cast<std::int64_t>(base + c) * kGroups + g];
        }
        __syncthreads();

        for (int c = 0; c < bc; ++c) {
            const int key_pos = base + c;
            if (causal && key_pos > q_pos) { break; }
            // QK dot: each lane covers 8 dims, one scale group each (8 dims inside one 64-group).
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d     = lane + i * 32;
                const int group = d / kQuantGroup;
                const float kd  = static_cast<float>(s_k[c][d]) * __half2float(s_ks[c][group]);
                partial += qr[i] * kd;
            }
            float dot = partial;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) { dot += __shfl_down_sync(0xffffffffu, dot, off); }
            dot = __shfl_sync(0xffffffffu, dot, 0);   // broadcast: every lane keeps the full dot
            const float s = dot * scale;

            // Online softmax (uniform across the lane because every lane has the same s).
            const float m_new = fmaxf(m_prev, s);
            const float alpha = __expf(m_prev - m_new);
            const float p     = __expf(s - m_new);
            l_prev            = l_prev * alpha + p;
            m_prev            = m_new;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d     = lane + i * 32;
                const int group = d / kQuantGroup;
                const float vd  = static_cast<float>(s_v[c][d]) * __half2float(s_vs[c][group]);
                acc[i] = acc[i] * alpha + p * vd;
            }
        }
        __syncthreads();
    }

    const float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    __half* dst = out + (static_cast<std::int64_t>(head) * T + row) * kHeadDim;
#pragma unroll
    for (int i = 0; i < 8; ++i) { dst[lane + i * 32] = __float2half_rn(acc[i] * inv_l); }
}

// ---------------------------------------------------------------------------
// prefill_attn_mma_ds2 -- the candidate: dim-split=2, 4 warps/CTA, Br=64, Bc=32.
//
// Structure copied from the in-tree Volta kernel (src/ops/kernel/gqa_attention_prefill_volta.cuh):
// the m8n8k4 fragment addressing, the per-8-key online-softmax recurrence with the r_lo/r_hi
// row split, and the PV alpha rescale are that kernel's, unchanged -- only the outer loops
// differ (KV tile loop outside, no row passes, no paged indirection, no partial/merge epilogue).
//
// What is new here is the reason this should beat the vendored kernel: dim-split=2 instead of
// 4. QK carries the same FLOPs as PV, so a dim-split of r duplicates the QK work r times; the
// ceiling is 2*eff/(r+1). Measured on this GPU: eff = 102 TFLOPS at 4 warps/SM, so r=2 caps at
// 68.3 TF against r=4's 44.3 TF. The cost is the PV accumulator: 128 dims / 8 per slice = 16
// slices x 8 floats = 128 registers/thread, which drops occupancy to 1 CTA (4 warps) per SM --
// measured to still leave 82% of the mma issue rate intact.
// ---------------------------------------------------------------------------
__global__ void dequant_kv(const std::int8_t* __restrict__ k, const __half* __restrict__ k_scale,
                           const std::int8_t* __restrict__ v, const __half* __restrict__ v_scale,
                           __half* __restrict__ k16, __half* __restrict__ v16, int P) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < P * kHeadDim; idx += gridDim.x * blockDim.x) {
        const int key = idx / kHeadDim, d = idx - key * kHeadDim;
        const int g   = d / kQuantGroup;
        k16[idx] = __float2half(float(k[idx]) * __half2float(k_scale[key * kGroups + g]));
        v16[idx] = __float2half(float(v[idx]) * __half2float(v_scale[key * kGroups + g]));
    }
}

__device__ unsigned long long g_cyc[5];   // 0 fill, 1 qk, 2 softmax, 3 pv, 4 total   // bit0 QK, bit1 PV, bit2 KV dequant fill

constexpr int kMmaRowsPerGrp = 32;  // rows covered by one warp (quadpair mapping)
constexpr int kMmaRowGroups  = 4;  // Br = RowGroups * 32 = 128 rows per CTA: the KV tile is then
                                   // shared by 4x the rows, so the per-row copy cost drops 4x
constexpr int kMmaBr        = kMmaRowGroups * kMmaRowsPerGrp;
// Template parameter, so r=2 and r=4 run the exact same code body.
constexpr int kMmaDimSplit  = 2;    // default instantiation (r=2)
constexpr int kMmaDSlice    = kHeadDim / kMmaDimSplit;
constexpr int kMmaDChunksLocal = kMmaDSlice / 8;
constexpr int kMmaWarps     = 4;
// 2, not 8: the shared-memory stride must be coprime with the 32 banks in *words*.
// 264 halfs = 132 words -> 132 % 32 = 4 -> four-way conflicts on every fragment load.
// 258 halfs = 129 words -> 129 % 32 = 1 -> conflict-free.
// Trade-off measured on this GPU: an 8-half pad keeps the row 16-byte aligned so the four
// half2 fragments become one LDS.128 (4-way bank conflict inside it), which beats a
// conflict-free but unaligned stride: 264 halfs -> 9.9 TF, 258 -> 5.1 TF.
constexpr int kMmaStridePad = 8;
constexpr int kMmaSmemStride = kHeadDim + kMmaStridePad;
__device__ __forceinline__ int mma_swz_plain(int key, int chunk) { return key * kMmaSmemStride + chunk * 8; }

template <int DimSplit>
__global__ void __launch_bounds__((kMmaRowGroups * DimSplit) * 32) prefill_attn_mma_tmpl(
        const __half* __restrict__ q,       // [heads][T][D]
        const __half* __restrict__ k16,     // [P][D] fp16, dequantised once by dequant_kv
        const __half* __restrict__ v16,     // [P][D] fp16
        __half* __restrict__ out,           // [heads][T][D]
        int T, int P, int base, float scale) {
    using namespace ninfer::ops;
    extern __shared__ __align__(16) char smem_raw[];
    __half* k_s = reinterpret_cast<__half*>(smem_raw);                    // [Bc][stride] fp16
    __half* v_s = k_s + kBc * kMmaSmemStride;                            // [Bc][stride] fp16

    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int row_group = warp / DimSplit;      // 0..kMmaRowGroups-1
    const int dim_warp  = warp % DimSplit;      // 0..1
    const int head      = blockIdx.y;
    const int row0      = blockIdx.x * kMmaBr + row_group * kMmaRowsPerGrp;   // this warp's first row
    const int q_pos_base = base + row0;             // absolute position of this warp's row 0

    __half* q_s = nullptr; (void)q_s;   // kept for the (now absent) staging path

    // Q fragments resident in registers for the whole key walk. The Q tile is identical for
    // every KV tile, so reloading it from shared memory once per tile was ~40% of all LDS
    // traffic. 32 k-chunks x 4 half2 = 128 registers, which still leaves room for the 64-register
    // fp16 accumulator at 2 CTAs/SM (65536/256 threads = 256 registers/thread available).
    // Q fragments straight from global: with Br=128 a shared staging buffer would need 67.6 KB.
    // The scatter is paid once per kernel, not once per KV tile.
    half2 qreg[kHeadDim / 8][4];
    {
        const int qi   = volta_qp_get_i();
        const bool qok = (row0 + qi) < T;
        const __half* qb = qok ? &q[static_cast<std::int64_t>(head * T + row0 + qi) * kHeadDim] : nullptr;
#pragma unroll
        for (int c = 0; c < kHeadDim / 8; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                qreg[c][l] = qok ? *reinterpret_cast<const half2*>(&qb[c * 8 + 2 * l])
                                 : __float2half2_rn(0.0f);
            }
        }
    }

    // Causal bound must be uniform across the CTA (the KV tile loop has __syncthreads()).
    int kv_end = base + blockIdx.x * kMmaBr + kMmaBr;
    if (kv_end > P) { kv_end = P; }

    // fp16 (half2) PV accumulator, as llama.cpp's Volta kernel does: 8 output values per lane
    // per 8-dim slice become 4 half2 = 4 registers instead of 8 floats. This is the change that
    // makes 2 CTAs/SM reachable -- the fp32 version needed 128 registers/thread and pinned the
    // kernel at one CTA per SM.
    constexpr int kAccChunks = (kHeadDim / DimSplit) / 8;
    half2 acc2[kAccChunks][4];
#pragma unroll
    for (int c = 0; c < kAccChunks; ++c) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc2[c][i] = __float2half2_rn(0.0f); }
    }
    unsigned long long c_fill = 0, c_qk = 0, c_sm = 0, c_pv = 0, c_all = 0, t0 = clock64();
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.0f, l_hi = 0.0f;
    constexpr float Log2E = 1.4426950408889634f;

    for (int k0 = 0; k0 < kv_end; k0 += kBc) {
        const int bc = (kv_end - k0 < kBc) ? (kv_end - k0) : kBc;
        unsigned long long ts = clock64();
        // Pure bulk copy in 16-byte units: one thread moves one 8-half chunk, so the global read
        // is a coalesced int4 (512 B per warp) instead of 32 separate 16-bit loads. The earlier
        // element-at-a-time version was still 48% of kernel time.
        constexpr int kChunks = kHeadDim / 8;
        constexpr int kThreadsCta = (kMmaRowGroups * DimSplit) * 32;
        for (int ci = threadIdx.x; ci < kBc * kChunks; ci += kThreadsCta) {
            const int key   = ci / kChunks;
            const int chunk = ci - key * kChunks;
            const int off   = mma_swz_plain(key, chunk);
            if (key < bc) {
                *reinterpret_cast<int4*>(&k_s[off]) =
                    *reinterpret_cast<const int4*>(&k16[static_cast<std::int64_t>(k0 + key) * kHeadDim + chunk * 8]);
                *reinterpret_cast<int4*>(&v_s[off]) =
                    *reinterpret_cast<const int4*>(&v16[static_cast<std::int64_t>(k0 + key) * kHeadDim + chunk * 8]);
            } else {
                *reinterpret_cast<int4*>(&k_s[off]) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&v_s[off]) = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        c_fill += clock64() - ts; ts = clock64();

        constexpr int kSubKeys = 8;
        constexpr int kMaxSubs  = kBc / kSubKeys;          // 4
        const int subs = (bc + kSubKeys - 1) / kSubKeys;   // ceil: a sub-8 tail still runs
        // QK for every subgroup with ONE Q fragment load per k-chunk (c outer, sub inner).
        // Loading Q inside the sub loop instead costs 4x the shared-memory traffic, which was
        // the dominant cost of the first version.
        float d_all[kMaxSubs][8];
#pragma unroll
        for (int s2 = 0; s2 < kMaxSubs; ++s2) {
#pragma unroll
            for (int l = 0; l < 8; ++l) { d_all[s2][l] = 0.0f; }
        }
        // K fragment prefetch: load the next k-chunk's fragment before issuing this chunk's mma, so
        // the shared-memory latency overlaps the tensor-core work. Costs 8 registers, not the 24+
        // that splitting into independent accumulator chains needed.
        {
            half2 kf_cur[kMaxSubs][4], kf_next[kMaxSubs][4];
#pragma unroll
            for (int s2 = 0; s2 < kMaxSubs; ++s2) {
                volta_load_k(kf_cur[s2], reinterpret_cast<const half2*>(&k_s[s2 * 8 * kMmaSmemStride + 0]), kMmaSmemStride / 2);
            }
#pragma unroll
            for (int c = 0; c < kHeadDim / 8; ++c) {
                if (c + 1 < kHeadDim / 8) {
#pragma unroll
                    for (int s2 = 0; s2 < kMaxSubs; ++s2) {
                        volta_load_k(kf_next[s2], reinterpret_cast<const half2*>(&k_s[s2 * 8 * kMmaSmemStride + (c + 1) * 8]), kMmaSmemStride / 2);
                    }
                }
#pragma unroll
                for (int s2 = 0; s2 < kMaxSubs; ++s2) {
                    volta_mma_qk(d_all[s2], qreg[c], kf_cur[s2]);
                }
#pragma unroll
                for (int s2 = 0; s2 < kMaxSubs; ++s2) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) { kf_cur[s2][l] = kf_next[s2][l]; }
                }
            }
        }
        for (int sub = 0; sub < subs; ++sub) {
            const int sub_k0 = k0 + sub * kSubKeys;
            float d_score[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) { d_score[l] = d_all[sub][l]; }
            c_qk += clock64() - ts; ts = clock64();
            // --- mask + scale ---
            const int r_lo = volta_d_get_i(0) & ~2;
            const int r_hi = volta_d_get_i(0) | 2;
            const int qabs_lo = q_pos_base + r_lo;
            const int qabs_hi = q_pos_base + r_hi;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int col   = volta_d_get_j(l);
                const int key   = sub_k0 + col;
                const int qabs  = ((l & 2) == 0) ? qabs_lo : qabs_hi;
                const bool ok   = key <= qabs && key < P;
                d_score[l]      = ok ? d_score[l] * scale : -INFINITY;
            }
            // --- online softmax ---
            float bm_lo = fmaxf(fmaxf(d_score[0], d_score[1]), fmaxf(d_score[4], d_score[5]));
            float bm_hi = fmaxf(fmaxf(d_score[2], d_score[3]), fmaxf(d_score[6], d_score[7]));
            bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffffu, bm_lo, 2));
            bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffffu, bm_hi, 2));
            const float new_m_lo = fmaxf(m_lo, bm_lo);
            const float new_m_hi = fmaxf(m_hi, bm_hi);
            const float alpha_lo = (m_lo == -INFINITY) ? 0.0f : exp2f((m_lo - new_m_lo) * Log2E);
            const float alpha_hi = (m_hi == -INFINITY) ? 0.0f : exp2f((m_hi - new_m_hi) * Log2E);
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float nm = ((l & 2) == 0) ? new_m_lo : new_m_hi;
                d_score[l] = (nm > -INFINITY && d_score[l] > -INFINITY)
                                 ? exp2f((d_score[l] - nm) * Log2E) : 0.0f;
            }
            float bl_lo = d_score[0] + d_score[1] + d_score[4] + d_score[5];
            float bl_hi = d_score[2] + d_score[3] + d_score[6] + d_score[7];
            bl_lo += __shfl_xor_sync(0xffffffffu, bl_lo, 2);
            bl_hi += __shfl_xor_sync(0xffffffffu, bl_hi, 2);
            l_lo = l_lo * alpha_lo + bl_lo;
            l_hi = l_hi * alpha_hi + bl_hi;
            m_lo = new_m_lo;
            m_hi = new_m_hi;
            const float alpha = ((lane & 2) == 0) ? alpha_lo : alpha_hi;
            const half2 alpha2 = __float2half2_rn(alpha);

            c_sm += clock64() - ts; ts = clock64();
            half2 p[4];
            volta_softmax_to_half2(p, d_score);
            // --- PV over this warp's 128-dim slice ---
#pragma unroll
            for (int c = 0; c < ((kHeadDim / DimSplit) / 8); ++c) {
                half2 vf[4];
                volta_load_v(vf, reinterpret_cast<const half2*>(
                                      &v_s[sub * 8 * kMmaSmemStride + dim_warp * (kHeadDim / DimSplit) + c * 8]),
                             kMmaSmemStride / 2);
                half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                volta_mma_pv(pv, p, vf);
#pragma unroll
                for (int n = 0; n < 4; ++n) { acc2[c][n] = __hfma2(acc2[c][n], alpha2, pv[n]); }
            }
        }
        c_pv += clock64() - ts;
        __syncthreads();
    }
    c_all += clock64() - t0;
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&g_cyc[0], c_fill); atomicAdd(&g_cyc[1], c_qk);
        atomicAdd(&g_cyc[2], c_sm);   atomicAdd(&g_cyc[3], c_pv);
        atomicAdd(&g_cyc[4], c_all);
    }

    // --- write-out: this thread owns output row = lane, its warp's 128-dim slice ---
    const int row = lane;
    const int grow = row0 + row;
    if (grow < T) {
        const float own_l = ((lane & 2) == 0) ? l_lo : l_hi;
        const float inv_l = (own_l > 0.0f) ? (1.0f / own_l) : 0.0f;
        __half* dst = out + (static_cast<std::int64_t>(head) * T + grow) * kHeadDim;
#pragma unroll
        for (int c = 0; c < ((kHeadDim / DimSplit) / 8); ++c) {
            const int d = dim_warp * (kHeadDim / DimSplit) + c * 8;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const float2 v2 = __half22float2(__hmul2(acc2[c][n], __float2half2_rn(inv_l)));
                dst[d + 2 * n + 0] = __float2half_rn(v2.x);
                dst[d + 2 * n + 1] = __float2half_rn(v2.y);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pure-MMA ceiling probe.
//
// The flash kernel's achievable rate is bounded by its tensor-core issue rate and by how much
// QK work each dim-split warp duplicates. If the mma stream alone sustains `eff` TFLOPS on the
// real fragment pattern, a kernel with QK redundancy r is capped at 2*eff/(r+1) (QK and PV carry
// equal FLOPs): r=4 (the in-tree topology) -> 0.4*eff, r=2 -> 0.667*eff. Measuring `eff` decides
// the design before a single line of softmax is written.
// ---------------------------------------------------------------------------
__global__ void mma_ceiling_qk(float* sink, int iters) {
    // No memory traffic: this measures the m8n8k4 issue rate of one warp, which is the ceiling
    // any dim-split design inherits. Each call is a real k=4 QK slice; the chain through d[8]
    // keeps them from being optimized away.
    float d0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float d1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int a0 = threadIdx.x | 0x3c003c00, a1 = a0 ^ 0x11, b0 = a0 ^ 0x22, b1 = a0 ^ 0x33;
    for (int it = 0; it < iters; ++it) {
#pragma unroll 8
        for (int c = 0; c < 32; ++c) {                       // D/8 = 32 k-steps
            asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                         : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]),
                           "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7])
                         : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                         : "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]),
                           "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7])
                         : "r"(a1), "r"(a0), "r"(b1), "r"(b0));
        }
    }
    if (d0[0] == 12345.0f) { sink[0] = d1[0]; }
}

// ---------------------------------------------------------------------------
// CPU reference for one (head, row): fp32, same causal rule, same int8 dequant.
// ---------------------------------------------------------------------------
void cpu_reference(const std::vector<__half>& q, const std::vector<std::int8_t>& k,
                   const std::vector<__half>& k_scale, const std::vector<std::int8_t>& v,
                   const std::vector<__half>& v_scale, std::vector<float>& out, int head, int row,
                   int T, int P, float scale, int causal) {
    const int kv_end = causal ? std::min(P, row + 1) : P;
    std::vector<float> s(kv_end), p(kv_end);
    float mx = -INFINITY;
    for (int c = 0; c < kv_end; ++c) {
        double dot = 0.0;
        for (int d = 0; d < kHeadDim; ++d) {
            const double kd = double(k[static_cast<std::int64_t>(c) * kHeadDim + d]) *
                              __half2float(k_scale[static_cast<std::int64_t>(c) * kGroups + d / kQuantGroup]);
            dot += __half2float(q[(static_cast<std::int64_t>(head) * T + row) * kHeadDim + d]) * kd;
        }
        s[c] = float(dot) * scale;
        mx   = std::max(mx, s[c]);
    }
    double sum = 0.0;
    for (int c = 0; c < kv_end; ++c) { p[c] = std::exp(s[c] - mx); sum += p[c]; }
    for (int d = 0; d < kHeadDim; ++d) {
        double acc = 0.0;
        for (int c = 0; c < kv_end; ++c) {
            const double vd = double(v[static_cast<std::int64_t>(c) * kHeadDim + d]) *
                              __half2float(v_scale[static_cast<std::int64_t>(c) * kGroups + d / kQuantGroup]);
            acc += p[c] * vd;
        }
        out[d] = float(acc / sum);
    }
}

struct Case { const char* name; int T; int P; };

} // namespace

int main(int argc, char** argv) {
    const float scale = 1.0f / std::sqrt(float(kHeadDim));

    // --- 1) correctness on a small case -------------------------------------
    // Metric: per-row max |got - ref| normalised by that row's max |ref|. A per-element
    // relative error is meaningless here -- attention outputs routinely have components that
    // sit near zero (fp16 PV accumulates to ~1e-4 absolute), which makes element-wise relative
    // error explode on correct output.
    const int Tc = 40, Pc = 128;
    std::vector<__half> q(kQHeads * Tc * kHeadDim), ks(Pc * kGroups), vs(Pc * kGroups);
    std::vector<std::int8_t> k(Pc * kHeadDim), v(Pc * kHeadDim);
    unsigned seed = 12345u;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    for (auto& x : q) x = __float2half((float(int(rnd() % 2000) - 1000)) / 2000.0f);
    for (auto& x : k) x = std::int8_t(int(rnd() % 127) - 63);
    for (auto& x : v) x = std::int8_t(int(rnd() % 127) - 63);
    for (auto& x : ks) x = __float2half(0.01f + float(rnd() % 50) / 5000.0f);
    for (auto& x : vs) x = __float2half(0.01f + float(rnd() % 50) / 5000.0f);

    __half *dq, *dks, *dvs, *dout, *dk16, *dv16;
    std::int8_t *dk, *dv;
    CK(cudaMalloc(&dq, q.size() * 2)); CK(cudaMalloc(&dk, k.size()));
    CK(cudaMalloc(&dv, v.size()));     CK(cudaMalloc(&dks, ks.size() * 2));
    CK(cudaMalloc(&dvs, vs.size() * 2)); CK(cudaMalloc(&dout, q.size() * 2));
    CK(cudaMalloc(&dk16, Pc * kHeadDim * 2)); CK(cudaMalloc(&dv16, Pc * kHeadDim * 2));
    CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, k.data(), k.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, v.data(), v.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice));
    dequant_kv<<<64, 256>>>(dk, dks, dv, dvs, dk16, dv16, Pc);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    const int smem_mma = 2 * kBc * kMmaSmemStride * int(sizeof(__half));
    CK(cudaFuncSetAttribute(prefill_attn_mma_tmpl<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_mma));

    // Row-normalised error of a [heads][T][D] output buffer against the CPU reference.
    auto row_err = [&](const std::vector<__half>& got, int T, int P) {
        std::vector<float> ref(kHeadDim);
        double worst = 0.0;
        int worst_row = -1;
        for (int h = 0; h < kQHeads; ++h) {
            for (int r = 0; r < T; ++r) {
                cpu_reference(q, k, ks, v, vs, ref, h, r, T, P, scale, 1);
                double rmax = 1e-6;
                for (int d = 0; d < kHeadDim; ++d) { rmax = std::max(rmax, std::fabs(double(ref[d]))); }
                double err = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    const float g = __half2float(got[(static_cast<std::int64_t>(h) * T + r) * kHeadDim + d]);
                    err = std::max(err, std::fabs(double(g) - ref[d]) / rmax);
                }
                if (err > worst) { worst = err; worst_row = r; }
            }
        }
        return std::pair<double, int>(worst, worst_row);
    };

    {
        std::vector<__half> o(Tc * kQHeads * kHeadDim, __float2half(0.f));
        dim3 g((Tc + kBr - 1) / kBr, kQHeads);
        prefill_attn_simt<<<g, kThreads>>>(dq, dk, dks, dv, dvs, dout, Tc, Pc, 0, scale, 1);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(o.data(), dout, o.size() * 2, cudaMemcpyDeviceToHost));
        auto e = row_err(o, Tc, Pc);
        std::printf("correctness T=%d P=%d [SIMT]: row-norm err %.2e (worst row %d)  %s\n", Tc, Pc, e.first,
                    e.second, e.first < 2e-2 ? "OK" : "FAIL");
    }
    {
        std::vector<__half> o(Tc * kQHeads * kHeadDim, __float2half(0.f));
        dim3 g((Tc + kMmaBr - 1) / kMmaBr, kQHeads);
        prefill_attn_mma_tmpl<2><<<g, (kMmaRowGroups * 2) * 32, smem_mma>>>(dq, dk16, dv16, dout, Tc, Pc, 0, scale);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(o.data(), dout, o.size() * 2, cudaMemcpyDeviceToHost));
        auto e = row_err(o, Tc, Pc);
        std::printf("correctness T=%d P=%d [MMA ]: row-norm err %.2e (worst row %d)  %s\n", Tc, Pc, e.first,
                    e.second, e.first < 2e-2 ? "OK" : "FAIL");
    }
    {
        // Escalating P: the first tile, a sub-8 tail, and multi-tile accumulation.
        for (int Pt : {1, 2, 4, 8, 32, 128}) {
            std::vector<__half> o(Tc * kQHeads * kHeadDim, __float2half(0.f));
            dim3 g((Tc + kMmaBr - 1) / kMmaBr, kQHeads);
            prefill_attn_mma_tmpl<2><<<g, (kMmaRowGroups * 2) * 32, smem_mma>>>(dq, dk16, dv16, dout, Tc, Pt, 0, scale);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(o.data(), dout, o.size() * 2, cudaMemcpyDeviceToHost));
            auto e = row_err(o, Tc, Pt);
            std::printf("   mma P=%-3d: row-norm err %.2e (worst row %d)\n", Pt, e.first, e.second);
        }
    }

    // --- 2) throughput at production geometry -------------------------------
    // A prefill chunk of T=4096 that starts at base = P-T (i.e. the LAST chunk of a P-token
    // prompt) -- the worst and most representative case at each context.
    const Case cases[] = {{"ctx 8k   (base 4k)", 4096, 8192},
                          {"ctx 32k  (base 28k)", 4096, 32768},
                          {"ctx 120k (base 116k)", 4096, 122880}};
    std::printf("\n%-22s %8s/%7s %7s/%6s %5s/%4s\n", "case", "ms", "mma", "TF", "mma", "%pk", "mma");
    double peak = 125.0;
    for (const Case& c : cases) {
        const int T = c.T, P = c.P;
        const size_t qn = size_t(kQHeads) * T * kHeadDim, kn = size_t(P) * kHeadDim,
                     sn = size_t(P) * kGroups;
        __half *dq, *dks, *dvs, *dout, *dk16, *dv16;
        std::int8_t *dk, *dv;
        CK(cudaMalloc(&dq, qn * 2)); CK(cudaMalloc(&dk, kn)); CK(cudaMalloc(&dv, kn));
        CK(cudaMalloc(&dks, sn * 2)); CK(cudaMalloc(&dvs, sn * 2)); CK(cudaMalloc(&dout, qn * 2));
        CK(cudaMalloc(&dk16, kn * 2)); CK(cudaMalloc(&dv16, kn * 2));
        CK(cudaMemset(dq, 0, qn * 2)); CK(cudaMemset(dk, 0, kn)); CK(cudaMemset(dv, 0, kn));
        CK(cudaMemset(dks, 0, sn * 2)); CK(cudaMemset(dvs, 0, sn * 2));
        dim3 grid((T + kBr - 1) / kBr, kQHeads);
        const int pbase = P - T;
        dequant_kv<<<256, 256>>>(dk, dks, dv, dvs, dk16, dv16, P);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());                       // this chunk starts here
        // Real causal work: row r (0-based in chunk) sees pbase + r + 1 keys.
        double keys = double(T) * double(pbase) + double(T) * double(T + 1) / 2.0;
        const double flops = 2.0 * 2.0 * keys * kHeadDim * kQHeads;   // QK + PV

        for (int i = 0; i < 2; ++i) { prefill_attn_simt<<<grid, kThreads>>>(dq, dk, dks, dv, dvs, dout, T, P, pbase, scale, 1); }
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int reps = 3;
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) { prefill_attn_simt<<<grid, kThreads>>>(dq, dk, dks, dv, dvs, dout, T, P, pbase, scale, 1); }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); ms /= reps;
        const double tf = flops / (ms * 1e-3) / 1e12;
        // mma kernel on the same problem
        const int smem_mma = 2 * kBc * kMmaSmemStride * int(sizeof(__half));
        CK(cudaFuncSetAttribute(prefill_attn_mma_tmpl<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_mma));
        dim3 gridm((T + kMmaBr - 1) / kMmaBr, kQHeads);
        prefill_attn_mma_tmpl<2><<<gridm, (kMmaRowGroups * 2) * 32, smem_mma>>>(dq, dk16, dv16, dout, T, P, pbase, scale);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        cudaEvent_t f0, f1; cudaEventCreate(&f0); cudaEventCreate(&f1);
        CK(cudaEventRecord(f0));
        for (int i = 0; i < reps; ++i) { prefill_attn_mma_tmpl<2><<<gridm, (kMmaRowGroups * 2) * 32, smem_mma>>>(dq, dk16, dv16, dout, T, P, pbase, scale); }
        CK(cudaEventRecord(f1)); CK(cudaEventSynchronize(f1));
        float msm = 0.f; cudaEventElapsedTime(&msm, f0, f1); msm /= reps;
        const double tfm = flops / (msm * 1e-3) / 1e12;
        // r=4 on the same problem: 8 warps/CTA, 64-register accumulator, 4x QK redundancy.
        CK(cudaFuncSetAttribute(prefill_attn_mma_tmpl<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_mma));
        dim3 grid4((T + kMmaBr - 1) / kMmaBr, kQHeads);
        prefill_attn_mma_tmpl<4><<<grid4, 8 * 32, smem_mma>>>(dq, dk16, dv16, dout, T, P, pbase, scale);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        {
            unsigned long long zero[5] = {0, 0, 0, 0, 0};
            CK(cudaMemcpyToSymbol(g_cyc, zero, sizeof(zero)));
            prefill_attn_mma_tmpl<2><<<grid4, (kMmaRowGroups * 2) * 32, smem_mma>>>(dq, dk16, dv16, dout, T, P, pbase, scale);
            CK(cudaDeviceSynchronize());
            unsigned long long cy[5] = {0, 0, 0, 0, 0};
            CK(cudaMemcpyFromSymbol(cy, g_cyc, sizeof(cy)));
            const double tot = double(cy[4]) > 0 ? double(cy[4]) : 1.0;
            std::printf("   [cyc] fill %4.1f%%  qk %4.1f%%  softmax %4.1f%%  pv %4.1f%%  (of total)\n",
                        100.0 * cy[0] / tot, 100.0 * cy[1] / tot, 100.0 * cy[2] / tot, 100.0 * cy[3] / tot);
        }
        cudaEvent_t h0, h1; cudaEventCreate(&h0); cudaEventCreate(&h1);
        CK(cudaEventRecord(h0));
        for (int i = 0; i < reps; ++i) { prefill_attn_mma_tmpl<4><<<grid4, 8 * 32, smem_mma>>>(dq, dk16, dv16, dout, T, P, pbase, scale); }
        CK(cudaEventRecord(h1)); CK(cudaEventSynchronize(h1));
        float ms4 = 0.f; cudaEventElapsedTime(&ms4, h0, h1); ms4 /= reps;
        const double tf4 = flops / (ms4 * 1e-3) / 1e12;
        std::printf("%-22s  r2 %6.1fms %6.2fTF | r4 %6.1fms %6.2fTF\n", c.name, msm, tfm, ms4, tf4);
        cudaEventDestroy(h0); cudaEventDestroy(h1);
        cudaEventDestroy(f0); cudaEventDestroy(f1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dks); cudaFree(dvs); cudaFree(dout);
        cudaFree(dk16); cudaFree(dv16);
    }
    // --- 3) pure-MMA ceiling: bounds any dim-split design -------------------
    {
        float* sink = nullptr;
        CK(cudaMalloc(&sink, sizeof(float)));
        const int iters = 200;
        const int blocks = 80 * 8;                     // 8 CTAs/SM; no smem in this probe
        // Occupancy sweep first: the mma issue rate of ONE warp is not the kernel's ceiling --
        // a 4-warps/SM design (dim-split=2, 128 accumulator registers) may not fill the pipes.
        // Reported as warps/SM = cta_per_sm * 4.
        double rate_at[4] = {0, 0, 0, 0};
        const int cta_per_sm[4] = {1, 2, 4, 8};
        for (int k2 = 0; k2 < 4; ++k2) {
            const int nb = 80 * cta_per_sm[k2];
            mma_ceiling_qk<<<nb, 128>>>(sink, iters);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            cudaEvent_t a2, b2; cudaEventCreate(&a2); cudaEventCreate(&b2);
            CK(cudaEventRecord(a2));
            mma_ceiling_qk<<<nb, 128>>>(sink, iters);
            CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
            float ms2 = 0.f; cudaEventElapsedTime(&ms2, a2, b2);
            const double f2 = double(nb) * 4.0 * double(iters) * 32.0 * 2.0 * (2.0 * 32 * 8 * 4);
            rate_at[k2] = f2 / (ms2 * 1e-3) / 1e12;
            std::printf("  mma rate @ %2d warps/SM: %6.1f TFLOPS (%0.0f%% peak)\n",
                        cta_per_sm[k2] * 4, rate_at[k2], 100.0 * rate_at[k2] / peak);
            cudaEventDestroy(a2); cudaEventDestroy(b2);
        }
        std::printf("\n");

        mma_ceiling_qk<<<blocks, 128>>>(sink, iters);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        CK(cudaEventRecord(a));
        mma_ceiling_qk<<<blocks, 128>>>(sink, iters);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
        // per warp-iteration: 32 QK mma calls x (2 * 32 rows x 8 keys x 4 k) = 32*2*32*8*4 flops
        // One m8n8k4 call in this quadpair mapping covers 32 rows x 8 keys x 4 k
        // (volta_mma_qk's D is [32x8]); FLOP = 2 * 32 * 8 * 4 per instruction.
        const double flops = double(blocks) * 4.0 * double(iters) * 32.0 * 2.0 * (2.0 * 32 * 8 * 4);
        const double eff = flops / (ms * 1e-3) / 1e12;
        const double eff_r4 = rate_at[1];   // r=4 设计：2 CTAs x 4 warps = 8 warps/SM
        const double eff_r2 = rate_at[0];   // r=2 设计：1 CTA  x 4 warps = 4 warps/SM（128 reg 累加器）
        std::printf("\npure QK mma stream: %.1f TFLOPS (%0.0f%% peak)  <- 一条 warp 的 QK 发射上限\n",
                    eff, 100.0 * eff / peak);
        std::printf("  => r=4 @8 warps/SM: 上限 2*eff/5 = %.1f TF (eff=%.1f)\n", 2.0 * eff_r4 / 5.0, eff_r4);
        std::printf("  => r=2 @4 warps/SM: 上限 2*eff/3 = %.1f TF (eff=%.1f)\n", 2.0 * eff_r2 / 3.0, eff_r2);
        std::printf("  => r=1 @? warps（需 256 reg）上限 ~= %.1f TF\n", rate_at[0]);
        cudaFree(sink);
    }

    std::printf("\nbaseline to beat (vendored fattn, in engine): 35.5 TFLOPS = 28.4%% peak @ avg_ctx>32k\n");
    std::printf("rewrite gate: 42.6 TFLOPS (1.2x)\n");
    return 0;
}
