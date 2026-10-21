// M2 prototype: chunked GDN core (WY solve + state scan) for the VOLTA path.
//
// Scope of this probe, deliberately narrow: validate and time the two pieces that do not exist
// anywhere in the tree yet --
//   kernel A: per (head, chunk)  Gram -> M -> (I+M)^-1 applied to [v | alpha*P_{m-1}*k]
//             by blocked forward substitution, giving U' = Mop v and W' = Mop (alpha P k).
//   kernel B: per (head, dv split) sequential scan over chunks
//             delta = beta * (U' - W' S),  S <- P_C S + K^T (coef * delta),
//             publishing delta and each chunk's starting state.
// The output kernel (q S + tril(coef q k^T) delta) is not implemented here; its cost is exactly
// the "ks" + "nd" shapes already measured by gdn_chunked_shape_bench.
//
// Dumps inputs and results to a raw binary so gdn_chunk_validate.py can compare against the FP64
// sequential recurrence.

#include "ninfer_bench_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace ninfer;
namespace wmma = nvcuda::wmma;

namespace {

constexpr int kD       = 128;   // head dim (qk and dv)
constexpr int kC       = 64;    // chunk size
constexpr int kB       = 16;    // substitution block
constexpr int kSplits  = 4;     // dv splits in the scan
constexpr int kDvSplit = kD / kSplits;
constexpr int kThreadsA = 256;
constexpr int kScanWarps = 8;               // 256 threads: 64 KiB of state / 256 = 64 floats each
constexpr int kScanSlots = 64 / kScanWarps; // tiles per warp

// ---------------------------------------------------------------- kernel A: WY solve
//
// One block per (head, chunk), 256 threads. Thread t owns column t of the solution matrix
// [U' | W'] (t < 128 -> U' column t, t >= 128 -> W' column t - 128), so the substitution runs
// with no cross-thread communication inside a row: every thread walks j = 0..i-1 reading
// M[i][j] from shared memory (a broadcast) and its own X[j] register.
__global__ void wy_solve_kernel(const __half* __restrict__ k, const __half* __restrict__ v,
                                const float* __restrict__ g, const float* __restrict__ beta,
                                __half* __restrict__ w_out, __half* __restrict__ u_out,
                                std::int32_t chunks) {
    extern __shared__ __align__(32) unsigned char smem_raw[];
    float* M    = reinterpret_cast<float*>(smem_raw);              // C x C, 16 KiB
    __half* Ks  = reinterpret_cast<__half*>(M + kC * kC);          // C x D, 8 KiB
    __half* Vs  = Ks + kC * kD;                                     // C x D, 8 KiB
    float* av   = reinterpret_cast<float*>(Vs + kC * kD);          // C: alpha, P_{m-1}
    float* pm1  = av + kC;

    const int chunk = blockIdx.y;
    const int head  = blockIdx.x;
    const int tid   = threadIdx.x;
    const std::int64_t col0 = static_cast<std::int64_t>(chunk) * kC;

    // stage k and v for this chunk
    for (int i = tid; i < kC * kD; i += blockDim.x) {
        const int row = i / kD;
        const int col = i % kD;
        Ks[i] = k[(col0 + row) * kD + col];
        Vs[i] = v[(static_cast<std::int64_t>(head) * gridDim.y * kC + col0 + row) * kD + col];
    }
    __syncthreads();

    // prefix decay products inside the chunk (single warp is enough)
    if (tid < kC) {
        const std::int64_t row = col0 + tid;
        const float alpha      = expf(g[static_cast<std::int64_t>(head) * gridDim.y * kC + row]);
        av[tid]                = alpha;
        float p                = 1.0f;
        for (int m = 0; m < tid; ++m) {
            p *= expf(g[static_cast<std::int64_t>(head) * gridDim.y * kC + col0 + m]);
        }
        pm1[tid] = p;  // P_{tid-1}
    }

    // Gram = K K^T by WMMA, written straight into M as fp32
    {
        const int tiles = (kC / 16) * (kC / 16);
        for (int tile = tid / 32; tile < tiles; tile += blockDim.x / 32) {
            const int ti = tile / (kC / 16);
            const int tj = tile % (kC / 16);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < kD; kk += 16) {
                wmma::load_matrix_sync(a, Ks + (ti * 16) * kD + kk, kD);
                wmma::load_matrix_sync(b, Ks + (tj * 16) * kD + kk, kD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(M + (ti * 16) * kC + tj * 16, acc, kC, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // M[i][j] = beta_i alpha_i (P_{i-1}/P_j) gram[i][j] for j < i, else 0. The Gram product is
    // already in M, so this is a single in-place scaling pass.
    for (int i = tid; i < kC * kC; i += blockDim.x) {
        const int r = i / kC;
        const int c = i % kC;
        if (c < r) {
            const std::int64_t base = static_cast<std::int64_t>(head) * gridDim.y * kC + col0;
            const float b_i         = beta[base + r];
            const float p_c         = pm1[c] * av[c];  // P_c
            M[i] = M[i] * b_i * av[r] * (pm1[r] / p_c);
        } else {
            M[i] = 0.0f;
        }
    }
    __syncthreads();

    // blocked forward substitution for X = (I+M)^-1 [v | alpha P_{m-1} k]
    float x[kC];
#pragma unroll 1
    for (int i = 0; i < kC; ++i) {
        const int dv_col = tid % kD;
        const bool is_u  = tid < kD;
        const float rhs  = is_u ? __half2float(Vs[i * kD + dv_col])
                               : av[i] * pm1[i] * __half2float(Ks[i * kD + dv_col]);
        float acc = rhs;
        for (int j = 0; j < i; ++j) { acc -= M[i * kC + j] * x[j]; }
        x[i] = acc;
    }

    // publish: U' = Mop v, W' = Mop (alpha P k)
    const int dv_col = tid % kD;
    const bool is_u  = tid < kD;
    __half* dst = is_u ? u_out + (static_cast<std::int64_t>(head) * chunks + chunk) * kC * kD
                       : w_out + (static_cast<std::int64_t>(head) * chunks + chunk) * kC * kD;
    for (int i = 0; i < kC; ++i) { dst[i * kD + dv_col] = __float2half(x[i]); }
}

// ---------------------------------------------------------------- kernel B: state scan
//
// One block per (head, dv split). The dv columns are independent in both the delta projection and
// the state update, so four splits give 48 blocks instead of 12. The state slice lives in
// registers (128 qk x 32 dv / 256 threads = 16 floats each) and is mirrored to shared fp16 for
// the WMMA products.
__global__ void scan_kernel(const __half* __restrict__ k, const __half* __restrict__ w,
                            const __half* __restrict__ u, const float* __restrict__ beta,
                            const float* __restrict__ g, const float* __restrict__ state_in,
                            float* __restrict__ state_out, __half* __restrict__ delta_out,
                            __half* __restrict__ chunk_state_out, std::int32_t chunks) {
    // One block per head, 256 threads. The 128x128 state lives in registers (64 floats per
    // thread: row t%128, one of two 64-column halves) and is mirrored to shared fp16 for the
    // WMMA products. W'/U'/K are read straight from global memory: staging them per chunk cost
    // 48 KiB per chunk and dominated the first version of this kernel.
    extern __shared__ __align__(32) unsigned char smem_raw[];
    __half* Sh = reinterpret_cast<__half*>(smem_raw);          // 128 x 128 fp16 = 32 KiB
    __half* Dh = Sh + kD * kD;                                 // 64 x 128 fp16 = 16 KiB
    __half* Kt = Dh + kC * kD;                                 // D x C fp16 = 16 KiB (K^T)
    float* Tiles = reinterpret_cast<float*>(Kt + kD * kC);     // kScanWarps x 256 fp32
    float* Pref  = Tiles + kScanWarps * 256;                    // C fp32 (prefix decay)

    const int head = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid / 32;

    // State ownership follows the update's tile assignment: warp w owns tiles {w, w+16, ...},
    // and lane l owns elements {l + 32i} of each of its tiles (4 tiles x 8 elements = 32 floats).
    // That way the epilogue of a tile updates exactly the registers of the warp that computed it.
    const int lane  = tid % 32;
    float s[kScanSlots * 8];
#pragma unroll
    for (int slot = 0; slot < kScanSlots; ++slot) {
        const int tile = warp + slot * kScanWarps;
        const int ti   = tile / (kD / 16);
        const int tj   = tile % (kD / 16);
        for (int i = 0; i < 8; ++i) {
            const int e = lane + 32 * i;
            const int q = ti * 16 + e / 16;
            const int d = tj * 16 + e % 16;
            s[slot * 8 + i] =
                state_in[static_cast<std::int64_t>(head) * kD * kD + q * kD + d];
        }
    }

    for (std::int32_t c = 0; c < chunks; ++c) {
        // publish state to shared fp16 and to the per-chunk state record
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
            const int tile = warp + slot * kScanWarps;
            const int ti   = tile / (kD / 16);
            const int tj   = tile % (kD / 16);
            for (int i = 0; i < 8; ++i) {
                const int e = lane + 32 * i;
                const int q = ti * 16 + e / 16;
                const int d = tj * 16 + e % 16;
                const __half h = __float2half(s[slot * 8 + i]);
                Sh[q * kD + d] = h;
                chunk_state_out[((static_cast<std::int64_t>(head) * chunks + c) * kD + q) * kD + d] =
                    h;
            }
        }
        __syncthreads();

        const std::int64_t wbase = (static_cast<std::int64_t>(head) * chunks + c) * kC * kD;
        const std::int64_t kbase = static_cast<std::int64_t>(c) * kC;
        const std::int64_t gbase = static_cast<std::int64_t>(head) * chunks * kC + c * kC;
        const __half* Wg = w + wbase;
        const __half* Ug = u + wbase;
        const __half* Kg = k + kbase * kD;
        for (int i = tid; i < kD * kC; i += blockDim.x) {
            const int q = i / kC;   // qk row
            const int r = i % kC;   // token inside the chunk
            Kt[i] = Kg[r * kD + q];
        }
        __syncthreads();

        // delta = beta * (U' - W' S): (C x D)(D x D) -> C x D
        {
            const int tiles = (kC / 16) * (kD / 16);
            for (int tile = warp; tile < tiles; tile += kScanWarps) {
                const int ti = tile / (kD / 16);
                const int tj = tile % (kD / 16);
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int kk = 0; kk < kD; kk += 16) {
                    wmma::load_matrix_sync(a, Wg + (ti * 16) * kD + kk, kD);
                    wmma::load_matrix_sync(b, Sh + kk * kD + tj * 16, kD);
                    wmma::mma_sync(acc, a, b, acc);
                }
                float* dst = Tiles + warp * 256;
                wmma::store_matrix_sync(dst, acc, 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = 0; e < 256; ++e) {
                    const int r = ti * 16 + e / 16;
                    const int d = tj * 16 + e % 16;
                    const float b_i = beta[gbase + r];
                    const float val = b_i * (__half2float(Ug[r * kD + d]) - dst[e]);
                    Dh[r * kD + d] = __float2half(val);
                }
                __syncwarp();
            }
        }
        __syncthreads();

        for (int i = tid; i < kC * kD; i += blockDim.x) {
            delta_out[wbase + i] = Dh[i];
        }
        __syncthreads();

        // S <- P_C S + K^T (coef * delta): (D x C)(C x D), coef folded into the delta copy
        {
            if (tid == 0) {
                float p = 1.0f;
                for (int r = 0; r < kC; ++r) {
                    p *= expf(g[gbase + r]);
                    Pref[r] = p;
                }
            }
            __syncthreads();
            const float pc = Pref[kC - 1];
            const float* pref = Pref;
            for (int i = tid; i < kC * kD; i += blockDim.x) {
                const int r = i / kD;
                Dh[i] = __float2half((pc / pref[r]) * __half2float(Dh[i]));
            }
            __syncthreads();

            const int tiles = (kD / 16) * (kD / 16);
            for (int tile = warp; tile < tiles; tile += kScanWarps) {
                const int ti = tile / (kD / 16);
                const int tj = tile % (kD / 16);
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int kk = 0; kk < kC; kk += 16) {
                    wmma::load_matrix_sync(a, Kt + ti * 16 * kC + kk, kC);
                    wmma::load_matrix_sync(b, Dh + kk * kD + tj * 16, kD);
                    wmma::mma_sync(acc, a, b, acc);
                }
                float* dst = Tiles + warp * 256;
                wmma::store_matrix_sync(dst, acc, 16, wmma::mem_row_major);
                __syncwarp();
                const int slot = tile / kScanWarps;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int e = lane + 32 * i;
                    s[slot * 8 + i] = pc * s[slot * 8 + i] + dst[e];
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int tile = warp + slot * kScanWarps;
        const int ti   = tile / (kD / 16);
        const int tj   = tile % (kD / 16);
        for (int i = 0; i < 8; ++i) {
            const int e = lane + 32 * i;
            const int q = ti * 16 + e / 16;
            const int d = tj * 16 + e % 16;
            state_out[static_cast<std::int64_t>(head) * kD * kD + q * kD + d] = s[slot * 8 + i];
        }
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
    const int chunks = tokens / kC;
    const char* dump = argc > 3 ? argv[3] : "/tmp/gdn_chunk_dump.bin";

    std::vector<__half> hq(static_cast<std::size_t>(hv) * tokens * kD);
    std::vector<__half> hk(hq.size());
    std::vector<__half> hv_(hq.size());
    std::vector<float> hg(static_cast<std::size_t>(hv) * tokens);
    std::vector<float> hbeta(hg.size());
    std::vector<float> hstate(static_cast<std::size_t>(hv) * kD * kD);

    std::uint32_t seed = 12345u;
    for (std::size_t t = 0; t < static_cast<std::size_t>(tokens); ++t) {
        // normalise each qk row so the recurrence sees the same magnitudes as production
        float nk = 0.0f, nq = 0.0f;
        for (int d = 0; d < kD; ++d) {
            const float a = frand(seed);
            const float b = frand(seed);
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
        for (int h = 0; h < hv; ++h) {
            for (int d = 0; d < kD; ++d) {
                hv_[static_cast<std::size_t>(h) * tokens * kD + t * kD + d] =
                    __float2half(0.5f * frand(seed));
            }
            hg[static_cast<std::size_t>(h) * tokens + t] = -0.05f;
            hbeta[static_cast<std::size_t>(h) * tokens + t] = 0.6f;
        }
    }
    for (int h = 0; h < hv; ++h) {
        for (int q = 0; q < kD; ++q) {
            for (int d = 0; d < kD; ++d) {
                hstate[(static_cast<std::size_t>(h) * kD + q) * kD + d] = 0.2f * frand(seed);
            }
        }
    }

    const std::size_t elems_qk = static_cast<std::size_t>(hv) * tokens * kD;
    const std::size_t elems_g  = static_cast<std::size_t>(hv) * tokens;
    ninfer::DeviceBuffer dq(elems_qk * sizeof(__half));
    ninfer::DeviceBuffer dk(elems_qk * sizeof(__half));
    ninfer::DeviceBuffer dv(elems_qk * sizeof(__half));
    ninfer::DeviceBuffer dg(elems_g * sizeof(float));
    ninfer::DeviceBuffer dbeta(elems_g * sizeof(float));
    ninfer::DeviceBuffer dstate_in(static_cast<std::size_t>(hv) * kD * kD * sizeof(float));
    ninfer::DeviceBuffer dstate_out(dstate_in.bytes);
    ninfer::DeviceBuffer dw(static_cast<std::size_t>(hv) * chunks * kC * kD * sizeof(__half));
    ninfer::DeviceBuffer du(dw.bytes);
    ninfer::DeviceBuffer ddelta(dw.bytes);
    ninfer::DeviceBuffer dchunk_state(
        static_cast<std::size_t>(hv) * chunks * kD * kD * sizeof(__half));

    CUDA_CHECK(cudaMemcpy(dq.p, hq.data(), dq.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk.p, hk.data(), dk.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.p, hv_.data(), dv.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.p, hg.data(), dg.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta.p, hbeta.data(), dbeta.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dstate_in.p, hstate.data(), dstate_in.bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const std::size_t smem_a = kC * kC * sizeof(float) + 2 * kC * kD * sizeof(__half) +
                               2 * kC * sizeof(float);
    const std::size_t smem_b =
        (kD * kD + kC * kD + kD * kC) * sizeof(__half) +
        kScanWarps * 256 * sizeof(float) + kC * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(wy_solve_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem_a)));
    CUDA_CHECK(cudaFuncSetAttribute(scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem_b)));

    const auto run = [&](cudaStream_t s) {
        const dim3 grid_a(hv, chunks);
        wy_solve_kernel<<<grid_a, kThreadsA, smem_a, s>>>(
            static_cast<const __half*>(dk.p), static_cast<const __half*>(dv.p),
            static_cast<const float*>(dg.p), static_cast<const float*>(dbeta.p),
            static_cast<__half*>(dw.p), static_cast<__half*>(du.p), chunks);
        const dim3 grid_b(hv);
        if (getenv("SKIP_B") == nullptr) scan_kernel<<<grid_b, 256, smem_b, s>>>(
            static_cast<const __half*>(dk.p), static_cast<const __half*>(dw.p),
            static_cast<const __half*>(du.p), static_cast<const float*>(dbeta.p),
            static_cast<const float*>(dg.p), static_cast<const float*>(dstate_in.p),
            static_cast<float*>(dstate_out.p), static_cast<__half*>(ddelta.p),
            static_cast<__half*>(dchunk_state.p), chunks);
    };

    run(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ninfer::DeviceBuffer flush(256ULL << 20);
    bench::TimedGraph graph;
    graph.capture(stream, run);
    const bench::ColdTiming timing = bench::measure_cold_graph(graph, flush, stream, 3, 20);

    std::vector<__half> ow(dw.bytes / sizeof(__half)), ou(dw.bytes / sizeof(__half));
    std::vector<__half> odelta(dw.bytes / sizeof(__half));
    std::vector<float> os(static_cast<std::size_t>(hv) * kD * kD);
    CUDA_CHECK(cudaMemcpy(ow.data(), dw.p, dw.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ou.data(), du.p, du.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(odelta.data(), ddelta.p, dw.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(os.data(), dstate_out.p, dstate_out.bytes, cudaMemcpyDeviceToHost));

    // Dump in a fixed order: meta, inputs, then outputs. All arrays little-endian, no padding.
    //   q[Hv][T][D] half, k[Hv][T][D] half, v[Hv][T][D] half, g[Hv][T] float, beta[Hv][T] float,
    //   state_in[Hv][D_qk][D_dv] float, w[Hv][chunks][C][D] half, u[same] half,
    //   delta[same] half, state_out[Hv][D_qk][D_dv] float
    std::ofstream out(dump, std::ios::binary);
    const std::int32_t meta[3] = {hv, tokens, kC};
    out.write(reinterpret_cast<const char*>(meta), sizeof(meta));
    out.write(reinterpret_cast<const char*>(hq.data()), hq.size() * sizeof(__half));
    out.write(reinterpret_cast<const char*>(hk.data()), hk.size() * sizeof(__half));
    out.write(reinterpret_cast<const char*>(hv_.data()), hv_.size() * sizeof(__half));
    out.write(reinterpret_cast<const char*>(hg.data()), hg.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(hbeta.data()), hbeta.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(hstate.data()), hstate.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(ow.data()), ow.size() * sizeof(__half));
    out.write(reinterpret_cast<const char*>(ou.data()), ou.size() * sizeof(__half));
    out.write(reinterpret_cast<const char*>(odelta.data()), odelta.size() * sizeof(__half));
    out.write(reinterpret_cast<const char*>(os.data()), os.size() * sizeof(float));
    out.close();
    std::printf("dump: %s\n", dump);

    std::printf("M2 core: Hv=%d T=%d chunks=%d  WY+scan median %.3f ms/layer\n", hv, tokens,
                chunks, timing.median_us / 1000.0);
    std::printf("  (48-layer GDN segment extrapolation: %.1f ms)\n", timing.median_us / 1000.0 * 48.0);
    return 0;
}
