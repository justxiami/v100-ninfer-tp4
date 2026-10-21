// Shape-matched throughput probe for a chunked SM70 GDN kernel.
//
// The chunked recurrence replaces the sequential state walk with per-chunk matmuls. Before
// writing those kernels, this probe answers the only question that decides whether the project
// can clear its gate (segment >= 3.5x, i.e. <= 95 ms for 48 layers at T=8192): what effective
// TFLOPS do the real chunk shapes reach on V100 in this regime?
//
// Every family is launched with the real per-layer batch counts for a TP4 shard
// (Hq=4, Hv=12, chunk C=64, head dim D=128, T=8192 -> 128 chunks):
//   gram_k   (C x D) x (C x D)^T  -> C x C   per (chunk, qk head)      512 ops
//   gram_qk  (C x D) x (C x D)^T  -> C x C   per (chunk, qk head)      512 ops
//   ks       (C x D) x (D x D)    -> C x D   per (chunk, value head)  1536 ops
//   qs       (C x D) x (D x D)    -> C x D   per (chunk, value head)  1536 ops
//   nd       (C x C) x (C x D)    -> C x D   per (chunk, value head)  1536 ops
//   ku       (D x C) x (C x D)    -> D x D   per (chunk, value head)  1536 ops
//   mm       (C x C) x (C x C)    -> C x C   per (chunk, value head)  1536 ops  (inverse step)
//
// Kernels are FP16 WMMA with FP32 accumulation (Volta tensor cores have no BF16 path; a real
// kernel needs a hi/lo split for the FP32 state, which doubles the state-side matmul cost).
// Timing uses the shared cold-graph harness: one captured graph per family, device events
// bracketing each replay, 256 MiB L2 flush outside the timed interval.

#include "ninfer_bench_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;
namespace wmma = nvcuda::wmma;

namespace {

constexpr int kChunk    = 64;   // C
constexpr int kHeadDim  = 128;  // D
constexpr int kChunks   = 128;  // T=8192 / C
constexpr int kQkHeads  = 4;    // TP4 shard
constexpr int kValHeads = 12;   // TP4 shard
constexpr int kLayers   = 48;
constexpr std::size_t kFlushBytes = 256ULL << 20;

// C[MxN] = A[MxK] * B[KxN], row-major half in, float out. One warp per 16x16 tile.
// a_mod / b_mod: when nonzero, the operand is shared across heads and indexed per chunk.
__global__ void matmul_row_major(const __half* __restrict__ A, const __half* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K, int tiles_n,
                                 int a_mod, int b_mod) {
    const int tile = blockIdx.y;
    const int b    = blockIdx.x;
    const int ti   = tile / tiles_n;
    const int tj   = tile % tiles_n;

    A += static_cast<std::size_t>(a_mod ? b % a_mod : b) * M * K;
    B += static_cast<std::size_t>(b_mod ? b % b_mod : b) * K * N;
    C += static_cast<std::size_t>(b) * M * N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + ti * 16 * K + k, K);
        wmma::load_matrix_sync(bb, B + k * N + tj * 16, N);
        wmma::mma_sync(acc, a, bb, acc);
    }
    wmma::store_matrix_sync(C + ti * 16 * N + tj * 16, acc, N, wmma::mem_row_major);
}

// C[MxM] = A[MxK] * A^T, i.e. the Gram product; B is A read column-major.
__global__ void matmul_gram(const __half* __restrict__ A, float* __restrict__ C, int M, int K,
                            int tiles_n, int a_mod) {
    const int tile = blockIdx.y;
    const int b    = blockIdx.x;
    const int ti   = tile / tiles_n;
    const int tj   = tile % tiles_n;

    A += static_cast<std::size_t>(a_mod ? b % a_mod : b) * M * K;
    C += static_cast<std::size_t>(b) * M * M;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bt;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + ti * 16 * K + k, K);
        wmma::load_matrix_sync(bt, A + tj * 16 * K + k, K);
        wmma::mma_sync(acc, a, bt, acc);
    }
    wmma::store_matrix_sync(C + ti * 16 * M + tj * 16, acc, M, wmma::mem_row_major);
}

// C[DxD] = A^T (D x C) * B (C x D); A is row-major M x K with M=D, K=C read column-major.
__global__ void matmul_transposed_left(const __half* __restrict__ A, const __half* __restrict__ B,
                                       float* __restrict__ C, int M, int N, int K, int tiles_n,
                                       int a_mod, int b_mod) {
    const int tile = blockIdx.y;
    const int b    = blockIdx.x;
    const int ti   = tile / tiles_n;
    const int tj   = tile % tiles_n;

    // A holds [K rows][M cols] row-major; A^T is M x K.
    A += static_cast<std::size_t>(a_mod ? b % a_mod : b) * K * M;
    B += static_cast<std::size_t>(b_mod ? b % b_mod : b) * K * N;
    C += static_cast<std::size_t>(b) * M * N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> at;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(at, A + k * M + ti * 16, M);
        wmma::load_matrix_sync(bb, B + k * N + tj * 16, N);
        wmma::mma_sync(acc, at, bb, acc);
    }
    wmma::store_matrix_sync(C + ti * 16 * N + tj * 16, acc, N, wmma::mem_row_major);
}

// 32x32 warp tile: four accumulator fragments share two A and two B loads per k-step, which
// doubles the MMA-per-load ratio against the 16x16 baseline above. This is the cheap way to read
// the shape's real ceiling before committing to a full chunked kernel.
__global__ void matmul_row_major_32(const __half* __restrict__ A, const __half* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K,
                                    int a_mod, int b_mod) {
    const int tiles_n = N / 32;
    const int tiles_m = M / 32;
    const int tile    = blockIdx.y;
    const int b       = blockIdx.x;
    const int ti      = tile / tiles_n;
    const int tj      = tile % tiles_n;
    if (ti >= tiles_m) { return; }

    A += static_cast<std::size_t>(a_mod ? b % a_mod : b) * M * K;
    B += static_cast<std::size_t>(b_mod ? b % b_mod : b) * K * N;
    C += static_cast<std::size_t>(b) * M * N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c00, c01, c10, c11;
    wmma::fill_fragment(c00, 0.0f);
    wmma::fill_fragment(c01, 0.0f);
    wmma::fill_fragment(c10, 0.0f);
    wmma::fill_fragment(c11, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a0, A + (ti * 32 + 0) * K + k, K);
        wmma::load_matrix_sync(a1, A + (ti * 32 + 16) * K + k, K);
        wmma::load_matrix_sync(b0, B + k * N + tj * 32 + 0, N);
        wmma::load_matrix_sync(b1, B + k * N + tj * 32 + 16, N);
        wmma::mma_sync(c00, a0, b0, c00);
        wmma::mma_sync(c01, a0, b1, c01);
        wmma::mma_sync(c10, a1, b0, c10);
        wmma::mma_sync(c11, a1, b1, c11);
    }
    wmma::store_matrix_sync(C + (ti * 32 + 0) * N + tj * 32 + 0, c00, N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + (ti * 32 + 0) * N + tj * 32 + 16, c01, N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + (ti * 32 + 16) * N + tj * 32 + 0, c10, N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + (ti * 32 + 16) * N + tj * 32 + 16, c11, N, wmma::mem_row_major);
}

// Shared-memory staged GEMM for one chunk-head: the operands for these shapes (C x D + D x D
// halves = 48 KiB) fit in one block's smem, so a block can stage them once and feed every warp
// from smem. Eight warps cover a 64 x 128 output as eight 32 x 32 tiles.
constexpr int kSmemHalfs = kChunk * kHeadDim + kHeadDim * kHeadDim;

__global__ void matmul_smem_staged(const __half* __restrict__ A, const __half* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K, int a_mod,
                                   int b_mod) {
    extern __shared__ __half smem[];
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* Ab = A + static_cast<std::size_t>(a_mod ? b % a_mod : b) * M * K;
    const __half* Bb = B + static_cast<std::size_t>(b_mod ? b % b_mod : b) * K * N;
    float* Cb       = C + static_cast<std::size_t>(b) * M * N;

    __half* As = smem;                 // M x K
    __half* Bs = smem + M * K;         // K x N
    for (int i = tid; i < M * K; i += blockDim.x) { As[i] = Ab[i]; }
    for (int i = tid; i < K * N; i += blockDim.x) { Bs[i] = Bb[i]; }
    __syncthreads();

    const int warp   = tid / 32;
    const int tiles_n = N / 32;
    const int ti     = warp / tiles_n;
    const int tj     = warp % tiles_n;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c00, c01, c10, c11;
    wmma::fill_fragment(c00, 0.0f);
    wmma::fill_fragment(c01, 0.0f);
    wmma::fill_fragment(c10, 0.0f);
    wmma::fill_fragment(c11, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a0, As + (ti * 32 + 0) * K + k, K);
        wmma::load_matrix_sync(a1, As + (ti * 32 + 16) * K + k, K);
        wmma::load_matrix_sync(b0, Bs + k * N + tj * 32 + 0, N);
        wmma::load_matrix_sync(b1, Bs + k * N + tj * 32 + 16, N);
        wmma::mma_sync(c00, a0, b0, c00);
        wmma::mma_sync(c01, a0, b1, c01);
        wmma::mma_sync(c10, a1, b0, c10);
        wmma::mma_sync(c11, a1, b1, c11);
    }
    wmma::store_matrix_sync(Cb + (ti * 32 + 0) * N + tj * 32 + 0, c00, N, wmma::mem_row_major);
    wmma::store_matrix_sync(Cb + (ti * 32 + 0) * N + tj * 32 + 16, c01, N, wmma::mem_row_major);
    wmma::store_matrix_sync(Cb + (ti * 32 + 16) * N + tj * 32 + 0, c10, N, wmma::mem_row_major);
    wmma::store_matrix_sync(Cb + (ti * 32 + 16) * N + tj * 32 + 16, c11, N, wmma::mem_row_major);
}

// 64x32 warp tile (eight accumulators): four A fragments and two B fragments per k-step feed
// eight MMAs, i.e. 1.33 loads per MMA against 2.0 for the 32x32 tile.
__global__ void matmul_row_major_64x32(const __half* __restrict__ A, const __half* __restrict__ B,
                                       float* __restrict__ C, int M, int N, int K, int a_mod,
                                       int b_mod) {
    const int tiles_n = N / 32;
    const int tiles_m = M / 64;
    const int tile    = blockIdx.y;
    const int b       = blockIdx.x;
    const int ti      = tile / tiles_n;
    const int tj      = tile % tiles_n;
    if (ti >= tiles_m) { return; }

    A += static_cast<std::size_t>(a_mod ? b % a_mod : b) * M * K;
    B += static_cast<std::size_t>(b_mod ? b % b_mod : b) * K * N;
    C += static_cast<std::size_t>(b) * M * N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) { wmma::fill_fragment(acc[i][j], 0.0f); }
    }
    for (int k = 0; k < K; k += 16) {
        for (int i = 0; i < 4; ++i) {
            wmma::load_matrix_sync(a[i], A + (ti * 64 + i * 16) * K + k, K);
        }
        for (int j = 0; j < 2; ++j) {
            wmma::load_matrix_sync(bb[j], B + k * N + tj * 32 + j * 16, N);
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) { wmma::mma_sync(acc[i][j], a[i], bb[j], acc[i][j]); }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(C + (ti * 64 + i * 16) * N + tj * 32 + j * 16, acc[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

struct Family {
    const char* name;
    int batch;       // ops per layer
    double macs;     // MACs per op
    int repeats;     // mm runs the inverse doubling step six times per chunk-head
};

struct Measurement {
    double us   = 0.0;
    double tflops = 0.0;
};

template <class Launch>
Measurement measure_family(ninfer::DeviceBuffer& flush, cudaStream_t stream, Launch&& launch) {
    launch(stream);  // resolve and validate once
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bench::TimedGraph graph;
    graph.capture(stream, launch);
    const bench::ColdTiming timing = bench::measure_cold_graph(graph, flush, stream, 3, 20);
    return {timing.median_us, 0.0};
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    // Per-layer batch counts: gram products are per qk head, everything else per value head.
    const int gram_batch = kChunks * kQkHeads;   // 512
    const int val_batch  = kChunks * kValHeads;  // 1536

    ninfer::DeviceBuffer k_buf(static_cast<std::size_t>(kChunks) * kChunk * kHeadDim * sizeof(__half));
    ninfer::DeviceBuffer q_buf(k_buf.bytes);
    ninfer::DeviceBuffer state_buf(static_cast<std::size_t>(val_batch) * kHeadDim * kHeadDim *
                                   sizeof(__half));
    ninfer::DeviceBuffer ms_buf(static_cast<std::size_t>(val_batch) * kChunk * kChunk *
                                sizeof(__half));
    ninfer::DeviceBuffer delta_buf(static_cast<std::size_t>(val_batch) * kChunk * kHeadDim *
                                   sizeof(__half));
    ninfer::DeviceBuffer out_big(static_cast<std::size_t>(val_batch) * kHeadDim * kHeadDim *
                                 sizeof(float));
    ninfer::DeviceBuffer out_chunk(static_cast<std::size_t>(val_batch) * kChunk * kHeadDim *
                                   sizeof(float));
    ninfer::DeviceBuffer out_small(static_cast<std::size_t>(val_batch) * kChunk * kChunk *
                                   sizeof(float));

    // Same per-chunk operand for every value head, exactly as the real kernel would read it.
    CUDA_CHECK(cudaMemset(k_buf.p, 0, k_buf.bytes));
    CUDA_CHECK(cudaMemset(q_buf.p, 0, q_buf.bytes));
    CUDA_CHECK(cudaMemset(state_buf.p, 0, state_buf.bytes));
    CUDA_CHECK(cudaMemset(ms_buf.p, 0, ms_buf.bytes));
    CUDA_CHECK(cudaMemset(delta_buf.p, 0, delta_buf.bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ninfer::DeviceBuffer flush(kFlushBytes);

    const __half* k_ptr = static_cast<const __half*>(k_buf.p);
    const __half* q_ptr = static_cast<const __half*>(q_buf.p);
    const __half* s_ptr = static_cast<const __half*>(state_buf.p);
    const __half* m_ptr = static_cast<const __half*>(ms_buf.p);
    const __half* d_ptr = static_cast<const __half*>(delta_buf.p);
    float* o_big        = static_cast<float*>(out_big.p);
    float* o_chunk      = static_cast<float*>(out_chunk.p);
    float* o_small      = static_cast<float*>(out_small.p);

    std::vector<Family> families;
    std::vector<Measurement> results;

    // gram_k: (C x D) x (C x D)^T -> C x C, per (chunk, qk head)
    families.push_back({"gram_k", gram_batch, double(kChunk) * kChunk * kHeadDim, 1});
    results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(gram_batch, (kChunk / 16) * (kChunk / 16));
        matmul_gram<<<grid, 32, 0, s>>>(k_ptr, o_small, kChunk, kHeadDim, kChunk / 16, kChunks);
    }));

    // gram_qk: same shape on Q
    families.push_back({"gram_qk", gram_batch, double(kChunk) * kChunk * kHeadDim, 1});
    results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(gram_batch, (kChunk / 16) * (kChunk / 16));
        matmul_gram<<<grid, 32, 0, s>>>(q_ptr, o_small, kChunk, kHeadDim, kChunk / 16, kChunks);
    }));

    // ks: (C x D) x (D x D) -> C x D, per (chunk, value head)
    families.push_back({"ks", val_batch, double(kChunk) * kHeadDim * kHeadDim, 1});
    results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kChunk / 16) * (kHeadDim / 16));
        matmul_row_major<<<grid, 32, 0, s>>>(k_ptr, s_ptr, o_chunk, kChunk, kHeadDim, kHeadDim,
                                             kHeadDim / 16, kChunks, 0);
    }));

    // qs: (C x D) x (D x D) -> C x D on Q
    families.push_back({"qs", val_batch, double(kChunk) * kHeadDim * kHeadDim, 1});
    results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kChunk / 16) * (kHeadDim / 16));
        matmul_row_major<<<grid, 32, 0, s>>>(q_ptr, s_ptr, o_chunk, kChunk, kHeadDim, kHeadDim,
                                             kHeadDim / 16, kChunks, 0);
    }));

    // nd: (C x C) x (C x D) -> C x D
    families.push_back({"nd", val_batch, double(kChunk) * kChunk * kHeadDim, 1});
    results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kChunk / 16) * (kHeadDim / 16));
        matmul_row_major<<<grid, 32, 0, s>>>(m_ptr, d_ptr, o_chunk, kChunk, kHeadDim, kChunk,
                                             kHeadDim / 16, 0, 0);
    }));

    // ku: (D x C) x (C x D) -> D x D, state update
    families.push_back({"ku", val_batch, double(kHeadDim) * kChunk * kHeadDim, 1});
    results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kHeadDim / 16) * (kHeadDim / 16));
        matmul_transposed_left<<<grid, 32, 0, s>>>(k_ptr, d_ptr, o_big, kHeadDim, kHeadDim, kChunk,
                                                   kHeadDim / 16, kChunks, 0);
    }));

    // mm: the inverse doubling step, six C x C products per chunk-head
    families.push_back({"mm", val_batch, double(kChunk) * kChunk * kChunk, 6});
    results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kChunk / 16) * (kChunk / 16));
        for (int r = 0; r < 6; ++r) {
            matmul_row_major<<<grid, 32, 0, s>>>(m_ptr, m_ptr, o_small, kChunk, kChunk, kChunk,
                                                 kChunk / 16, 0, 0);
        }
    }));

    std::vector<Family> tuned;
    std::vector<Measurement> tuned_results;
    // smem-staged ks shape: the realistic structure of a real chunked kernel.
    tuned.push_back({"ks_smem", val_batch, double(kChunk) * kHeadDim * kHeadDim, 1});
    tuned_results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch);
        matmul_smem_staged<<<grid, 256, kSmemHalfs * sizeof(__half), s>>>(k_ptr, s_ptr, o_chunk,
                                                                        kChunk, kHeadDim, kHeadDim,
                                                                        kChunks, 0);
    }));
    tuned.push_back({"ku_smem", val_batch, double(kHeadDim) * kChunk * kHeadDim, 1});
    tuned_results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch);
        // A = delta (D x C rows of K=D? ) -- ku is (D x C) x (C x D); use delta as M x K = D x C.
        matmul_smem_staged<<<grid, 256, kSmemHalfs * sizeof(__half), s>>>(d_ptr, k_ptr, o_big,
                                                                        kHeadDim, kHeadDim, kChunk,
                                                                        0, kChunks);
    }));

    tuned.push_back({"ks_64x32", val_batch, double(kChunk) * kHeadDim * kHeadDim, 1});
    tuned_results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kChunk / 64) * (kHeadDim / 32));
        matmul_row_major_64x32<<<grid, 32, 0, s>>>(k_ptr, s_ptr, o_chunk, kChunk, kHeadDim,
                                                   kHeadDim, kChunks, 0);
    }));
    tuned.push_back({"ku_64x32", val_batch, double(kHeadDim) * kChunk * kHeadDim, 1});
    tuned_results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kHeadDim / 64) * (kHeadDim / 32));
        matmul_row_major_64x32<<<grid, 32, 0, s>>>(d_ptr, k_ptr, o_big, kHeadDim, kHeadDim,
                                                   kChunk, 0, kChunks);
    }));

    // Same ks shape with the 32x32 warp tile, to separate "small tile" from "shape ceiling".
    tuned.push_back({"ks32", val_batch, double(kChunk) * kHeadDim * kHeadDim, 1});
    tuned_results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kChunk / 32) * (kHeadDim / 32));
        matmul_row_major_32<<<grid, 32, 0, s>>>(k_ptr, s_ptr, o_chunk, kChunk, kHeadDim, kHeadDim,
                                                kChunks, 0);
    }));
    tuned.push_back({"ku32", val_batch, double(kHeadDim) * kChunk * kHeadDim, 1});
    tuned_results.push_back(measure_family(flush, stream, [&](cudaStream_t s) {
        const dim3 grid(val_batch, (kHeadDim / 32) * (kHeadDim / 32));
        matmul_row_major_32<<<grid, 32, 0, s>>>(d_ptr, k_ptr, o_big, kHeadDim, kHeadDim, kChunk,
                                                0, kChunks);
    }));

    std::printf("Chunked GDN shape probe (C=64, D=128, TP4 Hq=4/Hv=12, T=8192 -> 128 chunks)\n");
    std::printf("FP16 WMMA, FP32 accumulate, cold L2, 3 warmup + 20 samples, median\n\n");
    std::printf("family     ops   repeats        MAC       ms/layer   TFLOPS\n");
    std::printf("--------  -----  -------  -----------  ----------  -------\n");

    double total_ms = 0.0;
    double total_flop = 0.0;
    for (std::size_t i = 0; i < families.size(); ++i) {
        const Family& f = families[i];
        const double flop = f.macs * f.batch * f.repeats * 2.0;
        const double tflops = flop / (results[i].us * 1.0e-6) / 1.0e12;
        total_ms += results[i].us / 1000.0;
        total_flop += flop;
        std::printf("%-8s  %5d  %7d  %11.3f  %10.3f  %7.2f\n", f.name, f.batch, f.repeats,
                    flop / 1.0e6, results[i].us / 1000.0, tflops);
    }
    std::printf("\n32x32 warp tile variants (fragment reuse):\n");
    for (std::size_t i = 0; i < tuned.size(); ++i) {
        const double flop = tuned[i].macs * tuned[i].batch * tuned[i].repeats * 2.0;
        std::printf("  %-6s  %10.3f ms/layer  %7.2f TFLOPS\n", tuned[i].name,
                    tuned_results[i].us / 1000.0, flop / (tuned_results[i].us * 1.0e-6) / 1.0e12);
    }

    std::printf("\nper-layer matmul time  %8.3f ms  (%.2f TFLOP total: %s)\n", total_ms,
                total_flop * kLayers / 1.0e12, "48 layers");
    std::printf("48-layer matmul time   %8.3f ms\n", total_ms * kLayers);
    std::printf("gate: sequential segment is 333.3 ms; 3.5x needs <= 95.2 ms\n");
    std::printf("projected segment      %8.3f ms  -> %.2fx vs sequential\n", total_ms * kLayers,
                333.3 / (total_ms * kLayers));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
