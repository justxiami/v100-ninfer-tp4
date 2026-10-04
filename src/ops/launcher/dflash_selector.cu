#include "ops/launcher/dflash_selector.h"
#include "core/device.h"

#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kTop = 16;
constexpr int kThreads = 256;
constexpr int kItems = 8;
constexpr int kTile = kDFlashSelectorTile;
using Sort = cub::BlockRadixSort<std::uint64_t, kThreads, kItems>;
using FinishSort = cub::BlockRadixSort<std::uint64_t, kThreads, kTop>;
using MergeSort = cub::BlockRadixSort<std::uint64_t, kThreads, 1>;

__device__ __forceinline__ std::uint64_t logit_key(float value, int id) {
    const std::uint32_t raw = __float_as_uint(value);
    const std::uint32_t ordered = (raw & 0x80000000u) ? ~raw : raw ^ 0x80000000u;
    return (static_cast<std::uint64_t>(ordered) << 32) |
           static_cast<std::uint32_t>(~id);
}

__global__ void dflash2_partial_topk(const __nv_bfloat16* logits, std::uint64_t* partial,
                                      int vocab, int parts) {
    const int part = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    const int first = part * kTile;
    std::uint64_t keys[kItems];
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int id = first + item * kThreads + static_cast<int>(threadIdx.x);
        keys[item] = id < vocab
                         ? logit_key(__bfloat162float(logits[static_cast<std::size_t>(col) * vocab + id]), id)
                         : 0ULL;
    }
    __shared__ Sort::TempStorage storage;
    Sort(storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * kItems + item;
        if (rank < kTop) {
            partial[(static_cast<std::size_t>(col) * parts + part) * kTop + rank] = keys[item];
        }
    }
}

__global__ void dflash2_local_partial_topk(const __nv_bfloat16* logits, std::uint64_t* partial,
                                           int vocab, int parts, int id_offset) {
    const int part = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    const int first = part * kTile;
    std::uint64_t keys[kItems];
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int local_id = first + item * kThreads + static_cast<int>(threadIdx.x);
        const int global_id = id_offset + local_id;
        keys[item] = local_id < vocab
                         ? logit_key(__bfloat162float(logits[static_cast<std::size_t>(col) * vocab + local_id]),
                                     global_id)
                         : 0ULL;
    }
    __shared__ Sort::TempStorage storage;
    Sort(storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * kItems + item;
        if (rank < kTop) {
            partial[(static_cast<std::size_t>(col) * parts + part) * kTop + rank] = keys[item];
        }
    }
}

__global__ void dflash2_finish_topk(const std::uint64_t* partial, int* topk, int parts) {
    const int col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    std::uint64_t winners[kTop];
#pragma unroll
    for (int i = 0; i < kTop; ++i) { winners[i] = 0ULL; }
    // A global top-16 member must belong to its thread's local top-16. Keep those local
    // winners, then merge them collectively; no floating arithmetic or tie policy changes.
    for (int item = tid; item < parts * kTop; item += kThreads) {
        const std::uint64_t key = partial[static_cast<std::size_t>(col) * parts * kTop + item];
        int insertion = kTop;
#pragma unroll
        for (int i = 0; i < kTop; ++i) {
            if (key > winners[i]) { insertion = i; break; }
        }
        if (insertion == kTop) { continue; }
        for (int i = kTop - 1; i > insertion; --i) { winners[i] = winners[i - 1]; }
        winners[insertion] = key;
    }
    __shared__ FinishSort::TempStorage storage;
    FinishSort(storage).SortDescending(winners);
#pragma unroll
    for (int i = 0; i < kTop; ++i) {
        const int rank = tid * kTop + i;
        if (rank < kTop) {
            topk[col * kTop + rank] = static_cast<int>(~static_cast<std::uint32_t>(winners[i]));
        }
    }
}

__global__ void dflash2_finish_topk_keys(const std::uint64_t* partial, std::uint64_t* topk,
                                         int parts) {
    const int col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    std::uint64_t winners[kTop];
#pragma unroll
    for (int i = 0; i < kTop; ++i) { winners[i] = 0ULL; }
    for (int item = tid; item < parts * kTop; item += kThreads) {
        const std::uint64_t key = partial[static_cast<std::size_t>(col) * parts * kTop + item];
        int insertion = kTop;
#pragma unroll
        for (int i = 0; i < kTop; ++i) {
            if (key > winners[i]) { insertion = i; break; }
        }
        if (insertion == kTop) { continue; }
        for (int i = kTop - 1; i > insertion; --i) { winners[i] = winners[i - 1]; }
        winners[insertion] = key;
    }
    __shared__ FinishSort::TempStorage storage;
    FinishSort(storage).SortDescending(winners);
#pragma unroll
    for (int i = 0; i < kTop; ++i) {
        const int rank = tid * kTop + i;
        if (rank < kTop) { topk[col * kTop + rank] = winners[i]; }
    }
}

__device__ __forceinline__ float key_value(std::uint64_t key) {
    const std::uint32_t ordered = static_cast<std::uint32_t>(key >> 32);
    const std::uint32_t raw = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(raw);
}

__device__ __forceinline__ int key_id(std::uint64_t key) {
    return static_cast<int>(~static_cast<std::uint32_t>(key));
}

__global__ void dflash2_merge_sharded_keys(const std::uint64_t* local, std::uint64_t* global,
                                           int gathered_count) {
    const int col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    std::uint64_t key[1] = {0ULL};
    // gather_columns_rank0 concatenates rank r's part into rows [r*16, (r+1)*16) of each
    // column: each column owns gathered_count (= 16 * tp) adjacent 16-entry candidate lists.
    if (tid < gathered_count) {
        key[0] = local[col * gathered_count + tid];
    }
    __shared__ MergeSort::TempStorage storage;
    MergeSort(storage).SortDescending(key);
    if (tid < kTop) { global[col * kTop + tid] = key[0]; }
}

__global__ void dflash2_walk(const __nv_bfloat16* logits, const __nv_bfloat16* gate,
                             const __nv_bfloat16* predecessor,
                             const __nv_bfloat16* successor, const int* anchors,
                             const int* topk, int* out, int vocab, int steps, int rank) {
    const int batch = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float scores[kTop];
    __shared__ int selected;
    if (tid == 0) { selected = anchors[batch]; }
    __syncthreads();
    for (int step = 0; step < steps; ++step) {
        const int col = batch * steps + step;
        const int* candidates = topk + col * kTop;
        const __nv_bfloat16* g = gate + static_cast<std::size_t>(col) * rank;
        for (int pass = 0; pass < 2; ++pass) {
            const int candidate = warp + pass * 8;
            const int id = candidates[candidate];
            float subtotal = 0.0f;
            for (int r = lane; r < rank; r += 32) {
                const float pred = __bfloat162float(predecessor[
                    static_cast<std::size_t>(selected) * rank + r]);
                const float factor = __bfloat162float(g[r]);
                const float succ = __bfloat162float(successor[
                    static_cast<std::size_t>(id) * rank + r]);
                subtotal += pred * factor * succ;
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                subtotal += __shfl_down_sync(0xffffffffu, subtotal, offset);
            }
            if (lane == 0) {
                scores[candidate] = __bfloat162float(logits[
                    static_cast<std::size_t>(col) * vocab + id]) + subtotal;
            }
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            for (int candidate = 1; candidate < kTop; ++candidate) {
                if (scores[candidate] > scores[best]) { best = candidate; }
            }
            selected = candidates[best];
            out[col] = selected;
        }
        __syncthreads();
    }
}

__global__ void dflash2_walk_keys(const std::uint64_t* global_keys, const __nv_bfloat16* gate,
                                  const __nv_bfloat16* predecessor,
                                  const __nv_bfloat16* successor, const int* anchors,
                                  int* out, int steps, int rank) {
    const int batch = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float scores[kTop];
    __shared__ int selected;
    if (tid == 0) { selected = anchors[batch]; }
    __syncthreads();
    for (int step = 0; step < steps; ++step) {
        const int col = batch * steps + step;
        const __nv_bfloat16* g = gate + static_cast<std::size_t>(col) * rank;
        for (int pass = 0; pass < 2; ++pass) {
            const int candidate = warp + pass * 8;
            const std::uint64_t key = global_keys[col * kTop + candidate];
            const int id = key_id(key);
            float subtotal = 0.0f;
            for (int r = lane; r < rank; r += 32) {
                const float pred = __bfloat162float(predecessor[static_cast<std::size_t>(selected) * rank + r]);
                const float factor = __bfloat162float(g[r]);
                const float succ = __bfloat162float(successor[static_cast<std::size_t>(id) * rank + r]);
                subtotal += pred * factor * succ;
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                subtotal += __shfl_down_sync(0xffffffffu, subtotal, offset);
            }
            if (lane == 0) { scores[candidate] = key_value(key) + subtotal; }
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            for (int candidate = 1; candidate < kTop; ++candidate) {
                if (scores[candidate] > scores[best]) { best = candidate; }
            }
            selected = key_id(global_keys[col * kTop + best]);
            out[col] = selected;
        }
        __syncthreads();
    }
}

} // namespace

void dflash2_select_launch(const Tensor& logits, const Tensor& gate, const Tensor& predecessor,
                           const Tensor& successor, const Tensor& anchors,
                           const Tensor& partial, Tensor& topk, Tensor& out, cudaStream_t stream) {
    const int vocab = logits.ne[0];
    const int steps = logits.ne[1];
    const int batch = logits.ne[2];
    const int parts = (vocab + kTile - 1) / kTile;
    dflash2_partial_topk<<<dim3(parts, steps * batch), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<std::uint64_t*>(partial.data), vocab, parts);
    CUDA_CHECK(cudaGetLastError());
    dflash2_finish_topk<<<steps * batch, kThreads, 0, stream>>>(
        static_cast<const std::uint64_t*>(partial.data), static_cast<int*>(topk.data), parts);
    CUDA_CHECK(cudaGetLastError());
    dflash2_walk<<<batch, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const __nv_bfloat16*>(successor.data),
        static_cast<const int*>(anchors.data),
        static_cast<const int*>(topk.data), static_cast<int*>(out.data), vocab, steps, gate.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_local_topk_launch(const Tensor& logits, Tensor& partial, Tensor& keys,
                               std::int32_t global_id_offset, cudaStream_t stream) {
    const int vocab = logits.ne[0];
    const int columns = logits.ne[1];
    const int parts = (vocab + kTile - 1) / kTile;
    dflash2_local_partial_topk<<<dim3(parts, columns), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::uint64_t*>(partial.data),
        vocab, parts, global_id_offset);
    CUDA_CHECK(cudaGetLastError());
    dflash2_finish_topk_keys<<<columns, kThreads, 0, stream>>>(
        static_cast<const std::uint64_t*>(partial.data), static_cast<std::uint64_t*>(keys.data),
        parts);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_select_sharded_launch(const Tensor& keys, const Tensor& gate,
                                   const Tensor& predecessor, const Tensor& successor,
                                   const Tensor& anchors, Tensor& global_keys, Tensor& out,
                                   cudaStream_t stream) {
    const int columns = gate.ne[1] * gate.ne[2];
    dflash2_merge_sharded_keys<<<columns, kThreads, 0, stream>>>(
        static_cast<const std::uint64_t*>(keys.data), static_cast<std::uint64_t*>(global_keys.data),
        keys.ne[0]);
    CUDA_CHECK(cudaGetLastError());
    dflash2_walk_keys<<<gate.ne[2], kThreads, 0, stream>>>(
        static_cast<const std::uint64_t*>(global_keys.data),
        static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const __nv_bfloat16*>(successor.data), static_cast<const int*>(anchors.data),
        static_cast<int*>(out.data), gate.ne[1], gate.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
