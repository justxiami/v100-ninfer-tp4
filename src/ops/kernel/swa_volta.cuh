#pragma once

#include "ops/kernel/bidirectional_gqa_attention.cuh"

namespace ninfer::ops {

// SM70 SWA: each warp owns a complete query row and KV split. Four independent
// warps share a CTA without per-key CTA barriers.
// QK, online softmax, the numerator and the private split accumulator stay
// FP32. The following reducer rounds only the final public output to BF16.
template <int Tokens>
__launch_bounds__(128, 2) __global__ void swa_volta_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k, const __nv_bfloat16* __restrict__ context_v,
    int window, int max_context, int split_capacity, float scale,
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    static_assert(Tokens >= 1 && Tokens <= 16);
    constexpr int Warps = 4;
    constexpr int D = kBidirectionalGqaHeadDim;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * Warps + threadIdx.x / 32;
    const int token = row / kBidirectionalGqaQHeads;
    const int q_head = row % kBidirectionalGqaQHeads;
    const int kv_head = q_head / kBidirectionalGqaGroup;
    const int split = blockIdx.y;
    const int batch = blockIdx.z;
    constexpr std::int64_t QueryElements = std::int64_t{D} * kBidirectionalGqaQHeads * Tokens;
    constexpr std::int64_t KvElements = std::int64_t{D} * kBidirectionalGqaKVHeads * Tokens;
    constexpr std::int64_t StatElements = std::int64_t{kBidirectionalGqaQHeads} * Tokens;
    q += QueryElements * batch;
    query_k += KvElements * batch;
    query_v += KvElements * batch;
    partial_acc += QueryElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    positions += std::int64_t{Tokens} * batch;
    const int length = positions[0];
    const int valid = valid_columns[batch];
    if (token >= Tokens || length < 0 || length > max_context || valid < 1 || valid > Tokens) {
        return;
    }
    if (token >= valid) {
        return;
    }
    const std::int64_t lane_elements = std::int64_t{D} * window * kBidirectionalGqaKVHeads;
    context_k += lane_elements * lanes[batch];
    context_v += lane_elements * lanes[batch];
    const int count = min(length, window - 1);
    const int start = length - count;
    const int tiles = (count + 31) / 32;
    const int active_splits = tiles > 0 ? min(tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }
    const int tile_begin = static_cast<int>((std::int64_t{tiles} * split) / active_splits);
    const int tile_end = static_cast<int>((std::int64_t{tiles} * (split + 1)) / active_splits);
    const int begin = max(start + 32 * tile_begin, positions[token] - (window - 1));
    const int end = min(length, start + 32 * tile_end);
    float query[4];
    float numerator[4] = {};
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        query[item] = __bfloat162float(q[bidirectional_gqa_q_index(q_head, lane + 32 * item, token)]);
    }
    float m = -CUDART_INF_F;
    float l = 0.0f;
    const auto consume = [&](const __nv_bfloat16* key, const __nv_bfloat16* value) {
        float dot = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            dot += query[item] * __bfloat162float(key[lane + 32 * item]);
        }
        const float score = warp_sum<32>(dot) * scale;
        const float next_m = fmaxf(m, score);
        const float alpha = m == -CUDART_INF_F ? 0.0f : expf(m - next_m);
        const float probability = expf(score - next_m);
        m = next_m;
        l = l * alpha + probability;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            numerator[item] = numerator[item] * alpha +
                              probability * __bfloat162float(value[lane + 32 * item]);
        }
    };
    for (int key = begin; key < end; ++key) {
        const auto index = bidirectional_gqa_cyclic_context_index(kv_head, 0, key & (window - 1), window);
        consume(context_k + index, context_v + index);
    }
    if (split == active_splits - 1) {
        for (int key = 0; key < valid; ++key) {
            const auto index = bidirectional_gqa_query_kv_index(kv_head, 0, key);
            consume(query_k + index, query_v + index);
        }
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + 32 * item;
        partial_acc[bidirectional_gqa_partial_index<Tokens>(q_head, d, token, split)] =
            numerator[item];
    }
    if (lane == 0) {
        const auto stat = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
        partial_m[stat] = m;
        partial_l[stat] = l;
    }
}

} // namespace ninfer::ops
