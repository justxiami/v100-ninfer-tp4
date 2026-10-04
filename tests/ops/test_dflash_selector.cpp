#include "ninfer/ops/dflash_selector.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int run_case(int V, int R, int K, int B) {
    std::vector<float> logits(static_cast<std::size_t>(V) * K * B);
    std::vector<float> gate(static_cast<std::size_t>(R) * K * B);
    std::vector<float> prev(static_cast<std::size_t>(V) * R);
    std::vector<float> next(static_cast<std::size_t>(V) * R);
    std::vector<std::int32_t> anchors(B);
    for (int b = 0; b < B; ++b) { anchors[b] = (4 + b * 13) % V; }
    for (int col = 0; col < K * B; ++col) {
        for (int v = 0; v < V; ++v) {
            const std::uint32_t mixed = (static_cast<std::uint32_t>(v) * 2654435761U) ^
                                        (static_cast<std::uint32_t>(col) * 2246822519U);
            logits[col * V + v] = static_cast<float>(static_cast<int>(mixed % 53) - 26) / 8.0f;
        }
        for (int r = 0; r < R; ++r) { gate[col * R + r] = static_cast<float>(r + 1) / 64.0f; }
    }
    for (int v = 0; v < V; ++v) {
        for (int r = 0; r < R; ++r) {
            prev[v * R + r] = static_cast<float>((v + 3 * r) % 7 - 3) / 16.0f;
            next[v * R + r] = static_cast<float>((v * 3 + r) % 9 - 4) / 16.0f;
        }
    }
    round_to_bf16(logits);
    round_to_bf16(gate);
    round_to_bf16(prev);
    round_to_bf16(next);
    std::vector<std::int32_t> expected(K * B);
    std::vector<std::int32_t> expected_topk(16 * K * B);
    for (int b = 0; b < B; ++b) {
        int parent = anchors[b];
        for (int t = 0; t < K; ++t) {
            const int col = b * K + t;
            std::vector<int> candidates(V);
            for (int v = 0; v < V; ++v) { candidates[v] = v; }
            std::partial_sort(candidates.begin(), candidates.begin() + 16, candidates.end(),
                              [&](int lhs, int rhs) {
                                  if (logits[col * V + lhs] != logits[col * V + rhs]) {
                                      return logits[col * V + lhs] > logits[col * V + rhs];
                                  }
                                  return lhs < rhs;
                              });
            std::copy_n(candidates.begin(), 16, expected_topk.begin() + col * 16);
            int best = candidates[0];
            double best_score = -1.0e30;
            for (int i = 0; i < 16; ++i) {
                const int id = candidates[i];
                double score = logits[col * V + id];
                for (int r = 0; r < R; ++r) {
                    score += static_cast<double>(prev[parent * R + r]) * gate[col * R + r] *
                             next[id * R + r];
                }
                if (score > best_score) { best_score = score; best = id; }
            }
            parent = expected[col] = best;
        }
    }
    auto bits = [](const std::vector<float>& src) {
        std::vector<std::uint16_t> dst(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) { dst[i] = f32_to_bf16(src[i]); }
        return dst;
    };
    DeviceBuffer d_logits = to_device(bits(logits));
    DeviceBuffer d_gate = to_device(bits(gate));
    DeviceBuffer d_prev = to_device(bits(prev));
    DeviceBuffer d_next = to_device(bits(next));
    DeviceBuffer d_anchor = to_device(anchors);
    const int parts = (V + ops::kDFlashSelectorTile - 1) / ops::kDFlashSelectorTile;
    DeviceBuffer d_partial(static_cast<std::size_t>(16 * parts * K * B) * sizeof(std::int64_t));
    DeviceBuffer d_topk(static_cast<std::size_t>(16 * K * B) * sizeof(std::int32_t));
    DeviceBuffer d_out(static_cast<std::size_t>(K * B) * sizeof(std::int32_t));
    Tensor logits_t(d_logits.p, DType::BF16, {V, K, B});
    Tensor gate_t(d_gate.p, DType::BF16, {R, K, B});
    Tensor prev_t(d_prev.p, DType::BF16, {V, R});
    Tensor next_t(d_next.p, DType::BF16, {V, R});
    Tensor anchor_t(d_anchor.p, DType::I32, {B});
    Tensor partial_t(d_partial.p, DType::I64, {16, parts, K * B});
    Tensor topk_t(d_topk.p, DType::I32, {16, K, B});
    Tensor out_t(d_out.p, DType::I32, {K, B});
    ops::dflash2_select(logits_t, gate_t, prev_t, next_t, anchor_t, partial_t, topk_t,
                        out_t, nullptr);
    cuda_synchronize();
    int failures = verify_exact("DFlash2 top-k", from_device<std::int32_t>(d_topk, 16 * K * B),
                                expected_topk);
    failures += verify_exact("DFlash2 selector", from_device<std::int32_t>(d_out, K * B), expected);

    // Check each TP shard and the merged lattice walk against the same independent CPU
    // sorting/FP64 oracle, not merely against the full-vocabulary production kernel. Odd V
    // exercises unequal shards; multiple columns catch transposed [32,C] gather layouts.
    const int columns = K * B;
    const int widths[2] = {V / 2, V - V / 2};
    const int offsets[2] = {0, V / 2};
    std::array<std::vector<std::uint64_t>, 2> local_keys;
    auto encode_key = [](float logit, int id) {
        const auto raw = std::bit_cast<std::uint32_t>(logit);
        const auto ordered = (raw & 0x80000000U) ? ~raw : (raw ^ 0x80000000U);
        return (static_cast<std::uint64_t>(ordered) << 32) |
               ~static_cast<std::uint32_t>(id);
    };
    for (int shard = 0; shard < 2; ++shard) {
        const int width = widths[shard], offset = offsets[shard];
        std::vector<float> shard_logits(static_cast<std::size_t>(width) * columns);
        std::vector<std::uint64_t> expected_keys(16 * columns);
        for (int col = 0; col < columns; ++col) {
            std::copy_n(logits.data() + col * V + offset, width,
                        shard_logits.data() + col * width);
            std::vector<int> candidates(width);
            for (int i = 0; i < width; ++i) { candidates[i] = offset + i; }
            std::partial_sort(candidates.begin(), candidates.begin() + 16, candidates.end(),
                              [&](int lhs, int rhs) {
                                  if (logits[col * V + lhs] != logits[col * V + rhs]) {
                                      return logits[col * V + lhs] > logits[col * V + rhs];
                                  }
                                  return lhs < rhs;
                              });
            for (int i = 0; i < 16; ++i) {
                expected_keys[col * 16 + i] =
                    encode_key(logits[col * V + candidates[i]], candidates[i]);
            }
        }
        DeviceBuffer d_shard = to_device(bits(shard_logits));
        const int shard_parts = (width + ops::kDFlashSelectorTile - 1) / ops::kDFlashSelectorTile;
        DeviceBuffer d_shard_partial(static_cast<std::size_t>(16 * shard_parts * columns) *
                                     sizeof(std::uint64_t));
        DeviceBuffer d_shard_keys(static_cast<std::size_t>(16 * columns) * sizeof(std::uint64_t));
        Tensor shard_t(d_shard.p, DType::BF16, {width, columns});
        Tensor shard_partial_t(d_shard_partial.p, DType::I64, {16, shard_parts, columns});
        Tensor shard_keys_t(d_shard_keys.p, DType::I64, {16, columns});
        ops::dflash2_local_topk(shard_t, shard_partial_t, shard_keys_t, offset, nullptr);
        cuda_synchronize();
        local_keys[shard] = from_device<std::uint64_t>(d_shard_keys, 16 * columns);
        failures += verify_exact("DFlash2 shard keys", local_keys[shard], expected_keys);
    }
    std::vector<std::uint64_t> gathered(32 * columns), expected_global_keys(16 * columns);
    for (int col = 0; col < columns; ++col) {
        for (int shard = 0; shard < 2; ++shard) {
            std::copy_n(local_keys[shard].data() + col * 16, 16,
                        gathered.data() + col * 32 + shard * 16);
        }
        for (int i = 0; i < 16; ++i) {
            const int id = expected_topk[col * 16 + i];
            expected_global_keys[col * 16 + i] = encode_key(logits[col * V + id], id);
        }
    }
    DeviceBuffer d_gathered = to_device(gathered);
    DeviceBuffer d_global_keys(static_cast<std::size_t>(16 * columns) * sizeof(std::uint64_t));
    Tensor gathered_t(d_gathered.p, DType::I64, {32, columns});
    Tensor global_keys_t(d_global_keys.p, DType::I64, {16, columns});
    ops::dflash2_select_sharded(gathered_t, gate_t, prev_t, next_t, anchor_t, global_keys_t,
                               out_t, nullptr);
    cuda_synchronize();
    failures += verify_exact("DFlash2 merged keys",
                             from_device<std::uint64_t>(d_global_keys, 16 * columns),
                             expected_global_keys);
    failures += verify_exact("DFlash2 sharded selector", from_device<std::int32_t>(d_out, columns),
                             expected);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "DFlash2 selector: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = run_case(1025, 8, 3, 2);
    failures += run_case(248320, 256, 7, 1);
    // More than one tile per thread in the final merge, with a non-full last vocabulary tile.
    failures += run_case(262145, 8, 3, 2);
    return failures == 0 ? 0 : 1;
}
