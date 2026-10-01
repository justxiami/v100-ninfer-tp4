#pragma once

// Global (tp1) row extents of the registered 27B fused objects, and the shard extent at a given
// tensor-parallel width.
//
// Until this header existed, every split wrapper in this directory spelled its shard extents as
// compile-time literals (`kShardXxx = <global> / 2`). Those literals are the global extents divided
// by a fixed parallel width, so naming the global extent once and dividing by the runtime width
// keeps a single source of truth and makes every divisor of the global extents a legal width.
//
// Split axes (see the design notes in include/ninfer/ops/*.h):
//   column-parallel  output rows / tp; the input extent K is never split (it is the hidden extent
//                    for every column shard registered here)
//   row-parallel     input extent K / tp; the output rows are never split
// A value that is never split has no shard form and is used at its global extent directly.
//
// The registered extents below are properties of the qwen3.6/3.8-27B fused objects (they were
// already hard-coded in the wrappers this replaces); a second target with a different hidden /
// intermediate / head count registers its own extents alongside its own ShardPlan.

#include "ninfer/types.h" // TpArray

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Never split: the input extent of every column shard.
constexpr std::int32_t kGlobalHiddenRows = 5120;

// Attention query_key_gate_value: q 6144 | k 1024 | v 1024 | gate 6144.
constexpr std::int32_t kGlobalAttnFusedRows = 14336;
constexpr std::int32_t kGlobalAttnQueryRows = 6144;
constexpr std::int32_t kGlobalAttnKvRows    = 1024; // k and v, each
constexpr std::int32_t kGlobalAttnSplitRows = 7168; // query_key / gate_value parent

// GDN query_key_value_z: q 2048 | k 2048 | v 6144 | z 6144.
constexpr std::int32_t kGlobalGdnQkvzRows     = 16384;
constexpr std::int32_t kGlobalGdnQkvRows      = 10240; // q | k | v
constexpr std::int32_t kGlobalGdnZRows        = 6144;
constexpr std::int32_t kGlobalGdnQueryKeyRows = 4096;  // q4 shard parent
constexpr std::int32_t kGlobalGdnValueZRows   = 12288; // q5 shard parent
constexpr std::int32_t kGlobalGdnQueryRows    = 2048;
constexpr std::int32_t kGlobalGdnKeyRows      = 2048;
constexpr std::int32_t kGlobalGdnValueRows    = 6144;
constexpr std::int32_t kGlobalGdnConvRows     = 10240; // q | k | v depthwise channels

// GDN gating heads (a_weight / b_weight rows, and the g / beta / A_log / dt_bias vectors).
constexpr std::int32_t kGlobalGdnHeads = 48;

// MLP gate_up 34816 (2 x intermediate) and down 17408.
constexpr std::int32_t kGlobalMlpGateUpRows = 34816;
constexpr std::int32_t kGlobalMlpDownRows   = 17408;

// attention/output and gdn/output: the residual projections' output rows (and the tp1 input extent
// of the same row-parallel shards).
constexpr std::int32_t kGlobalAttentionOutRows = 6144;

// A shard extent. The callers validate that the width divides the global extent before they ask.
constexpr std::int32_t shard_rows(std::int32_t global_rows, std::int32_t tp) {
    return global_rows / tp;
}

// A row-parallel shard's input extent, which is the global output extent divided by the width.
constexpr std::int32_t shard_columns(std::int32_t global_columns, std::int32_t tp) {
    return global_columns / tp;
}

// Copies a TpArray of tensors into a mutable slot array so a per-rank kernel that takes `Tensor&`
// can be handed an element of it. Slot r of the copy aliases slot r of the source; only the slots a
// live rank owns are populated.
template <class T>
TpArray<T> tp_array_copy(const TpArray<T>& source, std::int32_t tp) {
    TpArray<T> copy{};
    for (std::int32_t rank = 0; rank < tp; ++rank) {
        copy[static_cast<std::size_t>(rank)] = source[static_cast<std::size_t>(rank)];
    }
    return copy;
}

} // namespace ninfer::ops::detail
