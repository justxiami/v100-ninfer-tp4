#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "ops/linear_attention/gated_delta_net/common.cuh"

namespace ninfer::ops::detail::gated_delta_net {

// Chunked VOLTA path: FLA-style intra-chunk solve plus a chunk-sequential state scan. The
// sequential recurrent kernel remains the fallback for short sequences and tails.
inline constexpr std::int32_t kChunkedTokens    = 32;   // intra-chunk width
inline constexpr std::int32_t kChunkedMinTokens = 128;  // below this the sequential kernel wins
// The op-level wiring is not numerically clean yet: the same geometry validates in
// bench/ops/gdn_chunked_v3_probe.cu (out relL2 4.3e-4, state 3.0e-4) but the in-tree path differs,
// so it stays off until the plumbing (normalisation vs layout) is bisected. Flip to enable.
inline constexpr bool kChunkedEnabled = true;

// The chunked path's intermediate buffers. Capacity accounting must replay the exact same
// allocations, because the op suite requires the query and execution peaks to match exactly.
// Number of tokens the chunked route consumes as whole chunks; the remainder is a sequential tail.
inline constexpr std::int32_t chunked_body_tokens(std::int32_t tokens) {
    return (tokens / kChunkedTokens) * kChunkedTokens;
}

template <class Allocator>
void chunked_volta_allocate(Allocator& allocator, std::int32_t value_heads, std::int32_t tokens) {
    const std::int64_t chunks   = (static_cast<std::int64_t>(tokens) + kChunkedTokens - 1) /
                                  kChunkedTokens;
    const std::int64_t elems    = static_cast<std::int64_t>(value_heads) * tokens * kStateDim;
    const std::int64_t qk_elems = static_cast<std::int64_t>(value_heads) * chunks *
                                  kChunkedTokens * kChunkedTokens;
    allocator.alloc_bytes(elems * sizeof(std::uint16_t));                     // k_cumdecay
    allocator.alloc_bytes(elems * sizeof(std::uint16_t));                     // q * scale
    allocator.alloc_bytes(elems * sizeof(std::uint16_t));                     // k * decay
    allocator.alloc_bytes(elems * sizeof(float));                             // v_corr
    allocator.alloc_bytes(qk_elems * sizeof(std::uint16_t));                  // qk
    allocator.alloc_bytes(static_cast<std::int64_t>(value_heads) * tokens * sizeof(float));  // g_cum
}

void launch_chunked_volta(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
