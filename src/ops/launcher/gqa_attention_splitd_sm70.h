#pragma once

// ninfer::ops - route to the vendored Flash-V100 (SM70) Split-D attention kernel.
//
// The kernel itself lives in src/ops/vendor/flash_v100 (1CatAI, BSD-3). Unlike the
// vendored llama.cpp route this one computes the causal mask from (query_len, kv_len)
// internally and expects contiguous fp16 Q/K/V with a [row][head][D] layout:
//   q   [tokens][heads_q ][256]   row stride heads_q *256
//   k/v [n_kv  ][heads_kv][256]   row stride heads_kv*256
// The query rows are assumed to be the LAST `tokens` rows of the KV sequence, which is
// exactly a sequential prefill chunk.
//
// Preconditions (all checked): head_dim 256, causal, tokens % 64 == 0, n_kv % 32 == 0,
// heads_q % heads_kv == 0, tokens <= n_kv.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

bool splitd_supported(std::int32_t tokens, std::int32_t n_kv, std::int32_t heads_q,
                      std::int32_t heads_kv);

void splitd_launch(const __half* q, const __half* k, const __half* v, __half* out,
                   std::int32_t tokens, std::int32_t n_kv, std::int32_t heads_q,
                   std::int32_t heads_kv, float softmax_scale, cudaStream_t stream);

// q is bf16 at the call site; the kernel wants fp16. Same element order both ways.
void splitd_convert_q(const __nv_bfloat16* src, __half* dst, std::int64_t count,
                      cudaStream_t stream);
void splitd_convert_out(const __half* src, __nv_bfloat16* dst, std::int64_t count,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
