#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Volta-only cutlass fast path for the FUSED NVFP4 attention input projection. The generic
// NVFP4 -> FP16 CUTLASS GEMM writes one contiguous [n, T] projection, and a small split kernel
// scatters the rows into the shard's q/gate/k/v outputs in the parent's physical row order
// Q|K|Gate|V -- the same two-stage design fp8_attn_input_cutlass_sm70.cu proves at the
// official-v3 3470 tok/s prefill baseline. The fused NVFP4 shard previously had no path above the
// 32-token small_t chunk, so prefill hit the HBM bandwidth wall.
//
// Entry: nvfp4_attn_input_a16_dispatch_shard routes T >= 33 here when a caller workspace exists;
// T <= 32 keeps the decode/small_t kernels.
[[nodiscard]] std::size_t nvfp4_attn_input_cutlass_workspace_bytes(std::int32_t tokens);
[[nodiscard]] std::size_t nvfp4_attn_input_cutlass_shard_workspace_bytes(std::int32_t tokens);
void nvfp4_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v,
                                          WorkspaceArena& workspace, cudaStream_t stream);
void nvfp4_attn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& gate, Tensor& k, Tensor& v,
                                                WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail