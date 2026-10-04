#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
namespace ninfer::ops {
inline constexpr std::int32_t kDFlashSelectorTile = 2048;
// Greedy DFlash2 lattice walk. logits=[V,K,B], gate=[R,K,B], predecessor/successor=[V,R],
// anchors=[B], partial=[16,ceil(V/2048),K*B] I64, topk=[16,K,B] I32,
// output draft tokens=[K,B]. Scratch is caller-owned for graph address stability.
void dflash2_select(const Tensor& logits, const Tensor& gate, const Tensor& predecessor,
                    const Tensor& successor, const Tensor& anchors, const Tensor& partial,
                    Tensor& topk, Tensor& out,
                    cudaStream_t stream);

// Exact tensor-parallel proposal helpers. Local top-16 keys carry global token IDs and ordered
// FP32 representations of the BF16 logits; the rank-0 sharded selector merges the two local
// lists before performing the lattice walk. Local logits=[Vshard,C], partial=
// [16,ceil(Vshard/2048),C], keys=[16,C]. A key's high 32 bits encode the ordered logit and its
// low 32 bits encode the complemented global ID (lower IDs win logit ties).
// The selector consumes gathered keys=[32,K*B], gate=[R,K,B], codebooks=[V,R], anchors=[B],
// global_keys=[16,K*B], and output=[K,B]. All scratch is caller-owned and contiguous.
void dflash2_local_topk(const Tensor& logits, Tensor& partial, Tensor& keys,
                        std::int32_t global_id_offset, cudaStream_t stream);
void dflash2_select_sharded(const Tensor& keys, const Tensor& gate, const Tensor& predecessor,
                            const Tensor& successor, const Tensor& anchors, Tensor& global_keys,
                            Tensor& out, cudaStream_t stream);
}
