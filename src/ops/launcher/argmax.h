#pragma once

// ninfer::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

void argmax_with_value_launch(const Tensor& logits, Tensor& values, Tensor& indices,
                              std::int32_t valid_rows, cudaStream_t stream);
void merge_argmax_shards_launch(const Tensor& values, const Tensor& indices, Tensor& out,
                                std::int32_t first_shard_rows, cudaStream_t stream);

} // namespace ninfer::ops::detail
