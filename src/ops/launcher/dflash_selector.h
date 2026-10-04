#pragma once
#include "ninfer/ops/dflash_selector.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
void dflash2_select_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                           const Tensor&, Tensor&, Tensor&, cudaStream_t);
void dflash2_local_topk_launch(const Tensor&, Tensor&, Tensor&, std::int32_t, cudaStream_t);
void dflash2_select_sharded_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   const Tensor&, Tensor&, Tensor&, cudaStream_t);
}
