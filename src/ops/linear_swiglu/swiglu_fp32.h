#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Private fused-Op stage. Projected is FP32 [2*intermediate,T]. Optional BF16
// row_scales multiply the raw projections in FP32 before SiLU and the product.
void swiglu_fp32_launch(const Tensor& projected, const void* row_scales, Tensor& out,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
