#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using NarrowSchedule = Bf16GemvSchedule<4,
        Geometry::kOutputRows == 256 ? 4 : 1, 1, 8, 2,
        Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
        Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
    using Schedule = std::conditional_t<Geometry::kInputRows == 5120 &&
        (Geometry::kOutputRows == 1280 || Geometry::kOutputRows == 256),
        NarrowSchedule, Bf16LinearDecodeSchedule<Geometry>>;

    const Bf16ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (weight.k == 5120 && weight.n == 1280) {
        launch_geometry<Bf16GemvGeometry<1280, 5120>>(x, weight, out, stream);
        return;
    }
    if (weight.k == 5120 && weight.n == 256) {
        launch_geometry<Bf16GemvGeometry<256, 5120>>(x, weight, out, stream);
        return;
    }
#endif
    if (weight.n == 14336 && weight.k == 5120) {
        launch_geometry<Bf16GemvGeometry<14336, 5120>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 6144) {
        launch_geometry<Bf16GemvGeometry<5120, 6144>>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear decode: unsupported exact problem");
}

} // namespace ninfer::ops::detail
