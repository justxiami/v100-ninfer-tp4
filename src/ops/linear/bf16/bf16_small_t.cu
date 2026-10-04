#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using NarrowSchedule = Bf16SmallTInnerSchedule<4,
        Geometry::kOutputRows == 256 ? 4 : 1, 1, 8, 2, 4,
        Bf16SmallTActivationAccess::WarpPacked, Bf16WeightCache::Default,
        Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
    using Schedule = std::conditional_t<Geometry::kOutputRows == 1280 ||
                                         Geometry::kOutputRows == 256,
        NarrowSchedule, typename Bf16LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type>;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    const Bf16SmallTContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                            Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_small_t_inner_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Bf16Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kBf16SmallTMinTokens + static_cast<int>(Offsets)>...};
}

using ControlGeometry = Bf16GemvGeometry<14336, 5120>;
using OutputGeometry  = Bf16GemvGeometry<5120, 6144>;

constexpr auto kControlLaunchers = make_launchers<ControlGeometry>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});
constexpr auto kOutputLaunchers = make_launchers<OutputGeometry>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});
#ifdef NINFER_VOLTA_BUILD
constexpr auto kNarrow1280Launchers = make_launchers<Bf16GemvGeometry<1280, 5120>>(
    std::make_index_sequence<16 - kBf16SmallTMinTokens + 1>{});
constexpr auto kNarrow256Launchers = make_launchers<Bf16GemvGeometry<256, 5120>>(
    std::make_index_sequence<16 - kBf16SmallTMinTokens + 1>{});
#endif

} // namespace

void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kBf16SmallTMinTokens);
#ifdef NINFER_VOLTA_BUILD
    if (weight.k == 5120 && x.ne[1] >= 2 && x.ne[1] <= 16) {
        if (weight.n == 1280) { kNarrow1280Launchers[index](x, weight, out, stream); return; }
        if (weight.n == 256) { kNarrow256Launchers[index](x, weight, out, stream); return; }
    }
#endif
    if (weight.n == ControlGeometry::kOutputRows && weight.k == ControlGeometry::kInputRows) {
        kControlLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == OutputGeometry::kOutputRows && weight.k == OutputGeometry::kInputRows) {
        kOutputLaunchers[index](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
