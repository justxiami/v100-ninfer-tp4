#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_cutlass_epilogue.cuh"
#include "ops/linear/fp8/fp8_prepack_sm70.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

__global__ void dequant_fp8_row_to_fp16(const std::uint8_t* __restrict__ codes, int n, int k,
                                        bool prepacked, cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int pair_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n || pair_idx >= k / 2) { return; }
    const int k0 = pair_idx * 2;
    const std::int64_t offset = prepacked
                                    ? fp8_qpn_prepacked_offset(row, k0, k)
                                    : static_cast<std::int64_t>(row) * k + k0;
    const std::uint16_t packed =
        *reinterpret_cast<const std::uint16_t*>(codes + offset);
    const float2 weight = decode_fp8_e4m3x2(packed);
    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[pair_idx * 2]     = cutlass::half_t(weight.x);
    out_row[pair_idx * 2 + 1] = cutlass::half_t(weight.y);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

using ElementAccumulator     = float;
using ElementInput           = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;
template <class EpilogueOp, class Output = ElementOutput>
using Gemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    Output, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
using PlainGemm = Gemm<cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, float, float>>;
using RowScaledGemm = Gemm<Fp8RowScaledBf16Epilogue>;
using Fp32Gemm = Gemm<cutlass::epilogue::thread::LinearCombination<float, 4, float, float>, float>;

// The broadcast epilogue wins on wide output projections and gate/up. It slows
// narrow calls and long-K residuals, which retain the separate scaling kernel.
bool fuse_row_scale(int n, int k, int t) {
    return t >= 1024 && (k == 3072 || (k == 5120 && n >= 16384));
}

__global__ void scale_rows_kernel(__nv_bfloat16* __restrict__ data,
                                  const __nv_bfloat16* __restrict__ scales, std::int64_t count,
                                  int n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    data[i] = __float2bfloat16(__bfloat162float(data[i]) * __bfloat162float(scales[i % n]));
}

template <class Allocator>
struct Scratch {
    Tensor weight;
    Tensor input;
    DeviceSpan gemm;
};

template <class Allocator>
Scratch<Allocator> allocate_scratch(Allocator& allocator, int n, int k, int cols,
                                    std::size_t gemm_bytes) {
    Scratch<Allocator> out;
    out.weight = allocator.alloc(DType::FP16, {k, n});
    out.input  = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_bytes != 0) { out.gemm = allocator.alloc_bytes(gemm_bytes); }
    return out;
}

std::size_t gemm_workspace_bytes(int n, int k, int cols) {
    const cutlass::gemm::GemmCoord shape(cols, n, k);
    typename PlainGemm::Arguments args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    return PlainGemm::get_workspace_size(args);
}

template <bool Fused, bool UnscaledFp32 = false>
void run_gemm(const cutlass::half_t* input, const cutlass::half_t* weight, const Weight& w,
              Tensor& out, int t, DeviceSpan scratch, cudaStream_t stream) {
    using Operation = std::conditional_t<UnscaledFp32, Fp32Gemm,
                                         std::conditional_t<Fused, RowScaledGemm, PlainGemm>>;
    using Output = typename Operation::ElementC;
    const cutlass::gemm::GemmCoord shape(t, w.n, w.k);
    typename Operation::Arguments args{
        shape, {input, w.k}, {weight, w.k},
        {Fused ? static_cast<const Output*>(w.scales)
               : static_cast<const Output*>(out.data), Fused ? 0 : w.n},
        {static_cast<Output*>(out.data), w.n}, {1.0F, 0.0F}, 1};
    Operation op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = op.initialize(args, scratch.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS initialize failed");
    }
    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());
    if constexpr (!Fused && !UnscaledFp32) {
        const std::int64_t count = static_cast<std::int64_t>(t) * w.n;
        scale_rows_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(w.scales),
            count, w.n);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, cols, gemm_workspace_bytes(n, k, cols));
    return layout.peak_bytes(1);
}

namespace {

template <bool UnscaledFp32>
void launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
            cudaStream_t stream) {
    const int n = w.n;
    const int k = w.k;
    const int t = x.ne[1];
    const std::size_t gemm_bytes = gemm_workspace_bytes(n, k, t);
    auto scope = ws.scope();
    Scratch<WorkspaceArena> scratch = allocate_scratch(ws, n, k, t, gemm_bytes);
    auto* weight = static_cast<cutlass::half_t*>(scratch.weight.data);
    auto* input  = static_cast<cutlass::half_t*>(scratch.input.data);

    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((k / 2 + 255) / 256), static_cast<unsigned>(n), 1u);
    dequant_fp8_row_to_fp16<<<grid, block, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), n, k,
        w.layout == QuantLayout::VoltaQpnPrepacked, weight);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t input_count = static_cast<std::int64_t>(t) * k;
    bf16_to_fp16_kernel<<<static_cast<int>((input_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), input, input_count);
    CUDA_CHECK(cudaGetLastError());

    if constexpr (UnscaledFp32) {
        run_gemm<false, true>(input, weight, w, out, t, scratch.gemm, stream);
    } else if (fuse_row_scale(n, k, t)) {
        run_gemm<true>(input, weight, w, out, t, scratch.gemm, stream);
    } else {
        run_gemm<false>(input, weight, w, out, t, scratch.gemm, stream);
    }
}

} // namespace

void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream) {
    launch<false>(x, w, out, ws, stream);
}

void fp8_cutlass_sm70_unscaled_fp32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           WorkspaceArena& ws, cudaStream_t stream) {
    launch<true>(x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
