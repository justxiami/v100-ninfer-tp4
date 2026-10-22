// ninfer::ops - launcher for the vendored Flash-V100 (SM70) Split-D attention kernel.
// See gqa_attention_splitd_sm70.h for the contract and src/ops/vendor/flash_v100/README.md
// for provenance. This translation unit is the only one that sees the vendored headers.

#include "ops/launcher/gqa_attention_splitd_sm70.h"

#include "flash_fwd_d256_splitd_sm70.cu"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

constexpr int kHeadDim      = 256;
constexpr int kBlockM       = flash::Sm70D256SplitDTraits::kBlockM;
constexpr int kBlockN       = flash::Sm70D256SplitDTraits::kBlockN;
constexpr int kNThreads     = flash::Sm70D256SplitDTraits::kNThreads;
constexpr int kSmemBytes    = flash::Sm70D256SplitDTraits::kSmemBytes;

__global__ void splitd_convert_q_kernel(const __nv_bfloat16* __restrict__ src,
                                        __half* __restrict__ dst, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { dst[i] = __float2half(__bfloat162float(src[i])); }
}

__global__ void splitd_convert_out_kernel(const __half* __restrict__ src,
                                          __nv_bfloat16* __restrict__ dst, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { dst[i] = __float2bfloat16(__half2float(src[i])); }
}

} // namespace

bool splitd_supported(std::int32_t tokens, std::int32_t n_kv, std::int32_t heads_q,
                      std::int32_t heads_kv) {
    if (tokens <= 0 || n_kv < tokens || heads_q <= 0 || heads_kv <= 0) { return false; }
    if (tokens % kBlockM != 0 || n_kv % kBlockN != 0) { return false; }
    if (heads_q % heads_kv != 0) { return false; }
    return true;
}

void splitd_launch(const __half* q, const __half* k, const __half* v, __half* out,
                   std::int32_t tokens, std::int32_t n_kv, std::int32_t heads_q,
                   std::int32_t heads_kv, float softmax_scale, cudaStream_t stream) {
    if (!splitd_supported(tokens, n_kv, heads_q, heads_kv)) {
        throw std::invalid_argument("splitd_launch: preconditions not met");
    }
    auto kernel = flash::sm70_d256_splitd_dense_kernel<cutlass::half_t, /*PagedKV=*/false>;
    static const bool attr_ok = [] {
        return cudaFuncSetAttribute(
                   flash::sm70_d256_splitd_dense_kernel<cutlass::half_t, false>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes) == cudaSuccess;
    }();
    if (!attr_ok) { throw std::runtime_error("splitd_launch: dynamic shared memory opt-in failed"); }

    const dim3 grid(tokens / kBlockM, 1, heads_q);
    kernel<<<grid, kNThreads, kSmemBytes, stream>>>(
        reinterpret_cast<const cutlass::half_t*>(q),
        reinterpret_cast<const cutlass::half_t*>(k),
        reinterpret_cast<const cutlass::half_t*>(v),
        reinterpret_cast<cutlass::half_t*>(out),
        /*q_*_stride*/ heads_q * kHeadDim, heads_q * kHeadDim, kHeadDim,
        /*k_*_stride*/ heads_kv * kHeadDim, heads_kv * kHeadDim, kHeadDim,
        /*v_*_stride*/ heads_kv * kHeadDim, heads_kv * kHeadDim, kHeadDim,
        /*query_len*/ tokens, /*kv_len*/ n_kv, /*heads_q*/ heads_q, /*heads_kv*/ heads_kv,
        /*softmax_scale_log2*/ softmax_scale * 1.4426950408889634f,
        /*block_table*/ nullptr, /*page_size*/ 0, /*bt_batch_stride*/ 0,
        /*partial_out*/ nullptr, /*partial_max*/ nullptr, /*partial_sum*/ nullptr);
}

void splitd_convert_q(const __nv_bfloat16* src, __half* dst, std::int64_t count,
                      cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    splitd_convert_q_kernel<<<blocks, kThreads, 0, stream>>>(src, dst, count);
}

void splitd_convert_out(const __half* src, __nv_bfloat16* dst, std::int64_t count,
                        cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    splitd_convert_out_kernel<<<blocks, kThreads, 0, stream>>>(src, dst, count);
}

} // namespace ninfer::ops::detail
