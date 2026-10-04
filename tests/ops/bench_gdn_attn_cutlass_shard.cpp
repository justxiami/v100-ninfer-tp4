// Kernel-level prefill benchmark: fused gdn/attn input shard, tp4, T=4096 (the engine's
// prefill-chunk width). Compares the NVFP4 cutlass fast path (10/04 afternoon) against the
// official-v3 FP8 cutlass path (the 3470 tok/s baseline's projection) and against the
// pre-fix 32-token small_t chunk loop (null workspace -> chunk route), so the end-to-end
// prefill gap can be attributed to the projection or to everything else.
//
// Weights are the test patterned fixtures (not production artifacts): the point is kernel
// speed at identical shapes, not numerics (the op tests already validate the cutlass path
// against the FP64 oracle). Run: ninja ninfer_gdn_attn_cutlass_bench && ./tests/...

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "input_projection_test_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
using namespace ninfer::test::quantized_weight;

namespace {

constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kT      = 4096; // engine prefill-chunk width
constexpr int kIters          = 15;

std::vector<std::uint16_t> make_x_bits(std::uint32_t seed) {
    std::vector<std::uint16_t> out;
    out.reserve(std::size_t(kHidden) * kT);
    std::uint64_t s = seed;
    for (std::size_t i = 0; i < out.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const float v = float((s >> 40) & 0xFFFFFF) / float(1u << 20) - 1.0f;
        out.push_back(__float2bfloat16_rn(v));
    }
    return out;
}

double bench(std::string_view label, std::function<void()> run) {
    run(); // warmup
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) { run(); }
    cudaDeviceSynchronize();
    const auto t1    = std::chrono::steady_clock::now();
    const double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count() / kIters;
    const double tps = std::int64_t(kT) / (ms / 1000.0);
    std::printf("%-42s %8.2f ms  %8.0f tok/s\n", std::string(label).c_str(), ms, tps);
    return ms;
}

} // namespace

int main() {
    cudaSetDevice(0);
    const std::vector<std::uint16_t> x_bits = make_x_bits(20261004U);
    DeviceBuffer x_device(std::size_t(kHidden) * kT * sizeof(std::uint16_t));
    x_device.copy_from_host(x_bits.data(), x_bits.size() * sizeof(std::uint16_t));
    Tensor x(x_device.p, DType::BF16, {kHidden, kT});
    cudaStream_t stream = nullptr;

    // --- gdn tp4 shard (4096 rows = qkv 2560 + z 1536) -----------------------------------
    {
        PatternedWeightOptions options;
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
        DevicePackedWeight nvfp4(make_patterned_weight(QType::NVFP4, 4096, kHidden, 627U, options));
        DevicePackedWeight fp8(make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, 4096, kHidden, 613U));
        GuardedBf16Tensor qkv(2560, kT);
        GuardedBf16Tensor z(1536, kT);
        Tensor qkv_t = qkv.tensor();
        Tensor z_t   = z.tensor();
        const std::size_t cap_nvfp4 =
            ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, kHidden,
                                                         ops::LinearPolicy::A16Only, kT, kT);
        const std::size_t cap_fp8 =
            ops::gdn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, 16384, kHidden,
                                                         ops::LinearPolicy::A16Only, kT, kT);
        std::fprintf(stderr, "cap gdn nvfp4=%zu fp8=%zu\n", cap_nvfp4, cap_fp8);
        std::fprintf(stderr, "== gdn input tp4 shard (4096x5120), T=%d ==\n", kT);
        WorkspaceArena ws_nvfp4(cap_nvfp4);
        bench("nvfp4 gdn CUTLASS", [&] {
            ws_nvfp4.reset();
            ops::detail::nvfp4_gdn_input_dispatch_shard(x, nvfp4.view(), qkv_t, z_t,
                                                        ops::LinearPolicy::A16Only, &ws_nvfp4, stream);
        });
        WorkspaceArena ws_fp8(cap_fp8);
        bench("fp8 gdn CUTLASS (official path)", [&] {
            ws_fp8.reset();
            ops::detail::fp8_gdn_input_dispatch_shard(x, fp8.view(), qkv_t, z_t,
                                                      ops::LinearPolicy::A16Only, &ws_fp8, stream);
        });
        bench("nvfp4 gdn small_t chunks (pre-fix)", [&] {
            ops::detail::nvfp4_gdn_input_dispatch_shard(x, nvfp4.view(), qkv_t, z_t,
                                                        ops::LinearPolicy::A16Only, nullptr, stream);
        });
    }

    // --- attn tp4 shard (3584 rows = q/gate 1536 + k/v 256) ------------------------------
    {
        PatternedWeightOptions options;
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
        DevicePackedWeight nvfp4(make_patterned_weight(QType::NVFP4, 3584, kHidden, 367U, options));
        DevicePackedWeight fp8(make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, 3584, kHidden, 353U));
        GuardedBf16Tensor q(1536, kT);
        GuardedBf16Tensor gate(1536, kT);
        GuardedBf16Tensor k(256, kT);
        GuardedBf16Tensor v(256, kT);
        Tensor q_t = q.tensor();
        Tensor g_t = gate.tensor();
        Tensor k_t = k.tensor();
        Tensor v_t = v.tensor();
        const std::size_t cap_nvfp4 =
            ops::attn_input_proj_workspace_capacity_bytes(QType::NVFP4, 14336, kHidden,
                                                          ops::LinearPolicy::A16Only, kT, kT);
        const std::size_t cap_fp8 =
            ops::attn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, 14336, kHidden,
                                                          ops::LinearPolicy::A16Only, kT, kT);
        std::fprintf(stderr, "cap attn nvfp4=%zu fp8=%zu\n", cap_nvfp4, cap_fp8);
        std::fprintf(stderr, "== attn input tp4 shard (3584x5120), T=%d ==\n", kT);
        WorkspaceArena ws_nvfp4(cap_nvfp4);
        bench("nvfp4 attn CUTLASS", [&] {
            ws_nvfp4.reset();
            ops::detail::nvfp4_attn_input_dispatch_shard(x, nvfp4.view(), q_t, g_t, k_t, v_t,
                                                         ops::LinearPolicy::A16Only, &ws_nvfp4, stream);
        });
        WorkspaceArena ws_fp8(cap_fp8);
        bench("fp8 attn CUTLASS (official path)", [&] {
            ws_fp8.reset();
            ops::detail::fp8_attn_input_dispatch_shard(x, fp8.view(), q_t, g_t, k_t, v_t,
                                                       ops::LinearPolicy::A16Only, &ws_fp8, stream);
        });
        bench("nvfp4 attn small_t chunks (pre-fix)", [&] {
            ops::detail::nvfp4_attn_input_dispatch_shard(x, nvfp4.view(), q_t, g_t, k_t, v_t,
                                                         ops::LinearPolicy::A16Only, nullptr, stream);
        });
    }

    std::printf("done\n");
    return 0;
}