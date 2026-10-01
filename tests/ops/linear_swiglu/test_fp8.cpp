#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <cuda_runtime.h>

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 6> kA16Cases{1, 2, 32, 33, 34, 3072};
        constexpr std::array<std::int32_t, 6> kA8Cases{1, 2, 3, 48, 65, 1024};
        int failures = 0;
        bool skipped = false;
        for (const int result : {
                 run_profile(
                     "LinearSwiGLU FP8_A16",
                     {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1811U,
                      ActivationCompute::A16},
                     kA16Cases),
             }) {
            if (result == 77) {
                skipped = true;
            } else {
                failures += result;
            }
        }
        constexpr std::array<std::int32_t, 4> kShardCases{32, 33, 34, 3072};
        const int shard_result = run_column_parallel_profile(
            "LinearSwiGLU FP8_A16 TP2",
            {QType::FP8_E4M3FN_ROW_BF16S, 17408, 5120, 8704, 1817U,
             ActivationCompute::A16}, kShardCases);
        if (shard_result != 77) { failures += shard_result; }
        int device = 0;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
            properties.major >= 10) {
            failures += run_profile(
                "LinearSwiGLU FP8_A8",
                {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1813U,
                 ActivationCompute::A8}, kA8Cases);
        }
        if (failures == 0 && skipped) { return 77; }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU FP8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU FP8 test failed: " << error.what() << '\n';
        return 1;
    }
}
