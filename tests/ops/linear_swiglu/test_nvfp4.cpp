#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <cuda_runtime.h>

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 6> kA16Cases{1, 4, 8, 16, 128, 1024};
        constexpr std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024};
        int failures = 0;
        bool skipped = false;
        const int a16_result = run_profile(
            "LinearSwiGLU NVFP4_A16",
            {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16}, kA16Cases);
        for (const int result : {a16_result}) {
            if (result == 77) {
                skipped = true;
            } else {
                failures += result;
            }
        }
        constexpr std::array<std::int32_t, 3> kShardCases{31, 32, 3072};
        const int shard_result = run_column_parallel_profile(
            "LinearSwiGLU NVFP4_A16 TP2",
            {QType::NVFP4, 17408, 5120, 8704, 1807U, ActivationCompute::A16}, kShardCases);
        if (shard_result != 77) { failures += shard_result; }
        int device = 0;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
            properties.major >= 12) {
            failures += run_profile(
                "LinearSwiGLU NVFP4_A4",
                {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases);
        }
        if (failures == 0 && skipped) { return 77; }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
