#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a16() {
    constexpr std::array attn_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{20, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    constexpr std::array new_problem_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::Sampled, true, attn_invocations});
    // tp4 shard decode shapes (10/04, quasar NVFP4): attention input (t<=3 decode) and the
    // row-parallel output/down projections (t<8 decode) resolve through A16 to QPN2/SIMT at these
    // geometries; the parent suites above never instantiated them.
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {3584, 5120, 741U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {4096, 5120, 743U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 1536, 745U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 4352, 747U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::Sampled, true, new_problem_invocations});
#ifdef NINFER_VOLTA_BUILD
    constexpr std::array prefill_invocations{
        Invocation{128}, Invocation{1024}, Invocation{2047}, Invocation{2048}, Invocation{2049},
        Invocation{3072}, Invocation{4096},
    };
    for (const bool prepacked : {false, true}) {
        // The oracle decodes the original artifact bytes, independently of both the load-time
        // QPN permutation and the prefill FP16 materialization. Include the TP2 row shard.
        failures += run_shape("NVFP4_A16_PREFILL", ActivationCompute::A16, make_nvfp4_weight,
                              {5120, 8704, 709U, Comparison::Sampled, true,
                               prefill_invocations, prepacked});
        failures += run_shape("NVFP4_A16_PREFILL", ActivationCompute::A16, make_nvfp4_weight,
                              {17408, 5120, 711U, Comparison::Sampled, true,
                               prefill_invocations, prepacked});
    }
#endif
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
