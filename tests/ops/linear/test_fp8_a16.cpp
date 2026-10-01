#include "ops/linear/linear_test_common.h"
#include "ops/linear/fp8/fp8_format.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_fp8_a16() {
    constexpr std::array attn_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{11, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures = run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                             {14336, 5120, 811U, Comparison::Sampled, true, attn_invocations});
    constexpr std::array gdn_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{10, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {16384, 5120, 817U, Comparison::Sampled, true, gdn_invocations});
    constexpr std::array mlp_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {34816, 5120, 821U, Comparison::Sampled, true, mlp_invocations});
    constexpr std::array vocabulary_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{49, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {248320, 5120, 823U, Comparison::Sampled, true, vocabulary_invocations});
    constexpr std::array vocabulary_tp2_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {124160, 5120, 824U, Comparison::Sampled, true,
                           vocabulary_tp2_invocations});
    constexpr std::array vocabulary_policies{
        ops::LinearPolicy::A16Only,
        ops::LinearPolicy::AllowA8,
        ops::LinearPolicy::AllowA4,
    };
    for (const ops::LinearPolicy policy : vocabulary_policies) {
        try {
            const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, 248320, 5120, policy, 1, 2048);
            if (capacity != 0) {
                std::cerr << "FP8 vocabulary A16 route reported nonzero workspace\n";
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "FP8 vocabulary policy was rejected: " << error.what() << '\n';
            ++failures;
        }
    }
    constexpr std::array residual6144_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {5120, 6144, 827U, Comparison::Sampled, true, residual6144_invocations});
    constexpr std::array residual17408_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures +=
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {5120, 17408, 829U, Comparison::Sampled, true, residual17408_invocations});

#ifdef NINFER_VOLTA_BUILD
    // The resident output head is prepacked, including the first-token call and a T=1
    // remainder after a 32-token chunk. Check its represented FP8 values directly against
    // the same independent oracle used for the portable artifact layout.
    constexpr std::array prepacked_head_invocations{
        Invocation{1, CallForm::A16Convenience}, Invocation{4}, Invocation{33},
    };
    for (const int output_rows : {248320, 124160}) {
        failures += run_shape("FP8_A16_PREPACKED_HEAD", ActivationCompute::A16, make_fp8_weight,
                              {output_rows, 5120, 833U, Comparison::Sampled, true,
                               prepacked_head_invocations, true});
    }
    constexpr std::array prefill_invocations{
        Invocation{128}, Invocation{1023}, Invocation{1024}, Invocation{1025}, Invocation{4096},
    };
    for (const bool prepacked : {false, true}) {
        failures += run_shape("FP8_A16_PREFILL", ActivationCompute::A16, make_fp8_weight,
                              {5120, 3072, 835U, Comparison::Sampled, true,
                               prefill_invocations, prepacked});
        failures += run_shape("FP8_A16_PREFILL", ActivationCompute::A16, make_fp8_weight,
                              {5120, 8704, 837U, Comparison::Sampled, true,
                               prefill_invocations, prepacked});
    }
#endif
    auto packed = make_fp8_weight(14336, 5120, 831U);
    try {
        (void)ops::detail::validate_fp8_weight(packed.weight, "FP8 validator test");
    } catch (const std::exception& error) {
        std::cerr << "valid FP8 metadata was rejected: " << error.what() << '\n';
        ++failures;
    }
    const auto expect_invalid = [&](const char* label, Weight invalid) {
        try {
            (void)ops::detail::validate_fp8_weight(invalid, "FP8 validator test");
            std::cerr << "invalid FP8 " << label << " was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    Weight invalid = packed.weight;
    invalid.layout = QuantLayout::Contiguous;
    expect_invalid("layout", invalid);
    invalid             = packed.weight;
    invalid.scale_nb[1] = invalid.scale_nb[1] - 2;
    expect_invalid("scale stride", invalid);
    invalid               = packed.weight;
    invalid.payload_bytes = invalid.payload_bytes - 1;
    expect_invalid("payload bound", invalid);
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
