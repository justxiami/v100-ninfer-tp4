// Minimal probe for the tp > 2 issue-style question: nothing but the communicator, one buffer per
// rank, a single-threaded allreduce, and a bit-exact check after EVERY call.
//
// The full suite (tests/ops/test_allreduce_nccl4.cpp) fails in its 480 KiB case under
// NINFER_TP4_ISSUE=single, while a standalone program with the same loop does not -- so this probe
// exists to find out whether the trigger is the binary itself (this one links ninfer_core/ops) or
// the sequence of work the full suite performs (in which case adding pieces back will find it).
#include "core/dtype.h"
#include "core/tensor.h"
#include "core/tp_comm.h"
#include "ninfer/ops/allreduce.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr int kRanks = 4;
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kTokens = 48;

void fill(ninfer::Tensor tensor, int rank) {
    std::vector<__nv_bfloat16> host(static_cast<std::size_t>(tensor.ne[0]) * tensor.ne[1]);
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = __float2bfloat16_rn(static_cast<float>((rank + 1) * ((i % 5) + 1)));
    }
    CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
}

int check(const ninfer::ExecutionContext& ec, std::int32_t columns, int rounds) {
    const std::size_t count = static_cast<std::size_t>(kHidden) * columns;
    ninfer::TpArray<ninfer::Tensor> buffer;
    ninfer::TpArray<ninfer::Tensor> staging;
    std::vector<void*> allocations;
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        void* storage = nullptr;
        void* scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&storage, count * 2));
        CUDA_CHECK(cudaMalloc(&scratch, count * 2));
        allocations.push_back(storage);
        allocations.push_back(scratch);
        buffer[slot]  = ninfer::Tensor(storage, ninfer::DType::BF16, {kHidden, columns});
        staging[slot] = ninfer::Tensor(scratch, ninfer::DType::BF16, {kHidden, columns});
    }
    const ninfer::ops::PeerEvents events(ec);
    int failures = 0;
    for (int round = 0; round < rounds; ++round) {
        for (int rank = 0; rank < kRanks; ++rank) { fill(buffer[rank], rank); }
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        ninfer::ops::allreduce_sum(buffer, staging, ec, events);
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            std::vector<__nv_bfloat16> got(count);
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            CUDA_CHECK(cudaMemcpy(got.data(), buffer[rank].data, count * 2,
                                  cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < count; ++i) {
                float expected = 0.0f;
                for (int other = 0; other < kRanks; ++other) {
                    expected += static_cast<float>((other + 1) * ((i % 5) + 1));
                }
                const __nv_bfloat16 want = __float2bfloat16_rn(expected);
                if (std::memcmp(&got[i], &want, sizeof(want)) != 0) { ++failures; }
            }
        }
    }
    for (void* pointer : allocations) { CUDA_CHECK(cudaFree(pointer)); }
    return failures;
}

// Same as check() but allocating and freeing the buffers per round, which is what the full suite's
// check_allreduce does -- the one structural difference between it and check() above.
int check_churn(const ninfer::ExecutionContext& ec, std::int32_t columns, int rounds) {
    const std::size_t count = static_cast<std::size_t>(kHidden) * columns;
    const ninfer::ops::PeerEvents events(ec);
    int failures = 0;
    for (int round = 0; round < rounds; ++round) {
        ninfer::TpArray<ninfer::Tensor> buffer;
        ninfer::TpArray<ninfer::Tensor> staging;
        std::vector<void*> allocations;
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            void* storage = nullptr;
            void* scratch = nullptr;
            CUDA_CHECK(cudaMalloc(&storage, count * 2));
            CUDA_CHECK(cudaMalloc(&scratch, count * 2));
            allocations.push_back(storage);
            allocations.push_back(scratch);
            buffer[slot]  = ninfer::Tensor(storage, ninfer::DType::BF16, {kHidden, columns});
            staging[slot] = ninfer::Tensor(scratch, ninfer::DType::BF16, {kHidden, columns});
            fill(buffer[slot], rank);
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
        }
        ninfer::ops::allreduce_sum(buffer, staging, ec, events);
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            std::vector<__nv_bfloat16> got(count);
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            CUDA_CHECK(cudaMemcpy(got.data(), buffer[rank].data, count * 2,
                                  cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < count; ++i) {
                float expected = 0.0f;
                for (int other = 0; other < kRanks; ++other) {
                    expected += static_cast<float>((other + 1) * ((i % 5) + 1));
                }
                const __nv_bfloat16 want = __float2bfloat16_rn(expected);
                if (std::memcmp(&got[i], &want, sizeof(want)) != 0) { ++failures; }
            }
        }
        for (void* pointer : allocations) { CUDA_CHECK(cudaFree(pointer)); }
    }
    return failures;
}

int check_churn_device(const ninfer::ExecutionContext& ec, std::int32_t columns, int rounds) {
    const std::size_t count = static_cast<std::size_t>(kHidden) * columns;
    const ninfer::ops::PeerEvents events(ec);
    int failures = 0;
    for (int round = 0; round < rounds; ++round) {
        ninfer::TpArray<ninfer::Tensor> buffer;
        ninfer::TpArray<ninfer::Tensor> staging;
        std::vector<void*> allocations;
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            void* storage = nullptr;
            void* scratch = nullptr;
            CUDA_CHECK(cudaMalloc(&storage, count * 2));
            CUDA_CHECK(cudaMalloc(&scratch, count * 2));
            allocations.push_back(storage);
            allocations.push_back(scratch);
            buffer[slot]  = ninfer::Tensor(storage, ninfer::DType::BF16, {kHidden, columns});
            staging[slot] = ninfer::Tensor(scratch, ninfer::DType::BF16, {kHidden, columns});
            fill(buffer[slot], rank);
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        ninfer::ops::allreduce_sum(buffer, staging, ec, events);
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            std::vector<__nv_bfloat16> got(count);
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            CUDA_CHECK(cudaMemcpy(got.data(), buffer[rank].data, count * 2,
                                  cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < count; ++i) {
                float expected = 0.0f;
                for (int other = 0; other < kRanks; ++other) {
                    expected += static_cast<float>((other + 1) * ((i % 5) + 1));
                }
                const __nv_bfloat16 want = __float2bfloat16_rn(expected);
                if (std::memcmp(&got[i], &want, sizeof(want)) != 0) { ++failures; }
            }
        }
        for (void* pointer : allocations) { CUDA_CHECK(cudaFree(pointer)); }
    }
    return failures;
}

} // namespace

int main() {
    ninfer::ExecutionContext ec({0, 1, 2, 3});
    ec.comm = ninfer::TpComm::create(ec);

    // Same payloads as the full suite, in the same order, but with no other work in between.
    const int small = check(ec, 1, 12);
    std::cout << "probe [5120,1]  10 KiB x12 : " << (small == 0 ? "bit-exact" : "WRONG") << '\n';
    const int large = check(ec, kTokens, 12);
    std::cout << "probe [5120,48] 480 KiB x12: " << (large == 0 ? "bit-exact" : "WRONG") << '\n';
    const int churn_small = check_churn(ec, 1, 6);
    std::cout << "probe churn [5120,1]  x6 : " << (churn_small == 0 ? "bit-exact" : "WRONG") << '\n';
    const int churn_large = check_churn(ec, kTokens, 12);
    std::cout << "probe churn [5120,48] x12: " << (churn_large == 0 ? "bit-exact" : "WRONG") << '\n';
    const int churn_dev = check_churn_device(ec, kTokens, 12);
    std::cout << "probe churn+devsync x12  : " << (churn_dev == 0 ? "bit-exact" : "WRONG") << '\n';
    return (small == 0 && large == 0 && churn_small == 0 && churn_large == 0 && churn_dev == 0) ? 0 : 1;
}
