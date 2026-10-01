// Opt-in four-device test for the tp > 2 collective backend (src/core/tp_comm.h).
//
// The tp == 2 path has its own suite (ops/test_allreduce.cpp) exercising the hand-written pull
// protocol. This one covers the other backend, and the things that are specific to it:
//
//   * the communicator plus its WARMUP collective come up at all. The warmup is the part NCCL
//     cannot do lazily: the first collective's channel handshake needs every rank inside the call
//     concurrently, so TpComm drives it from one host thread per rank. A regression there hangs
//     (rather than fails), which is why this test exists as an end-to-end smoke of construction.
//   * allreduce_sum is bit-exact at four ranks for exactly representable inputs, at the real decode
//     shape (5120 BF16 values, all-reduced 128 times per token) and at the 2-D prefill shape.
//   * allgather_rows reproduces the Op's [C, R] destination layout with C == 1 -- the shape the
//     logits head actually asks for (one vocabulary column at a time, piece [1, shard_vocab]).
//     NCCL's rank-major concatenation is byte-identical to that layout, which is why the tp > 2
//     implementation needs no scatter kernel.
//   * an unbounded back-to-back sequence of collectives is safe with no host synchronization in
//     between (the ordering the graph-captured decode program depends on).
//
// Skipped (exit 77) unless four CUDA devices are visible, so it costs nothing in a default ctest
// run on a smaller box.
#include "core/dtype.h"
#include "core/tensor.h"
#include "core/tp_comm.h"
#include "ninfer/ops/allreduce.h"

#include <nccl.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {

constexpr int kRanks = 4;
constexpr std::int32_t kHidden = 5120;            // real decode allreduce width
constexpr std::int32_t kTokens = 48;              // real prefill chunk width
constexpr std::int32_t kVocab = 248320;           // real vocabulary
constexpr std::int32_t kShardVocab = kVocab / kRanks;

// Integer-valued inputs so the 4-rank sum is exactly representable in BF16 and the comparison can
// be bitwise (max sum: 4 ranks * 5 * 3 = 60).
inline float input_value(int rank, std::int32_t ne0_index, std::int32_t ne1_index) {
    return static_cast<float>((rank + 1) * ((ne0_index % 5) + 1) * ((ne1_index % 3) + 1));
}

std::uint16_t bf16_bits(float value) {
    const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &rounded, sizeof(bits));
    return bits;
}

void fill_input(ninfer::Tensor tensor, int rank) {
    std::vector<__nv_bfloat16> host(static_cast<std::size_t>(tensor.ne[0]) * tensor.ne[1]);
    for (std::int32_t row = 0; row < tensor.ne[1]; ++row) {
        for (std::int32_t col = 0; col < tensor.ne[0]; ++col) {
            host[static_cast<std::size_t>(row) * tensor.ne[0] + col] =
                __float2bfloat16_rn(input_value(rank, col, row));
        }
    }
    CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
}

void sync_all(const ninfer::ExecutionContext& ec) {
    for (int rank = 0; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
    }
}

// The Op contract's caller obligation (include/ninfer/ops/allreduce.h): inputs staged with the plain
// cudaMemcpy/cudaMemset forms land on the LEGACY DEFAULT stream, which does NOT implicitly
// synchronize with DeviceContext::stream. A stream sync is NOT enough to retire them for a large
// collective: measured here, a 480 KiB allreduce reading freshly staged (and freshly allocated)
// buffers came back with a wrong tail under stream sync alone and was bit-exact under a device
// sync. The tp2 suite's retire_staging does exactly this for the same reason.
void retire_staging(const ninfer::ExecutionContext& ec) {
    for (int rank = 0; rank < ec.tp; ++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[static_cast<std::size_t>(rank)]->device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// `ne0` is the contiguous dimension (see core/tensor.h), so the real decode case is [5120, 1].
int check_allreduce(const char* label, std::int32_t ne0, std::int32_t ne1,
                    const ninfer::ExecutionContext& ec, const ninfer::ops::PeerEvents& events) {
    ninfer::TpArray<ninfer::Tensor> buffer;
    ninfer::TpArray<ninfer::Tensor> staging;
    std::vector<void*> allocations;
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        void* storage = nullptr;
        void* scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&storage, static_cast<std::size_t>(ne0) * ne1 * 2));
        CUDA_CHECK(cudaMalloc(&scratch, static_cast<std::size_t>(ne0) * ne1 * 2));
        allocations.push_back(storage);
        allocations.push_back(scratch);
        buffer[slot]  = ninfer::Tensor(storage, ninfer::DType::BF16, {ne0, ne1});
        staging[slot] = ninfer::Tensor(scratch, ninfer::DType::BF16, {ne0, ne1});
        fill_input(buffer[slot], rank);
    }
    retire_staging(ec);  // caller obligation: retire the legacy-default-stream staging

    ninfer::ops::allreduce_sum(buffer, staging, ec, events);
    sync_all(ec);

    int failures = 0;
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        std::vector<__nv_bfloat16> got(static_cast<std::size_t>(ne0) * ne1);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        CUDA_CHECK(cudaMemcpy(got.data(), buffer[slot].data,
                              got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        // Report the mismatch distribution rather than the first hit: "which third of the buffer
        // is wrong" is what distinguishes a chunking problem from a shape problem.
        std::size_t bad = 0;
        std::size_t first_bad = static_cast<std::size_t>(-1);
        std::size_t last_bad = 0;
        for (std::int32_t row = 0; row < ne1; ++row) {
            for (std::int32_t col = 0; col < ne0; ++col) {
                float expected = 0.0f;
                for (int other = 0; other < kRanks; ++other) {
                    expected += input_value(other, col, row);
                }
                const std::size_t index = static_cast<std::size_t>(row) * ne0 + col;
                std::uint16_t got_bits = 0;
                std::memcpy(&got_bits, &got[index], sizeof(got_bits));
                if (got_bits != bf16_bits(expected)) {
                    if (bad == 0) {
                        first_bad = index;
                        std::cerr << label << ": rank " << rank << " first mismatch [" << col << ","
                                  << row << "] index " << index << " got 0x" << std::hex << got_bits
                                  << " want 0x" << bf16_bits(expected) << std::dec << '\n';
                    }
                    last_bad = index;
                    ++bad;
                }
            }
        }
        if (bad != 0) {
            std::cerr << label << ": rank " << rank << " -> " << bad << " / "
                      << (static_cast<std::size_t>(ne0) * ne1) << " elements wrong, indices ["
                      << first_bad << ", " << last_bad << "], first_bad/ne0 = "
                      << (first_bad / static_cast<std::size_t>(ne0)) << '\n';
            ++failures;
        }
    }
    for (void* pointer : allocations) {
        CUDA_CHECK(cudaFree(pointer));
    }
    std::cout << label << ": " << (failures == 0 ? "bit-exact" : "MISMATCH") << '\n';
    return failures;
}

// The logits gather exactly as the model issues it: C == 1, one vocabulary column at a time.
int check_allgather_logits(const ninfer::ExecutionContext& ec,
                           const ninfer::ops::PeerEvents& events) {
    ninfer::TpArray<ninfer::Tensor> piece;
    ninfer::TpArray<ninfer::Tensor> whole;
    std::vector<void*> allocations;
    std::vector<std::vector<std::uint16_t>> expected(kRanks);
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        void* shard = nullptr;
        void* full = nullptr;
        CUDA_CHECK(cudaMalloc(&shard, static_cast<std::size_t>(kShardVocab) * 2));
        CUDA_CHECK(cudaMalloc(&full, static_cast<std::size_t>(kVocab) * 2));
        allocations.push_back(shard);
        allocations.push_back(full);
        piece[slot] = ninfer::Tensor(shard, ninfer::DType::BF16, {1, kShardVocab});
        whole[slot] = ninfer::Tensor(full, ninfer::DType::BF16, {1, kVocab});
        // Distinct, deterministic bit patterns: any layout mistake is visible byte for byte.
        expected[slot].resize(static_cast<std::size_t>(kShardVocab));
        for (std::int32_t i = 0; i < kShardVocab; ++i) {
            expected[slot][static_cast<std::size_t>(i)] =
                static_cast<std::uint16_t>((0x3F00u + rank * 0x40u + (i % 0x3Fu)) & 0x7FFFu);
        }
        CUDA_CHECK(cudaMemcpy(piece[slot].data, expected[slot].data(),
                              expected[slot].size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice));
    }
    retire_staging(ec);

    ninfer::ops::allgather_rows(whole, piece, ec, events);
    sync_all(ec);

    int failures = 0;
    for (int rank = 0; rank < kRanks && failures == 0; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        std::vector<std::uint16_t> got(static_cast<std::size_t>(kVocab));
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        CUDA_CHECK(cudaMemcpy(got.data(), whole[slot].data, got.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        for (int source = 0; source < kRanks && failures == 0; ++source) {
            for (std::int32_t i = 0; i < kShardVocab; ++i) {
                // rank `source` owns rows [source * shard, (source+1) * shard); C == 1 means the
                // destination row index IS the flat index.
                const std::size_t index =
                    static_cast<std::size_t>(source) * kShardVocab + static_cast<std::size_t>(i);
                if (got[index] != expected[static_cast<std::size_t>(source)][static_cast<std::size_t>(i)]) {
                    std::cerr << "allgather (C=1): rank " << rank << " row " << index
                              << " got 0x" << std::hex << got[index] << " want 0x"
                              << expected[static_cast<std::size_t>(source)][static_cast<std::size_t>(i)]
                              << std::dec << '\n';
                    ++failures;
                    break;
                }
            }
        }
    }
    for (void* pointer : allocations) {
        CUDA_CHECK(cudaFree(pointer));
    }
    std::cout << "allgather_rows [1," << kVocab << "]: " << (failures == 0 ? "bit-exact" : "MISMATCH")
              << '\n';
    return failures;
}

// 50 rounds of four allreduces plus one gather with no host synchronization between them: the
// ordering (and buffer reuse) the captured decode program relies on. Inputs are zero so repeated
// in-place allreduces stay zero.
int check_back_to_back_sequence(const ninfer::ExecutionContext& ec,
                                const ninfer::ops::PeerEvents& events) {
    ninfer::TpArray<ninfer::Tensor> buffer;
    ninfer::TpArray<ninfer::Tensor> staging;
    std::vector<void*> allocations;
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        void* storage = nullptr;
        void* scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&storage, static_cast<std::size_t>(kHidden) * 2));
        CUDA_CHECK(cudaMalloc(&scratch, static_cast<std::size_t>(kHidden) * 2));
        allocations.push_back(storage);
        allocations.push_back(scratch);
        buffer[slot]  = ninfer::Tensor(storage, ninfer::DType::BF16, {kHidden, 1});
        staging[slot] = ninfer::Tensor(scratch, ninfer::DType::BF16, {kHidden, 1});
        CUDA_CHECK(cudaMemset(storage, 0, static_cast<std::size_t>(kHidden) * 2));
    }
    retire_staging(ec);

    constexpr int kRounds = 50;
    for (int round = 0; round < kRounds; ++round) {
        for (int repeat = 0; repeat < 4; ++repeat) {
            ninfer::ops::allreduce_sum(buffer, staging, ec, events);
        }
    }
    sync_all(ec);

    int failures = 0;
    for (int rank = 0; rank < kRanks && failures == 0; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        std::vector<__nv_bfloat16> got(static_cast<std::size_t>(kHidden));
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        CUDA_CHECK(cudaMemcpy(got.data(), buffer[slot].data, got.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        const std::uint16_t zero = bf16_bits(0.0f);
        for (const __nv_bfloat16& value : got) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            if (bits != zero) {
                std::cerr << "back-to-back: rank " << rank << " drifted to 0x" << std::hex << bits
                          << std::dec << '\n';
                ++failures;
                break;
            }
        }
    }
    for (void* pointer : allocations) {
        CUDA_CHECK(cudaFree(pointer));
    }
    std::cout << "back-to-back " << kRounds << "x4 allreduces: "
              << (failures == 0 ? "stable" : "MISMATCH") << '\n';
    return failures;
}

// EXPERIMENT: the same 2-D allreduce issued from one thread per rank instead of one thread total.
// If this is exact while the single-threaded issue is not, then a large allreduce needs concurrent
// host entry (NCCL's own multi-process usage always has it), and the tp > 2 backend has to be
// restructured -- ninfer drives every rank from one thread.
int check_allreduce_threaded(const ninfer::ExecutionContext& ec, std::int32_t ne0, std::int32_t ne1,
                             int /*unused*/, const ninfer::ops::PeerEvents&) {
    ninfer::TpArray<ninfer::Tensor> buffer;
    std::vector<void*> allocations;
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        void* storage = nullptr;
        CUDA_CHECK(cudaMalloc(&storage, static_cast<std::size_t>(ne0) * ne1 * 2));
        allocations.push_back(storage);
        buffer[slot] = ninfer::Tensor(storage, ninfer::DType::BF16, {ne0, ne1});
        fill_input(buffer[slot], rank);
    }
    std::vector<std::thread> threads;
    for (int rank = 0; rank < kRanks; ++rank) {
        threads.emplace_back([&, rank] {
            const std::size_t slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            const std::size_t count = static_cast<std::size_t>(ne0) * ne1;
            ncclAllReduce(buffer[slot].data, buffer[slot].data, count, ncclBfloat16, ncclSum,
                          static_cast<ncclComm_t>(ec.comm->comm_handle(rank)),
                          ec.dev[slot]->stream);
            CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
        });
    }
    for (std::thread& thread : threads) { thread.join(); }

    int failures = 0;
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        std::vector<__nv_bfloat16> got(static_cast<std::size_t>(ne0) * ne1);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        CUDA_CHECK(cudaMemcpy(got.data(), buffer[slot].data, got.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        for (std::int32_t row = 0; row < ne1; ++row) {
            for (std::int32_t col = 0; col < ne0; ++col) {
                float expected = 0.0f;
                for (int other = 0; other < kRanks; ++other) { expected += input_value(other, col, row); }
                std::uint16_t bits = 0;
                std::memcpy(&bits, &got[static_cast<std::size_t>(row) * ne0 + col], sizeof(bits));
                if (bits != bf16_bits(expected)) { ++failures; }
            }
        }
    }
    for (void* pointer : allocations) { CUDA_CHECK(cudaFree(pointer)); }
    std::cout << "allreduce_sum [5120,48] issued from " << kRanks << " threads: "
              << (failures == 0 ? "bit-exact" : "MISMATCH(" + std::to_string(failures) + ")") << '\n';
    return failures == 0 ? 0 : 1;
}

// Timing, so the multi-threaded issue cost is a measured number rather than a guess: this is the
// overhead the tp > 2 path pays per collective on top of NCCL's own cost.
int time_collectives(const ninfer::ExecutionContext& ec, const ninfer::ops::PeerEvents& events) {
    ninfer::TpArray<ninfer::Tensor> buffer;
    ninfer::TpArray<ninfer::Tensor> staging;
    std::vector<void*> allocations;
    const std::int32_t ne0 = kHidden;
    for (int rank = 0; rank < kRanks; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        void* storage = nullptr;
        void* scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&storage, static_cast<std::size_t>(ne0) * kTokens * 2));
        CUDA_CHECK(cudaMalloc(&scratch, static_cast<std::size_t>(ne0) * kTokens * 2));
        allocations.push_back(storage);
        allocations.push_back(scratch);
        buffer[slot]  = ninfer::Tensor(storage, ninfer::DType::BF16, {ne0, kTokens});
        staging[slot] = ninfer::Tensor(scratch, ninfer::DType::BF16, {ne0, kTokens});
        CUDA_CHECK(cudaMemset(storage, 0, static_cast<std::size_t>(ne0) * kTokens * 2));
    }
    sync_all(ec);
    auto timed = [&](const char* label, std::int32_t columns, int repeats) {
        ninfer::TpArray<ninfer::Tensor> view;
        for (int rank = 0; rank < kRanks; ++rank) {
            const std::size_t slot = static_cast<std::size_t>(rank);
            view[slot] = ninfer::Tensor(buffer[slot].data, ninfer::DType::BF16, {ne0, columns});
        }
        ninfer::ops::allreduce_sum(view, staging, ec, events);  // warm
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) {
            ninfer::ops::allreduce_sum(view, staging, ec, events);
        }
        sync_all(ec);
        const double micros =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                .count() / repeats;
        std::cout << "eager allreduce [" << ne0 << "," << columns << "] (" << (columns * 10) / 1024
                  << " KiB): " << micros << " us/collective\n";
    };
    timed("decode", 1, 128);
    timed("prefill", kTokens, 20);
    for (void* pointer : allocations) { CUDA_CHECK(cudaFree(pointer)); }
    return 0;
}

} // namespace

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < kRanks) {
        std::cout << "SKIP: the tp > 2 collective test needs " << kRanks
                  << " CUDA devices, found " << device_count << '\n';
        return 77;
    }

    ninfer::ExecutionContext ec({0, 1, 2, 3});
    ec.comm = ninfer::TpComm::create(ec);
    std::cout << "tp transport: NCCL communicators up for tp=" << ec.tp << '\n';
    const ninfer::ops::PeerEvents events(ec);

    int failures = 0;
    failures += check_allreduce_threaded(ec, kHidden, kTokens, 0, events);
    failures += check_allreduce("allreduce_sum [5120,1] (decode shape)", kHidden, 1, ec, events);
    // Called twice on purpose: the first call in a size class is the one that can be cold.
    failures += check_allreduce("allreduce_sum [5120,48] (prefill chunk, 1st)", kHidden, kTokens, ec,
                                events);
    failures += check_allreduce("allreduce_sum [5120,48] (prefill chunk, 2nd)", kHidden, kTokens, ec,
                                events);
    failures += check_allgather_logits(ec, events);
    failures += check_back_to_back_sequence(ec, events);
    failures += time_collectives(ec, events);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
