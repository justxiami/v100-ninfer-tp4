// Which transport should carry which payload at tp = 4?
//
// Background (see ~/work/ninfer-tp4-m2/changelog.md): NCCL issued from ninfer's single-threaded
// runtime is bit-exact for small payloads (a 10 KiB allreduce, and a 496 KiB allgather -- one
// phase) but SILENTLY produces a partial reduction for a large allreduce (the real prefill shape,
// 5120 x 48 BF16 = 480 KiB, ~20% of the tail wrong, non-deterministically). Creating one thread per
// rank per collective fixes correctness and costs ~4-5x per collective. So the question is whether
// ninfer's own pull protocol -- single-threaded by construction, no NCCL internals, no thread
// churn -- is the better carrier for the large payloads.
//
// This program measures, on the same buffers and the same machine state:
//   * pull  (4-rank mesh: rank r pulls all three peers into private staging slots, one event
//            protocol, then one local combine) -- the tp2 design generalized, single-threaded
//   * NCCL, single-threaded issue
//   * NCCL, one fresh thread per rank per collective
// for three payloads: [5120,1] (10 KiB, decode), [5120,48] (480 KiB, prefill chunk), and the
// vocabulary gather ([1,248320], 496 KiB, one shape only: C == 1).
//
// Correctness first: integer-valued BF16 inputs make the 4-rank sum exactly representable, so every
// transport is compared bit for bit, and a wrong tail shows up as a mismatch rather than as a
// slightly-off float.
#include <nccl.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t status_ = (x);                                                   \
        if (status_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorName(status_));                                 \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

constexpr int kRanks = 4;
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kTokens = 48;
constexpr std::int32_t kVocab = 248320;
constexpr std::int32_t kShardVocab = kVocab / kRanks;

int g_devices[kRanks] = {0, 1, 2, 3};
cudaStream_t g_stream[kRanks] = {};
ncclComm_t g_comm[kRanks] = {};
cudaEvent_t g_ir[kRanks] = {};  // inputs_ready
cudaEvent_t g_pd[kRanks] = {};  // pull_done

// Per-rank work area. `main` is the operand (and the in-place result); `staging[p]` receives peer
// p's operand. Only this rank's stream ever writes its own staging slots (the tp2 ownership rule).
struct RankBuf {
    __nv_bfloat16* main = nullptr;
    __nv_bfloat16* staging[kRanks] = {};
    std::size_t capacity = 0;
};

RankBuf g_buf[kRanks];

float input_value(int rank, std::size_t index) {
    return static_cast<float>((rank + 1) * ((index % 5) + 1));
}

void fill_main(int rank, std::size_t count) {
    std::vector<__nv_bfloat16> host(count);
    for (std::size_t i = 0; i < count; ++i) host[i] = __float2bfloat16(input_value(rank, i));
    CK(cudaSetDevice(g_devices[rank]));
    CK(cudaMemcpy(g_buf[rank].main, host.data(), count * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
}

void sync_all() {
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        CK(cudaStreamSynchronize(g_stream[rank]));
    }
}

// Four BF16 operands accumulated in FP32 with a single round-to-nearest store (the tp2 local
// combine's arithmetic, generalized to four ranks).
__global__ void combine4(__nv_bfloat16* main, const __nv_bfloat16* a, const __nv_bfloat16* b,
                         const __nv_bfloat16* c, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = __bfloat162float(main[i]);
        value += __bfloat162float(a[i]);
        value += __bfloat162float(b[i]);
        value += __bfloat162float(c[i]);
        main[i] = __float2bfloat16_rn(value);
    }
}

// ---- the pull protocol, 4 ranks (faithful to src/ops/common/allreduce.cu's tp2 shape) ----
void pull_allreduce(std::size_t count) {
    const std::size_t bytes = count * sizeof(__nv_bfloat16);
    // Phase A: every rank publishes its operand before any wait observes it.
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        CK(cudaEventRecord(g_ir[rank], g_stream[rank]));
    }
    // Phase B: wait every peer, pull every peer into my own staging, publish pull_done.
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer != rank) { CK(cudaStreamWaitEvent(g_stream[rank], g_ir[peer], 0)); }
        }
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer == rank) { continue; }
            CK(cudaMemcpyAsync(g_buf[rank].staging[peer], g_buf[peer].main, bytes,
                               cudaMemcpyDeviceToDevice, g_stream[rank]));
        }
        CK(cudaEventRecord(g_pd[rank], g_stream[rank]));
    }
    // Phase C: my operand may be overwritten only after every peer finished reading it.
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer != rank) { CK(cudaStreamWaitEvent(g_stream[rank], g_pd[peer], 0)); }
        }
        int peers[kRanks - 1] = {};
        int found = 0;
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer != rank) { peers[found++] = peer; }
        }
        const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
        combine4<<<blocks, 256, 0, g_stream[rank]>>>(
            g_buf[rank].main, g_buf[rank].staging[peers[0]], g_buf[rank].staging[peers[1]],
            g_buf[rank].staging[peers[2]], count);
    }
}

// C == 1 only: one row per rank, so a rank's block is contiguous and NCCL's rank-major order equals
// the [1, vocab] destination order. Same shape the model actually gathers.
void pull_allgather(std::int32_t rows_per_rank, __nv_bfloat16* const destination[]) {
    const std::size_t bytes = static_cast<std::size_t>(rows_per_rank) * sizeof(__nv_bfloat16);
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        CK(cudaEventRecord(g_ir[rank], g_stream[rank]));
    }
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer != rank) { CK(cudaStreamWaitEvent(g_stream[rank], g_ir[peer], 0)); }
        }
        // Own block locally, peers pulled straight into their destination offsets.
        CK(cudaMemcpyAsync(destination[rank] + static_cast<std::size_t>(rank) * rows_per_rank,
                           g_buf[rank].main, bytes, cudaMemcpyDeviceToDevice, g_stream[rank]));
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer == rank) { continue; }
            CK(cudaMemcpyAsync(destination[rank] + static_cast<std::size_t>(peer) * rows_per_rank,
                               g_buf[peer].main, bytes, cudaMemcpyDeviceToDevice,
                               g_stream[rank]));
        }
        CK(cudaEventRecord(g_pd[rank], g_stream[rank]));
    }
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer != rank) { CK(cudaStreamWaitEvent(g_stream[rank], g_pd[peer], 0)); }
        }
    }
}

// ---- NCCL, two issue styles ----
void nccl_allreduce_single(std::size_t count) {
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        ncclAllReduce(g_buf[rank].main, g_buf[rank].main, count, ncclBfloat16, ncclSum, g_comm[rank],
                      g_stream[rank]);
    }
}

void nccl_allreduce_threads(std::size_t count) {
    std::vector<std::thread> threads;
    for (int rank = 0; rank < kRanks; ++rank) {
        threads.emplace_back([rank, count] {
            CK(cudaSetDevice(g_devices[rank]));
            ncclAllReduce(g_buf[rank].main, g_buf[rank].main, count, ncclBfloat16, ncclSum,
                          g_comm[rank], g_stream[rank]);
            CK(cudaStreamSynchronize(g_stream[rank]));
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
}

bool verify_allreduce(std::size_t count, const char* label) {
    std::size_t bad_total = 0;
    for (int rank = 0; rank < kRanks; ++rank) {
        std::vector<__nv_bfloat16> got(count);
        CK(cudaSetDevice(g_devices[rank]));
        CK(cudaMemcpy(got.data(), g_buf[rank].main, count * sizeof(__nv_bfloat16),
                      cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < count; ++i) {
            float expected = 0.0f;
            for (int other = 0; other < kRanks; ++other) { expected += input_value(other, i); }
            const __nv_bfloat16 want = __float2bfloat16_rn(expected);
            if (std::memcmp(&got[i], &want, sizeof(want)) != 0) { ++bad_total; }
        }
    }
    if (bad_total != 0) {
        std::cout << "    " << label << ": MISMATCH (" << bad_total << " elements over 4 ranks)\n";
    }
    return bad_total == 0;
}

double time_it(int repeats, const std::function<void()>& body) {
    body();
    sync_all();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) { body(); }
    sync_all();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
               .count() /
           repeats;
}

} // namespace

int main() {
    int device_count = 0;
    CK(cudaGetDeviceCount(&device_count));
    if (device_count < kRanks) {
        std::cout << "SKIP: needs " << kRanks << " devices, found " << device_count << '\n';
        return 77;
    }

    const std::size_t kCapacity = static_cast<std::size_t>(kHidden) * kTokens;  // 245760 elements
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        CK(cudaStreamCreateWithFlags(&g_stream[rank], cudaStreamNonBlocking));
        CK(cudaMalloc(&g_buf[rank].main, kCapacity * sizeof(__nv_bfloat16)));
        for (int peer = 0; peer < kRanks; ++peer) {
            if (peer != rank) {
                CK(cudaMalloc(&g_buf[rank].staging[peer], kCapacity * sizeof(__nv_bfloat16)));
            }
        }
        g_buf[rank].capacity = kCapacity;
        CK(cudaEventCreateWithFlags(&g_ir[rank], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&g_pd[rank], cudaEventDisableTiming));
    }
    // NCCL comms + the warmup collective from one thread per rank (the first collective's channel
    // handshake is a host rendezvous; see src/core/tp_comm.h in the ninfer tree).
    if (ncclCommInitAll(g_comm, kRanks, g_devices) != ncclSuccess) { return 1; }
    {
        std::vector<void*> warm(kRanks, nullptr);
        for (int rank = 0; rank < kRanks; ++rank) {
            CK(cudaSetDevice(g_devices[rank]));
            CK(cudaMalloc(&warm[rank], 1u << 20));
        }
        std::vector<std::thread> threads;
        for (int rank = 0; rank < kRanks; ++rank) {
            threads.emplace_back([rank, &warm] {
                CK(cudaSetDevice(g_devices[rank]));
                ncclAllReduce(warm[rank], warm[rank], (1u << 20) / 4, ncclInt32, ncclSum,
                              g_comm[rank], g_stream[rank]);
                CK(cudaStreamSynchronize(g_stream[rank]));
            });
        }
        for (std::thread& thread : threads) { thread.join(); }
        for (int rank = 0; rank < kRanks; ++rank) {
            CK(cudaSetDevice(g_devices[rank]));
            CK(cudaFree(warm[rank]));
        }
    }

    const std::size_t count = static_cast<std::size_t>(kHidden) * kTokens;
    std::cout << "=== allreduce, payload sweep (4 ranks, bit-exactness + us/collective) ===\n";
    const std::int32_t column_counts[] = {1, kTokens};
    for (std::int32_t columns : column_counts) {
        const std::size_t count = static_cast<std::size_t>(kHidden) * columns;
        std::cout << "  payload [" << kHidden << "," << columns << "] = " << (count * 2) / 1024
                  << " KiB\n";
        for (int rank = 0; rank < kRanks; ++rank) { fill_main(rank, count); }
        sync_all();
        pull_allreduce(count);
        sync_all();
        const bool pull_ok = verify_allreduce(count, "pull");
        const double pull_us = time_it(kRanks == 1 ? 1 : (columns == 1 ? 128 : 30), [&] {
            pull_allreduce(count);
        });
        std::cout << "    pull (single-threaded mesh)     : " << (pull_ok ? "bit-exact" : "BROKEN")
                  << "   " << pull_us << " us\n";

        for (int rank = 0; rank < kRanks; ++rank) { fill_main(rank, count); }
        sync_all();
        nccl_allreduce_single(count);
        sync_all();
        const bool nccl1_ok = verify_allreduce(count, "nccl single");
        const double nccl1_us = time_it(columns == 1 ? 128 : 30, [&] { nccl_allreduce_single(count); });
        std::cout << "    NCCL (single-threaded issue)    : " << (nccl1_ok ? "bit-exact" : "BROKEN")
                  << "   " << nccl1_us << " us\n";

        for (int rank = 0; rank < kRanks; ++rank) { fill_main(rank, count); }
        sync_all();
        nccl_allreduce_threads(count);
        sync_all();
        const bool ncclN_ok = verify_allreduce(count, "nccl threads");
        const double ncclN_us =
            time_it(columns == 1 ? 128 : 30, [&] { nccl_allreduce_threads(count); });
        std::cout << "    NCCL (fresh thread per rank)    : " << (ncclN_ok ? "bit-exact" : "BROKEN")
                  << "   " << ncclN_us << " us\n";
    }

    std::cout << "=== allgather [1," << kVocab << "] = " << (kVocab * 2) / 1024 << " KiB ===\n";
    std::vector<__nv_bfloat16*> destination(kRanks, nullptr);
    for (int rank = 0; rank < kRanks; ++rank) {
        CK(cudaSetDevice(g_devices[rank]));
        CK(cudaMalloc(&destination[rank], static_cast<std::size_t>(kVocab) * sizeof(__nv_bfloat16)));
    }
    __nv_bfloat16* dest_raw[kRanks] = {};
    for (int rank = 0; rank < kRanks; ++rank) { dest_raw[rank] = destination[rank]; }
    for (int rank = 0; rank < kRanks; ++rank) { fill_main(rank, kShardVocab); }
    sync_all();
    pull_allgather(kShardVocab, dest_raw);
    sync_all();
    bool allgather_ok = true;
    for (int rank = 0; rank < kRanks && allgather_ok; ++rank) {
        std::vector<__nv_bfloat16> got(static_cast<std::size_t>(kVocab));
        CK(cudaSetDevice(g_devices[rank]));
        CK(cudaMemcpy(got.data(), destination[rank], got.size() * sizeof(__nv_bfloat16),
                      cudaMemcpyDeviceToHost));
        for (int source = 0; source < kRanks && allgather_ok; ++source) {
            for (std::int32_t i = 0; i < kShardVocab; ++i) {
                const std::size_t index = static_cast<std::size_t>(source) * kShardVocab +
                                          static_cast<std::size_t>(i);
                if (std::memcmp(&got[index], &got[0], 0) != 0) { /* placeholder */ }
                const __nv_bfloat16 want = __float2bfloat16(input_value(source, i));
                if (std::memcmp(&got[index], &want, sizeof(want)) != 0) {
                    allgather_ok = false;
                    break;
                }
            }
        }
    }
    const double pull_ag_us = time_it(60, [&] { pull_allgather(kShardVocab, dest_raw); });
    std::cout << "    pull (single-threaded mesh)     : " << (allgather_ok ? "bit-exact" : "BROKEN")
              << "   " << pull_ag_us << " us\n";
    // NCCL's own gather, single-threaded, for the same shape (measured bit-exact in the ninfer tree).
    const double nccl_ag_us = time_it(60, [&] {
        for (int rank = 0; rank < kRanks; ++rank) {
            CK(cudaSetDevice(g_devices[rank]));
            ncclAllGather(g_buf[rank].main, destination[rank], kShardVocab, ncclBfloat16,
                          g_comm[rank], g_stream[rank]);
        }
    });
    std::cout << "    NCCL (single-threaded issue)    : (correctness covered by the ninfer test)   "
              << nccl_ag_us << " us\n";
    // --- Which condition makes NCCL's single-threaded issue lose the tail? The ninfer-tree test
    // re-allocates its buffers per check (so the base address moves every time) and failed; this
    // program keeps one allocation per rank and passed. Sweep the base offset and verify EVERY
    // call, so the answer is a failure RATE rather than one anecdote.
    std::cout << "=== NCCL single-threaded, 480 KiB, per-call verification vs base offset ===\n";
    for (int offset_elements : {0, 1, 2, 4, 8, 16, 64, 256}) {
        std::vector<void*> bases(kRanks, nullptr);
        for (int rank = 0; rank < kRanks; ++rank) {
            CK(cudaSetDevice(g_devices[rank]));
            CK(cudaMalloc(&bases[rank],
                          (count + 512) * sizeof(__nv_bfloat16)));
            g_buf[rank].main = reinterpret_cast<__nv_bfloat16*>(bases[rank]) + offset_elements;
        }
        int failures = 0;
        const int rounds = 12;
        for (int round = 0; round < rounds; ++round) {
            for (int rank = 0; rank < kRanks; ++rank) { fill_main(rank, count); }
            sync_all();
            nccl_allreduce_single(count);
            sync_all();
            if (!verify_allreduce(count, "nccl single")) { ++failures; }
        }
        std::cout << "  base offset " << offset_elements << " elements (" << offset_elements * 2
                  << " B): " << failures << " / " << rounds << " calls wrong\n";
        for (int rank = 0; rank < kRanks; ++rank) {
            CK(cudaSetDevice(g_devices[rank]));
            CK(cudaFree(bases[rank]));
        }
    }
    // Last mechanism candidate: reusing one address region for a DIFFERENT payload size (which is
    // what the ninfer-tree test does -- it allocates/frees per check, and the 10 KiB case runs
    // before the 480 KiB one), versus keeping one buffer per size. If the churn is the trigger,
    // the failure rate below will separate the two.
    std::cout << "=== NCCL single-threaded: alloc/free churn (alternating 10 KiB / 480 KiB) ===\n";
    {
        int failures = 0;
        const int rounds = 12;
        for (int round = 0; round < rounds; ++round) {
            for (int which = 0; which < 2; ++which) {
                const std::size_t elements =
                    which == 0 ? static_cast<std::size_t>(kHidden)
                               : static_cast<std::size_t>(kHidden) * kTokens;
                std::vector<void*> fresh(kRanks, nullptr);
                for (int rank = 0; rank < kRanks; ++rank) {
                    CK(cudaSetDevice(g_devices[rank]));
                    CK(cudaMalloc(&fresh[rank], elements * sizeof(__nv_bfloat16)));
                    g_buf[rank].main = static_cast<__nv_bfloat16*>(fresh[rank]);
                }
                for (int rank = 0; rank < kRanks; ++rank) { fill_main(rank, elements); }
                sync_all();
                nccl_allreduce_single(elements);
                sync_all();
                if (!verify_allreduce(elements, "churn")) { ++failures; }
                for (int rank = 0; rank < kRanks; ++rank) {
                    CK(cudaSetDevice(g_devices[rank]));
                    CK(cudaFree(fresh[rank]));
                }
            }
        }
        std::cout << "  churned allocations: " << failures << " / " << (rounds * 2)
                  << " calls wrong\n";
    }
    // Hypothesis that fits every observation so far: the 4 ranks' launches must reach the device
    // within some skew, and a slow/jittery host (huge binary, per-call allocations, diag loops, a
    // descheduled pool thread) exceeds it. Inject a controlled delay between rank launches.
    std::cout << "=== NCCL single-threaded, 480 KiB, injected launch skew between ranks ===\n";
    for (int skew_us : {0, 20, 50, 100, 200, 500}) {
        int failures = 0;
        const int rounds = 8;
        for (int round = 0; round < rounds; ++round) {
            for (int rank = 0; rank < kRanks; ++rank) { fill_main(rank, count); }
            sync_all();
            for (int rank = 0; rank < kRanks; ++rank) {
                CK(cudaSetDevice(g_devices[rank]));
                ncclAllReduce(g_buf[rank].main, g_buf[rank].main, count, ncclBfloat16, ncclSum,
                              g_comm[rank], g_stream[rank]);
                if (skew_us > 0 && rank + 1 < kRanks) {
                    std::this_thread::sleep_for(std::chrono::microseconds(skew_us));
                }
            }
            sync_all();
            if (!verify_allreduce(count, "skew")) { ++failures; }
        }
        std::cout << "  skew " << skew_us << " us between rank launches: " << failures << " / "
                  << rounds << " calls wrong\n";
    }
    return 0;
}
