// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// Both collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "core/tp_comm.h"           // tp > 2 collectives (NCCL)
#include "ops/common/split_launch.h" // detail::require_split_context
#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>
#include <vector>
#include <stdexcept>
#include <string>
#include "ninfer/types.h" // TpArray, kMaximumDevices

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04, Task
// 4.2's capture probe), which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void startup_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("peer transport startup: ") + operation + ": " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

// Linux CUDA PCIe P2P is unsupported behind a translated IOMMU domain. A small
// allocation can nevertheless pass a copy probe while other mappings silently
// lose writes, so the domain restriction takes precedence over that probe.
std::string translated_iommu_domain(const ExecutionContext& ec) {
    std::string reason;
    // Diagnostic escape hatches, both OFF by default; same pair as the v3 fork carries.
    // The verdict below is conservative on purpose -- it refuses direct peer access on a
    // translated IOMMU domain, which also refuses the NVLink transport for the whole tp2 run.
    // NINFER_FORCE_DIRECT_P2P=1 skips the domain verdict so the runtime transfer probe decides;
    // NINFER_FORCE_HOST_STAGED=1 forces the pinned host-staged path back.
    if (std::getenv("NINFER_FORCE_HOST_STAGED") != nullptr) {
        return std::string("NINFER_FORCE_HOST_STAGED is set");
    }
    if (std::getenv("NINFER_FORCE_DIRECT_P2P") != nullptr) { return reason; }
#if defined(__linux__)
    for (int rank = 0; rank < ec.tp; ++rank) {
        char pci_bus_id[32]{};
        startup_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ec.dev[rank]->device),
                      "cudaDeviceGetPCIBusId");
        // CUDA may emit uppercase hexadecimal; Linux PCI sysfs names are lowercase.
        for (char& c : pci_bus_id) {
            if (c >= 'A' && c <= 'F') { c += 'a' - 'A'; }
        }
        std::ifstream domain_file(std::string("/sys/bus/pci/devices/") + pci_bus_id +
                                  "/iommu_group/type");
        std::string domain;
        domain_file >> domain;
        if (domain == "DMA" || domain == "DMA-FQ") {
            if (!reason.empty()) { reason += "; "; }
            reason += std::string("PCI ") + pci_bus_id + " uses translated IOMMU domain " + domain;
        }
    }
#else
    (void)ec;
#endif
    return reason;
}

// Startup-only storage. The exact payload catches drivers that advertise peer access
// but silently drop DMA writes (observed with V100s behind an IOMMU). Use the same
// pull API and destination compute stream as the collectives, without GPU peer loads.
class PeerTransferProbe {
public:
    explicit PeerTransferProbe(const ExecutionContext& ec) : ec_(ec) {
        startup_check(cudaGetDevice(&previous_), "cudaGetDevice");
    }

    ~PeerTransferProbe() {
        for (int rank = 0; rank < ec_.tp; ++rank) {
            cleanup(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
            if (source_[rank] != nullptr) { cleanup(cudaFree(source_[rank]), "cudaFree source"); }
            if (destination_[rank] != nullptr) {
                cleanup(cudaFree(destination_[rank]), "cudaFree destination");
            }
        }
        cleanup(cudaSetDevice(previous_), "restore device");
    }

    PeerTransferProbe(const PeerTransferProbe&) = delete;
    PeerTransferProbe& operator=(const PeerTransferProbe&) = delete;

    void set_device(int rank) const {
        startup_check(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
    }

    void initialize() {
        for (int rank = 0; rank < ec_.tp; ++rank) {
            set_device(rank);
            startup_check(cudaMalloc(&source_[rank], kBytes), "cudaMalloc source");
            startup_check(cudaMalloc(&destination_[rank], kBytes), "cudaMalloc destination");
            std::vector<std::uint32_t> values(kWords);  // heap: 1 MiB of payload, not a stack array
            for (std::size_t i = 0; i < kWords; ++i) { values[i] = pattern(rank, i); }
            startup_check(cudaMemcpyAsync(source_[rank], values.data(), kBytes,
                                           cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                          "initialize source");
            startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream), "retire source");
        }
    }

    // Empty means both complete copies matched their independent host patterns exactly.
    std::string qualify() {
        std::string mismatch;
        for (int rank = 0; rank < ec_.tp; ++rank) {
            set_device(rank);
            cudaStream_t stream = ec_.dev[rank]->stream;
            startup_check(cudaMemsetAsync(destination_[rank], 0xcd, kBytes, stream),
                          "clear destination");
            startup_check(pull_peer(destination_[rank], source_[1 - rank], kBytes, stream),
                          "cross-device copy");
            startup_check(cudaStreamSynchronize(stream), "retire cross-device copy");
            std::vector<std::uint32_t> actual(kWords);  // heap, matches initialize()
            startup_check(cudaMemcpy(actual.data(), destination_[rank], kBytes,
                                      cudaMemcpyDeviceToHost), "read destination");
            for (std::size_t i = 0; i < kWords; ++i) {
                if (actual[i] != pattern(1 - rank, i)) {
                    if (mismatch.empty()) {
                        mismatch = "device " + std::to_string(ec_.dev[1 - rank]->device) +
                                   " -> " + std::to_string(ec_.dev[rank]->device) +
                                   " data mismatch at word " + std::to_string(i);
                    }
                    break;
                }
            }
        }
        return mismatch;
    }

    void disable_peer_access() const {
        for (int rank = 0; rank < ec_.tp; ++rank) {
            set_device(rank);
            const cudaError_t status = cudaDeviceDisablePeerAccess(ec_.dev[1 - rank]->device);
            if (status == cudaErrorPeerAccessNotEnabled) {
                (void)cudaGetLastError();
            } else {
                startup_check(status, "cudaDeviceDisablePeerAccess");
            }
        }
    }

public:
    // Probe payload size. Reported when direct P2P is qualified, so an operator can tell from the
    // log which transport the process actually chose.
    static constexpr std::size_t probe_bytes() noexcept { return kBytes; }

private:
    // 1 MiB, not 16 KiB. The failure this probe exists to catch is a driver silently dropping peer
    // writes (observed with V100s behind an IOMMU), and a payload small enough to live in one
    // mapping cannot speak for the multi-megabyte allocations the collectives actually move.
    // pattern() is a bijection over the index space (multiplier 65537 is odd), so every word of the
    // 1 MiB still carries a distinct expected value.
    static constexpr std::size_t kWords = 262144;  // * sizeof(uint32_t) = 1 MiB
    static constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

    static std::uint32_t pattern(int rank, std::size_t index) {
        return 0x4f000000U ^ (std::uint32_t(rank) << 20U) ^
               (static_cast<std::uint32_t>(index) * 65537U);
    }

    static void cleanup(cudaError_t status, const char* operation) noexcept {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during peer probe %s: %s: %s\n",
                         operation, cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    const ExecutionContext& ec_;
    int previous_ = 0;
    TpArray<void*> source_{};
    TpArray<void*> destination_{};
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }

    PeerTransferProbe probe(ec);
    try {
        std::string direct_failure = translated_iommu_domain(ec);
        int forward = 0;
        int reverse = 0;
        if (direct_failure.empty()) {
            startup_check(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]),
                          "cudaDeviceCanAccessPeer forward");
            startup_check(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]),
                          "cudaDeviceCanAccessPeer reverse");
        }
        const bool supported = direct_failure.empty() && forward != 0 && reverse != 0;
        if (supported) {
            for (int rank = 0; rank < ec.tp; ++rank) {
                probe.set_device(rank);
                const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                } else {
                    startup_check(status, "cudaDeviceEnablePeerAccess");
                }
            }
        } else {
            if (direct_failure.empty()) { direct_failure = "peer access unavailable"; }
            probe.disable_peer_access();
        }
        probe.initialize();
        if (supported) {
            direct_failure = probe.qualify();
            if (direct_failure.empty()) {
                std::fprintf(stderr,
                             "[ninfer] direct P2P enabled (peer access qualified, %zu KiB/rank "
                             "probe)\n",
                             PeerTransferProbe::probe_bytes() >> 10);
                return true;
            }
            probe.disable_peer_access();
        }

        // With peer access disabled (or prohibited by the IOMMU domain), the same UVA D2D
        // operation is handled by CUDA's driver-managed staging path. Qualify that exact API
        // rather than maintaining a second explicit D2H/H2D transport in every collective.
        const std::string fallback_failure = probe.qualify();
        if (!fallback_failure.empty()) {
            throw std::runtime_error("peer transport startup: UVA D2D fallback validation failed: " +
                                     fallback_failure);
        }
        std::fprintf(stderr,
                     "[ninfer] direct P2P disabled (%s); using verified CUDA UVA D2D staging\n",
                     direct_failure.c_str());
        return false;
    } catch (...) {
        // Failed startup must not leave a partially enabled pair behind. Clear the
        // runtime's last error before cleanup; a fatal context error still prevents
        // further use, and the original startup exception remains authoritative.
        (void)cudaGetLastError();
        try { probe.disable_peer_access(); } catch (...) {}
        throw;
    }
}

PeerEvents::PeerEvents(const ExecutionContext& ec) {
    detail::require_split_context(
        ec, "PeerEvents: requires at least two distinct devices with one context per rank");
    const CurrentDeviceGuard guard;
    const int tp = ec.tp;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    std::array<cudaEvent_t, static_cast<std::size_t>(2) * kMaximumDevices> created{};
    for (int slot = 0; slot < tp * 2; ++slot) {
        const int rank             = slot / 2;
        const cudaError_t creation = cudaSetDevice(ec.dev[static_cast<std::size_t>(rank)]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[static_cast<std::size_t>(slot)],
                                              cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) {
                cudaEventDestroy(created[static_cast<std::size_t>(done)]);
            }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    tp_ = tp;
    for (int rank = 0; rank < tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        inputs_ready_[slot]    = created[2 * slot];
        pull_done_[slot]       = created[2 * slot + 1];
    }
    // Seed pull_done with completed events so the first collective may use the same inter-call
    // lifetime edge as every later one.
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[static_cast<std::size_t>(rank)]->device);
        CUDA_CHECK(cudaEventRecord(pull_done_[static_cast<std::size_t>(rank)],
                                   ec.dev[static_cast<std::size_t>(rank)]->stream));
    }
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[static_cast<std::size_t>(rank)]->device);
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[static_cast<std::size_t>(rank)]->stream));
    }
}

PeerEvents::~PeerEvents() {
    for (TpArray<cudaEvent_t>* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_), tp_(other.tp_) {
    other.inputs_ready_ = {};
    other.pull_done_    = {};
    other.tp_           = 0;
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    return *this;
}

void allreduce_sum(const TpArray<Tensor>& buffer, const TpArray<Tensor>& staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    // Two transports, one contract (see src/core/tp_comm.h): tp == 2 uses the hand-written pull
    // below, tp > 2 hands the same in-place sum to NCCL. `staging` is the pull protocol's scratch
    // and NCCL ignores it.
    if (ec.tp > 2) {
        ops::nccl_allreduce_sum(ec, buffer, staging);
        return;
    }
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < ec.tp; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");

    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }

#ifndef NDEBUG
    for (int rank = 0; rank < ec.tp; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < ec.tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each rank pulls the peer operand through CUDA's UVA device-to-device path. On
    // translated IOMMU domains the driver transparently stages this copy through host memory;
    // keeping it as one captured D2D node avoids the extra D2H/H2D event chain and is materially
    // faster on the V100 PCIe bridge.
    for (int rank = 0; rank < ec.tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < ec.tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allgather_rows(const TpArray<Tensor>& destination, const TpArray<Tensor>& part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    if (ec.tp > 2) {
        ops::nccl_allgather_rows(ec, destination, part);
        return;
    }
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < ec.tp; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};

#ifndef NDEBUG
    for (int rank = 0; rank < ec.tp; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < ec.tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls the peer block, both on its stream.
    for (int rank = 0; rank < ec.tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                             part[1 - rank].data, block[1 - rank], local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < ec.tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

void gather_columns_rank0(const Tensor& destination, const TpArray<Tensor>& part,
                          const ExecutionContext& ec, const PeerEvents& events) {
    // Rank0-centric by construction: the DFlash selector reads its proposal logits on rank 0 only.
    // Generalized from the two-rank pull: rank 0 records inputs_ready(0), every non-zero rank
    // records its own, then rank 0 waits on each and memcpy2D-asyncs its part into the
    // concatenated destination. Every copy stays a single captured node (capture-safe).
    const DType dtype             = destination.dtype;
    const std::int32_t full_width = destination.ne[0];
    const std::int32_t columns    = destination.ne[1];
    require(full_width > 0 && columns > 0,
            "gather_columns_rank0: dimensions must be positive");
    require(destination.data != nullptr && destination.is_contiguous(),
            "gather_columns_rank0: destination must be contiguous and non-null");
    require(destination.ne[2] == 1 && destination.ne[3] == 1,
            "gather_columns_rank0: destination must be two-dimensional [C,T]");
    for (int rank = 0; rank < ec.tp; ++rank) {
        require(part[rank].dtype == dtype && part[rank].data != nullptr &&
                    part[rank].is_contiguous(),
                "gather_columns_rank0: parts must share dtype and be contiguous/non-null");
        require(part[rank].ne[1] == columns && part[rank].ne[0] > 0 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "gather_columns_rank0: parts must be two-dimensional [C_r,T]");
    }
    std::int32_t width_sum = 0;
    for (int rank = 0; rank < ec.tp; ++rank) { width_sum += part[rank].ne[0]; }
    require(width_sum == full_width,
            "gather_columns_rank0: owned widths must sum to destination width");
    require(events.live(), "gather_columns_rank0: events must be live");

    const std::size_t element_bytes = dtype_size(dtype);
    const std::size_t destination_pitch =
        static_cast<std::size_t>(full_width) * element_bytes;

#ifndef NDEBUG
    require_resident_on(destination.data, ec.dev[0]->device,
                        "gather_columns_rank0: destination must be resident on rank 0");
    for (int rank = 0; rank < ec.tp; ++rank) {
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "gather_columns_rank0: part must be resident on its rank");
        require_disjoint(destination.data, destination.bytes(), part[rank].data, part[rank].bytes(),
                         "gather_columns_rank0: part must not overlap destination");
    }
#endif

    const CurrentDeviceGuard guard;
    const DeviceContext& rank0 = *ec.dev[0];
    // Publish every rank's operand before any wait observes it.
    for (int rank = 0; rank < ec.tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[static_cast<std::size_t>(rank)]->device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank),
                                   ec.dev[static_cast<std::size_t>(rank)]->stream));
    }

    CurrentDeviceGuard::set(rank0.device);
    std::size_t offset = 0;
    for (int rank = 0; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        const std::size_t block =
            static_cast<std::size_t>(part[slot].ne[0]) * element_bytes;
        CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(rank), 0));
        CUDA_CHECK(cudaMemcpy2DAsync(byte_offset(destination.data, offset), destination_pitch,
                                     part[slot].data, block, block,
                                     static_cast<std::size_t>(columns),
                                     cudaMemcpyDeviceToDevice, rank0.stream));
        offset += block;
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(0), rank0.stream));

    // Every peer may overwrite its part only after rank 0 has consumed it. No reciprocal
    // destination write is needed because the selector and all subsequent DFlash logic run on
    // rank 0.
    for (int rank = 1; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[slot]->device);
        CUDA_CHECK(cudaStreamWaitEvent(ec.dev[slot]->stream, events.pull_done(0), 0));
    }
}

void broadcast_rank0(const Tensor& source, const TpArray<Tensor>& destination,
                     const ExecutionContext& ec, const PeerEvents& events) {
    // Rank 0-centric by construction: the DFlash draft tokens and draft-model output hidden
    // are produced on rank 0 alone. Generalized from the two-rank pull to every non-zero rank:
    // one single-copy UVA D2D per peer, all ordered after the same inputs_ready(0) record.
    // Each single copy stays a one-shot node so the whole exchange is capture-safe (a memcpy
    // node is captured into the graph; a host-side loop around it is not, see
    // decode_graph.cpp's capture contract).
    require(events.live(), "broadcast_rank0: events must be live");
    require(source.data != nullptr && source.is_contiguous(),
            "broadcast_rank0: source must be contiguous and non-null");
    const std::size_t bytes = source.bytes();
    if (bytes == 0) { return; }
#ifndef NDEBUG
    require_resident_on(source.data, ec.dev[0]->device,
                        "broadcast_rank0: source is not on rank 0");
#endif
    const CurrentDeviceGuard guard;
    const DeviceContext& origin = *ec.dev[0];
    CurrentDeviceGuard::set(origin.device);
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), origin.stream));
    for (int rank = 1; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        // [10/02 diagnostic] granular report: the merged condition cannot tell a null arena
        // allocation from a dtype/layout mismatch at tp4, where each rank's operand comes from a
        // different source (a fresh arena alloc on the propose path vs. a frame view on the
        // decode path). Remove once the tp4 path is qualified.
        if (destination[slot].data == nullptr || destination[slot].dtype != source.dtype ||
            !destination[slot].is_contiguous()) {
            const auto describe = [](const Tensor& t) {
                return std::string("data=") + (t.data == nullptr ? "null" : "ok") +
                       " dtype=" + std::to_string(static_cast<int>(t.dtype)) + " ne=[" +
                       std::to_string(t.ne[0]) + "," + std::to_string(t.ne[1]) + "," +
                       std::to_string(t.ne[2]) + "," + std::to_string(t.ne[3]) + "] nb=[" +
                       std::to_string(t.nb[0]) + "," + std::to_string(t.nb[1]) + "," +
                       std::to_string(t.nb[2]) + "," + std::to_string(t.nb[3]) +
                       "] contig=" + (t.is_contiguous() ? "1" : "0");
            };
            throw std::invalid_argument("broadcast_rank0: rank " + std::to_string(rank) +
                                        " destination invalid (" + describe(destination[slot]) +
                                        "); source (" + describe(source) + ")");
        }
        for (int d = 0; d < 4; ++d) {
            require(source.ne[d] == destination[slot].ne[d],
                    "broadcast_rank0: shapes must match on every rank");
        }
#ifndef NDEBUG
        require_resident_on(destination[slot].data, ec.dev[slot]->device,
                            "broadcast_rank0: destination is not on its rank");
        require_disjoint(destination[slot].data, bytes, source.data, bytes,
                         "broadcast_rank0: destination must not overlap the source");
#endif
        const DeviceContext& peer = *ec.dev[slot];
        CurrentDeviceGuard::set(peer.device);
        CUDA_CHECK(cudaStreamWaitEvent(peer.stream, events.inputs_ready(0), 0));
        CUDA_CHECK(pull_peer(destination[slot].data, source.data, bytes, peer.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(slot), peer.stream));
        CurrentDeviceGuard::set(origin.device);
        CUDA_CHECK(cudaStreamWaitEvent(origin.stream, events.pull_done(slot), 0));
    }
}

} // namespace ninfer::ops
