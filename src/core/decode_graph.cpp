// Implements: src/core/decode_graph.h
//
// THE MULTI-DEVICE CONSTRAINTS THAT ACTUALLY BIND, and what each one forces.
// Measured on CUDA 13.1 / driver 580.178.04 by tools/tp2/capture_probe.cu and
// tools/tp2/transport_probe.cu; re-run both on any new two-GPU topology.
//
//  1. TWO LIVE CAPTURES CANNOT BE LINKED. cudaStreamWaitEvent on an event recorded inside a
//     *different* live capture fails with cudaErrorStreamCaptureMerge. Since a tensor-parallel
//     decode program's only cross-device ordering IS such an event edge, "one graph per device"
//     is not expressible. Forces: ONE graph holding both devices' nodes, with the peer's stream
//     forked into the origin's capture (DecodeGraphPeerBridge).
//
//  2. MEMCPY NODES MUST NAME THEIR MEMORY BY UVA POINTER, not by device id. cudaMemcpyPeerAsync
//     -- the explicit (ptr, device, ptr, device) form -- is rejected inside a capture region with
//     cudaErrorStreamCaptureUnsupported. cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA
//     pointers expresses the same cross-device transfer and IS captured, because a UVA pointer
//     already resolves to its device and the node needs no separate context argument. Forces:
//     src/ops/common/allreduce.cu pulls with the UVA form (see its pull_peer()).
//
//  3. cudaGraphExecUpdate REQUIRES THE SAME TOPOLOGY, INCLUDING NODE DEVICE RESIDENCY. Swapping
//     an installed profile into an existing executable only works while every node keeps the
//     device and context it was captured on. That holds here because all profiles of a family
//     are captured from the same body against the same two DeviceContexts; it is why the
//     profile-boundary swap is exercised deliberately in tests/targets/qwen3_6_27b/
//     test_graph_tp2.cpp rather than left to chance.
//
//  4. NO DEVICE-SIDE LAUNCH. cudaGraphInstantiate is deliberately called with flags 0, NOT
//     cudaGraphInstantiateFlagDeviceLaunch: device-launchable graphs must be single-device, so
//     requesting that flag would reject exactly the cross-device graph this file exists to build.
//
//  5. EVENT RECORDS AND WAITS BECOME EDGES, NOT NODES. A captured graph therefore holds only real
//     device work; the collectives' four-event choreography costs zero nodes. This is what makes
//     the node count a usable measurement of "did both devices' work get captured".
#include "core/decode_graph.h"

#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

// Current-device save/restore. Enrolling and retiring the peer stream requires that device to be
// current for the record and this one for the wait, and the caller's device must survive.
class ScopedDevice {
public:
    ScopedDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~ScopedDevice() { log_cuda_error("cudaSetDevice(restore)", cudaSetDevice(previous_)); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

// Forks every peer stream into the capture that `stream` (the origin) has already begun.
void fork_peer(cudaStream_t stream, const DecodeGraphPeerCapture& peer) {
    const ScopedDevice scope;
    ScopedDevice::set(peer.bridge->origin_device());
    for (int rank = 1; rank < peer.bridge->rank_count(); ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer.streams[slot] == nullptr) { continue; }
        CUDA_CHECK(cudaEventRecord(peer.bridge->fork_event(rank), stream));
    }
    for (int rank = 1; rank < peer.bridge->rank_count(); ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer.streams[slot] == nullptr) { continue; }
        ScopedDevice::set(peer.bridge->peer_device(rank));
        CUDA_CHECK(cudaStreamWaitEvent(peer.streams[slot], peer.bridge->fork_event(rank), 0));
    }
    ScopedDevice::set(peer.bridge->origin_device());
}

// Joins every peer stream back into the origin. Without this cudaStreamEndCapture reports
// cudaErrorStreamCaptureUnjoined and the whole capture is discarded.
void join_peer(cudaStream_t stream, const DecodeGraphPeerCapture& peer) {
    const ScopedDevice scope;
    for (int rank = 1; rank < peer.bridge->rank_count(); ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer.streams[slot] == nullptr) { continue; }
        ScopedDevice::set(peer.bridge->peer_device(rank));
        CUDA_CHECK(cudaEventRecord(peer.bridge->join_event(rank), peer.streams[slot]));
    }
    ScopedDevice::set(peer.bridge->origin_device());
    for (int rank = 1; rank < peer.bridge->rank_count(); ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer.streams[slot] == nullptr) { continue; }
        CUDA_CHECK(cudaStreamWaitEvent(stream, peer.bridge->join_event(rank), 0));
    }
}

void discard_capture(cudaStream_t stream, const DecodeGraphPeerCapture* peer,
                     bool peer_forked) noexcept {
    // Best effort: rejoin the peer streams so the origin's EndCapture is well formed. If the
    // capture was already invalidated these calls fail harmlessly and EndCapture then returns a
    // null graph and clears ALL streams' capture state, which is the outcome that matters.
    // Hand-rolled rather than ScopedDevice because this runs on an exception path and must not
    // throw.
    if (peer != nullptr && peer_forked) {
        log_cuda_error("cudaSetDevice(origin)", cudaSetDevice(peer->bridge->origin_device()));
        for (int rank = 1; rank < peer->bridge->rank_count(); ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (peer->streams[slot] == nullptr) { continue; }
            log_cuda_error("cudaSetDevice(peer)",
                           cudaSetDevice(peer->bridge->peer_device(rank)));
            log_cuda_error("cudaEventRecord(join)",
                           cudaEventRecord(peer->bridge->join_event(rank), peer->streams[slot]));
        }
        log_cuda_error("cudaSetDevice(origin)", cudaSetDevice(peer->bridge->origin_device()));
        for (int rank = 1; rank < peer->bridge->rank_count(); ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (peer->streams[slot] == nullptr) { continue; }
            log_cuda_error("cudaStreamWaitEvent(join)",
                           cudaStreamWaitEvent(stream, peer->bridge->join_event(rank), 0));
        }
    }
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
    if (peer != nullptr && peer_forked) {
        // A stream left in capture mode would poison every later launch on it, so say so loudly
        // rather than failing mysteriously later.
        for (int rank = 1; rank < peer->bridge->rank_count(); ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (peer->streams[slot] == nullptr) { continue; }
            cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(peer->streams[slot], &status) == cudaSuccess &&
                status != cudaStreamCaptureStatusNone) {
                std::fprintf(stderr,
                             "CUDA cleanup failed: peer stream is still capturing after a "
                             "discarded multi-device capture\n");
            }
        }
    }
}

} // namespace

DecodeGraphPeerBridge::DecodeGraphPeerBridge(int origin_device,
                                             std::span<const int> peer_devices)
    : origin_device_(origin_device) {
    if (peer_devices.size() < 2) {
        throw std::invalid_argument(
            "DecodeGraphPeerBridge requires an origin and at least one peer rank");
    }
    rank_count_ = static_cast<int>(peer_devices.size());
    for (int rank = 1; rank < rank_count_; ++rank) {
        const int device = peer_devices[static_cast<std::size_t>(rank)];
        if (device == origin_device) {
            throw std::invalid_argument(
                "DecodeGraphPeerBridge requires distinct devices; a capture cannot fork a stream "
                "into itself");
        }
        peer_devices_[static_cast<std::size_t>(rank)] = device;
    }
    // fork_event(r) is recorded on the origin; join_event(r) and gate_ are on rank r's device.
    const ScopedDevice scope;
    int created = 0;
    const auto create = [&](int device, cudaEvent_t* destination) {
        cudaError_t status = cudaSetDevice(device);
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(destination, cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("DecodeGraphPeerBridge: event creation failed: ") +
                cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
        ++created;
    };
    try {
        for (int rank = 1; rank < rank_count_; ++rank) {
            create(origin_device, &forks_[static_cast<std::size_t>(rank)]);
        }
        for (int rank = 1; rank < rank_count_; ++rank) {
            create(peer_devices_[static_cast<std::size_t>(rank)],
                   &joins_[static_cast<std::size_t>(rank)]);
        }
        for (int rank = 1; rank < rank_count_; ++rank) {
            create(peer_devices_[static_cast<std::size_t>(rank)],
                   &gates_[static_cast<std::size_t>(rank)]);
        }
    } catch (...) {
        // The destructor is not reached when the constructor throws, so release here.
        for (int rank = 1; rank < rank_count_; ++rank) {
            destroy_event(forks_[static_cast<std::size_t>(rank)]);
            destroy_event(joins_[static_cast<std::size_t>(rank)]);
            destroy_event(gates_[static_cast<std::size_t>(rank)]);
        }
        throw;
    }
    (void)created;
}

DecodeGraphPeerBridge::~DecodeGraphPeerBridge() {
    for (int rank = 1; rank < rank_count_; ++rank) {
        destroy_event(forks_[static_cast<std::size_t>(rank)]);
        destroy_event(joins_[static_cast<std::size_t>(rank)]);
        destroy_event(gates_[static_cast<std::size_t>(rank)]);
    }
}

bool DecodeGraphPeerBridge::live() const noexcept {
    if (rank_count_ < 2) { return false; }
    for (int rank = 1; rank < rank_count_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (forks_[slot] == nullptr || joins_[slot] == nullptr || gates_[slot] == nullptr) {
            return false;
        }
    }
    return true;
}

DecodeGraphPeerBridge::DecodeGraphPeerBridge(DecodeGraphPeerBridge&& other) noexcept
    : origin_device_(other.origin_device_), rank_count_(other.rank_count_),
      peer_devices_(other.peer_devices_), forks_(other.forks_), joins_(other.joins_),
      gates_(other.gates_) {
    other.rank_count_ = 1;
    other.forks_      = {};
    other.joins_      = {};
    other.gates_      = {};
}

DecodeGraphPeerBridge& DecodeGraphPeerBridge::operator=(DecodeGraphPeerBridge&& other) noexcept {
    if (this == &other) { return *this; }
    for (int rank = 1; rank < rank_count_; ++rank) {
        destroy_event(forks_[static_cast<std::size_t>(rank)]);
        destroy_event(joins_[static_cast<std::size_t>(rank)]);
        destroy_event(gates_[static_cast<std::size_t>(rank)]);
    }
    origin_device_ = other.origin_device_;
    rank_count_    = other.rank_count_;
    peer_devices_  = other.peer_devices_;
    forks_         = other.forks_;
    joins_         = other.joins_;
    gates_         = other.gates_;
    other.rank_count_ = 1;
    other.forks_      = {};
    other.joins_      = {};
    other.gates_      = {};
    return *this;
}

void DecodeGraphPeerBridge::gate_launch(const TpArray<cudaStream_t>& peer_streams,
                                        cudaStream_t origin_stream) const {
    if (!live()) {
        throw std::logic_error("a moved-from DecodeGraphPeerBridge cannot gate a graph launch");
    }
    const ScopedDevice scope;
    for (int rank = 1; rank < rank_count_; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer_streams[slot] == nullptr) { continue; }
        ScopedDevice::set(peer_device(rank));
        CUDA_CHECK(cudaEventRecord(gates_[slot], peer_streams[slot]));
        ScopedDevice::set(origin_device_);
        CUDA_CHECK(cudaStreamWaitEvent(origin_stream, gates_[slot], 0));
    }
}

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    capture(stream, body, DecodeGraphPeerCapture{});
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body,
                                    const DecodeGraphPeerCapture& peer) {
    const bool dual = peer.bridge != nullptr;
    if (dual) {
        if (!peer.bridge->live()) {
            throw std::invalid_argument("multi-device capture requires a live peer bridge");
        }
        for (int rank = 1; rank < peer.bridge->rank_count(); ++rank) {
            if (peer.streams[static_cast<std::size_t>(rank)] == nullptr) {
                throw std::invalid_argument(
                    "multi-device capture requires every peer rank's stream");
            }
        }
    }
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    bool peer_forked = false;
    try {
        if (dual) {
            fork_peer(stream, peer);
            peer_forked = true;
        }
        body();
        if (dual) { join_peer(stream, peer); }
    } catch (...) {
        discard_capture(stream, dual ? &peer : nullptr, peer_forked);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

std::size_t DecodeGraphDefinition::node_count() const {
    if (graph_ == nullptr) { return 0; }
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes));
    return nodes;
}

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        CUDA_CHECK(err);
    }
    exec_ = exec;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace ninfer
