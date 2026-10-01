#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <functional>
#include <span>

#include "ninfer/types.h" // TpArray, kMaximumDevices

namespace ninfer {

// The events that enroll every PEER device's stream in a capture that began on the origin device's
// stream, and that order a replay of the result. Tensor-parallel decode issues work on every
// rank's stream and orders them against each other with cross-device event edges; those edges only
// become graph edges if all the streams belong to the SAME capture. Two independent captures
// cannot be linked -- a cudaStreamWaitEvent across two live captures is rejected with
// cudaErrorStreamCaptureMerge -- so a tensor-parallel decode program is ONE graph holding every
// rank's nodes, not one graph per device.
//
// Per peer rank r:
//   record(fork) on the origin stream  ->  wait(fork) on rank r's stream    (rank r joins)
//   ...the whole multi-device decode body...
//   record(join) on rank r's stream    ->  wait(join) on the origin stream  (rank r rejoins)
//
// The joins are mandatory: cudaStreamEndCapture fails with cudaErrorStreamCaptureUnjoined if any
// forked stream is still outstanding. One further event serves gate_launch() below, which is about
// replay rather than capture. All of them are created once (cudaEventCreate is not capturable) and
// one instance serves an unbounded number of sequential captures and launches.
class DecodeGraphPeerBridge {
public:
    // `peer_devices` is INDEXED BY RANK and has at least `origin rank + 1` entries: slot 0 is the
    // origin (ignored; it must equal `origin_device` if supplied), slot r > 0 is rank r's device.
    DecodeGraphPeerBridge(int origin_device, std::span<const int> peer_devices);
    ~DecodeGraphPeerBridge();

    DecodeGraphPeerBridge(const DecodeGraphPeerBridge&)            = delete;
    DecodeGraphPeerBridge& operator=(const DecodeGraphPeerBridge&) = delete;
    DecodeGraphPeerBridge(DecodeGraphPeerBridge&& other) noexcept;
    DecodeGraphPeerBridge& operator=(DecodeGraphPeerBridge&& other) noexcept;

    [[nodiscard]] int origin_device() const noexcept { return origin_device_; }
    [[nodiscard]] int rank_count() const noexcept { return rank_count_; }
    [[nodiscard]] int peer_device(int rank) const noexcept {
        return peer_devices_[static_cast<std::size_t>(rank)];
    }
    [[nodiscard]] cudaEvent_t fork_event(int rank) const noexcept {
        return forks_[static_cast<std::size_t>(rank)];
    }
    [[nodiscard]] cudaEvent_t join_event(int rank) const noexcept {
        return joins_[static_cast<std::size_t>(rank)];
    }
    [[nodiscard]] bool live() const noexcept;

    // REPLAY-side ordering, not capture-side. A multi-device graph is launched on the ORIGIN
    // device's stream; the graph's own edges then order its peer-device nodes after the graph
    // root, but they say nothing about work the caller already enqueued on a peer device's own
    // stream (at decode time: the mirrored KV page materialization). Eager execution gets that
    // ordering for free because it issues the peer's kernels on that same stream; a graph launch
    // does not, so the origin stream is explicitly ordered after EVERY peer stream's outstanding
    // work before each launch, which transitively orders the whole graph after it.
    // `peer_streams` is indexed by rank (slot 0 unused); a null entry is skipped.
    void gate_launch(const TpArray<cudaStream_t>& peer_streams,
                     cudaStream_t origin_stream) const;

private:
    int origin_device_ = 0;
    int rank_count_    = 1;
    TpArray<int> peer_devices_{};
    TpArray<cudaEvent_t> forks_{};
    TpArray<cudaEvent_t> joins_{};
    // One gate event PER PEER, each created on that peer's own device: an event may only be
    // recorded on a stream of the device it was created on, and every gate is recorded on its own
    // rank's stream before the origin waits for it.
    TpArray<cudaEvent_t> gates_{};
};

// The peer half of a multi-device capture: which streams to enroll, and the bridge that enrolls
// them. `streams` is indexed by rank (slot 0 unused; a null entry is skipped).
struct DecodeGraphPeerCapture {
    const DecodeGraphPeerBridge* bridge = nullptr;
    TpArray<cudaStream_t> streams{};
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    // Single-device capture: `stream` is both the origin and the only stream captured.
    void capture(cudaStream_t stream, const std::function<void()>& body);
    // Multi-device capture: `stream` is the origin (rank 0's stream) and `peer.streams` names every
    // peer rank's stream, each of which is forked into the same capture for the duration of `body`
    // and joined back before the capture ends. A null `peer.bridge` is the single-device form
    // above.
    void capture(cudaStream_t stream, const std::function<void()>& body,
                 const DecodeGraphPeerCapture& peer);
    [[nodiscard]] bool ready() const noexcept;
    // Node count of the captured graph, 0 when empty. Cross-device event edges are edges, not
    // nodes, so this counts real device work on BOTH devices.
    [[nodiscard]] std::size_t node_count() const;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
