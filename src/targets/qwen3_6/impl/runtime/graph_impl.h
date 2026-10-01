#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// Every peer rank's stream, indexed by rank (slot 0 is null). The graph bridge and the capture
// both need all of them, and both skip null slots.
inline TpArray<cudaStream_t> peer_graph_streams(const ExecutionCore& execution) {
    TpArray<cudaStream_t> streams{};
    for (std::size_t rank = 1; rank < execution.peers.size(); ++rank) {
        const TpPeerCore* peer = execution.peer_at(rank);
        if (peer == nullptr) { continue; }
        streams[rank] = peer->device->stream;
    }
    return streams;
}

// A multi-device graph is launched ONCE, on rank 0's stream, and holds every rank's nodes inside
// it. Launching it also makes rank 0's stream wait for the whole graph, so the caller's existing
// "synchronize every peer, then synchronize rank 0" is still sufficient to retire the round.
template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        const TpPeerCore* first = state.execution.peer_at(1);
        if (first != nullptr) {
            if (first->graph_bridge == nullptr) {
                throw std::logic_error("tensor-parallel graph launch requires a peer bridge");
            }
            // See DecodeGraphPeerBridge::gate_launch: the eager path issues each rank's kernels on
            // that rank's own stream and is therefore automatically ordered after the round's
            // mirrored KV page materialization; a graph launched on rank 0's stream is not.
            first->graph_bridge->gate_launch(peer_graph_streams(state.execution),
                                             state.execution.device.stream);
        }
        executable->launch(state.execution.device.stream);
    } else {
        body();
    }
}

// One capture site for every decode family, at tp1 and tp2 alike.
//
// At tp2 the body issues work on BOTH devices' streams and orders them with the collectives'
// cross-device events. Those events only become graph edges if both streams belong to the same
// capture, so rank 1's stream is forked into rank 0's capture for the duration of the body and
// joined back before it ends: one cudaGraph holding both devices' nodes, launched once on rank 0's
// stream. Two independent per-device captures are not an option -- CUDA rejects an event wait that
// would cross two live captures (cudaErrorStreamCaptureMerge).
//
// Both workspace arenas are reset first, for the same reason at both ranks: the body allocates its
// activations at deterministic arena offsets, and those addresses are baked into the graph.
template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    const TpPeerCore* first = state.execution.peer_at(1);
    if (first == nullptr) {
        definition.capture(state.execution.device.stream, body);
        return;
    }
    if (first->graph_bridge == nullptr) {
        throw std::logic_error("tensor-parallel graph capture requires a peer capture bridge");
    }
    for (std::size_t rank = 1; rank < state.execution.peers.size(); ++rank) {
        const TpPeerCore* peer = state.execution.peer_at(rank);
        if (peer == nullptr) { continue; }
        peer->work->reset();
    }
    definition.capture(state.execution.device.stream, body,
                       DecodeGraphPeerCapture{first->graph_bridge,
                                              peer_graph_streams(state.execution)});
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
