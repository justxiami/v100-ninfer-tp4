#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include "ninfer/types.h" // TpArray, kMaximumDevices

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_decode || state.dflash_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame  = *state.execution.io.dflash_decode;
            Tensor count = frame.append_counts.slice(0, 0, 1);
            Tensor lane  = frame.lanes.slice(0, 0, 1);
            Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
            ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            if (rewrite_checkpoint) {
                state.dflash->save_rewrite_checkpoint(state.dflash_host_ingress->lanes[0],
                                                      state.execution.device.stream);
            }
        });
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t rewrite_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_linear_state_slots(current_state_slot, rewrite_checkpoint_state_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier, bool finalize_at_end) {
    TpPeers peers = tp_executions(state.execution);
    for (std::size_t r = 1; r < peers.size(); ++r) {
        if (!peers[r].has_value()) { continue; }
        // The per-sequence MTP KV window is request state, so it comes from the PrefillContext
        // rather than from the process-lifetime peer core: the MTP prefill appends and reads
        // through the execution view, unlike the text prefill, which drives the batch cache.
        peers[r]->mtp_kv = state.mtp_kv_peers[r];
        if (peers[r]->mtp_kv.valid() != state.mtp_kv.valid()) {
            throw std::logic_error("tensor-parallel MTP KV windows disagree between ranks");
        }
    }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache, peers);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_rewrite_checkpoint_frontier(
        rewrite_checkpoint_capture_frontier
            ? static_cast<std::int64_t>(*rewrite_checkpoint_capture_frontier)
            : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier,
                         bool finalize_at_end) {
    if (state.dflash != nullptr) {
        throw std::logic_error("DFlash staged multimodal prefill is unavailable");
    }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_rewrite_checkpoint_frontier(
        rewrite_checkpoint_capture_frontier
            ? static_cast<std::int64_t>(*rewrite_checkpoint_capture_frontier)
            : -1);
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

TpArray<Tensor> resume_hidden(ExecutionCore& execution, const Tensor& hidden) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("prefix resume requires BF16 [hidden,1]");
    }
    TpArray<Tensor> result{};
    result[0] = hidden;
    // Rank 1's copy runs over the shared PeerEvents handshake: the tp2 pull protocol is the one
    // path that needs that choreography, and keeping rank 1's half byte-for-byte identical to what
    // the shipped tp2 path always did is worth more than the symmetry.
    const TpPeerCore* peer = execution.peers[1];
    if (peer == nullptr) { return result; }
    result[1] = peer->prefill_hidden->slice(1, 0, 1);
    const CurrentDevice restore;
    CUDA_CHECK(cudaSetDevice(execution.device.device));
    CUDA_CHECK(cudaEventRecord(peer->events->inputs_ready(0), execution.device.stream));
    CUDA_CHECK(cudaSetDevice(peer->device->device));
    CUDA_CHECK(cudaStreamWaitEvent(peer->device->stream, peer->events->inputs_ready(0), 0));
    CUDA_CHECK(cudaMemcpyAsync(result[1].data, hidden.data, hidden.bytes(),
                               cudaMemcpyDeviceToDevice, peer->device->stream));
    CUDA_CHECK(cudaEventRecord(peer->events->pull_done(1), peer->device->stream));
    CUDA_CHECK(cudaSetDevice(execution.device.device));
    CUDA_CHECK(cudaStreamWaitEvent(execution.device.stream, peer->events->pull_done(1), 0));
    // Ranks 2..tp-1 need the same [hidden,1] staging slice, into THEIR OWN `prefill_hidden`
    // buffers: that staging is what the bridge reads back as the previous hidden and what the
    // suffix prefill replaces afterwards, so borrowing rank 0's tensor across a peer mapping is
    // not equivalent. They have no pull-protocol choreography of their own -- at tp > 2 every
    // collective goes through NCCL -- so the copies ride rank 0's stream and are retired with one
    // synchronize before returning. That is safe here and only here: this call site is eager
    // prefill (`mtp_bridge_and_propose`, `sample_from_hidden`), never a captured region, and every
    // rank consumes the slice on its own stream strictly afterwards.
    bool extra_copies = false;
    for (std::size_t rank = 2; rank < execution.peers.size(); ++rank) {
        const TpPeerCore* lane = execution.peers[rank];
        if (lane == nullptr) { continue; }
        result[rank] = lane->prefill_hidden->slice(1, 0, 1);
        CUDA_CHECK(cudaMemcpyAsync(result[rank].data, hidden.data, hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, execution.device.stream));
        extra_copies = true;
    }
    if (extra_copies) { execution.device.synchronize(); }
    return result;
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    const auto restored = resume_hidden(state.execution, hidden);
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    TpPeers peers = tp_executions(state.execution);
    if (!peers[1].has_value()) { peers = TpPeers{}; }
    if (peers[1].has_value()) {
        TpArray<Tensor> destinations;
        destinations[0] = logits;
        for (std::size_t r = 1; r < peers.size(); ++r) {
            if (!peers[r].has_value()) { continue; }
            peers[r]->work->reset();
            destinations[r] = peers[r]->io->logits.slice(1, 0, 1);
        }
        TpArray<Tensor> restored_logits;
        for (std::size_t r = 0; r < peers.size(); ++r) {
            if (r != 0 && !peers[r].has_value()) { continue; }
            restored_logits[r] = restored[r];
        }
        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, state.text_kv,
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk,
                         state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                         peers);
        card.target_logits(restored_logits, destinations);
    } else {
        ops::linear(hidden, state.execution.model.output_head, logits,
                     state.execution.device.stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token, TextConfig::token_domain, state.sampling,
                state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    state.execution.work.reset();
    for (std::size_t r = 1; r < peers.size(); ++r) {
        if (peers[r].has_value()) { peers[r]->work->reset(); }
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
