#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens);
    }
    ops::speculative_accept_greedy_drafts(frame.target_tokens, frame.target_logits, frame.drafts,
                                          frame.current_extents, frame.frontiers, frame.anchors,
                                          frame.licensed_tokens, frame.licensed_counts,
                                          frame.accepted_drafts, TextConfig::token_domain,
                                          frame.sampling, execution.work, execution.device.stream);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

// One frame per rank, INDEXED BY RANK (slot 0 is rank 0's). The acceptance arithmetic is
// replicated on every rank rather than transferred, because each of its inputs is either the
// ingress record -- uploaded to every rank's own frame -- or the gathered logits, which are
// bit-identical everywhere. What is NOT replicated is rank 0's bookkeeping: the
// continuation-hidden scatter and the egress transfer stay on rank 0 alone.
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, const TpArray<TargetVerifyFrameView>& frames,
                          ops::GqaExecutionEnvelope envelope) {
    const TpPeerCore* first = execution.peer_at(1);
    if (first == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires a peer");
    }
    const ExecutionContext& ec = *first->execution;
    if (static_cast<int>(frames.size()) < ec.tp) {
        throw std::logic_error("tensor-parallel target verify needs one frame per rank");
    }

    TpArray<Tensor> ids;
    TpArray<Tensor> cache_positions;
    TpArray<Tensor> rope_positions;
    TpArray<Tensor> valid_columns;
    TpArray<Tensor> kv_table_rows;
    TpArray<Tensor> lanes;
    TpArray<Tensor> target_hidden;
    TpArray<Tensor> target_logits;
    TpArray<Tensor> target_tokens;
    TpArray<WorkspaceArena*> work;
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot                = static_cast<std::size_t>(rank);
        const TargetVerifyFrameView& v = frames[slot];
        if (v.replay_records == nullptr) {
            throw std::logic_error("speculative target verify has no ReplaySSM record storage");
        }
        if (rank > 0 && v.feature_sink != nullptr) {
            throw std::logic_error("tensor-parallel DFlash peer feature sink is unexpected");
        }
        ids[slot]             = v.ids;
        cache_positions[slot] = v.cache_positions;
        rope_positions[slot]  = v.rope_positions;
        valid_columns[slot]   = v.valid_columns;
        kv_table_rows[slot]   = v.kv_table_rows;
        lanes[slot]           = v.lanes;
        target_hidden[slot]   = v.target_hidden;
        target_logits[slot]   = v.target_logits;
        target_tokens[slot]   = v.target_tokens;
        work[slot] = rank == 0 ? &execution.work : execution.peer_at(slot)->work;
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frames[0].replay_records);
    if (frames[0].feature_sink != nullptr) {
        card.target_verify_batch(ids, cache_positions, rope_positions, valid_columns,
                                 kv_table_rows, lanes, envelope, target_hidden, target_logits,
                                 target_tokens, *frames[0].feature_sink);
    } else {
        card.target_verify_batch(ids, cache_positions, rope_positions, valid_columns,
                                 kv_table_rows, lanes, envelope, target_hidden, target_logits,
                                 target_tokens);
    }
    for_each_rank(ec, [&](int rank) {
        const auto slot          = static_cast<std::size_t>(rank);
        // A mutable COPY: the two Ops below write through their destination arguments.
        TargetVerifyFrameView v  = frames[slot];
        ops::speculative_accept_greedy_drafts(v.target_tokens, v.target_logits, v.drafts,
                                              v.current_extents, v.frontiers, v.anchors,
                                              v.licensed_tokens, v.licensed_counts,
                                              v.accepted_drafts, TextConfig::token_domain,
                                              v.sampling, *work[slot], ec.dev[slot]->stream);
        ops::speculative_select_accepted_hidden(v.target_hidden, v.accepted_drafts,
                                                v.selected_hidden, ec.dev[slot]->stream);
    });
    ops::scatter(frames[0].selected_hidden, frames[0].lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
