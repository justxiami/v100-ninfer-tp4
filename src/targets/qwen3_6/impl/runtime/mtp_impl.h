#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "ninfer/types.h" // TpArray, kMaximumDevices

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

// One rank's window into ITS OWN MtpDecodeState, sliced to the round's batch. Both ranks are
// sliced by the same function so a shape mistake cannot apply to one device only -- which matters
// because rank 1's buffers live on the other GPU, where an out-of-bounds write is silent: a
// peer-side write that is exactly in bounds at batch 1 runs off the end of the buffer at batch 2
// with nothing on the local device noticing.
struct MtpRoundView {
    Tensor anchors;
    Tensor frontiers;
    Tensor budgets;
    Tensor current_extents;
    Tensor target_valid;
    Tensor current_drafts;
    Tensor target_rope;
    Tensor text_rows;
    Tensor mtp_rows;
    Tensor lanes;
    Tensor rope_deltas;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_tokens;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor selected_hidden;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted;
    Tensor next_extents;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;
    Tensor next_drafts;
    Tensor proposal_logits;
    const ops::SamplingConfig* sampling = nullptr;
};

MtpRoundView slice_mtp_frame(qwen3_6::MtpDecodeState& frame, std::int32_t batch_size) {
    MtpRoundView out;
    out.anchors           = frame.anchors.slice(0, 0, batch_size);
    out.frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
    out.budgets           = frame.remaining_budgets.slice(0, 0, batch_size);
    out.current_extents   = frame.current_extents.slice(0, 0, batch_size);
    out.target_valid      = frame.target_valid_columns.slice(0, 0, batch_size);
    out.current_drafts    = frame.current_drafts.slice(1, 0, batch_size);
    out.target_rope       = frame.target_rope_positions.slice(1, 0, batch_size);
    out.text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
    out.mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
    out.lanes             = frame.lanes.slice(0, 0, batch_size);
    out.rope_deltas       = frame.rope_deltas.slice(0, 0, batch_size);
    out.verify_ids        = frame.verify_ids.slice(1, 0, batch_size);
    out.target_positions  = frame.target_positions.slice(1, 0, batch_size);
    out.target_tokens     = frame.target_argmax.slice(1, 0, batch_size);
    out.target_logits     = frame.target_logits.slice(2, 0, batch_size);
    out.target_hidden     = frame.target_hidden.slice(2, 0, batch_size);
    out.selected_hidden   = frame.target_continuation_hidden.slice(1, 0, batch_size);
    out.licensed_tokens   = frame.licensed_tokens.slice(1, 0, batch_size);
    out.licensed_counts   = frame.licensed_counts.slice(0, 0, batch_size);
    out.accepted          = frame.accepted_drafts.slice(0, 0, batch_size);
    out.next_extents      = frame.next_extents.slice(0, 0, batch_size);
    out.alignment_ids     = frame.alignment_ids.slice(1, 0, batch_size);
    out.alignment_hidden  = frame.alignment_hidden.slice(2, 0, batch_size);
    out.ar_hidden         = frame.ar_hidden.slice(1, 0, batch_size);
    out.next_hidden       = frame.next_hidden.slice(1, 0, batch_size);
    out.ar_positions      = frame.ar_positions.slice(0, 0, batch_size);
    out.ar_rope_positions = frame.ar_rope_positions.slice(0, 0, batch_size);
    out.ar_valid_columns  = frame.ar_valid_columns.slice(0, 0, batch_size);
    out.next_drafts       = frame.next_drafts.slice(0, 0, batch_size);
    out.proposal_logits   = frame.proposal_logits.slice(1, 0, batch_size);
    out.sampling          = frame.sampling;
    return out;
}

// Debug-only round trace, enabled by NINFER_TP4_MTP_TRACE=1. Prints, for row 0 of every rank, the
// integers that decide acceptance: the drafts the MTP head proposed, the target's own argmax at the
// same columns (from the gathered verify logits), what the accept op licensed, and how many it
// accepted. `drafts` vs `target` answers whether the PROPOSAL is wrong; `licensed` vs `target`
// answers whether the ACCEPT/commit path is wrong. Both were open at tp4 and they need different
// fixes, so guessing between them costs more than these two memcpys per round.
inline void mtp_trace_round(const ExecutionContext& ec, const TpArray<MtpRoundView>& views,
                            std::int32_t k) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TP4_MTP_TRACE");
        return value != nullptr && *value != '0';
    }();
    if (!enabled) { return; }
    const std::int32_t width = k + 1;
    const CurrentDevice restore;
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const MtpRoundView& v = views[slot];
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        // MUST be the async form on THIS rank's stream followed by a stream sync. A plain
        // cudaMemcpy goes to the legacy default stream, which this engine's streams (created
        // cudaStreamNonBlocking) do not synchronize with -- the dump would then read whatever the
        // buffer held before the round, which is exactly the kind of stale read this trace exists
        // to rule out.
        auto read = [&](const Tensor& t, std::int32_t count) {
            std::vector<std::int32_t> out(static_cast<std::size_t>(count));
            CUDA_CHECK(cudaMemcpyAsync(out.data(), t.data,
                                       static_cast<std::size_t>(count) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, ec.dev[slot]->stream));
            CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
            return out;
        };
        const auto drafts  = read(v.current_drafts, k);
        const auto verify  = read(v.verify_ids, width);
        const auto target  = read(v.target_tokens, width);
        const auto licensed = read(v.licensed_tokens, width);
        const auto accepted = read(v.accepted, 1);
        const auto counts   = read(v.licensed_counts, 1);
        const auto anchors  = read(v.anchors, 1);
        const auto extents  = read(v.current_extents, 1);
        const auto valid    = read(v.target_valid, 1);
        const auto text_row = read(v.text_rows, 1);
        const auto mtp_row  = read(v.mtp_rows, 1);
        const auto tpos     = read(v.target_positions, width);
        const auto trope    = read(v.target_rope, width);
        const auto arope    = read(v.ar_rope_positions, k);
        auto join = [](const std::vector<std::int32_t>& xs) {
            std::string out = "[";
            for (std::size_t i = 0; i < xs.size(); ++i) {
                out += (i ? "," : "") + std::to_string(xs[i]);
            }
            return out + "]";
        };
        if (rank == 0) {
            // Column aliasing probe: the verify logits are [vocab, width, batch] with the vocab on
            // the fast axis, so column i's rows are data[i * vocab + v]. Printing a few raw values
            // per column (and the value AT each column's argmax) shows immediately whether two
            // columns share one buffer image -- which would make their argmaxes agree by
            // construction instead of by computation.
            // The vocab extent comes from the tensor itself: the 27b and 35b targets have
            // different padded widths, and a literal here would silently read the wrong column.
            const std::int32_t vocab_rows = v.target_logits.ne[0];
            std::vector<unsigned short> raw(4);
            for (std::int32_t c = 0; c < width; ++c) {
                CUDA_CHECK(cudaMemcpyAsync(
                    raw.data(),
                    static_cast<const unsigned short*>(v.target_logits.data) + c * vocab_rows,
                    raw.size() * sizeof(unsigned short), cudaMemcpyDeviceToHost,
                    ec.dev[slot]->stream));
                CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
                std::int32_t at_argmax[1] = {0};
                CUDA_CHECK(cudaMemcpyAsync(
                    at_argmax,
                    static_cast<const unsigned short*>(v.target_logits.data) + c * vocab_rows +
                        target[static_cast<std::size_t>(c)],
                    sizeof(unsigned short), cudaMemcpyDeviceToHost, ec.dev[slot]->stream));
                CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
                auto to_float = [](unsigned short bits) {
                    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
                    float out                = 0.0F;
                    std::memcpy(&out, &wide, sizeof(out));
                    return out;
                };
                std::fprintf(stderr, "    col%d rows0..3=%.4f,%.4f,%.4f,%.4f atArgmax(%d)=%.4f\n", c,
                             to_float(raw[0]), to_float(raw[1]), to_float(raw[2]), to_float(raw[3]),
                             target[static_cast<std::size_t>(c)],
                             to_float(static_cast<unsigned short>(at_argmax[0])));
            }
        }
        std::fprintf(stderr,
                     "[mtp-trace] r%d anchor=%d accept=%d count=%d extent=%d valid=%d text_row=%d "
                     "mtp_row=%d\n    drafts=%s verify=%s target=%s licensed=%s\n    tpos=%s "
                     "trope=%s arope=%s\n",
                     rank, anchors[0], accepted[0], counts[0], extents[0], valid[0], text_row[0],
                     mtp_row[0], join(drafts).c_str(), join(verify).c_str(), join(target).c_str(),
                     join(licensed).c_str(), join(tpos).c_str(), join(trope).c_str(),
                     join(arope).c_str());
    }
}

TargetVerifyFrameView verify_view(const MtpRoundView& v, const GdnReplayRecords* records) {
    return TargetVerifyFrameView{
        .ids             = v.verify_ids,
        .cache_positions = v.target_positions,
        .rope_positions  = v.target_rope,
        .valid_columns   = v.target_valid,
        .kv_table_rows   = v.text_rows,
        .lanes           = v.lanes,
        .target_hidden   = v.target_hidden,
        .target_logits   = v.target_logits,
        .target_tokens   = v.target_tokens,
        .drafts          = v.current_drafts,
        .current_extents = v.current_extents,
        .frontiers       = v.frontiers,
        .anchors         = v.anchors,
        .licensed_tokens = v.licensed_tokens,
        .licensed_counts = v.licensed_counts,
        .accepted_drafts = v.accepted,
        .selected_hidden = v.selected_hidden,
        .replay_records  = records,
        .sampling        = v.sampling,
    };
}

// Debug-only proposal-head A/B (NINFER_TP4_MTP_HIDDEN_DUMP=path / ..._LOAD=path).
//
// Acceptance depends on two things that are entangled in a live round: the MTP module's forward
// (stem -> attention -> MLP) and the proposal head's column-parallel GEMM plus its logits gather.
// Both are sharded differently at tp2 and tp4, so "the drafts are worse at tp4" does not say which
// one is at fault. This probe breaks the tie: dump the [hidden, batch] proposal input of the FIRST
// round (from whichever width), then load that exact buffer into every rank's proposal input on
// another run and print the resulting argmax. Same activation through both heads: an equal argmax
// means the head is fine and the module is the degraded half.
inline void mtp_proposal_input_probe(const ExecutionContext& ec,
                                     const TpArray<MtpRoundView>& views) {
    static const char* dump_path = std::getenv("NINFER_TP4_MTP_HIDDEN_DUMP");
    static const char* load_path = std::getenv("NINFER_TP4_MTP_HIDDEN_LOAD");
    if (dump_path == nullptr && load_path == nullptr) { return; }
    static bool done = false; // first round only: later rounds have advanced state
    if (done) { return; }
    done = true;
    const Tensor& source  = views[0].ar_hidden;
    const std::size_t bytes = static_cast<std::size_t>(source.bytes());
    const CurrentDevice restore;
    if (dump_path != nullptr) {
        std::vector<unsigned char> host(bytes);
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        CUDA_CHECK(cudaMemcpyAsync(host.data(), source.data, bytes, cudaMemcpyDeviceToHost,
                                   ec.dev[0]->stream));
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[0]->stream));
        FILE* file = std::fopen(dump_path, "wb");
        if (file == nullptr) { throw std::runtime_error("MTP hidden probe: dump path unreadable"); }
        std::fwrite(host.data(), 1, bytes, file);
        std::fclose(file);
        std::fprintf(stderr, "[mtp-hidden] dumped %zu bytes from rank 0\n", bytes);
    }
    if (load_path != nullptr) {
        std::vector<unsigned char> host(bytes);
        FILE* file = std::fopen(load_path, "rb");
        if (file == nullptr) { throw std::runtime_error("MTP hidden probe: load path unreadable"); }
        const std::size_t got = std::fread(host.data(), 1, bytes, file);
        std::fclose(file);
        if (got != bytes) {
            throw std::runtime_error("MTP hidden probe: file size does not match the frame");
        }
        // Every rank gets the SAME image: the residual stream is replicated by construction, and a
        // per-rank shard here would make the comparison measure two different inputs.
        for (int rank = 0; rank < ec.tp; ++rank) {
            CUDA_CHECK(cudaSetDevice(ec.dev[static_cast<std::size_t>(rank)]->device));
            CUDA_CHECK(cudaMemcpyAsync(views[static_cast<std::size_t>(rank)].ar_hidden.data,
                                       host.data(), bytes, cudaMemcpyHostToDevice,
                                       ec.dev[static_cast<std::size_t>(rank)]->stream));
        }
        std::fprintf(stderr, "[mtp-hidden] loaded %zu bytes into every rank\n", bytes);
    }
}

void mtp_bridge_tp2(PrefillContext& state, const Tensor& next_token,
                     const Tensor& previous_hidden, std::int32_t position,
                     std::span<const std::int32_t> rope_position, bool build_proposal) {
    TpPeers peers = tp_executions(state.execution);
    // Every non-zero rank runs the bridge over ITS OWN MTP KV window and its own weight shards.
    // The bridge is not a rank-0-only prelude: the MTP module's projections are split across the
    // whole width, so each rank appends its own slice of the bridge token to its own window and
    // contributes its own shard to the proposal gather. The tp2 era ran this on one peer because
    // there was only one.
    TpExecution* tp = peers[1].has_value() ? &*peers[1] : nullptr;
    if (tp == nullptr) {
        throw std::logic_error("tensor-parallel MTP bridge requires a peer");
    }
    const ExecutionContext& ec = *tp->execution;
    TpArray<WorkspaceArena*> work{};
    TpArray<qwen3_6::RoundState*> io{};
    work[0] = &state.execution.work;
    io[0]   = &state.execution.io;
    for (int rank = 1; rank < ec.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        peers[slot]->mtp_kv = state.mtp_kv_peers[slot];
        if (!peers[slot]->mtp_kv.valid() || !peers[slot]->io->mtp) {
            throw std::logic_error(
                "tensor-parallel MTP bridge requires both KV windows on every rank");
        }
        peers[slot]->work->reset();
        work[slot] = peers[slot]->work;
        io[slot]   = peers[slot]->io;
    }
    state.execution.work.reset();
    const auto restored = resume_hidden(state.execution, previous_hidden);
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache, peers);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    TpArray<Tensor> positions, rope, ar_hidden, logits, ar_positions;
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        const auto stream = ec.dev[r]->stream;
        positions[r] = io[r]->mtp->target_positions.slice(0, 0, 1);
        ops::set_i32_scalar(positions[r], position, stream);
        rope[r] = work[r]->alloc(DType::I32, {1, 3});
        CUDA_CHECK(cudaMemcpyAsync(rope[r].data, rope_position.data(), rope_position.size_bytes(),
                                   cudaMemcpyHostToDevice, stream));
        ar_hidden[r] = io[r]->mtp->ar_hidden;
        logits[r] = io[r]->logits.slice(1, 0, 1);
        ar_positions[r] = io[r]->mtp->position.slice(0, 0, 1);
    });
    Tensor draft0 = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    const auto visible = static_cast<std::uint32_t>(position + 1);
    mtp_stage_barrier(state.execution, "bridge: entered");
    card.mtp_forward_batch(next_token, restored, positions, rope, {visible, visible}, ar_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr);
    mtp_stage_barrier(state.execution, "bridge: forward batch + proposal argmax");
    if (build_proposal) {
        for_each_rank(ec, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::set_i32_scalar(ar_positions[r], position + 1, ec.dev[r]->stream);
        });
        for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
            Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
            Tensor next_draft = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
            TpArray<Tensor> next_hidden;
            for (int rank = 0; rank < ec.tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                next_hidden[slot] = rank == 0
                                        ? state.execution.prefill_hidden.slice(1, i, 1)
                                        : peers[slot]->prefill_hidden->slice(1, i, 1);
            }
            const auto ar_visible = static_cast<std::uint32_t>(position + i + 1);
            card.mtp_forward_ar_step(previous_token, ar_hidden, ar_positions,
                                     {ar_visible, ar_visible}, next_hidden, logits, next_draft);
            for_each_rank(ec, [&](int rank) {
                const auto r = static_cast<std::size_t>(rank);
                const auto stream = ec.dev[r]->stream;
                CUDA_CHECK(cudaMemcpyAsync(ar_hidden[r].data, next_hidden[r].data,
                                           ar_hidden[r].bytes(), cudaMemcpyDeviceToDevice, stream));
                ops::increment_i32_scalar(ar_positions[r], stream);
            });
            mtp_stage_barrier(state.execution, "bridge: AR step");
        }
    }
    // The following suffix prefill resets both arenas and may replace the staging hidden. Retire
    // every bridge stream here, including a bridge with no proposal or cross-rank logit gather.
    state.execution.device.synchronize();
    for (int rank = 1; rank < ec.tp; ++rank) {
        peers[static_cast<std::size_t>(rank)]->device->synchronize();
    }
    state.execution.work.reset();
    for (int rank = 1; rank < ec.tp; ++rank) {
        peers[static_cast<std::size_t>(rank)]->work->reset();
    }
}

} // namespace

void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    if (build_proposal &&
        (state.mtp_proposal_extent == 0 ||
         state.mtp_proposal_extent >
             static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0]))) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }
    if (state.execution.peer_at(1) != nullptr) {
        if (next_embedding != nullptr) {
            throw std::logic_error("tensor-parallel MTP bridge supports text inputs only");
        }
        mtp_bridge_tp2(state, next_token, previous_hidden, position, rope_position, build_proposal);
        return;
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
        TpPeers peers = tp_executions(state.execution);
        TpExecution* tp = peers[1].has_value() ? &*peers[1] : nullptr;
        // Every non-zero rank runs the round from ITS OWN copy of the same ingress record, so all
        // ranks read identical anchors, drafts, positions, KV rows and lane ids without a
        // cross-device copy. The copy is that rank's OWN pinned record, not rank 0's: the record
        // ends with a SamplingConfig per row whose `token_counts` names a penalty counter lane,
        // and `speculative_accept_greedy_drafts` reads and atomically writes that pointer on every
        // rank. Handing every rank rank 0's record therefore double-counts rank 0's lane and leaves
        // ranks 2..n-1 sampling with no repetition penalty at all -- which is exactly the silent
        // corruption this loop prevents.
        if (tp != nullptr) {
            const ExecutionContext& ec = *tp->execution;
            for (int rank = 1; rank < ec.tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                TpExecution& lane = peers[slot].value();
                if (!lane.io->mtp_decode.has_value()) {
                    throw std::logic_error(
                        "tensor-parallel MTP decode requires a peer frame on every rank");
                }
                if (state.peer_host_ingress[slot] == nullptr) {
                    throw std::logic_error(
                        "tensor-parallel MTP decode requires an ingress record on every rank");
                }
                int previous = 0;
                CUDA_CHECK(cudaGetDevice(&previous));
                CUDA_CHECK(cudaSetDevice(lane.device->device));
                CUDA_CHECK(cudaMemcpyAsync(lane.io->mtp_decode->ingress.data,
                                           state.peer_host_ingress[slot],
                                           sizeof(qwen3_6::MtpDecodeIngress),
                                           cudaMemcpyHostToDevice, lane.device->stream));
                CUDA_CHECK(cudaSetDevice(previous));
            }
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache,
                         &state.mtp_cache, peers);

        MtpRoundView v = slice_mtp_frame(frame, batch_size);

        if (!tp) {
            ops::speculative_prepare_verify_inputs(v.anchors, v.current_drafts, v.frontiers,
                                                   v.current_extents, v.verify_ids,
                                                   v.target_positions,
                                                   state.execution.device.stream);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(v, state.execution.replay_records),
                                 envelopes.target_verify);

            ops::mtp_prepare_next_round(v.verify_ids, v.anchors, v.accepted, v.frontiers,
                                        v.budgets, v.licensed_counts, v.rope_deltas,
                                        v.alignment_ids, v.next_extents, v.ar_positions,
                                        v.ar_rope_positions, v.ar_valid_columns,
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        state.execution.device.stream);
            card.mtp_forward_decode_batch(v.alignment_ids, v.target_hidden, v.target_positions,
                                          v.target_rope, v.licensed_counts, v.mtp_rows,
                                          envelopes.batch, v.alignment_hidden);
            ops::speculative_select_accepted_hidden(v.alignment_hidden, v.accepted, v.ar_hidden,
                                                    state.execution.device.stream);

            Tensor draft0 = v.next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch(v.ar_hidden, v.proposal_logits, draft0);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    v.next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next = v.next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                                  .view({batch_size});
                Tensor position =
                    v.ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1,
                                                                                      batch_size});
                Tensor rope = v.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = v.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch    = previous.view({1, batch_size});
                Tensor hidden_batch      = v.ar_hidden.view({TextConfig::hidden, 1, batch_size});
                Tensor next_hidden_batch = v.next_hidden.view({TextConfig::hidden, 1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              v.mtp_rows, envelopes.ar[step], next_hidden_batch);
                card.mtp_propose_batch(v.next_hidden, v.proposal_logits, next);
                CUDA_CHECK(cudaMemcpyAsync(v.ar_hidden.data, v.next_hidden.data,
                                           v.ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
            }
        } else {
            const ExecutionContext& ec = *tp->execution;
            // One round view per rank: rank 0's is the round's own frame, every other rank's is ITS
            // OWN MtpDecodeState. All of them are sliced by the same function, so a shape mistake
            // cannot apply to one device only -- which matters because another rank's buffers live
            // on another GPU, where an out-of-bounds write is silent.
            TpArray<MtpRoundView> views{};
            for (int rank = 0; rank < ec.tp; ++rank) {
                const auto slot             = static_cast<std::size_t>(rank);
                qwen3_6::MtpDecodeState& f =
                    rank == 0 ? frame : *state.execution.peer_at(slot)->io->mtp_decode;
                views[slot]                 = slice_mtp_frame(f, batch_size);
            }

            for_each_rank(ec, [&](int rank) {
                MtpRoundView& r = views[static_cast<std::size_t>(rank)];
                ops::speculative_prepare_verify_inputs(r.anchors, r.current_drafts, r.frontiers,
                                                       r.current_extents, r.verify_ids,
                                                       r.target_positions, ec.dev[rank]->stream);
            });

            TpArray<TargetVerifyFrameView> verify_frames{};
            for (int rank = 0; rank < ec.tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                const GdnReplayRecords* records =
                    rank == 0 ? state.execution.replay_records
                              : state.execution.peer_at(slot)->replay_records;
                verify_frames[slot] = verify_view(views[slot], records);
            }
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_frames, envelopes.target_verify);
            mtp_trace_round(ec, views, static_cast<std::int32_t>(k));

            for_each_rank(ec, [&](int rank) {
                MtpRoundView& r = views[static_cast<std::size_t>(rank)];
                ops::mtp_prepare_next_round(
                    r.verify_ids, r.anchors, r.accepted, r.frontiers, r.budgets, r.licensed_counts,
                    r.rope_deltas, r.alignment_ids, r.next_extents, r.ar_positions,
                    r.ar_rope_positions, r.ar_valid_columns,
                    static_cast<std::int32_t>(state.text_cache.max_context()),
                    ec.dev[rank]->stream);
            });

            {
                TpArray<Tensor> alignment_hidden;
                TpArray<Tensor> target_hidden;
                TpArray<Tensor> target_positions;
                TpArray<Tensor> target_rope;
                TpArray<Tensor> licensed_counts;
                TpArray<Tensor> mtp_rows;
                for (int rank = 0; rank < ec.tp; ++rank) {
                    const auto slot       = static_cast<std::size_t>(rank);
                    MtpRoundView& r       = views[slot];
                    alignment_hidden[slot] = r.alignment_hidden;
                    target_hidden[slot]    = r.target_hidden;
                    target_positions[slot] = r.target_positions;
                    target_rope[slot]      = r.target_rope;
                    licensed_counts[slot]  = r.licensed_counts;
                    mtp_rows[slot]         = r.mtp_rows;
                }
                card.mtp_forward_decode_batch(views[0].alignment_ids, target_hidden,
                                              target_positions, target_rope, licensed_counts,
                                              mtp_rows, envelopes.batch, alignment_hidden);
            }
            for_each_rank(ec, [&](int rank) {
                MtpRoundView& r = views[static_cast<std::size_t>(rank)];
                ops::speculative_select_accepted_hidden(r.alignment_hidden, r.accepted, r.ar_hidden,
                                                        ec.dev[rank]->stream);
            });

            {
                TpArray<Tensor> ar_hidden;
                TpArray<Tensor> proposal_logits;
                for (int rank = 0; rank < ec.tp; ++rank) {
                    const auto slot        = static_cast<std::size_t>(rank);
                    ar_hidden[slot]        = views[slot].ar_hidden;
                    proposal_logits[slot]  = views[slot].proposal_logits;
                }
                Tensor draft0 = views[0].next_drafts.slice(1, 0, 1).view({batch_size});
                mtp_proposal_input_probe(ec, views);
                card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
                if (std::getenv("NINFER_TP4_MTP_HIDDEN_DUMP") != nullptr ||
                    std::getenv("NINFER_TP4_MTP_HIDDEN_LOAD") != nullptr) {
                    std::int32_t token = 0;
                    CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
                    CUDA_CHECK(cudaMemcpyAsync(&token, draft0.data, sizeof(token),
                                               cudaMemcpyDeviceToHost, ec.dev[0]->stream));
                    CUDA_CHECK(cudaStreamSynchronize(ec.dev[0]->stream));
                    std::fprintf(stderr, "[mtp-hidden] tp%d proposal argmax=%d\n",
                                 ec.tp, token);
                }
            }

            TpArray<Tensor> proposal_logits_step;
            for (int rank = 0; rank < ec.tp; ++rank) {
                proposal_logits_step[static_cast<std::size_t>(rank)] =
                    views[static_cast<std::size_t>(rank)].proposal_logits;
            }
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    views[0].next_drafts.slice(1, static_cast<std::int32_t>(step), 1)
                        .view({batch_size});
                Tensor next = views[0].next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                                  .view({batch_size});
                TpArray<Tensor> position;
                TpArray<Tensor> rope;
                TpArray<Tensor> valid;
                TpArray<Tensor> hidden_batch;
                TpArray<Tensor> next_hidden_batch;
                TpArray<Tensor> mtp_rows;
                TpArray<Tensor> next_hidden;
                for (int rank = 0; rank < ec.tp; ++rank) {
                    const auto slot = static_cast<std::size_t>(rank);
                    MtpRoundView& v = views[slot];
                    position[slot] = v.ar_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                         .view({1, batch_size});
                    rope[slot] = v.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                     .view({1, batch_size});
                    valid[slot] = v.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                      .view({batch_size});
                    hidden_batch[slot]      = v.ar_hidden.view({TextConfig::hidden, 1, batch_size});
                    next_hidden_batch[slot] = v.next_hidden.view({TextConfig::hidden, 1, batch_size});
                    mtp_rows[slot]          = v.mtp_rows;
                    next_hidden[slot]       = v.next_hidden;
                }
                Tensor previous_batch = previous.view({1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              mtp_rows, envelopes.ar[step], next_hidden_batch);
                card.mtp_propose_batch(next_hidden, proposal_logits_step, next);
                for_each_rank(ec, [&](int rank) {
                    MtpRoundView& r = views[static_cast<std::size_t>(rank)];
                    CUDA_CHECK(cudaMemcpyAsync(r.ar_hidden.data, r.next_hidden.data,
                                               r.ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                               ec.dev[rank]->stream));
                });
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
