#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/swa.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>
#include "ninfer/types.h" // TpArray, kMaximumDevices

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

using qwen3_6::PreparedPromptData;
using qwen3_6::PromptModality;

// Current-device save/restore around per-rank issue. Kernel launches go to the CURRENT device, so
// every op issued for rank r must run with ec.dev[r]->device current; the calls are enqueue-only,
// so the two ranks' kernels still overlap even though the host issues them in sequence. This
// mirrors ops::detail::for_each_rank without reaching into an Op's private header, and lives here
// rather than in one impl header because both the schedule and the MTP round need it.
class CurrentDevice {
public:
    CurrentDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDevice() { (void)cudaSetDevice(previous_); }

    CurrentDevice(const CurrentDevice&)            = delete;
    CurrentDevice& operator=(const CurrentDevice&) = delete;

private:
    int previous_ = 0;
};

template <class Body>
void for_each_rank(const ExecutionContext& ec, Body&& body) {
    const CurrentDevice restore;
    for (int rank = 0; rank < ec.tp; ++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
        body(rank);
    }
}

// Rank 1's half of a tensor-parallel execution. Null at tp == 1, in which case every schedule
// entry point below behaves exactly as it always has.
struct TpPeerCore {
    const ExecutionContext* execution          = nullptr;
    const ops::PeerEvents* events              = nullptr;
    DeviceContext* device                      = nullptr;
    const LoadedModelData* model               = nullptr;
    WorkspaceArena* work                       = nullptr;
    LinearAttentionStatePool* linear_attention = nullptr;
    qwen3_6::RoundState* io                    = nullptr;
    Tensor* prefill_hidden                     = nullptr;
    const qwen3_6::PagedKVCache* text_cache    = nullptr;
    // Present only when the sequence plan enables MTP.
    const qwen3_6::PagedKVCache* mtp_cache  = nullptr;
    const GdnReplayRecords* replay_records  = nullptr;
    // Rank 1's own pinned MTP ingress record (see PeerRuntime::token_counts). It differs from
    // rank 0's only in the per-row `sampling[row].token_counts` pointer, which must name rank 1's
    // counter lane: `speculative_accept_greedy_drafts` READS and atomically WRITES that pointer
    // in sampling mode, and a pointer into the other device's arena is an illegal access without
    // peer mapping and a silent double-increment with it.
    const qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    const qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    // Enrolls rank 1's stream in rank 0's capture. Null when graphs are disabled; the eager path
    // never reads it.
    const DecodeGraphPeerBridge* graph_bridge = nullptr;
};

struct ExecutionCore {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    qwen3_6::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
    // YaRN rotary override, indexed by RANK: `[r]` names the table resident on rank r's own device
    // (`ProgramImplCore::rope_frequency`). Every TextContext built from this core forwards it to
    // its text rope call sites, MTP ones included. All-null is the native constant-table path,
    // bit-for-bit, which is why the default value is the pre-YaRN behavior.
    TpArray<ops::RopeFrequencyOverride> rope_frequency{};
    // One core per non-zero rank, indexed by RANK; slot 0 is unused and slots at or above the
    // width stay null. `peer_at(1)` is the single peer the tp2-only paths (MTP, the graph bridge)
    // read; DFlash loops 1..tp-1. There is deliberately no second, separately-maintained `peer`
    // field, because two copies of the same fact is exactly how a tp2 path silently keeps
    // running at tp4.
    TpArray<const TpPeerCore*> peers{};
    [[nodiscard]] const TpPeerCore* peer_at(std::size_t rank) const noexcept {
        return rank < peers.size() ? peers[rank] : nullptr;
    }
    // Tensor-parallel width (1 = single rank). tp2-only paths (MTP, the graph bridge) still
    // read peer_at(1) directly; the widened DFlash path loops 1..tp-1.
    std::int32_t tp = 1;
};

// Builds the per-rank core array an ExecutionCore is initialised with: rank 1's core at slot 1.
// (A single-rank-per-slot table is what the two-rank paths below need; the program layer widens it
// once every rank has its own core.)
[[nodiscard]] inline TpArray<const TpPeerCore*> peer_lanes(const TpPeerCore* rank_one) {
    TpArray<const TpPeerCore*> out{};
    out[1] = rank_one;
    return out;
}

// The same idea for a per-rank PagedKVCacheView (the MTP prefill's per-sequence window).
[[nodiscard]] inline TpArray<qwen3_6::PagedKVCacheView>
peer_mtp_kv_lanes(const qwen3_6::PagedKVCacheView& rank_one) {
    TpArray<qwen3_6::PagedKVCacheView> out{};
    out[1] = rank_one;
    return out;
}

// ... and for a per-rank pinned ordinary-decode ingress record.
[[nodiscard]] inline TpArray<const qwen3_6::OrdinaryDecodeIngress*>
peer_ingress_lanes(const qwen3_6::OrdinaryDecodeIngress* rank_one) {
    TpArray<const qwen3_6::OrdinaryDecodeIngress*> out{};
    out[1] = rank_one;
    return out;
}

// Assembles the TextContext-side views of every non-zero rank. All slots are empty at tp == 1.
[[nodiscard]] inline TpPeers tp_executions(const ExecutionCore& execution) {
    TpPeers out{};
    for (std::size_t rank = 1; rank < execution.peers.size(); ++rank) {
        const TpPeerCore* peer = execution.peers[rank];
        if (peer == nullptr) { continue; }
        TpExecution lane;
        lane.execution      = peer->execution;
        lane.events         = peer->events;
        lane.device         = peer->device;
        lane.weights        = peer->model;
        lane.work           = peer->work;
        lane.state          = peer->linear_attention;
        lane.io             = peer->io;
        lane.prefill_hidden = peer->prefill_hidden;
        lane.batch_kv       = peer->text_cache;
        lane.batch_mtp_kv   = peer->mtp_cache;
        lane.replay_records = peer->replay_records;
        out[rank]           = lane;
    }
    return out;
}

// Debug-only stage barrier, enabled by NINFER_TP4_MTP_STAGE_SYNC=1. A device fault is STICKY and
// only surfaces at the next synchronize, so the surviving error message always names whatever
// synced last (for a prefill chunk, its very end) and never the kernel that actually went out of
// bounds. These barriers synchronize every rank after one named stage, which turns "illegal memory
// access somewhere in this 400-line path" into "illegal memory access in the stem's KV append".
// Off unless the variable is set, so the shipped path pays one predictable branch per stage.
[[nodiscard]] inline bool mtp_stage_barrier_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TP4_MTP_STAGE_SYNC");
        return value != nullptr && *value != '0';
    }();
    return enabled;
}

inline void mtp_stage_barrier(const ExecutionContext& ec, const char* stage) {
    if (!mtp_stage_barrier_enabled()) { return; }
    for (int rank = 0; rank < ec.tp; ++rank) {
        if (ec.dev[static_cast<std::size_t>(rank)].has_value()) {
            ec.dev[static_cast<std::size_t>(rank)]->synchronize();
        }
    }
    std::fprintf(stderr, "[mtp-stage] %s ok\n", stage);
}

inline void mtp_stage_barrier(const ExecutionCore& core, const char* stage) {
    if (!mtp_stage_barrier_enabled()) { return; }
    core.device.synchronize();
    for (std::size_t rank = 1; rank < core.peers.size(); ++rank) {
        const TpPeerCore* peer = core.peers[rank];
        if (peer != nullptr) { peer->device->synchronize(); }
    }
    std::fprintf(stderr, "[mtp-stage] %s ok\n", stage);
}

struct PrefillContext {
    ExecutionCore execution;
    qwen3_6::PagedKVCacheView text_kv;
    qwen3_6::PagedKVCacheView mtp_kv;
    // Each non-zero rank's per-sequence MTP KV window; empty at tp == 1 and when MTP is off. The
    // text prefill needs no peer twin because it drives the BATCH cache view plus table rows,
    // but the MTP prefill appends and reads through the per-sequence execution view.
    TpArray<qwen3_6::PagedKVCacheView> mtp_kv_peers{};
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t current_state_slot                         = 0;
    std::int32_t rewrite_checkpoint_state_slot              = 0;
    std::uint32_t mtp_proposal_extent                       = 0;
    const qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    qwen3_6::OrdinaryDecodeState& frame;
    const qwen3_6::OrdinaryDecodeIngress& host_ingress;
    qwen3_6::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // Each non-zero rank's own pinned copy of `host_ingress`, with every row's sampling counter
    // pointer nulled (ProgramImplCore::publish_peer_ordinary_ingress). All null at tp1.
    TpArray<qwen3_6::OrdinaryDecodeIngress*> peer_host_ingress{};
};

struct MtpBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache& mtp_cache;
    qwen3_6::MtpDecodeState& frame;
    const qwen3_6::MtpDecodeIngress& host_ingress;
    qwen3_6::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // One pinned MTP ingress record per NON-ZERO rank, indexed by rank (slot 0 unused, slots at
    // or above the width null). Each is byte-for-byte rank 0's record except that every row's
    // `sampling[row].token_counts` names THAT rank's counter lane -- the round is replicated across
    // every rank and `speculative_accept_greedy_drafts` reads and atomically writes that pointer on
    // each of them, so handing a rank another rank's lane is a silent double-increment. See
    // ProgramImplCore::publish_peer_mtp_ingress. All null at tp1.
    TpArray<qwen3_6::MtpDecodeIngress*> peer_host_ingress{};
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    DFlashPersistentState& dflash;
    qwen3_6::DFlashDecodeState& frame;
    const qwen3_6::DFlashDecodeIngress& host_ingress;
    qwen3_6::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // Op-layer execution context (dev[]/tp/comm) for the collective Ops; the ExecutionCore above
    // is the target-layer core and does not name the devices the collectives switch between.
    const ExecutionContext& ec;
    bool greedy_target = false;
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct MtpGqaEnvelopes {
    ops::GqaExecutionEnvelope target_verify;
    ops::GqaExecutionEnvelope batch;
    std::array<ops::GqaExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

struct DFlashEnvelopes {
    ops::SwaContextExecutionEnvelope local;
    ops::GqaContextExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor lanes;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    Tensor drafts;
    Tensor current_extents;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor selected_hidden;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t rewrite_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent);
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target = false);
// Tensor-parallel form: one frame per rank, INDEXED BY RANK. Each rank's frame is its OWN
// identically-shaped view; the acceptance arithmetic is replicated rather than transferred,
// because every one of its inputs is either the ingress record (uploaded to every frame) or the
// gathered logits (bit-identical on every rank). What is NOT replicated is rank 0's bookkeeping:
// the continuation-hidden scatter and the egress transfer stay on rank 0 alone.
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, const TpArray<TargetVerifyFrameView>& frames,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target = false);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier, bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier,
                         bool finalize_at_end);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
// Retained hidden is authoritative on rank 0, including partial MTP commit corrections. Copy it
// into rank 1's prefill scratch only on resume; both streams protect its producer/read lifetime.
[[nodiscard]] TpArray<Tensor> resume_hidden(ExecutionCore& execution, const Tensor& hidden);
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope, DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
