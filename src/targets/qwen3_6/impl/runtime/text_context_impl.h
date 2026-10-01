#include "targets/qwen3_6/impl/runtime/instance.h"

#include <cmath>
#include <cstring>
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "ninfer/types.h" // TpArray, kMaximumDevices

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(const Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::GqaExecutionEnvelope*& slot,
                   const ops::GqaExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::GqaExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

// Binds one per-call control tensor on every non-zero rank for the enclosing scope, restoring the
// previous bindings on exit. `member` names the TextRankBinding field; `values[r]` is rank r's own
// tensor. At tp == 1 there are no peers and this is a no-op.
class ScopedRankTensorBinding {
public:
    ScopedRankTensorBinding(std::int32_t tp, std::array<TextRankBinding, kMaximumDevices>& ranks,
                            const Tensor* TextRankBinding::*member,
                            const std::array<const Tensor*, kMaximumDevices>& values)
        : tp_(tp), ranks_(&ranks), member_(member) {
        for (std::int32_t r = 1; r < tp_; ++r) {
            const auto slot         = static_cast<std::size_t>(r);
            previous_[slot]         = (*ranks_)[slot].*member_;
            (*ranks_)[slot].*member_ = values[slot];
        }
    }

    ScopedRankTensorBinding(const ScopedRankTensorBinding&)            = delete;
    ScopedRankTensorBinding& operator=(const ScopedRankTensorBinding&) = delete;

    ~ScopedRankTensorBinding() {
        for (std::int32_t r = 1; r < tp_; ++r) {
            const auto slot         = static_cast<std::size_t>(r);
            (*ranks_)[slot].*member_ = previous_[slot];
        }
    }

private:
    std::int32_t tp_;
    std::array<TextRankBinding, kMaximumDevices>* ranks_;
    const Tensor* TextRankBinding::*member_;
    std::array<const Tensor*, kMaximumDevices> previous_{};
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(
    DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
    const TpArray<ops::RopeFrequencyOverride>& rope_frequency,
    qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state, qwen3_6::RoundState& io,
    Tensor& prefill_hidden, std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
    qwen3_6::PagedKVCacheView mtp_kv, const qwen3_6::PagedKVCache* batch_text_kv,
    const qwen3_6::PagedKVCache* batch_mtp_kv, const TpPeers& peers)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      rope_frequency_(rope_frequency), batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv) {
    // One lane per non-zero rank, in order. All slots empty is tp == 1, where nothing below changes.
    for (std::int32_t r = 1; r < static_cast<std::int32_t>(kMaximumDevices); ++r) {
        const auto slot = static_cast<std::size_t>(r);
        if (!peers[slot].has_value()) { break; }
        ranks_[slot].execution = *peers[slot];
        tp_count_              = r + 1;
    }
    if (tp_count_ > 1) {
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            const TpExecution& lane = ranks_[static_cast<std::size_t>(r)].execution;
            if (!lane.complete() || lane.execution == nullptr || lane.execution->tp != tp_count_) {
                throw std::invalid_argument("tensor-parallel TextContext binding is incomplete");
            }
            // The pull protocol's event choreography only exists at a width of exactly two.
            if (tp_count_ == 2 && !(lane.events != nullptr && lane.events->live())) {
                throw std::invalid_argument("tensor-parallel TextContext binding is incomplete");
            }
            // Every rank must agree about the MTP KV storage: one rank carrying an MTP window while
            // another does not would have them run different schedules.
            if (mtp_enabled() != (lane.mtp_kv.valid() || lane.batch_mtp_kv != nullptr)) {
                throw std::invalid_argument("tensor-parallel MTP storage disagrees between ranks");
            }
        }
    }
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, state_.slot_count() > 1 ? 1 : 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t current_slot,
                                         std::int32_t rewrite_checkpoint_slot) {
    if (current_slot < 0 || current_slot >= state_.slot_count() || rewrite_checkpoint_slot < 0 ||
        rewrite_checkpoint_slot >= state_.slot_count() || current_slot == rewrite_checkpoint_slot) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_current_slot_            = current_slot;
    linear_state_rewrite_checkpoint_slot_ = rewrite_checkpoint_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    // Every non-zero rank's bindings point into ITS OWN model view, whose sharded extents are
    // already divided by the loader. Norms and the embedding table are replicated, so all ranks
    // bind structurally identical -- but physically distinct, per-device -- objects.
    for (std::int32_t r = 1; r < tp_count_; ++r) {
        TextRankBinding& binding    = ranks_[static_cast<std::size_t>(r)];
        const LoadedModelData& peer = *binding.execution.weights;
        binding.embed               = &peer.token_embedding;
        binding.final_norm          = &peer.final_norm;
        binding.lm_head             = &peer.output_head;
        for (int layer = 0; layer < kCfg.n_layers; ++layer) {
            if (ModelConfig::is_full(layer)) {
                const std::size_t fidx = static_cast<std::size_t>(ModelConfig::full_idx(layer));
                FullLayerW& out        = binding.full[fidx];
                const auto& source     = peer.full_layers[fidx];
                out.input_norm         = &source.input_norm;
                out.projection         = &source.projection;
                out.o_proj             = &source.output;
                out.q_norm             = &source.query_norm;
                out.k_norm             = &source.key_norm;
                out.post_attn_norm     = &source.post_attention_norm;
                out.mlp                = bind_mlp(source.post_mixer);
            } else {
                const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
                GdnLayerW& out         = binding.gdn[gidx];
                const auto& source     = peer.gdn_layers[gidx];
                out.input_norm         = &source.input_norm;
                out.projection         = &source.projection;
                out.conv1d             = &source.convolution;
                out.gdn_norm           = &source.norm;
                out.out_proj           = &source.output;
                out.post_attn_norm     = &source.post_attention_norm;
                out.mlp                = bind_mlp(source.post_mixer);
            }
        }
    }
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            TextRankBinding& binding = ranks_[static_cast<std::size_t>(r)];
            if (!binding.execution.weights->optimized_proposal) {
                throw std::invalid_argument("tensor-parallel peer has no proposal head shard");
            }
            const auto& peer_proposal = *binding.execution.weights->optimized_proposal;
            binding.proposal_head     = &peer_proposal.head;
            // `draft_head_token_ids` is REPLICATED, so this is the peer's own device copy of
            // the whole [131072] map, not a 65536-entry slice. A non-zero rank's copy
            // is DEAD STORAGE in this build: the remap runs where the argmax runs, which is rank
            // 0. It is bound anyway, and its presence checked below, because that check is what
            // proves the loader actually replicated the map rather than sharding it -- 512 KiB
            // against a ~400 MiB draft head, and the alternative is a loader special case whose
            // only effect would be to make the placement asymmetric.
            binding.proposal_head_ids =
                static_cast<const std::int32_t*>(peer_proposal.token_ids.data);
        }
    }

    const auto bind_mtp_weights = [](const auto& source) {
        return MtpW{&source,
                    &source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.final_norm};
    };
    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        mtp_ = bind_mtp_weights(*weights_.mtp);
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            TextRankBinding& binding = ranks_[static_cast<std::size_t>(r)];
            if (!binding.execution.weights->mtp) {
                throw std::invalid_argument("tensor-parallel peer has no MTP weight shard");
            }
            binding.mtp = bind_mtp_weights(*binding.execution.weights->mtp);
        }
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

const MtpW& TextContext::mtp_weights_for(int rank) const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    if (rank == 0) { return mtp_; }
    const MtpW& weights = rank_binding(rank).mtp;
    if (weights.payload == nullptr) {
        throw std::logic_error("tensor-parallel peer MTP weights are unbound");
    }
    return weights;
}

const GdnReplayRecords* TextContext::replay_records_for(int rank) const {
    if (rank == 0) { return replay_records_; }
    if (tp_count_ < 2) { throw std::logic_error("TextContext has no tensor-parallel context"); }
    // The ranks must agree: a peer with no record storage while rank 0 records would fold a
    // stale half of the GDN state on its device and diverge silently from the next round on.
    if ((replay_records_ == nullptr) != (rank_binding(1).execution.replay_records == nullptr)) {
        throw std::logic_error("tensor-parallel replay-record bindings disagree between ranks");
    }
    return rank_binding(rank).execution.replay_records;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, rope_frequency_[0], s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
                           *active_backend_kv_table_rows_, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                    Tensor* final_hidden, Tensor* logits, Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, rope_frequency_[0], s);
        ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, rope_frequency_[0],
                  s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::gqa_attention_cached(qn, last_position, kAttnScale, mtp_kv_.layer_view(0), envelope,
                                  work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, ops::GqaExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, ops::GqaExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_slots,
                                        ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                        Tensor& logits) {
    if (tp2()) {
        ordinary_decode_batch_tp2(ids, cache_positions, rope_positions, kv_table_rows,
                                  linear_state_slots, envelope, hidden, logits);
        return;
    }
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
        ops::linear(hidden, *lm_head_, logits, stream);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_slots,
                                           ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                           Tensor& logits, Tensor& target_tokens, Tap& tap) {
    if (tp2()) {
        throw std::logic_error("use tensor-parallel target_verify_batch overload");
    }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {kCfg.hidden, columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({kCfg.hidden, columns});
        Tensor flat_logits = logits.view({kCfg.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, flat_hidden, stream);
        ops::linear(flat_hidden, *lm_head_, flat_logits, stream);
        ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, rope_frequency_[0], s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
                           kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                           *active_gqa_envelope_, work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows, kAttnScale,
                           batch_text_kv_->batch_layer_view(fidx), *active_gqa_envelope_, work_, a,
                           s);
    }
    ops::sigmoid_mul(gate, a, s);

    Variant::attention_output_projection(a.view({kCfg.q_size, T}), *w.o_proj, x, ph, work_, s);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, s);

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor& conv_states     = state_.conv.at(static_cast<std::size_t>(gidx));
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            Variant::gdn_input_projection_record(projection_input, *w.projection, *w.conv1d,
                                                 conv_states, valid, *active_linear_state_slots_,
                                                 records.conv, query_output, key_output,
                                                 value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_slots_, *active_linear_state_slots_, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        }
    } else {
        const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Tensor qkv      = conv.projected;
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor qkv_c = conv.convolved;
        Tensor conv_state =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qkv_c, s);
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& recurrent_states = state_.recurrent.at(static_cast<std::size_t>(gidx));
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                                          /*normalize_qk=*/true, recurrent_states, valid,
                                          *active_linear_state_slots_, *active_linear_state_slots_,
                                          out_batch, s);
        }
    } else {
        Tensor recurrent_state =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent_state, o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    Variant::gdn_output_projection(on.view({kCfg.value_dim, T}), *w.out_proj, x, ph, work_, s);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, fidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64         = static_cast<std::int64_t>(base);
    const std::int64_t checkpoint_abs = prefill_rewrite_checkpoint_frontier_;
    const bool has_rewrite_checkpoint =
        checkpoint_abs > base64 && checkpoint_abs <= base64 + static_cast<std::int64_t>(T);
    const int checkpoint_rel =
        has_rewrite_checkpoint ? static_cast<int>(checkpoint_abs - base64) : -1;
    const std::int32_t rewrite_checkpoint_slot = linear_state_rewrite_checkpoint_slot_;

    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (checkpoint_rel > 0 && t0 < checkpoint_rel && t0 + len > checkpoint_rel) {
            len = checkpoint_rel - t0;
        }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                for (int j = 0; j < prompt_columns; ++j) {
                    mtp_ids_host[static_cast<std::size_t>(j)] =
                        alignment_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                      static_cast<std::size_t>(j)];
                }
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }

                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                copy_i32(mtp_ids_host.data(), mtp_ids, s);
                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_6::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (checkpoint_rel > 0 && t0 + len == checkpoint_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {kCfg.hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, checkpoint_rel > 0 && t0 + len == checkpoint_rel);
        }

        if (checkpoint_rel > 0 && t0 + len == checkpoint_rel) {
            state_.copy_slot(linear_state_current_slot_, rewrite_checkpoint_slot, s);
        }

        t0 += len;
        break;
    }

    prefill_rewrite_checkpoint_frontier_ = -1;

    ctx_.synchronize();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    if (tp2()) {
        return prefill_impl_tp2(full_ids.subspan(begin, nominal_length), text_prefill,
                                finalize_at_end);
    }
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    if (tp2()) {
        const TextPrefill text_prefill{full_ids, begin};
        return prefill_impl_tp2(full_ids.subspan(begin, nominal_length), text_prefill,
                                finalize_at_end, &sink);
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_6::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    if (tp2()) {
        throw std::logic_error("multimodal prefill has no tensor-parallel path in this build");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}


// =================================================================================================
// tp == 2 forward
// =================================================================================================
//
// SHAPE OF THE SCHEDULE. Every layer runs the same pattern:
//
//   1. a replicated elementwise stage (RMSNorm, RoPE, gating multiply) issued once per rank on
//      that rank's own stream over that rank's own copy of the data;
//   2. a COLUMN-parallel projection, one call driving both ranks, no communication;
//   3. head-local mixing -- attention over this device's 12 of 24 query heads against its own 2 of
//      4 KV heads, GDN over its own 24 of 48 value heads -- again per rank, no communication;
//   4. a ROW-parallel output projection whose single `allreduce_sum` is the ONLY collective in the
//      mixer, and which folds the residual in exactly once (rank 0, pre-reduce);
//   5. the MLP tail, whose `down` projection carries the layer's second and last collective.
//
// Two all-reduces per layer, 128 per token for the 64-layer model.
//
// STREAMS AND EVENTS. There is no host synchronization anywhere inside the layer loop. Rank r's
// work is enqueued on `ec.dev[r]->stream`; the collectives' own four-event choreography
// (`inputs_ready` / `pull_done`, include/ninfer/ops/allreduce.h) is what orders the two streams
// against each other, both within a call and across calls. The single `PeerEvents` instance lives
// in the Program, is created once, and is reused by every collective of every layer. Kernel
// launches go to the CURRENT device, so every per-rank issue runs inside `for_each_rank`, which
// sets and restores it.
//
// WHY THE TWO HALVES STAY IN LOCKSTEP. The residual is the only tensor both ranks must agree on
// bit-for-bit, and they do: `allreduce_sum` leaves rank 0 holding `p0 + p1` and rank 1 holding
// `p1 + p0`, and IEEE addition is commutative, so both store the identical BF16. Everything
// derived from the residual -- the next layer's norm, the GDN convolution and recurrent state, the
// KV pages -- is therefore identical on both devices with no further agreement protocol. Prefill
// chunk boundaries are driven by the token count alone, so both ranks also see the same chunks.

const ExecutionContext& TextContext::ec() const {
    if (tp_count_ < 2) { throw std::logic_error("TextContext has no tensor-parallel context"); }
    return *ranks_[1].execution.execution;
}

const ops::PeerEvents& TextContext::peer_events() const {
    if (tp_count_ < 2 || ranks_[1].execution.events == nullptr) {
        throw std::logic_error("TextContext has no tensor-parallel PeerEvents");
    }
    return *ranks_[1].execution.events;
}

TpArray<WorkspaceArena*> TextContext::workspaces() const {
    TpArray<WorkspaceArena*> arenas{};
    arenas[0] = &work_;
    for (std::int32_t r = 1; r < tp_count_; ++r) {
        arenas[static_cast<std::size_t>(r)] = ranks_[static_cast<std::size_t>(r)].execution.work;
    }
    return arenas;
}

TextContext::RankScopes TextContext::rank_scopes() const {
    RankScopes scopes;
    const TpArray<WorkspaceArena*> arenas = workspaces();
    for (std::int32_t r = 0; r < tp_count_; ++r) {
        const auto slot = static_cast<std::size_t>(r);
        scopes[slot].emplace(arenas[slot]->scope());
    }
    return scopes;
}

void TextContext::synchronize_all() const {
    ctx_.synchronize();
    for (std::int32_t r = 1; r < tp_count_; ++r) {
        ranks_[static_cast<std::size_t>(r)].execution.device->synchronize();
    }
}

// --- per-rank shard views ------------------------------------------------------------------------
//
// Every one of these mirrors rank 0's own `*_` member at rank r. They exist so an Op that takes a
// TpArray gets a complete argument list at any width; nothing below is allowed to assume two ranks.

TpArray<const Weight*> TextContext::embed_weights() const {
    TpArray<const Weight*> out{};
    out[0] = embed_;
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = ranks_[r].embed; }
    return out;
}

TpArray<const Tensor*> TextContext::final_norms() const {
    TpArray<const Tensor*> out{};
    out[0] = final_norm_;
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = ranks_[r].final_norm; }
    return out;
}

TpArray<Weight> TextContext::lm_heads() const {
    TpArray<Weight> out{};
    out[0] = *lm_head_;
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = *ranks_[r].lm_head; }
    return out;
}

TpArray<const FullLayerW*> TextContext::full_layers(int index) const {
    const auto idx = static_cast<std::size_t>(index);
    TpArray<const FullLayerW*> out{};
    out[0] = &full_[idx];
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = &ranks_[r].full[idx]; }
    return out;
}

TpArray<const GdnLayerW*> TextContext::gdn_layers(int index) const {
    const auto idx = static_cast<std::size_t>(index);
    TpArray<const GdnLayerW*> out{};
    out[0] = &gdn_[idx];
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = &ranks_[r].gdn[idx]; }
    return out;
}

TpArray<const FullAttentionProjectionWeights*>
TextContext::attention_projections(int index) const {
    const TpArray<const FullLayerW*> layers = full_layers(index);
    TpArray<const FullAttentionProjectionWeights*> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) { out[r] = layers[r]->projection; }
    return out;
}

TpArray<Weight> TextContext::attention_output_weights(int index) const {
    const TpArray<const FullLayerW*> layers = full_layers(index);
    TpArray<Weight> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) { out[r] = *layers[r]->o_proj; }
    return out;
}

TpArray<const GdnProjectionWeights*> TextContext::gdn_projections(int index) const {
    const TpArray<const GdnLayerW*> layers = gdn_layers(index);
    TpArray<const GdnProjectionWeights*> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) { out[r] = layers[r]->projection; }
    return out;
}

TpArray<Weight> TextContext::gdn_output_weights(int index) const {
    const TpArray<const GdnLayerW*> layers = gdn_layers(index);
    TpArray<Weight> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) { out[r] = *layers[r]->out_proj; }
    return out;
}

TpArray<const MlpWeights*> TextContext::full_post_mixers(int index) const {
    const TpArray<const FullLayerW*> layers = full_layers(index);
    TpArray<const MlpWeights*> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) { out[r] = layers[r]->mlp.payload; }
    return out;
}

TpArray<const MlpWeights*> TextContext::gdn_post_mixers(int index) const {
    const TpArray<const GdnLayerW*> layers = gdn_layers(index);
    TpArray<const MlpWeights*> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) { out[r] = layers[r]->mlp.payload; }
    return out;
}

TpArray<Weight> TextContext::proposal_heads() const {
    TpArray<Weight> out{};
    out[0] = *proposal_head_;
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = *ranks_[r].proposal_head; }
    return out;
}

TpArray<const std::int32_t*> TextContext::proposal_head_id_maps() const {
    TpArray<const std::int32_t*> out{};
    out[0] = proposal_head_ids_;
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = ranks_[r].proposal_head_ids; }
    return out;
}

TpArray<const typename Variant::MtpAttentionProjectionWeights*>
TextContext::mtp_attention_weights() const {
    TpArray<const typename Variant::MtpAttentionProjectionWeights*> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) {
        out[static_cast<std::size_t>(r)] = &mtp_weights_for(r).payload->attention;
    }
    return out;
}

TpArray<Weight> TextContext::mtp_output_weights() const {
    TpArray<Weight> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) {
        out[static_cast<std::size_t>(r)] = *mtp_weights_for(r).o_proj;
    }
    return out;
}

TpArray<const typename Variant::MtpPostMixerWeights*> TextContext::mtp_post_mixer_weights() const {
    TpArray<const typename Variant::MtpPostMixerWeights*> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) {
        out[static_cast<std::size_t>(r)] = &mtp_weights_for(r).payload->post_mixer;
    }
    return out;
}

TpArray<Tensor> TextContext::attention_flat(const TpArray<Tensor>& a, std::int32_t tokens) const {
    TpArray<Tensor> out{};
    for (std::int32_t r = 0; r < tp_count_; ++r) {
        const auto slot = static_cast<std::size_t>(r);
        out[slot]        = a[slot].view({shard_q_size(), tokens});
    }
    return out;
}

TpArray<qwen3_6::RoundState*> TextContext::rank_ios() const {
    TpArray<qwen3_6::RoundState*> out{};
    out[0] = &io_;
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = ranks_[r].execution.io; }
    return out;
}

TpArray<LinearAttentionStatePool*> TextContext::rank_states() const {
    TpArray<LinearAttentionStatePool*> out{};
    out[0] = &state_;
    for (std::int32_t r = 1; r < tp_count_; ++r) { out[r] = ranks_[r].execution.state; }
    return out;
}

const Tensor& TextContext::rank_cache_positions(int rank) const {
    if (rank == 0) {
        return active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    }
    const Tensor* binding = ranks_[static_cast<std::size_t>(rank)].cache_positions;
    if (binding == nullptr) {
        throw std::logic_error("tensor-parallel peer cache positions are unbound");
    }
    return *binding;
}

const Tensor& TextContext::rank_rope_positions(int rank) const {
    if (rank == 0) {
        return active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    }
    const Tensor* binding = ranks_[static_cast<std::size_t>(rank)].rope_positions;
    if (binding == nullptr) {
        throw std::logic_error("tensor-parallel peer RoPE positions are unbound");
    }
    return *binding;
}

const Tensor& TextContext::rank_kv_table_rows(int rank) const {
    if (rank == 0) {
        return active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    }
    const Tensor* binding = ranks_[static_cast<std::size_t>(rank)].kv_table_rows;
    if (binding == nullptr) {
        throw std::logic_error("tensor-parallel peer KV table rows are unbound");
    }
    return *binding;
}

Tensor TextContext::rank_valid_columns(int rank) const {
    // Unlike its siblings, an ABSENT binding is legal here: the prefill and ordinary-decode paths
    // never bind valid columns, and every consumer reads an empty Tensor as "every column counts".
    // What is not legal is the ranks DISAGREEING -- a peer binding present while rank 0's is absent
    // (or the reverse) would have the devices mask different columns and diverge silently, which is
    // exactly the trap the sibling accessors' throws exist to prevent.
    if ((active_valid_columns_ == nullptr) != (ranks_[1].valid_columns == nullptr)) {
        throw std::logic_error("tensor-parallel valid-column bindings disagree between ranks");
    }
    for (std::int32_t r = 2; r < tp_count_; ++r) {
        if ((ranks_[r].valid_columns == nullptr) != (ranks_[1].valid_columns == nullptr)) {
            throw std::logic_error("tensor-parallel valid-column bindings disagree between ranks");
        }
    }
    if (rank == 0) {
        return active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
    }
    const Tensor* binding = ranks_[static_cast<std::size_t>(rank)].valid_columns;
    return binding != nullptr ? *binding : Tensor{};
}

const Tensor& TextContext::rank_backend_kv_table_rows(int rank) const {
    if (rank == 0) {
        if (active_backend_kv_table_rows_ == nullptr) {
            throw std::logic_error("MTP backend KV table rows are unbound");
        }
        return *active_backend_kv_table_rows_;
    }
    const Tensor* binding = ranks_[static_cast<std::size_t>(rank)].backend_kv_table_rows;
    if (binding == nullptr) {
        throw std::logic_error("tensor-parallel peer backend KV table rows are unbound");
    }
    return *binding;
}

const Tensor& TextContext::rank_linear_state_slots(int rank) const {
    if (rank == 0) {
        if (active_linear_state_slots_ == nullptr) {
            throw std::logic_error("Linear Attention state slots are unbound");
        }
        return *active_linear_state_slots_;
    }
    const Tensor* binding = ranks_[static_cast<std::size_t>(rank)].linear_state_slots;
    if (binding == nullptr) {
        throw std::logic_error("tensor-parallel peer state slots are unbound");
    }
    return *binding;
}

void TextContext::attn_mix_tp2(TpArray<Tensor>& x, int fidx, Phase ph,
                               const TpArray<Tensor>& staging) {
    const ExecutionContext& execution = ec();
    const int T                       = x[0].ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }
    const TpArray<const FullLayerW*> w = full_layers(fidx);
    const TpArray<WorkspaceArena*> ws  = workspaces();

    TpArray<Tensor> h;
    TpArray<Tensor> q;
    TpArray<Tensor> gate;
    TpArray<Tensor> k;
    TpArray<Tensor> v;
    TpArray<Tensor> q_flat;
    TpArray<Tensor> gate_flat;
    TpArray<Tensor> k_flat;
    TpArray<Tensor> v_flat;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        const auto projection =
            workspace_recipe::text_attention_projection<TextConfig>(*ws[r], T,
                                                                    ec().tp);
        h[r]         = projection.hidden;
        q[r]         = projection.query.view({kCfg.head_dim, shard_q_heads(), T});
        gate[r]      = projection.gate.view({kCfg.head_dim, shard_q_heads(), T});
        k[r]         = projection.key.view({kCfg.head_dim, shard_kv_heads(), T});
        v[r]         = projection.value.view({kCfg.head_dim, shard_kv_heads(), T});
        q_flat[r]    = q[r].view({shard_q_size(), T});
        gate_flat[r] = gate[r].view({shard_q_size(), T});
        k_flat[r]    = k[r].view({shard_kv_size(), T});
        v_flat[r]    = v[r].view({shard_kv_size(), T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], *w[r]->input_norm, kCfg.rms_eps, true, h[r], stream_for(rank));
    });
    Variant::attention_projection(h, attention_projections(fidx), q_flat, gate_flat, k_flat, v_flat,
                                  ph, ws, execution);

    TpArray<Tensor> qn;
    TpArray<Tensor> kn;
    TpArray<Tensor> a;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        const auto results =
            workspace_recipe::text_attention_results<TextConfig>(*ws[r], T, ec().tp);
        qn[r] = results.normalized_query.view({kCfg.head_dim, shard_q_heads(), T});
        kn[r] = results.normalized_key.view({kCfg.head_dim, shard_kv_heads(), T});
        a[r]  = results.attention.view({kCfg.head_dim, shard_q_heads(), T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t s      = stream_for(rank);
        const Tensor& cache = rank_cache_positions(rank);
        const Tensor& rope  = rank_rope_positions(rank);
        ops::rmsnorm(q[r], *w[r]->q_norm, kCfg.rms_eps, true, qn[r], s);
        ops::rmsnorm(k[r], *w[r]->k_norm, kCfg.rms_eps, true, kn[r], s);
        Tensor rope_for_op = active_sequence_batch_ != 0 ? rope.view({T}) : rope;
        ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn[r], kn[r],
                  rope_frequency_[r], s);

        const qwen3_6::PagedKVCache& pages = rank_batch_kv(rank);
        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            if (width <= 0 || width * active_sequence_batch_ != T) {
                throw std::logic_error(
                    "Text sequence batch binding does not match aggregate columns");
            }
            Tensor q_batch =
                qn[r].view({kCfg.head_dim, shard_q_heads(), width, active_sequence_batch_});
            Tensor k_batch =
                kn[r].view({kCfg.head_dim, shard_kv_heads(), width, active_sequence_batch_});
            Tensor v_batch =
                v[r].view({kCfg.head_dim, shard_kv_heads(), width, active_sequence_batch_});
            Tensor a_batch =
                a[r].view({kCfg.head_dim, shard_q_heads(), width, active_sequence_batch_});
            Tensor position_batch = cache.view({width, active_sequence_batch_});
            ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, rank_valid_columns(rank),
                               rank_kv_table_rows(rank), kAttnScale, pages.batch_layer_view(fidx),
                               *active_gqa_envelope_, *ws[r], a_batch, s);
        } else {
            ops::gqa_attention(qn[r], kn[r], v[r], cache, Tensor{}, rank_kv_table_rows(rank),
                               kAttnScale, pages.batch_layer_view(fidx), *active_gqa_envelope_,
                               *ws[r], a[r], s);
        }
        ops::sigmoid_mul(gate[r], a[r], s);
    });

    TpArray<Tensor> a_flat;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        a_flat[r] = a[r].view({shard_q_size(), T});
    }
    Variant::attention_output_projection(a_flat, attention_output_weights(fidx), x, staging, ph, ws,
                                         execution, peer_events());
}

void TextContext::gdn_mix_tp2(TpArray<Tensor>& x, int gidx, Phase ph,
                              const TpArray<Tensor>& staging) {
    const ExecutionContext& execution = ec();
    const int T                       = x[0].ne[1];
    const TpArray<const GdnLayerW*> w = gdn_layers(gidx);
    const TpArray<WorkspaceArena*> ws = workspaces();

    TpArray<Tensor> h;
    TpArray<Tensor> g;
    TpArray<Tensor> beta;
    TpArray<Tensor> z;
    TpArray<Tensor> qc;
    TpArray<Tensor> kc;
    TpArray<Tensor> vc;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        const auto control = workspace_recipe::gdn_control<TextConfig>(*ws[r], T,
                                                                      ec().tp);
        h[r]               = control.hidden;
        g[r]               = control.g;
        beta[r]            = control.beta;
        const auto projection =
            workspace_recipe::gdn_projection<TextConfig>(*ws[r], T, ec().tp);
        z[r]  = projection.output_gate.view({kCfg.gdn_v_dim, shard_gdn_v_heads(), T});
        qc[r] = projection.query;
        kc[r] = projection.key;
        vc[r] = projection.value;
    }
    // The tp1 leaf fuses this RMSNorm into the gating GEMM. There is no split form of the fused
    // kernel and no need for one: the norm is replicated elementwise work over the full-width
    // residual, so it runs per rank and the gating GEMM is the column-parallel leaf.
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], *w[r]->input_norm, kCfg.rms_eps, true, h[r], stream_for(rank));
    });
    Variant::gdn_control_projection(h, gdn_projections(gidx), g, beta, ws, execution);

    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0) {
            throw std::logic_error("Verify GDN requires an explicit sequence batch");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        TpArray<Tensor> projection_input;
        TpArray<Tensor> query_output;
        TpArray<Tensor> key_output;
        TpArray<Tensor> value_output;
        TpArray<Tensor> gate_output;
        TpArray<Tensor> conv_weight;
        TpArray<Tensor> conv_states;
        TpArray<Tensor> valid;
        TpArray<Tensor> slots;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            const int rank      = static_cast<int>(r);
            projection_input[r] = h[r].view({kCfg.hidden, width, active_sequence_batch_});
            query_output[r]     = qc[r].view({shard_key_dim(), width, active_sequence_batch_});
            key_output[r]       = kc[r].view({shard_key_dim(), width, active_sequence_batch_});
            value_output[r]     = vc[r].view({shard_value_dim(), width, active_sequence_batch_});
            gate_output[r]      = z[r].view({shard_value_dim(), width, active_sequence_batch_});
            conv_weight[r]      = *w[r]->conv1d;
            conv_states[r]      = state_for(rank).conv.at(static_cast<std::size_t>(gidx));
            valid[r]            = rank_valid_columns(rank);
            slots[r]            = rank_linear_state_slots(rank);
        }
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            // The speculative verify round records this device's own head/channel shard instead
            // of committing it: the round's accepted prefix is only known after the target's
            // argmax, so the recurrent and conv updates are replayed afterwards by
            // ops::gdn_replay_fold on EACH device, at the registered
            // FoldGeometry<48, 8, 24, 5120>. Both ranks record the same
            // columns of the same rows -- only the head range differs -- so the two folds commit
            // the same accepted prefix without any agreement protocol.
            TpArray<Tensor> conv_record;
            for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
                const GdnReplayRecords* records = replay_records_for(static_cast<int>(r));
                if (records == nullptr) {
                    throw std::logic_error("Replay-record GDN has no record storage");
                }
                conv_record[r] =
                    records->layer(gidx, active_sequence_batch_).conv;
            }
            Variant::gdn_input_projection_record(projection_input, gdn_projections(gidx),
                                                 conv_weight, conv_states, valid, slots,
                                                 conv_record, query_output, key_output,
                                                 value_output, gate_output, ph, ws, execution);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, gdn_projections(gidx), conv_weight, conv_states, valid, slots,
                slots, query_output, key_output, value_output, gate_output, ph, ws, execution);
        }
    } else {
        TpArray<Tensor> qkv;
        TpArray<Tensor> qkv_c;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            const auto conv =
                workspace_recipe::gdn_prefill_conv<TextConfig>(*ws[r], T, ec().tp);
            qkv[r]   = conv.projected;
            qkv_c[r] = conv.convolved;
        }
        Variant::gdn_input_projection(h, gdn_projections(gidx), qkv, z, ph, ws, execution);
        for_each_rank(execution, [&](int rank) {
            const auto r      = static_cast<std::size_t>(rank);
            cudaStream_t s    = stream_for(rank);
            Tensor conv_state = state_for(rank).conv_slot(static_cast<std::uint32_t>(gidx),
                                                          linear_state_current_slot_);
            ops::causal_conv1d_silu(qkv[r], *w[r]->conv1d, conv_state, conv_state, qkv_c[r], s);
            // Shard-local section offsets: this device's convolved block is its own
            // q(1024) | k(1024) | v(3072), not the model's 2048 | 2048 | 6144.
            ops::extract_bf16_columns(qkv_c[r], 0, qc[r], s);
            ops::extract_bf16_columns(qkv_c[r], shard_key_dim(), kc[r], s);
            ops::extract_bf16_columns(qkv_c[r], 2 * shard_key_dim(), vc[r], s);
        });
    }

    TpArray<Tensor> o;
    TpArray<Tensor> on;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        o[r] = workspace_recipe::gdn_recurrent_output<TextConfig>(*ws[r], T, ec().tp)
                   .view({kCfg.gdn_v_dim, shard_gdn_v_heads(), T});
        on[r] = workspace_recipe::gdn_normalized_output<TextConfig>(*ws[r], T, ec().tp)
                    .view({kCfg.gdn_v_dim, shard_gdn_v_heads(), T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r       = static_cast<std::size_t>(rank);
        cudaStream_t s     = stream_for(rank);
        Tensor q_recurrent = qc[r].view({kCfg.gdn_k_dim, shard_gdn_k_heads(), T});
        Tensor k_recurrent = kc[r].view({kCfg.gdn_k_dim, shard_gdn_k_heads(), T});
        Tensor vv          = vc[r].view({kCfg.gdn_v_dim, shard_gdn_v_heads(), T});
        if (ph == Phase::Verify) {
            Tensor& recurrent_states = state_for(rank).recurrent.at(static_cast<std::size_t>(gidx));
            const std::int32_t width = active_sequence_width_;
            Tensor q_batch =
                q_recurrent.view({kCfg.gdn_k_dim, shard_gdn_k_heads(), width, active_sequence_batch_});
            Tensor k_batch =
                k_recurrent.view({kCfg.gdn_k_dim, shard_gdn_k_heads(), width, active_sequence_batch_});
            Tensor v_batch =
                vv.view({kCfg.gdn_v_dim, shard_gdn_v_heads(), width, active_sequence_batch_});
            Tensor g_batch    = g[r].view({shard_gdn_v_heads(), width, active_sequence_batch_});
            Tensor beta_batch = beta[r].view({shard_gdn_v_heads(), width, active_sequence_batch_});
            Tensor out_batch =
                o[r].view({kCfg.gdn_v_dim, shard_gdn_v_heads(), width, active_sequence_batch_});
            const Tensor valid = rank_valid_columns(rank);
            const Tensor slots = rank_linear_state_slots(rank);
            if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
                GdnReplayRecordLayer records =
                    replay_records_for(rank)->layer(gidx, active_sequence_batch_);
                ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                                   kGdnScale, recurrent_states, valid, slots,
                                                   records.key, records.value, records.gate,
                                                   out_batch, s);
            } else {
                ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                              kGdnScale,
                                              /*normalize_qk=*/true, recurrent_states, valid,
                                              slots, slots, out_batch, s);
            }
        } else {
            Tensor recurrent_state = state_for(rank).recurrent_slot(
                static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
            ops::gated_delta_net(q_recurrent, k_recurrent, vv, g[r], beta[r], kGdnScale,
                                 /*normalize_qk=*/true, *ws[r], recurrent_state, o[r], s);
        }
        // `gdn_norm` is the per-head-DIMENSION gain {128}: replicated, so each rank applies the
        // whole weight over its own 24 value heads.
        ops::gated_rmsnorm(o[r], *w[r]->gdn_norm, z[r], kCfg.rms_eps, on[r], s);
    });

    TpArray<Tensor> on_flat;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        on_flat[r] = on[r].view({shard_value_dim(), T});
    }
    Variant::gdn_output_projection(on_flat, gdn_output_weights(gidx), x, staging, ph, ws, execution,
                                   peer_events());
}

void TextContext::mlp_tail_tp2(TpArray<Tensor>& x, int index, bool full_attention, Phase ph,
                               const TpArray<Tensor>& staging) {
    const ExecutionContext& execution     = ec();
    const int T                           = x[0].ne[1];
    const TpArray<WorkspaceArena*> ws     = workspaces();
    const TpArray<const FullLayerW*> full = full_layers(index);
    const TpArray<const GdnLayerW*> gdn   = gdn_layers(index);
    TpArray<const Tensor*> norm;
    for (std::int32_t r = 0; r < execution.tp; ++r) {
        norm[r] = full_attention ? full[r]->post_attn_norm : gdn[r]->post_attn_norm;
    }
    TpArray<Tensor> h;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        h[r] = workspace_recipe::post_mixer_hidden<TextConfig>(*ws[r], T);
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], *norm[r], kCfg.rms_eps, true, h[r], stream_for(rank));
    });
    const TpArray<const MlpWeights*> mixers =
        full_attention ? full_post_mixers(index) : gdn_post_mixers(index);
    Variant::post_mixer(h, mixers, x, staging, ph, ws, execution, peer_events());
}
void TextContext::run_layers_tp2(TpArray<Tensor>& x, Phase ph,
                                 const TpArray<Tensor>& staging,
                                 DFlashFeatureSink* dflash_sink) {
    if (dflash_sink != nullptr) { dflash_sink->begin(x[0]); }
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const auto fidx = static_cast<std::size_t>(ModelConfig::full_idx(layer));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto scopes = rank_scopes();
                attn_mix_tp2(x, static_cast<int>(fidx), ph, staging);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto scopes = rank_scopes();
                mlp_tail_tp2(x, static_cast<int>(fidx), true, ph, staging);
            }
        } else {
            const auto gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto scopes = rank_scopes();
                gdn_mix_tp2(x, static_cast<int>(gidx), ph, staging);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto scopes = rank_scopes();
                mlp_tail_tp2(x, static_cast<int>(gidx), false, ph, staging);
            }
        }
        if (dflash_sink != nullptr) {
            dflash_sink->capture_layer(layer, x[0], stream_for(0));
        }
    }
    if (dflash_sink != nullptr) {
        dflash_sink->capture_positions(rank_cache_positions(0), stream_for(0));
    }
}

void TextContext::target_logits(const TpArray<Tensor>& hidden,
                                const TpArray<Tensor>& logits) {
    if (!tp2()) { throw std::logic_error("tensor-parallel target logits require a peer"); }
    const std::int32_t columns = hidden[0].ne[1];
    if (columns <= 0) {
        throw std::invalid_argument("target logits require at least one hidden column");
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, columns},
                             "target logits hidden");
    }
    logits_tp2(hidden, logits);
}

void TextContext::logits_tp2(const TpArray<Tensor>& hidden,
                             const TpArray<Tensor>& logits) {
    const ExecutionContext& execution = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const std::int32_t columns        = hidden[0].ne[1];
    // Every destination is caller-supplied because there is no single "the logits buffer": the
    // prefill path writes RoundState's one-column scalar logits, the decode path writes the
    // ordinary frame's [vocab, batch] logits. Validating all of them, rather than reaching for one
    // here, is what keeps a non-zero rank's write in bounds -- an undersized destination is an
    // out-of-bounds write on ANOTHER device, which nothing local would notice.
    for (std::int32_t r = 0; r < execution.tp; ++r) {
        const Tensor& destination = logits[r];
        if (destination.dtype != DType::BF16 || destination.ne[0] != kCfg.vocab ||
            destination.ne[1] != columns || destination.ne[2] != 1 || destination.ne[3] != 1 ||
            !destination.is_contiguous() || destination.data == nullptr) {
            throw std::logic_error(
                "tensor-parallel logits destination does not match the vocabulary");
        }
    }
    auto scope_0 = work_.scope();
    auto scope_1 = rank_scopes();
    TpArray<Tensor> part;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        part[r] = ws[r]->alloc(DType::BF16, {shard_vocab(), columns});
    }
    ops::linear_column_parallel(hidden, lm_heads(), part, execution);

    // `allgather_rows` gathers along ne[1] and the vocabulary is ne[0], so the gather runs one
    // column at a time -- which needs no transpose: one column of a [V, C] BF16 matrix is a
    // contiguous V-element run, and viewed as [1, V] that is exactly the Op's [row length 1,
    // row count V] layout. C is 1 in prefill and the decode batch size (at most 8) otherwise.
    for (std::int32_t column = 0; column < columns; ++column) {
        TpArray<Tensor> piece;
        TpArray<Tensor> whole;
        for (std::int32_t r = 0; r < execution.tp; ++r) {
            piece[r] = part[r].slice(1, column, 1).view({1, shard_vocab()});
            whole[r] = logits[r].slice(1, column, 1).view({1, kCfg.vocab});
        }
        ops::allgather_rows(whole, piece, execution, peer_events());
    }
}

void TextContext::reset_peer_workspaces() const {
    for (std::int32_t r = 1; r < tp_count_; ++r) {
        ranks_[static_cast<std::size_t>(r)].execution.work->reset();
    }
}

PrefillChunkResult TextContext::prefill_impl_tp2(std::span<const int> ids,
                                                 const TextPrefill& text_prefill,
                                                 bool finalize_at_end,
                                                 DFlashFeatureSink* dflash_sink) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    // The same chunk-belongs-to-this-prompt check the tp1 path makes: this context was built for
    // one cache base, and a chunk that starts somewhere else would append at the wrong positions.
    if (text_kv_base_ != text_prefill.begin ||
        text_prefill.token_ids.size() < static_cast<std::size_t>(text_kv_base_) + ids.size()) {
        throw std::invalid_argument("text prefill chunk does not match its full prompt");
    }
    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const int T                             = static_cast<int>(ids.size());
    const int chunk                         = static_cast<int>(prefill_chunk_);
    // Prefix-append prefill continues an existing cache, so positions are absolute and can leave
    // int32 even when the chunk itself is small.
    if (static_cast<std::uint64_t>(text_kv_base_) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(text_kv_base_);
    // tp1 zeroes this only when starting a fresh cache, and otherwise carries whatever a
    // multimodal chunk set. tp2 has no multimodal path, so a nonzero delta here means a caller
    // assumption this path does not implement -- say so instead of silently prefilling with 0.
    if (rope_delta_ != 0) {
        throw std::logic_error("tensor-parallel prefill does not support a nonzero RoPE delta");
    }
    for_each_rank(execution, [&](int rank) {
        ops::set_i32_scalar(io_for(rank).rope_delta, 0, stream_for(rank));
    });

    const std::int64_t base64         = static_cast<std::int64_t>(text_kv_base_);
    const std::int64_t checkpoint_abs = prefill_rewrite_checkpoint_frontier_;
    const bool has_rewrite_checkpoint =
        checkpoint_abs > base64 && checkpoint_abs <= base64 + static_cast<std::int64_t>(T);
    const int checkpoint_rel =
        has_rewrite_checkpoint ? static_cast<int>(checkpoint_abs - base64) : -1;

    int len = std::min(chunk, T);
    if (checkpoint_rel > 0 && len > checkpoint_rel) { len = checkpoint_rel; }
    const bool is_last                = finalize_at_end && len == T;
    const bool prepare_mtp_prompt     = mtp_enabled() && io_.mtp.has_value();
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    work_.reset();
    reset_peer_workspaces();
    {
        auto scope_0 = work_.scope();
        auto scope_1 = rank_scopes();
        TpArray<Tensor> ids_device;
        TpArray<Tensor> positions;
        TpArray<Tensor> x;
        TpArray<Tensor> staging;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            const auto roots = workspace_recipe::text_prefill_roots<TextConfig>(*ws[r], len, 0, 0);
            ids_device[r]    = roots.ids;
            positions[r]     = roots.positions;
            x[r]             = roots.residual;
            staging[r]       = ws[r]->alloc(DType::BF16, {kCfg.hidden, len});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r   = static_cast<std::size_t>(rank);
            cudaStream_t s = stream_for(rank);
            copy_i32(ids.data(), ids_device[r], s);
            ops::fill_i32_positions(positions[r], base_i, s);
            ops::embedding(ids_device[r], rank_embed(rank), x[r], s);
        });

        std::array<const Tensor*, kMaximumDevices> peer_cache_values{};
        std::array<const Tensor*, kMaximumDevices> peer_row_values{};
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            peer_cache_values[static_cast<std::size_t>(r)] = &positions[r];
            peer_row_values[static_cast<std::size_t>(r)] =
                &ranks_[static_cast<std::size_t>(r)].execution.io->text_kv_table_row;
        }
        ScopedRankTensorBinding peer_cache(tp_count_, ranks_, &TextRankBinding::cache_positions,
                                           peer_cache_values);
        ScopedRankTensorBinding peer_rope(tp_count_, ranks_, &TextRankBinding::rope_positions,
                                          peer_cache_values);
        ScopedRankTensorBinding peer_rows(tp_count_, ranks_, &TextRankBinding::kv_table_rows,
                                          peer_row_values);
        ScopedPositions scoped_cache(active_cache_positions_, positions[0]);
        ScopedPositions scoped_rope(active_rope_positions_, positions[0]);
        const auto visible = static_cast<std::uint32_t>(base_i + len);
        const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
        ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

        run_layers_tp2(x, Phase::Prefill, staging, dflash_sink);

        TpArray<Tensor> xf;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            const int rank = static_cast<int>(r);
            const Tensor& retention = rank == 0 ? prefill_hidden_ : rank_prefill_hidden(rank);
            xf[r] = retention.data != nullptr ? matrix_window(retention, len)
                                              : ws[r]->alloc(DType::BF16, {kCfg.hidden, len});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], rank_final_norm(rank), kCfg.rms_eps, true,
                         xf[r], stream_for(rank));
        });

        if (is_last) {
            TpArray<Tensor> last;
            TpArray<Tensor> chunk_logits;
            for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
                const int rank  = static_cast<int>(r);
                last[r]         = xf[r].slice(1, len - 1, 1);
                chunk_logits[r] = matrix_window(io_for(rank).logits, 1);
            }
            logits_tp2(last, chunk_logits);
            // Sampling belongs to rank 0 alone: it consumes the reconstructed FULL logits and
            // writes the single committed token.
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(ctx_.device));
            ops::set_i32_scalar(io_.pos, base_i + T, ctx_.stream);
            ops::set_i32_scalar(io_.rope_pos, base_i + T, ctx_.stream);
            if (sampling_config_ != nullptr) {
                ops::sample(chunk_logits[0], io_.token, kCfg.token_domain, sampling_config_,
                            io_.pos, ops::kSamplePurposePrefill, work_, ctx_.stream);
            } else {
                ops::argmax(chunk_logits[0], io_.token, kCfg.token_domain, ctx_.stream);
            }
        }


        // MTP prompt alignment, chunk by chunk, exactly as the tp1 path drives it: the MTP head
        // consumes the SHIFTED token stream against the text model's own final hidden, so its KV
        // is built one column behind the target's. Only rank 0 needs the shifted ids (its fc
        // shard is the embedding half); rank 1 works from its own copy of the final hidden.
        if (prepare_mtp_prompt) {
            for (int rank = 1; rank < tp_count_; ++rank) {
                if (!io_for(rank).mtp.has_value()) {
                    throw std::logic_error(
                        "tensor-parallel MTP prefill requires an MTP frame on every rank");
                }
            }
            const auto alignment_tokens = static_cast<std::uint32_t>(text_prefill.token_ids.size());
            const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                alignment_tokens, text_kv_base_, static_cast<std::uint32_t>(len));
            std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
            const int prompt_columns =
                len - static_cast<int>(mtp_window.final_column_uses_generated_token);
            for (int j = 0; j < prompt_columns; ++j) {
                mtp_ids_host[static_cast<std::size_t>(j)] =
                    text_prefill
                        .token_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                   static_cast<std::size_t>(j)];
            }
            Tensor mtp_ids = ws[0]->alloc(DType::I32, {len});
            {
                const CurrentDevice restore;
                CUDA_CHECK(cudaSetDevice(ctx_.device));
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }
                copy_i32(mtp_ids_host.data(), mtp_ids, ctx_.stream);
            }

            TpArray<Tensor> ar_hidden;
            TpArray<Tensor> mtp_logits;
            for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
                const int rank = static_cast<int>(r);
                ar_hidden[r]   = io_for(rank).mtp->ar_hidden;
                mtp_logits[r]  = matrix_window(io_for(rank).logits, 1);
            }
            if (is_last && mtp_proposal_extent_ != 0) {
                if (mtp_proposal_extent_ >
                    static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
                    throw std::logic_error("MTP proposal extent exceeds the configured window");
                }
                Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                mtp_prefill_chunk_tp2(mtp_ids, xf, positions, positions, chunk_envelope,
                                      /*final_chunk=*/true, &ar_hidden, &mtp_logits, &draft0);

                TpArray<Tensor> ar_position;
                for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
                    ar_position[r] = io_for(static_cast<int>(r)).mtp->position.slice(0, 0, 1);
                }
                for_each_rank(execution, [&](int rank) {
                    const auto r = static_cast<std::size_t>(rank);
                    ops::set_i32_scalar(const_cast<Tensor&>(ar_position[r]), base_i + T,
                                        stream_for(rank));
                });
                for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                    auto ar_scope_0 = work_.scope();
                    auto ar_scope_1 = rank_scopes();
                    Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                    Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                    TpArray<Tensor> next_hidden;
                    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
                        next_hidden[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
                    }
                    const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                    const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                    mtp_forward_ar_step(prev_token, ar_hidden, ar_position, ar_envelope,
                                        next_hidden, mtp_logits, next_token);
                    for_each_rank(execution, [&](int rank) {
                        const auto r   = static_cast<std::size_t>(rank);
                        cudaStream_t s = stream_for(rank);
                        CUDA_CHECK(cudaMemcpyAsync(ar_hidden[r].data, next_hidden[r].data,
                                                   ar_hidden[r].bytes(), cudaMemcpyDeviceToDevice,
                                                   s));
                        ops::increment_i32_scalar(const_cast<Tensor&>(ar_position[r]), s);
                    });
                }
            } else {
                mtp_prefill_chunk_tp2(mtp_ids, xf, positions, positions, chunk_envelope,
                                      /*final_chunk=*/false, nullptr, nullptr, nullptr);
            }
        }

        if (checkpoint_rel > 0 && len == checkpoint_rel &&
            rewrite_checkpoint_hidden_output_ != nullptr) {
            require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16, {kCfg.hidden, 1},
                                 "rewrite checkpoint hidden output");
            const Tensor checkpoint_hidden = xf[0].slice(1, len - 1, 1);
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(ctx_.device));
            CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                       checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, ctx_.stream));
        }
    }

    if (checkpoint_rel > 0 && len == checkpoint_rel) {
        for_each_rank(execution, [&](int rank) {
            state_for(rank).copy_slot(linear_state_current_slot_,
                                      linear_state_rewrite_checkpoint_slot_, stream_for(rank));
        });
    }

    prefill_rewrite_checkpoint_frontier_ = -1;
    synchronize_all();
    if (dflash_sink != nullptr) {
        dflash_sink->consume_prefill_chunk(len, checkpoint_rel > 0 && len == checkpoint_rel);
        synchronize_all();
    }
    work_.reset();
    reset_peer_workspaces();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = finalize_at_end && len == T};
}

void TextContext::ordinary_decode_batch_tp2(const Tensor& ids, const Tensor& cache_positions,
                                            const Tensor& rope_positions,
                                            const Tensor& kv_table_rows,
                                            const Tensor& linear_state_slots,
                                            ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                            Tensor& logits) {
    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const std::int32_t batch                = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");
    for (int rank = 1; rank < tp_count_; ++rank) {
        if (!io_for(rank).ordinary.has_value()) {
            throw std::logic_error(
                "tensor-parallel decode requires an ordinary frame on every rank");
        }
    }
    // Rank 1's control tensors are its OWN mirror of the decode frame. The Program uploads a
    // SEPARATE pinned record into it (`ordinary_peer_host_ingress`), byte-for-byte rank 0's except
    // that every row's `sampling[row].token_counts` is null: that field is a rank-0 DEVICE address
    // and must never reach rank 1's frame. Rank 1 never samples in the ordinary round -- the
    // vocabulary-split output head gathers to rank 0, which is where `ops::sample` runs -- so the
    // fields read below (tokens, positions, KV rows, lanes) are the ones that matter, and they
    // agree by construction because both records are written from the same host state.
    qwen3_6::OrdinaryDecodeState& peer = *io_for(1).ordinary;
    const Tensor peer_ids              = peer.tokens.slice(0, 0, batch);
    const Tensor peer_cache            = peer.cache_positions.slice(0, 0, batch);
    const Tensor peer_rope             = peer.rope_positions.slice(0, 0, batch);
    const Tensor peer_rows             = peer.text_kv_table_rows.slice(0, 0, batch);
    const Tensor peer_lanes            = peer.lanes.slice(0, 0, batch);
    Tensor peer_hidden                 = peer.hidden.slice(1, 0, batch);

    work_.reset();
    reset_peer_workspaces();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);
        // Every non-zero rank's own mirror of the decode frame, bound for this round and restored
        // on exit. Rank r's is a SEPARATE pinned record uploaded into rank r's device.
        std::array<const Tensor*, kMaximumDevices> peer_cache_values{};
        std::array<const Tensor*, kMaximumDevices> peer_row_values{};
        std::array<const Tensor*, kMaximumDevices> peer_slot_values{};
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            const auto slot = static_cast<std::size_t>(r);
            peer_cache_values[slot] = &ranks_[slot].execution.io->ordinary->cache_positions;
            peer_row_values[slot]   = &ranks_[slot].execution.io->ordinary->text_kv_table_rows;
            peer_slot_values[slot]  = &ranks_[slot].execution.io->ordinary->lanes;
        }
        ScopedRankTensorBinding peer_cache_binding(tp_count_, ranks_,
                                                   &TextRankBinding::cache_positions,
                                                   peer_cache_values);
        ScopedRankTensorBinding peer_rope_binding(tp_count_, ranks_, &TextRankBinding::rope_positions,
                                                  peer_cache_values);
        ScopedRankTensorBinding peer_rows_binding(tp_count_, ranks_, &TextRankBinding::kv_table_rows,
                                                  peer_row_values);
        ScopedRankTensorBinding peer_slots_binding(tp_count_, ranks_,
                                                   &TextRankBinding::linear_state_slots,
                                                   peer_slot_values);

        auto scope_0 = work_.scope();
        auto scope_1 = rank_scopes();
        TpArray<Tensor> x;
        TpArray<Tensor> staging;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            x[r]       = ws[r]->alloc(DType::BF16, {kCfg.hidden, batch});
            staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, batch});
        }
        TpArray<Tensor> rank_ids;
        rank_ids[0] = ids;
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            rank_ids[r] = ranks_[static_cast<std::size_t>(r)].execution.io->ordinary->tokens.slice(
                0, 0, batch);
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::embedding(rank_ids[r], rank_embed(rank), x[r], stream_for(rank));
        });
        run_layers_tp2(x, Phase::Verify, staging);

        TpArray<Tensor> flat_hidden;
        TpArray<Tensor> flat_logits;
        flat_hidden[0] = hidden;
        flat_logits[0] = logits;
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            const auto slot = static_cast<std::size_t>(r);
            flat_hidden[slot] = ranks_[slot].execution.io->ordinary->hidden.slice(1, 0, batch);
            // A non-zero rank's destination is ITS ordinary frame's logits, not its scalar
            // RoundState logits: the decode round produces one column per lane, and
            // RoundState::logits is a single column. Getting this wrong is an out-of-bounds write
            // on that device at any batch above 1.
            flat_logits[slot] = ranks_[slot].execution.io->ordinary->logits.slice(1, 0, batch);
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], rank_final_norm(rank), kCfg.rms_eps, true, flat_hidden[r],
                         stream_for(rank));
        });
        logits_tp2(flat_hidden, flat_logits);
    }
    work_.reset();
    reset_peer_workspaces();
}

// --- tp == 2 MTP -------------------------------------------------------------------------------
//
// The MTP head is a single decoder layer plus a vocabulary head, so its split schedule is the
// text layer's split schedule with one instance of each stage. Three collectives per MTP call:
// the stem's fc all-reduce, the attention output projection's all-reduce, and the post-mixer's
// all-reduce. The residual `x` is replicated and bit-identical on both ranks after every one of
// them, exactly as in the text layers, which is what keeps the MTP KV pages and the proposal
// argmax in lockstep without any further agreement protocol.

// Debug-only (NINFER_TP4_MTP_STEM_REF_DUMP=<prefix>): everything a HOST-side reference needs to
// recompute the MTP stem, captured from the SAME call -- the call's ids, the target hidden, the
// looked-up embedding, the stem's outputs x/ah, the three norms, and every rank's fc weight payload.
// Recomputing in fp32 on the host and diffing against x/ah localizes a stem defect with no
// cross-width A/B and no pin (see work/ninfer-tp4-m2/README_tp4_mtp_rca_2026-10-01.md section 10).
// File layout, little endian:
//   header[8] i32 = {version, T, hidden, tp, half, ids_bytes, embedding_bytes, x_bytes}
//   ids | hidden | embedding | x | ah | norm_embedding | norm_hidden | norm_input
//   per rank: {rank, n, k, group, payload_bytes} i64[5] | fc payload
void mtp_stem_reference_dump(const ExecutionContext& ec, const Tensor& ids,
                             const TpArray<Tensor>& hidden,
                             const TpArray<workspace_recipe::MtpStemRoots>& roots,
                             const TpArray<Tensor>& x, const TpArray<Tensor>& ah, const MtpW& mtp0,
                             const TpArray<const Weight*>& fc_weights,
                             const TpArray<Tensor>& fc_input, std::int32_t half) {
    static const char* prefix = std::getenv("NINFER_TP4_MTP_STEM_REF_DUMP");
    if (prefix == nullptr) { return; }
    static bool done = false;
    if (done) { return; }
    done = true;
    const std::int32_t T = ids.ne[0] * ids.ne[1];
    const CurrentDevice restore;
    const auto read_device = [&](const void* data, std::size_t bytes, int device) {
        std::vector<unsigned char> out(bytes);
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMemcpyAsync(out.data(), data, bytes, cudaMemcpyDeviceToHost,
                                   ec.dev[0]->stream));
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[0]->stream));
        return out;
    };
    const auto read_tensor = [&](const Tensor& t) {
        return read_device(t.data, static_cast<std::size_t>(t.bytes()), ec.dev[0]->device);
    };
    const auto ids_bytes    = read_device(ids.data, static_cast<std::size_t>(ids.bytes()),
                                          ec.dev[0]->device);
    const auto hidden_bytes = read_tensor(hidden[0]);
    const auto embed_bytes =
        roots[0].embedding.data != nullptr ? read_tensor(roots[0].embedding)
                                           : std::vector<unsigned char>{};
    const auto x_bytes  = read_tensor(x[0]);
    const auto ah_bytes = read_tensor(ah[0]);
    // PER RANK, because the embedding side and the hidden side write different buffers: at tp2 only
    // rank 0's normalized_embedding and rank 1's normalized_hidden are ever written, so dumping
    // rank 0's pair and assuming both are meaningful compares a real buffer against garbage.
    std::vector<std::vector<unsigned char>> norm_emb_out(static_cast<std::size_t>(ec.tp));
    std::vector<std::vector<unsigned char>> norm_hid_out(static_cast<std::size_t>(ec.tp));
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const int device = ec.dev[slot]->device;
        if (roots[slot].normalized_embedding.data != nullptr) {
            norm_emb_out[slot] = read_device(roots[slot].normalized_embedding.data,
                                             static_cast<std::size_t>(
                                                 roots[slot].normalized_embedding.bytes()),
                                             device);
        }
        if (roots[slot].normalized_hidden.data != nullptr) {
            norm_hid_out[slot] = read_device(
                roots[slot].normalized_hidden.data,
                static_cast<std::size_t>(roots[slot].normalized_hidden.bytes()), device);
        }
    }
    std::vector<std::vector<unsigned char>> fc_input_bytes(static_cast<std::size_t>(ec.tp));
    for (int rank = 0; rank < ec.tp; ++rank) {
        if (fc_input[static_cast<std::size_t>(rank)].data != nullptr) {
            fc_input_bytes[static_cast<std::size_t>(rank)] =
                read_tensor(fc_input[static_cast<std::size_t>(rank)]);
        }
    }
    const auto norm_e   = read_tensor(*mtp0.pre_fc_norm_embedding);
    const auto norm_h   = read_tensor(*mtp0.pre_fc_norm_hidden);
    const auto norm_i   = read_tensor(*mtp0.input_norm);
    FILE* file = std::fopen(prefix, "wb");
    if (file == nullptr) { throw std::runtime_error("MTP stem reference dump: unreadable"); }
    const std::int32_t header[8] = {1, T, kCfg.hidden, ec.tp, half,
                                    static_cast<std::int32_t>(ids_bytes.size()),
                                    static_cast<std::int32_t>(embed_bytes.size()),
                                    static_cast<std::int32_t>(x_bytes.size())};
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(ids_bytes.data(), 1, ids_bytes.size(), file);
    // Self-describing sections: a 8-byte ASCII tag then the length, so a reader can never mistake
    // one buffer for another (an earlier revision relied on write order and mis-parsed a buffer that
    // the engine never wrote at this width).
    const auto write_tagged = [&](const char* tag, const std::vector<unsigned char>& blob) {
        // Exactly 8 bytes, space padded: a shorter c_str would write whatever follows the NUL and a
        // reader that keys on the tag would then miss the section.
        char padded[8] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
        for (int i = 0; i < 8 && tag[i] != '\0'; ++i) { padded[i] = tag[i]; }
        std::fwrite(padded, 1, sizeof(padded), file);
        const std::int64_t length = static_cast<std::int64_t>(blob.size());
        std::fwrite(&length, sizeof(length), 1, file);
        std::fwrite(blob.data(), 1, blob.size(), file);
    };
    write_tagged("hidden  ", hidden_bytes);
    write_tagged("embed   ", embed_bytes);
    write_tagged("stemX   ", x_bytes);
    write_tagged("stemAH  ", ah_bytes);
    write_tagged("normE   ", norm_e);
    write_tagged("normH   ", norm_h);
    write_tagged("normI   ", norm_i);
    for (int rank = 0; rank < ec.tp; ++rank) {
        write_tagged(("nEmb" + std::to_string(rank)).c_str(),
                     norm_emb_out[static_cast<std::size_t>(rank)]);
        write_tagged(("nHid" + std::to_string(rank)).c_str(),
                     norm_hid_out[static_cast<std::size_t>(rank)]);
    }
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto& blob = fc_input_bytes[static_cast<std::size_t>(rank)];
        const std::string tag = "fcIn" + std::to_string(rank);
        write_tagged(tag.c_str(), blob);
    }
    for (int rank = 0; rank < ec.tp; ++rank) {
        const Weight& fc  = *fc_weights[static_cast<std::size_t>(rank)];
        const int device  = ec.dev[static_cast<std::size_t>(rank)]->device;
        const auto payload =
            read_device(fc.payload, static_cast<std::size_t>(fc.payload_bytes), device);
        const std::string fctag = "fcShard" + std::to_string(rank);
        std::fwrite(fctag.c_str(), 1, fctag.size(), file);
        const std::int64_t fcheader[5] = {rank, fc.n, fc.k, fc.group,
                                          static_cast<std::int64_t>(fc.payload_bytes)};
        std::fwrite(fcheader, sizeof(fcheader), 1, file);
        std::fwrite(payload.data(), 1, payload.size(), file);
    }
    std::fclose(file);
    std::fprintf(stderr,
                 "[mtp-stem-ref] T=%d half=%d hidden=%zuB embed=%zuB x=%zuB ah=%zuB roots=%zu\n", T,
                 half, hidden_bytes.size(), embed_bytes.size(), x_bytes.size(), ah_bytes.size(),
                 static_cast<std::size_t>(roots.size()));
    // In-process trio check: shapes plus a few (raw, weight, out) values, so "which buffer is which"
    // cannot be an artefact of how the dump file is interpreted.
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot    = static_cast<std::size_t>(rank);
        const bool embedside = rank < half;
        const Tensor& raw = embedside ? roots[slot].embedding : hidden[slot];
        const Tensor& out = embedside ? roots[slot].normalized_embedding
                                      : roots[slot].normalized_hidden;
        const Tensor& weight = embedside ? *mtp0.pre_fc_norm_embedding : *mtp0.pre_fc_norm_hidden;
        if (raw.data == nullptr || out.data == nullptr) { continue; }
        const auto raw_v  = read_device(raw.data, static_cast<std::size_t>(raw.bytes()),
                                        ec.dev[slot]->device);
        const auto out_v  = read_device(out.data, static_cast<std::size_t>(out.bytes()),
                                        ec.dev[slot]->device);
        const auto w_v    = read_device(weight.data, static_cast<std::size_t>(weight.bytes()),
                                        ec.dev[slot]->device);
        const auto to_f = [](unsigned short bits) {
            const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
            float value              = 0.0F;
            std::memcpy(&value, &wide, sizeof(value));
            return value;
        };
        const auto* raw_b = reinterpret_cast<const unsigned short*>(raw_v.data());
        const auto* out_b = reinterpret_cast<const unsigned short*>(out_v.data());
        const auto* w_b   = reinterpret_cast<const unsigned short*>(w_v.data());
        const std::int32_t rows = embedside ? raw.ne[0] : raw.ne[0];
        double sum = 0.0;
        for (std::int32_t i = 0; i < raw.ne[0]; ++i) {
            const double v = to_f(raw_b[i]);
            sum += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum / static_cast<double>(raw.ne[0]) + kCfg.rms_eps);
        const double g0  = to_f(w_b[0]) + 1.0;
        std::fprintf(stderr,
                     "[mtp-trio] rank %d %s: raw.ne=%d,%d out.ne=%d,%d  raw[0]=%+.5f w[0]=%+.5f "
                     "out[0]=%+.5f  expected(raw*inv*gain)=%+.5f  inv=%.5f  rows=%d\n",
                     rank, embedside ? "embed" : "hidden", raw.ne[0], raw.ne[1], out.ne[0],
                     out.ne[1], to_f(raw_b[0]), to_f(w_b[0]), to_f(out_b[0]),
                     to_f(raw_b[0]) * inv * g0, inv, rows);
    }
}

void TextContext::mtp_forward_stem_tp2(const Tensor& ids, const TpArray<Tensor>& hidden,
                                       TpArray<Tensor>& x, TpArray<Tensor>& ah,
                                       const TpArray<Tensor>& staging) {
    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const int T                             = ids.ne[0] * ids.ne[1];
    // The caller's hidden is [hidden, T] in the AR/bridge shapes and [hidden, width, batch] in the
    // decode-batch one, so the aggregate column count is what has to match -- comparing ne[1]
    // alone silently accepts a batch>1 frame and then reads only its first lane.
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        if (hidden[r].ne[0] != kCfg.hidden || !hidden[r].is_contiguous() ||
            hidden[r].numel() != static_cast<std::int64_t>(kCfg.hidden) * T) {
            throw std::logic_error(
                "tensor-parallel MTP stem hidden does not match the token count");
        }
    }
    Tensor flat_ids = ids.view({T});

    // The packed fc input is [embedding_norm (hidden rows) | hidden_norm (hidden rows)], and
    // `append_row_parallel` hands rank r packed rows [r*K/tp, (r+1)*K/tp). With half = tp/2 that is
    //   ranks [0, half):   the r-th 1/half slice of embedding_norm
    //   ranks [half, tp):  the (r - half)-th 1/half slice of hidden_norm
    // so an embedding-side rank embeds a token and normalizes the EMBEDDING, a hidden-side rank
    // normalizes the HIDDEN, and each then contracts only its own slice. At tp == 2 this degenerates
    // to "rank 0 the embedding, rank 1 the hidden", which is the mapping the tp2 shard always used.
    //
    // Only the embedding-side ranks need an embedding root; the others never embed a token here.
    // The startup capacity query plans for the embedding side on every rank, which over-plans the
    // hidden-side ranks rather than under-planning them.
    const std::int32_t half       = tp_count_ / 2;
    const std::int32_t shard_rows = kCfg.hidden / half;
    // `ids` is ONE buffer on rank 0 for every rank (one model input, one host copy per chunk), but
    // only the EMBEDDING-side ranks read it -- and at tp > 2 that set is wider than rank 0. With
    // half = tp/2 the embedding side is ranks [0, half), so at tp4 rank 1 embeds the same token
    // stream out of rank 0's buffer.
    //
    // Rank 0 wrote that buffer on ITS OWN stream (the shifted-token copy in the prefill, the
    // alignment ids of a decode round, the draft token of an AR step, the bridge token). A peer
    // stream has no implicit ordering against it, so without the handshake below the peer's
    // embedding gather can read the buffer's PREVIOUS contents -- and a garbage token id is an
    // out-of-bounds read of the embedding table, i.e. a sticky illegal-access fault rather than a
    // wrong token. Measured: at tp4 that fault fired on roughly every second load-time warmup, and
    // one synchronize anywhere inside the stem made it disappear, which is the signature of a
    // missing cross-stream edge rather than of a bad address computation.
    //
    // At tp == 2 half == 1: rank 0 is the only reader, on the stream that wrote it, which is why
    // the tp2 path never needed this. The record is enqueued on the WRITING stream, so it is also
    // a barrier for every earlier write on rank 0's stream, and it is capture-safe (an event wait
    // becomes a graph edge), unlike the synchronize it replaces.
    if (half > 1) {
        const ops::PeerEvents& events = peer_events();
        const CurrentDevice restore;
        CUDA_CHECK(cudaSetDevice(ec().dev[0]->device));
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), stream_for(0)));
        for (std::int32_t rank = 1; rank < half; ++rank) {
            CUDA_CHECK(cudaSetDevice(ec().dev[static_cast<std::size_t>(rank)]->device));
            CUDA_CHECK(cudaStreamWaitEvent(stream_for(rank), events.inputs_ready(0), 0));
        }
    }
    TpArray<workspace_recipe::MtpStemRoots> roots;
    for (std::int32_t r = 0; r < tp_count_; ++r) {
        const auto slot = static_cast<std::size_t>(r);
        roots[slot]     = workspace_recipe::mtp_stem<TextConfig>(*ws[slot], T,
                                                                 /*allocate_embedding=*/r < half,
                                                                 ec().tp);
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        x[r]  = roots[r].residual;
        ah[r] = roots[r].attention_hidden;
    }

    // `ops::mtp_pack_fc_input` is NOT called here, and the packed [10240, T] buffer is not even
    // allocated. Device r's fc shard contracts packed rows [5120r, 5120r + 5120), which by that
    // Op's own definition are the normalized EMBEDDING on rank 0 and the normalized HIDDEN on
    // rank 1 (bindings.cpp `ends("mtp/input_projection")` -> append_row_parallel, whose
    // even_chunk gives device 0 the low half). So each rank computes only the half it will
    // contract, and one all-reduce completes the product.
    for_each_rank(execution, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        cudaStream_t s  = stream_for(rank);
        if (rank < half) {
            ops::embedding(flat_ids, rank_embed(rank), roots[slot].embedding, s);
            ops::rmsnorm(roots[slot].embedding, *mtp_weights_for(rank).pre_fc_norm_embedding,
                         kCfg.rms_eps, true, roots[slot].normalized_embedding, s);
        } else {
            const Tensor flat_hidden = hidden[slot].view({kCfg.hidden, T});
            ops::rmsnorm(flat_hidden, *mtp_weights_for(rank).pre_fc_norm_hidden, kCfg.rms_eps, true,
                         roots[slot].normalized_hidden, s);
        }
    });
    TpArray<Tensor> fc_input;
    TpArray<Weight> fc;
    for (std::int32_t r = 0; r < tp_count_; ++r) {
        const auto slot           = static_cast<std::size_t>(r);
        const bool embedding_side = r < half;
        const std::int32_t within = embedding_side ? r : r - half;
        const Tensor source       = embedding_side ? roots[slot].normalized_embedding
                                                   : roots[slot].normalized_hidden;
        fc[slot]                  = *mtp_weights_for(r).fc;
        if (half == 1) {
            // tp 2: the rank's K-slice IS the whole normalized buffer, already contiguous.
            fc_input[slot] = source;
            continue;
        }
        // tp > 2: the rank's K-slice is a ROW RANGE of one normalized buffer. Those rows are
        // adjacent, so the range is one contiguous run of bytes -- but the [K/tp, T] view of it is
        // strided as a whole and the GEMM needs a compact activation. Copy the run once.
        //
        // ONE COPY PER COLUMN, because the buffer is COLUMN-MAJOR: ne[0] (= hidden) is the FAST
        // axis, so rows [a, a + shard_rows) of every column are strided by `hidden`, NOT one
        // contiguous run. The single memcpy this replaced read the range as if rows were
        // contiguous, which handed every tp > 2 rank a slice of the wrong elements -- a silent
        // defect, since the values are still real activations. tp2 never ran it (the `half == 1`
        // shortcut below assigns the whole normalized buffer), which is why only the tp4 MTP was
        // degraded; measured against a host reference: fc_input rel_l2 1.2-1.36 at tp4 versus
        // 0.0018 at tp2, and every downstream stage inherits it.
        constexpr std::int64_t kElementBytes = sizeof(std::uint16_t);
        Tensor slice = ws[slot]->alloc(DType::BF16, {shard_rows, T});
        {
            const CurrentDevice restore;
            // `ranks_` holds slots 1..tp-1 only, so rank 0's device comes from the context.
            CUDA_CHECK(cudaSetDevice(
                ec().dev[static_cast<std::size_t>(r)]->device));
            for (std::int32_t column = 0; column < T; ++column) {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint8_t*>(slice.data) +
                        static_cast<std::int64_t>(column) * shard_rows * kElementBytes,
                    static_cast<const std::uint8_t*>(source.data) +
                        (static_cast<std::int64_t>(column) * kCfg.hidden +
                         static_cast<std::int64_t>(within) * shard_rows) * kElementBytes,
                    static_cast<std::size_t>(shard_rows) *
                        static_cast<std::size_t>(kElementBytes),
                    cudaMemcpyDeviceToDevice, stream_for(r)));
            }
        }
        fc_input[slot] = slice;
    }
    ops::linear_row_parallel(fc_input, fc, x, staging, execution, peer_events());
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], *mtp_weights_for(rank).input_norm, kCfg.rms_eps, true, ah[r],
                     stream_for(rank));
    });
    {
        TpArray<const Weight*> fc_weights{};
        for (int r = 0; r < ec().tp; ++r) {
            fc_weights[static_cast<std::size_t>(r)] = mtp_weights_for(r).fc;
        }
        mtp_stem_reference_dump(ec(), ids, hidden, roots, x, ah, mtp_weights_for(0), fc_weights,
                                fc_input, static_cast<std::int32_t>(half));
    }
}

// See mtp_stem_probe; this one dumps the upper half of the tail's output for elementwise diffing.
void mtp_tail_output_probe(const ExecutionContext& ec, const TpArray<Tensor>& mtp_hidden) {
    static const char* dump_path = std::getenv("NINFER_TP4_MTP_TAILOUT_DUMP");
    if (dump_path == nullptr) { return; }
    static bool done = false;
    if (done) { return; }
    done = true;
    const Tensor& source  = mtp_hidden[0];
    const std::size_t one = static_cast<std::size_t>(source.bytes());
    std::vector<unsigned char> host(one);
    const CurrentDevice restore;
    CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), source.data, one, cudaMemcpyDeviceToHost,
                               ec.dev[0]->stream));
    CUDA_CHECK(cudaStreamSynchronize(ec.dev[0]->stream));
    FILE* file = std::fopen(dump_path, "wb");
    if (file == nullptr) { throw std::runtime_error("MTP tailout probe: dump path unreadable"); }
    std::fwrite(host.data(), 1, host.size(), file);
    std::fclose(file);
    std::fprintf(stderr, "[mtp-tailout] dumped %zu bytes (ne=%d,%d,%d,%d)\n", host.size(),
                 source.ne[0], source.ne[1], source.ne[2], source.ne[3]);
}

// See mtp_stem_probe: same idea one stage later, over the post-mixer's (residual, hidden) pair.
void mtp_post_mixer_probe(const ExecutionContext& ec, const TpArray<Tensor>& residual,
                          const TpArray<Tensor>& hidden, std::int32_t rank_count) {
    static const char* dump_path = std::getenv("NINFER_TP4_MTP_POSTMIX_DUMP");
    static const char* load_path = std::getenv("NINFER_TP4_MTP_POSTMIX_LOAD");
    if (dump_path == nullptr && load_path == nullptr) { return; }
    static bool done = false;
    if (done) { return; }
    done = true;
    const std::size_t one = static_cast<std::size_t>(residual[0].bytes());
    std::vector<unsigned char> host(2 * one);
    const CurrentDevice restore;
    if (dump_path != nullptr) {
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        CUDA_CHECK(cudaMemcpyAsync(host.data(), residual[0].data, one, cudaMemcpyDeviceToHost,
                                   ec.dev[0]->stream));
        CUDA_CHECK(cudaMemcpyAsync(host.data() + one, hidden[0].data, one, cudaMemcpyDeviceToHost,
                                   ec.dev[0]->stream));
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[0]->stream));
        FILE* file = std::fopen(dump_path, "wb");
        if (file == nullptr) { throw std::runtime_error("MTP postmix probe: dump unreadable"); }
        std::fwrite(host.data(), 1, host.size(), file);
        std::fclose(file);
        std::fprintf(stderr, "[mtp-postmix] dumped x+mh = %zu bytes\n", host.size());
    }
    if (load_path != nullptr) {
        FILE* file = std::fopen(load_path, "rb");
        if (file == nullptr) { throw std::runtime_error("MTP postmix probe: load unreadable"); }
        const std::size_t got = std::fread(host.data(), 1, host.size(), file);
        std::fclose(file);
        if (got != host.size()) {
            throw std::runtime_error("MTP postmix probe: file size does not match this frame");
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            CUDA_CHECK(cudaMemcpyAsync(residual[slot].data, host.data(), one,
                                       cudaMemcpyHostToDevice, ec.dev[slot]->stream));
            CUDA_CHECK(cudaMemcpyAsync(hidden[slot].data, host.data() + one, one,
                                       cudaMemcpyHostToDevice, ec.dev[slot]->stream));
        }
        std::fprintf(stderr, "[mtp-postmix] loaded x+mh = %zu bytes into every rank\n",
                     host.size());
    }
}

void TextContext::mtp_forward_tail_tp2(TpArray<Tensor>& x, const TpArray<Tensor>& ah,
                                       const TpArray<Tensor>& positions,
                                       const TpArray<Tensor>& rope_positions,
                                       ops::GqaExecutionEnvelope envelope,
                                       const TpArray<Tensor>& mtp_hidden,
                                       const TpArray<Tensor>& staging) {
    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const int T                             = x[0].ne[1];

    TpArray<Tensor> q;
    TpArray<Tensor> k;
    TpArray<Tensor> gate;
    TpArray<Tensor> v;
    TpArray<Tensor> q_flat;
    TpArray<Tensor> gate_flat;
    TpArray<Tensor> k_flat;
    TpArray<Tensor> v_flat;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        const auto projection =
            workspace_recipe::mtp_attention_projection<TextConfig>(*ws[r], T,
                                                                   ec().tp);
        q[r]         = projection.query.view({kCfg.head_dim, shard_q_heads(), T});
        k[r]         = projection.key.view({kCfg.head_dim, shard_kv_heads(), T});
        gate[r]      = projection.gate.view({kCfg.head_dim, shard_q_heads(), T});
        v[r]         = projection.value.view({kCfg.head_dim, shard_kv_heads(), T});
        q_flat[r]    = q[r].view({shard_q_size(), T});
        gate_flat[r] = gate[r].view({shard_q_size(), T});
        k_flat[r]    = k[r].view({shard_kv_size(), T});
        v_flat[r]    = v[r].view({shard_kv_size(), T});
    }
    Variant::mtp_attention_projection(
        ah, mtp_attention_weights(),
        q_flat, gate_flat, k_flat, v_flat, ws, execution);

    TpArray<Tensor> qn;
    TpArray<Tensor> kn;
    TpArray<Tensor> a;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        const auto results =
            workspace_recipe::mtp_attention_results<TextConfig>(*ws[r], T, ec().tp);
        qn[r] = results.normalized_query.view({kCfg.head_dim, shard_q_heads(), T});
        kn[r] = results.normalized_key.view({kCfg.head_dim, shard_kv_heads(), T});
        a[r]  = results.attention.view({kCfg.head_dim, shard_q_heads(), T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r    = static_cast<std::size_t>(rank);
        cudaStream_t s  = stream_for(rank);
        const MtpW& mtp = mtp_weights_for(rank);
        ops::rmsnorm(q[r], *mtp.q_norm, kCfg.rms_eps, true, qn[r], s);
        ops::rmsnorm(k[r], *mtp.k_norm, kCfg.rms_eps, true, kn[r], s);
        Tensor rope_for_op =
            active_sequence_batch_ != 0 ? rope_positions[r].view({T}) : rope_positions[r];
        ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn[r], kn[r],
                  rope_frequency_[r], s);

        const qwen3_6::PagedKVCache& pages = rank_batch_mtp_kv(rank);
        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            if (width <= 0 || width * active_sequence_batch_ != T) {
                throw std::logic_error("MTP sequence batch binding is incomplete");
            }
            Tensor q_batch =
                qn[r].view({kCfg.head_dim, shard_q_heads(), width, active_sequence_batch_});
            Tensor k_batch =
                kn[r].view({kCfg.head_dim, shard_kv_heads(), width, active_sequence_batch_});
            Tensor v_batch =
                v[r].view({kCfg.head_dim, shard_kv_heads(), width, active_sequence_batch_});
            Tensor a_batch =
                a[r].view({kCfg.head_dim, shard_q_heads(), width, active_sequence_batch_});
            Tensor position_batch = positions[r].view({width, active_sequence_batch_});
            ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, rank_valid_columns(rank),
                               rank_backend_kv_table_rows(rank), kAttnScale,
                               pages.batch_layer_view(0), envelope, *ws[r], a_batch, s);
        } else {
            ops::gqa_attention(qn[r], kn[r], v[r], positions[r], Tensor{},
                               io_for(rank).backend_kv_table_row, kAttnScale,
                               pages.batch_layer_view(0), envelope, *ws[r], a[r], s);
        }
        ops::sigmoid_mul(gate[r], a[r], s);
    });

    // The MTP output projection is `linear` + `residual_add`, not the fused `linear_add` the text
    // layers use -- the tp1 MTP leaf composes it the same way, because W8G32_F16S has no
    // linear_add profile. So the row-parallel split is `linear_row_parallel` + a replicated
    // per-rank `residual_add` over the all-reduced result.
    TpArray<Tensor> o;
    TpArray<Tensor> mh;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        const auto post   = workspace_recipe::mtp_post_attention<TextConfig>(*ws[r], T);
        o[r]              = post.output;
        mh[r]             = post.post_mixer_hidden;
    }
    ops::linear_row_parallel(attention_flat(a, T),
                             mtp_output_weights(), o, staging,
                             execution, peer_events());
    for_each_rank(execution, [&](int rank) {
        const auto r   = static_cast<std::size_t>(rank);
        cudaStream_t s = stream_for(rank);
        ops::residual_add(o[r], x[r], s);
        ops::rmsnorm(x[r], *mtp_weights_for(rank).post_attn_norm, kCfg.rms_eps, true, mh[r], s);
    });

    // Debug-only (NINFER_TP4_MTP_POSTMIX_DUMP=path / ..._LOAD=path): pins the post-mixer's input --
    // the residual `x` and the post-attention hidden `mh`, both replicated [hidden, T] because the
    // output projection's all-reduce has already run. The pair splits the tail in two: if the same
    // pinned pair yields the same proposal at tp2 and tp4, the post-mixer is fine and the
    // attention stage above it is the defective half.
    mtp_post_mixer_probe(ec(), x, mh, ec().tp);
    {
        auto scope_0 = work_.scope();
        auto scope_1 = rank_scopes();
        Variant::mtp_post_mixer(
            mh, mtp_post_mixer_weights(),
            x, staging, ws, execution, peer_events());
    }

    for_each_rank(execution, [&](int rank) {
        const auto r           = static_cast<std::size_t>(rank);
        Tensor flat_mtp_hidden = mtp_hidden[r].view({kCfg.hidden, T});
        ops::rmsnorm(x[r], *mtp_weights_for(rank).norm, kCfg.rms_eps, true, flat_mtp_hidden,
                     stream_for(rank));
    });
    // Debug-only (NINFER_TP4_MTP_TAILOUT_DUMP=path): the tail's output AFTER the final norm, which
    // is replicated, so the tp2 and tp4 images are directly comparable elementwise. Comparing the
    // proposal argmax instead is confounded: the argmax is taken over the column the VERIFY's
    // accepted count selects, and the verify's own batched logits differ slightly between widths,
    // so a different column can be selected for reasons that have nothing to do with the tail.
    mtp_tail_output_probe(ec(), mtp_hidden);
}

// Debug-only (NINFER_TP4_MTP_STEM_DUMP=path / ..._LOAD=path): pins the MTP module's stem OUTPUT --
// the residual `x` and the attention input `ah`, both replicated [hidden, T] -- so the tail
// (attention + post-mixer) can be exercised at two widths on one identical activation. Together
// with mtp_proposal_input_probe this splits the MTP module into stem / tail / head, one A/B each.
// The stem consumes `ids` and the target hidden (both rank-invariant for the hidden side), so a
// per-rank difference here is a shard-geometry defect, not a data difference.
void mtp_stem_probe(const ExecutionContext& ec, const TpArray<Tensor>& x, const TpArray<Tensor>& ah,
                    std::int32_t rank_count) {
    static const char* dump_path = std::getenv("NINFER_TP4_MTP_STEM_DUMP");
    static const char* load_path = std::getenv("NINFER_TP4_MTP_STEM_LOAD");
    if (dump_path == nullptr && load_path == nullptr) { return; }
    static bool done = false;
    if (done) { return; }
    done = true;
    const std::size_t one = static_cast<std::size_t>(x[0].bytes());
    std::vector<unsigned char> host(2 * one);
    const CurrentDevice restore;
    if (dump_path != nullptr) {
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        CUDA_CHECK(cudaMemcpyAsync(host.data(), x[0].data, one, cudaMemcpyDeviceToHost,
                                   ec.dev[0]->stream));
        CUDA_CHECK(cudaMemcpyAsync(host.data() + one, ah[0].data, one, cudaMemcpyDeviceToHost,
                                   ec.dev[0]->stream));
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[0]->stream));
        FILE* file = std::fopen(dump_path, "wb");
        if (file == nullptr) { throw std::runtime_error("MTP stem probe: dump path unreadable"); }
        std::fwrite(host.data(), 1, host.size(), file);
        std::fclose(file);
        std::fprintf(stderr, "[mtp-stem] dumped x+ah = %zu bytes\n", host.size());
    }
    if (load_path != nullptr) {
        FILE* file = std::fopen(load_path, "rb");
        if (file == nullptr) { throw std::runtime_error("MTP stem probe: load path unreadable"); }
        const std::size_t got = std::fread(host.data(), 1, host.size(), file);
        std::fclose(file);
        if (got != host.size()) {
            throw std::runtime_error(
                "MTP stem probe: file size does not match this round's x+ah (same --draft-tokens?)");
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
            CUDA_CHECK(cudaMemcpyAsync(x[slot].data, host.data(), one, cudaMemcpyHostToDevice,
                                       ec.dev[slot]->stream));
            CUDA_CHECK(cudaMemcpyAsync(ah[slot].data, host.data() + one, one, cudaMemcpyHostToDevice,
                                       ec.dev[slot]->stream));
        }
        std::fprintf(stderr, "[mtp-stem] loaded x+ah = %zu bytes into every rank\n", host.size());
    }
}

void TextContext::mtp_forward_core_tp2(const Tensor& ids, const TpArray<Tensor>& hidden,
                                       const TpArray<Tensor>& positions,
                                       const TpArray<Tensor>& rope_positions,
                                       ops::GqaExecutionEnvelope envelope,
                                       const TpArray<Tensor>& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    for (int rank = 1; rank < tp_count_; ++rank) {
        if (rank_binding(rank).execution.batch_mtp_kv == nullptr) {
            throw std::runtime_error("MTP forward is not enabled on every rank");
        }
    }
    const TpArray<WorkspaceArena*> ws = workspaces();
    auto scope_0                            = work_.scope();
    auto scope_1 = rank_scopes();
    const int T                             = ids.ne[0] * ids.ne[1];
    TpArray<Tensor> staging;
    for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
        staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, T});
    }
    TpArray<Tensor> x;
    TpArray<Tensor> ah;
    mtp_forward_stem_tp2(ids, hidden, x, ah, staging);
    mtp_stem_probe(ec(), x, ah, ec().tp);
    mtp_forward_tail_tp2(x, ah, positions, rope_positions, envelope, mtp_hidden, staging);
}

void TextContext::proposal_argmax_tp2(const TpArray<Tensor>& hidden,
                                      const TpArray<Tensor>& logits,
                                      Tensor& proposal_tokens) {
    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const std::int32_t T                    = hidden[0].ne[1];
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, T}, "proposal hidden");
        require_tensor_window(logits[r], DType::BF16, kCfg.vocab, T, "proposal logits");
    }
    if (proposal_head_ == nullptr) {
        // Full LM head: the gathered [vocab, T] logits ARE the caller's destinations, so this is
        // the ordinary logits path plus rank 0's argmax.
        TpArray<Tensor> output_logits;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            output_logits[r] = matrix_window(logits[r], T);
        }
        logits_tp2(hidden, output_logits);
        const CurrentDevice restore;
        CUDA_CHECK(cudaSetDevice(ctx_.device));
        ops::argmax(output_logits[0], proposal_tokens, kCfg.token_domain, ctx_.stream);
        return;
    }
    const std::int32_t shard_rows = proposal_head_->n;
    for (int rank = 1; rank < tp_count_; ++rank) {
        const TextRankBinding& binding = rank_binding(rank);
        if (binding.proposal_head == nullptr || binding.proposal_head_ids == nullptr) {
            throw std::logic_error("tensor-parallel proposal head bindings disagree between ranks");
        }
        if (binding.proposal_head->n != shard_rows) {
            throw std::logic_error("tensor-parallel proposal head shards disagree on width");
        }
    }
    // `proposal_head_n_` is THIS RANK'S shard row count -- `set_proposal_head` is called with
    // `proposal.head.n`, which the loader already halved -- so the logical vocabulary of the draft
    // head is the two shards summed. The winning index is a global row in that logical space,
    // which is why the gather has to run before the argmax and why `draft_head_token_ids` is
    // replicated.
    const std::int32_t total_rows = proposal_head_n_ * tp_count_;
    auto scope_0                  = work_.scope();
    auto scope_1 = rank_scopes();
    TpArray<Tensor> part;
    TpArray<Tensor> whole;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        part[r]  = ws[r]->alloc(DType::BF16, {shard_rows, T});
        whole[r] = ws[r]->alloc(DType::BF16, {total_rows, T});
    }
    ops::linear_column_parallel(hidden, proposal_heads(), part, execution);
    // As in logits_tp2: allgather_rows gathers along ne[1] while the vocabulary is ne[0], so the
    // gather runs one column at a time over contiguous V-element runs -- no transpose.
    for (std::int32_t column = 0; column < T; ++column) {
        TpArray<Tensor> piece;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            piece[r] = part[r].slice(1, column, 1).view({1, shard_rows});
        }
        TpArray<Tensor> full;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            full[r] = whole[r].slice(1, column, 1).view({1, total_rows});
        }
        ops::allgather_rows(full, piece, execution, peer_events());
    }
    const CurrentDevice restore;
    CUDA_CHECK(cudaSetDevice(ctx_.device));
    ops::argmax(whole[0], proposal_tokens, total_rows, ctx_.stream);
    ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, total_rows, ctx_.stream);
}

void TextContext::target_verify_batch(const TpArray<Tensor>& ids,
                                      const TpArray<Tensor>& cache_positions,
                                      const TpArray<Tensor>& rope_positions,
                                      const TpArray<Tensor>& valid_columns,
                                      const TpArray<Tensor>& kv_table_rows,
                                      const TpArray<Tensor>& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope,
                                      const TpArray<Tensor>& hidden,
                                      const TpArray<Tensor>& logits,
                                      const TpArray<Tensor>& target_tokens) {
    if (!tp2()) { throw std::logic_error("tensor-parallel target verify requires a peer"); }
    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const std::int32_t width                = ids[0].ne[0];
    const std::int32_t batch                = ids[0].ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    // Both ranks' extents are validated, not just rank 0's: rank 1's destinations live on the
    // other device, where an undersized buffer is a silent out-of-bounds write -- a defect of
    // exactly this class was once live here, invisible at batch 1 and out of bounds at batch > 1.
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        require_tensor_shape(ids[r], DType::I32, {width, batch}, "target verify batch ids");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "target verify batch cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "target verify batch RoPE positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "target verify batch valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch}, "target verify batch KV rows");
        require_tensor_shape(linear_state_slots[r], DType::I32, {batch},
                             "target verify batch Linear Attention slots");
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "target verify batch hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, width, batch},
                             "target verify batch logits");
        require_tensor_shape(target_tokens[r], DType::I32, {width, batch},
                             "target verify batch tokens");
    }

    work_.reset();
    reset_peer_workspaces();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions[0]);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions[0]);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows[0]);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_,
                                                 &linear_state_slots[0]);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
        // SYMMETRIC valid-column binding. `rank_valid_columns` throws if only one rank is bound,
        // precisely because two devices masking different columns would diverge silently; that
        // invariant is only exercised from here.
        std::array<const Tensor*, kMaximumDevices> peer_cache_values{};
        std::array<const Tensor*, kMaximumDevices> peer_row_values{};
        std::array<const Tensor*, kMaximumDevices> peer_slot_values{};
        std::array<const Tensor*, kMaximumDevices> peer_valid_values{};
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            const auto slot         = static_cast<std::size_t>(r);
            peer_cache_values[slot] = &cache_positions[slot];
            peer_row_values[slot]   = &kv_table_rows[slot];
            peer_slot_values[slot]  = &linear_state_slots[slot];
            peer_valid_values[slot] = &valid_columns[slot];
        }
        ScopedRankTensorBinding peer_cache_binding(tp_count_, ranks_,
                                                   &TextRankBinding::cache_positions,
                                                   peer_cache_values);
        ScopedRankTensorBinding peer_rope_binding(tp_count_, ranks_, &TextRankBinding::rope_positions,
                                                  peer_cache_values);
        ScopedRankTensorBinding peer_rows_binding(tp_count_, ranks_, &TextRankBinding::kv_table_rows,
                                                  peer_row_values);
        ScopedRankTensorBinding peer_slots_binding(tp_count_, ranks_,
                                                   &TextRankBinding::linear_state_slots,
                                                   peer_slot_values);
        ScopedRankTensorBinding peer_valid_binding(tp_count_, ranks_,
                                                   &TextRankBinding::valid_columns,
                                                   peer_valid_values);

        auto scope_0 = work_.scope();
        auto scope_1 = rank_scopes();
        TpArray<Tensor> x;
        TpArray<Tensor> staging;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            x[r]       = ws[r]->alloc(DType::BF16, {kCfg.hidden, columns});
            staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, columns});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r    = static_cast<std::size_t>(rank);
            Tensor flat_ids = ids[r].view({columns});
            ops::embedding(flat_ids, rank_embed(rank), x[r], stream_for(rank));
        });
        run_layers_tp2(x, Phase::Verify, staging);

        TpArray<Tensor> flat_hidden;
        TpArray<Tensor> flat_logits;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            flat_hidden[r] = hidden[r].view({kCfg.hidden, columns});
            flat_logits[r] = logits[r].view({kCfg.vocab, columns});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], rank_final_norm(rank), kCfg.rms_eps, true,
                         flat_hidden[r], stream_for(rank));
        });
        logits_tp2(flat_hidden, flat_logits);
        // The argmax is REPLICATED, not rank 0's alone: both ranks hold the identical gathered
        // logits (the gather is an exact relocation and IEEE addition is commutative, so the two
        // buffers are bit-identical), and rank 1 needs its own target tokens to run the same
        // acceptance arithmetic without a control-tensor transfer.
        for_each_rank(execution, [&](int rank) {
            const auto r       = static_cast<std::size_t>(rank);
            Tensor flat_tokens = target_tokens[r].view({columns});
            ops::argmax(flat_logits[r], flat_tokens, kCfg.token_domain, stream_for(rank));
        });
    }
    work_.reset();
    reset_peer_workspaces();
}

void TextContext::target_verify_batch(const TpArray<Tensor>& ids,
                                      const TpArray<Tensor>& cache_positions,
                                      const TpArray<Tensor>& rope_positions,
                                      const TpArray<Tensor>& valid_columns,
                                      const TpArray<Tensor>& kv_table_rows,
                                      const TpArray<Tensor>& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope,
                                      const TpArray<Tensor>& hidden,
                                      const TpArray<Tensor>& logits,
                                      const TpArray<Tensor>& target_tokens,
                                      DFlashFeatureSink& sink) {
    if (!tp2()) { throw std::logic_error("tensor-parallel target verify requires a peer"); }
    const ExecutionContext& execution = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    const std::int32_t width = ids[0].ne[0];
    const std::int32_t batch = ids[0].ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("tensor-parallel DFlash target verify shape is invalid");
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        require_tensor_shape(ids[r], DType::I32, {width, batch}, "DFlash target verify ids");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "DFlash target verify cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "DFlash target verify rope positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "DFlash target verify valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch},
                             "DFlash target verify KV rows");
        require_tensor_shape(linear_state_slots[r], DType::I32, {batch},
                             "DFlash target verify state slots");
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "DFlash target verify hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, width, batch},
                             "DFlash target verify logits");
        require_tensor_shape(target_tokens[r], DType::I32, {width, batch},
                             "DFlash target verify tokens");
    }
    work_.reset();
    reset_peer_workspaces();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions[0]);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions[0]);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows[0]);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots[0]);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
        std::array<const Tensor*, kMaximumDevices> peer_cache_values{};
        std::array<const Tensor*, kMaximumDevices> peer_row_values{};
        std::array<const Tensor*, kMaximumDevices> peer_slot_values{};
        std::array<const Tensor*, kMaximumDevices> peer_valid_values{};
        for (std::int32_t r = 1; r < tp_count_; ++r) {
            const auto slot         = static_cast<std::size_t>(r);
            peer_cache_values[slot] = &cache_positions[slot];
            peer_row_values[slot]   = &kv_table_rows[slot];
            peer_slot_values[slot]  = &linear_state_slots[slot];
            peer_valid_values[slot] = &valid_columns[slot];
        }
        ScopedRankTensorBinding peer_cache_binding(tp_count_, ranks_,
                                                   &TextRankBinding::cache_positions,
                                                   peer_cache_values);
        ScopedRankTensorBinding peer_rope_binding(tp_count_, ranks_, &TextRankBinding::rope_positions,
                                                  peer_cache_values);
        ScopedRankTensorBinding peer_rows_binding(tp_count_, ranks_, &TextRankBinding::kv_table_rows,
                                                  peer_row_values);
        ScopedRankTensorBinding peer_slots_binding(tp_count_, ranks_,
                                                   &TextRankBinding::linear_state_slots,
                                                   peer_slot_values);
        ScopedRankTensorBinding peer_valid_binding(tp_count_, ranks_,
                                                   &TextRankBinding::valid_columns,
                                                   peer_valid_values);
        auto scope_0 = work_.scope();
        auto scope_1 = rank_scopes();
        TpArray<Tensor> x;
        TpArray<Tensor> staging;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            x[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, width * batch});
            staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, width * batch});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::embedding(ids[r].view({width * batch}), rank_embed(rank),
                           x[r], stream_for(rank));
        });
        run_layers_tp2(x, Phase::Verify, staging, &sink);
        TpArray<Tensor> flat_hidden;
        TpArray<Tensor> flat_logits;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            flat_hidden[r] = hidden[r].view({kCfg.hidden, width * batch});
            flat_logits[r] = logits[r].view({kCfg.vocab, width * batch});
            for_each_rank(execution, [&](int rank) {
                if (static_cast<std::size_t>(rank) == r) {
                    ops::rmsnorm(x[r], rank_final_norm(rank), kCfg.rms_eps,
                                 true, flat_hidden[r], stream_for(rank));
                }
            });
        }
        logits_tp2(flat_hidden, flat_logits);
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            Tensor flat_tokens = target_tokens[r].view({width * batch});
            ops::argmax(flat_logits[r], flat_tokens, kCfg.token_domain, stream_for(rank));
        });
    }
    work_.reset();
    reset_peer_workspaces();
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids,
                                           const TpArray<Tensor>& hidden,
                                           const TpArray<Tensor>& cache_positions,
                                           const TpArray<Tensor>& rope_positions,
                                           const TpArray<Tensor>& valid_columns,
                                           const TpArray<Tensor>& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope,
                                           const TpArray<Tensor>& mtp_hidden) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP decode requires a peer"); }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    for (int rank = 1; rank < tp_count_; ++rank) {
        if (rank_binding(rank).execution.batch_mtp_kv == nullptr) {
            throw std::runtime_error("MTP forward is not enabled on every rank");
        }
    }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "MTP decode batch target hidden");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "MTP decode batch cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "MTP decode batch RoPE positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "MTP decode batch valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch}, "MTP decode batch KV rows");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "MTP decode batch hidden");
    }

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows[0]);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
    std::array<const Tensor*, kMaximumDevices> peer_backend_values{};
    std::array<const Tensor*, kMaximumDevices> peer_valid_values{};
    for (std::int32_t r = 1; r < tp_count_; ++r) {
        const auto slot              = static_cast<std::size_t>(r);
        peer_backend_values[slot]    = &kv_table_rows[slot];
        peer_valid_values[slot]      = &valid_columns[slot];
    }
    ScopedRankTensorBinding peer_backend_binding(tp_count_, ranks_,
                                                 &TextRankBinding::backend_kv_table_rows,
                                                 peer_backend_values);
    ScopedRankTensorBinding peer_valid_binding(tp_count_, ranks_, &TextRankBinding::valid_columns,
                                               peer_valid_values);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core_tp2(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_propose_batch(const TpArray<Tensor>& hidden,
                                    const TpArray<Tensor>& logits, Tensor& draft_tokens) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP proposal requires a peer"); }
    const std::int32_t batch = hidden[0].ne[1];
    for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, batch},
                             "MTP proposal batch hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, batch},
                             "MTP proposal batch logits");
    }
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    auto scope_0 = work_.scope();
    auto scope_1 = rank_scopes();
    proposal_argmax_tp2(hidden, logits, draft_tokens);
}

void TextContext::mtp_forward_batch(const Tensor& ids, const TpArray<Tensor>& hidden,
                                    const TpArray<Tensor>& positions,
                                    const TpArray<Tensor>& rope_positions,
                                    ops::GqaExecutionEnvelope envelope,
                                    const TpArray<Tensor>& mtp_hidden, int logits_column,
                                    const TpArray<Tensor>* logits, Tensor* draft_token) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP batch requires a peer"); }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    for (int rank = 1; rank < tp_count_; ++rank) {
        if (rank_binding(rank).execution.batch_mtp_kv == nullptr) {
            throw std::runtime_error("MTP forward is not enabled on every rank");
        }
    }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP positions");
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, T}, "MTP hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
        const Tensor& rope = rope_positions[r];
        if (rope.dtype != DType::I32 || rope.ne[0] != T ||
            (rope.ne[1] != 1 && rope.ne[1] != 3) || rope.ne[2] != 1 || rope.ne[3] != 1 ||
            !rope.is_contiguous() || rope.data == nullptr) {
            throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
        }
    }
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        for (const Tensor& destination : *logits) {
            require_tensor_shape(destination, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        }
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    mtp_forward_core_tp2(ids, hidden, positions, rope_positions, envelope, mtp_hidden);
    if (logits_column >= 0) {
        // One column per rank, not two: the proposal gather reads every rank's shard, so a
        // two-element list here would hand ranks 2..tp-1 default-constructed (null) tensors and the
        // gather would read through them.
        TpArray<Tensor> columns;
        for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
            columns[r] = mtp_hidden[r].slice(1, logits_column, 1);
        }
        proposal_argmax_tp2(columns, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token,
                                      const TpArray<Tensor>& previous_hidden,
                                      const TpArray<Tensor>& position,
                                      ops::GqaExecutionEnvelope envelope,
                                      const TpArray<Tensor>& mtp_hidden,
                                      const TpArray<Tensor>& logits, Tensor& draft_token) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP AR step requires a peer"); }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    for (int rank = 1; rank < tp_count_; ++rank) {
        if (rank_binding(rank).execution.batch_mtp_kv == nullptr) {
            throw std::runtime_error("MTP forward is not enabled on every rank");
        }
    }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");
    for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
        require_tensor_shape(position[r], DType::I32, {1}, "MTP AR position");
        require_tensor_shape(previous_hidden[r], DType::BF16, {kCfg.hidden, 1},
                             "MTP AR previous hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    }
    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    auto position_scope_0                   = work_.scope();
    auto position_scope_1 = rank_scopes();
    TpArray<Tensor> rope_position;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) { rope_position[r] = ws[r]->alloc(DType::I32, {1}); }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::offset_i32_positions(position[r], io_for(rank).rope_delta, rope_position[r],
                                  stream_for(rank));
    });
    mtp_forward_core_tp2(token, previous_hidden, position, rope_position, envelope, mtp_hidden);
    auto logits_scope_0 = work_.scope();
    auto logits_scope_1 = rank_scopes();
    proposal_argmax_tp2(mtp_hidden, logits, draft_token);
}

void TextContext::mtp_prefill_chunk_tp2(const Tensor& ids, const TpArray<Tensor>& hidden,
                                        const TpArray<Tensor>& positions,
                                        const TpArray<Tensor>& rope_positions,
                                        ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                        const TpArray<Tensor>* final_hidden,
                                        const TpArray<Tensor>* logits,
                                        Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    for (int rank = 1; rank < tp_count_; ++rank) {
        if (!rank_mtp_kv(rank).valid()) {
            throw std::runtime_error("MTP prefill is not enabled on every rank");
        }
    }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    // member function: the ExecutionContext comes from the accessor, not a local alias
    for (std::size_t r = 0; r < static_cast<std::size_t>(ec().tp); ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP prefill positions");
        require_tensor_shape(rope_positions[r], DType::I32, {T}, "MTP prefill rope positions");
    }
    if (final_chunk && (final_hidden == nullptr || logits == nullptr || draft_token == nullptr)) {
        throw std::invalid_argument("MTP final prefill outputs are required");
    }

    const ExecutionContext& execution       = ec();
    const TpArray<WorkspaceArena*> ws = workspaces();
    auto scratch_scope_0                    = work_.scope();
    auto scratch_scope_1 = rank_scopes();
    TpArray<Tensor> staging;
    TpArray<Tensor> last_staging;
    TpArray<Tensor> x_last;
    TpArray<Tensor> ah_last;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, T});
        if (final_chunk) {
            // The final-chunk stage reduces ONE column, and `linear_row_parallel` requires the
            // staging buffer to match its output's shape exactly on both devices -- the chunk's
            // [hidden, T] staging would be the wrong shape for it.
            last_staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
            x_last[r]       = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
            ah_last[r]      = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
        }
    }

    {
        auto bulk_scope_0 = work_.scope();
        auto bulk_scope_1 = rank_scopes();
        TpArray<Tensor> x;
        TpArray<Tensor> ah;
        mtp_forward_stem_tp2(ids, hidden, x, ah, staging);

        TpArray<Tensor> k_flat;
        TpArray<Tensor> v_flat;
        for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
            k_flat[r] = ws[r]->alloc(DType::BF16, {shard_kv_size(), T});
            v_flat[r] = ws[r]->alloc(DType::BF16, {shard_kv_size(), T});
        }
        Variant::mtp_kv_projection(
            ah, mtp_attention_weights(),
            k_flat, v_flat, ws, execution);
        for_each_rank(execution, [&](int rank) {
            const auto r    = static_cast<std::size_t>(rank);
            cudaStream_t s  = stream_for(rank);
            const MtpW& mtp = mtp_weights_for(rank);
            Tensor k        = k_flat[r].view({kCfg.head_dim, shard_kv_heads(), T});
            Tensor v        = v_flat[r].view({kCfg.head_dim, shard_kv_heads(), T});
            Tensor kn       = ws[r]->alloc(DType::BF16, {kCfg.head_dim, shard_kv_heads(), T});
            ops::rmsnorm(k, *mtp.k_norm, kCfg.rms_eps, true, kn, s);
            ops::rope(rope_positions[r], kCfg.rotary_dim, kCfg.rope_theta, kn,
                      rope_frequency_[r], s);
            qwen3_6::PagedKVCacheView pages = rank_mtp_kv(rank);
            ops::gqa_kv_append(kn, v, positions[r], pages.layer_view(0), s);
            if (final_chunk) {
                const std::size_t column_bytes =
                    static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
                const auto* x_src = static_cast<const unsigned char*>(x[r].data) +
                                    static_cast<std::size_t>(T - 1) * column_bytes;
                const auto* ah_src = static_cast<const unsigned char*>(ah[r].data) +
                                     static_cast<std::size_t>(T - 1) * column_bytes;
                CUDA_CHECK(cudaMemcpyAsync(x_last[r].data, x_src, column_bytes,
                                           cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(ah_last[r].data, ah_src, column_bytes,
                                           cudaMemcpyDeviceToDevice, s));
            }
        });
    }
    mtp_stage_barrier(execution, "mtp_prefill: stem + kv append");

    if (!final_chunk) { return; }

    TpArray<Tensor> q_flat;
    TpArray<Tensor> gate_flat;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        q_flat[r]    = ws[r]->alloc(DType::BF16, {shard_q_size(), 1});
        gate_flat[r] = ws[r]->alloc(DType::BF16, {shard_q_size(), 1});
    }
    Variant::mtp_q_gate_projection(
        ah_last, mtp_attention_weights(),
        q_flat, gate_flat, ws, execution);
    TpArray<Tensor> a;
    TpArray<Tensor> o;
    TpArray<Tensor> mh;
    for (std::size_t r = 0; r < static_cast<std::size_t>(execution.tp); ++r) {
        a[r]  = ws[r]->alloc(DType::BF16, {kCfg.head_dim, shard_q_heads(), 1});
        o[r]  = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
        mh[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r    = static_cast<std::size_t>(rank);
        cudaStream_t s  = stream_for(rank);
        const MtpW& mtp = mtp_weights_for(rank);
        Tensor q        = q_flat[r].view({kCfg.head_dim, shard_q_heads(), 1});
        Tensor gate     = gate_flat[r].view({kCfg.head_dim, shard_q_heads(), 1});
        Tensor qn       = ws[r]->alloc(DType::BF16, {kCfg.head_dim, shard_q_heads(), 1});
        ops::rmsnorm(q, *mtp.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions[r].slice(0, T - 1, 1);
        Tensor last_rope     = rope_positions[r].slice(0, T - 1, 1);
        ops::rope(last_rope, kCfg.rotary_dim, kCfg.rope_theta, qn, rope_frequency_[r], s);
        qwen3_6::PagedKVCacheView pages = rank_mtp_kv(rank);
        ops::gqa_attention_cached(qn, last_position, kAttnScale, pages.layer_view(0), envelope,
                                  *ws[r], a[r], s);
        ops::sigmoid_mul(gate, a[r], s);
    });
    ops::linear_row_parallel(attention_flat(a, 1), mtp_output_weights(), o, last_staging,
                             execution, peer_events());
    for_each_rank(execution, [&](int rank) {
        const auto r   = static_cast<std::size_t>(rank);
        cudaStream_t s = stream_for(rank);
        ops::residual_add(o[r], x_last[r], s);
        ops::rmsnorm(x_last[r], *mtp_weights_for(rank).post_attn_norm, kCfg.rms_eps, true, mh[r],
                     s);
    });
    mtp_stage_barrier(execution, "mtp_prefill: attention + output projection");
    {
        auto post_scope_0 = work_.scope();
        auto post_scope_1 = rank_scopes();
        Variant::mtp_post_mixer(
            mh, mtp_post_mixer_weights(),
            x_last, last_staging, ws, execution, peer_events());
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        require_tensor_shape((*final_hidden)[r], DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        ops::rmsnorm(x_last[r], *mtp_weights_for(rank).norm, kCfg.rms_eps, true,
                     const_cast<Tensor&>((*final_hidden)[r]), stream_for(rank));
    });
    mtp_stage_barrier(execution, "mtp_prefill: post mixer + final norm");
    proposal_argmax_tp2(*final_hidden, *logits, *draft_token);
    mtp_stage_barrier(execution, "mtp_prefill: proposal argmax");
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
