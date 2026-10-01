#include "targets/qwen3_6_27b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"
#include "ninfer/types.h" // TpArray, kMaximumDevices

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

#if defined(NINFER_SM8X_COMPAT) || defined(NINFER_VOLTA_BUILD)
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::AllowA8;
#endif

ops::LinearPolicy text_policy(const Weight& weight) {
    switch (weight.qtype) {
    case QType::NVFP4:
        return kNvfp4TextPolicy;
    case QType::FP8_E4M3FN_ROW_BF16S:
        return kFp8TextPolicy;
    default:
        return ops::LinearPolicy::A16Only;
    }
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                  std::uint32_t draft_window,
                                                                  std::uint32_t) {
    if (capacity == 0 || draft_window == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8192U, 16384U, 32768U}) {
        if (visible_end > draft_window + 1U) {
            ends.push_back(visible_end - draft_window - 1U);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1U, ends);
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    if (weight.qtype == QType::GGML_K) {
        ops::ggml_k_gdn_output(hidden, weight, residual, workspace, stream);
        return;
    }
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up), workspace,
                       stream);
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return 0;
    case WeightsProfile::Qwen38GgmlK:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::GGML_K, 14336, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 14336, TextConfig::hidden, kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return ops::linear_add_workspace_capacity_bytes(
            QType::GGML_K, TextConfig::hidden, TextConfig::value_dim,
            ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::query_size,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::GGML_K, 16384, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return 0;
    case WeightsProfile::Qwen36Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::GGML_K, 16384, TextConfig::hidden,
                            ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::GGML_K, 16384, TextConfig::hidden,
                            ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return ops::linear_add_workspace_capacity_bytes(
            QType::GGML_K, TextConfig::hidden, TextConfig::value_dim,
            ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::value_dim,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    if (weights_profile == WeightsProfile::Qwen38GgmlK) {
        return ops::linear_workspace_capacity_bytes(
            QType::GGML_K, TextConfig::gdn_value_heads, TextConfig::hidden,
            ops::LinearPolicy::A16Only, first, last);
    }
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return post_mixer_workspace_bytes(QType::GGML_K, QType::GGML_K,
                                          ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return post_mixer_workspace_bytes(QType::Q4G64_F16S, QType::Q5G64_F16S,
                                          ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first,
                                          last);
    case WeightsProfile::Qwen38Nvfp4: {
        const std::size_t nvfp4 =
            post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first, last);
        const std::size_t fp8 = post_mixer_workspace_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, kFp8TextPolicy, first, last);
        return std::max(nvfp4, fp8);
    }
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

// --- tp == 2 split leaves ----------------------------------------------------------------------
//
// Each of these is the tp1 leaf above with every per-rank argument taken as an array and the
// matching `*_column_parallel` / `*_row_parallel` Op called ONCE for both ranks. The weight
// variant (fused vs split storage) is resolved from rank 0 and required to agree on rank 1: both
// ranks bind the same artifact objects through the same profile, so a disagreement is a loader
// bug, not a supported configuration.

namespace {

// Every live rank's payload, after requiring that all of them use the SAME storage form. The
// variant is resolved from every rank rather than from rank 0 with the rest assumed: a rank that
// bound a different form would otherwise run a different kernel from its peers.
template <class Payload, class Weights>
TpArray<const Payload*> require_same_alternative(const TpArray<const Weights*>& w,
                                                 const char* label, const ExecutionContext& ec) {
    TpArray<const Payload*> out{};
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot     = static_cast<std::size_t>(rank);
        const auto* payload = std::get_if<Payload>(w[slot]);
        if (payload == nullptr) {
            throw std::logic_error(std::string(label) +
                                   ": ranks disagree on weight storage form");
        }
        out[slot] = payload;
    }
    return out;
}

// One entry per live rank, selected out of that rank's payload.
template <class Result, class Payload, class Select>
TpArray<Result> rank_shards(const ExecutionContext& ec, const TpArray<const Payload*>& payloads,
                            Select&& select) {
    TpArray<Result> out{};
    for (int rank = 0; rank < ec.tp; ++rank) {
        out[rank] = select(*payloads[static_cast<std::size_t>(rank)]);
    }
    return out;
}

// Current-device save/restore around a per-rank kernel issue. Only ONE tp2 leaf below needs it --
// `mtp_attention_projection`, whose second stage (`ops::mtp_split_attn_in`) is a purely
// elementwise remap with no cross-device form and no reason for one, so it is issued once per
// rank on that rank's own stream. Every other split leaf delegates wholly to an Op that owns its
// own per-rank issue. This mirrors ops::detail::for_each_rank rather than including an Op's
// private header, exactly as the family schedule does (text_context_impl.h).
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

// Both registered MTP codecs use A16 projections without transient Linear storage.
void require_mtp_shard(const Weight& a, const Weight& b, const char* label) {
    if (a.qtype != b.qtype ||
        (a.qtype != QType::W8G32_F16S && a.qtype != QType::GGML_K)) {
        throw std::logic_error(std::string(label) +
                               ": unsupported or inconsistent MTP weight format");
    }
    if (a.k != b.k || a.n != b.n) {
        throw std::logic_error(std::string(label) + ": MTP shards disagree on shape");
    }
}

// Requires every rank's shard of one member to agree, checked against rank 0's. `get(rank)` names
// that rank's shard.
template <class Getter>
void require_agreeing_shards(const ExecutionContext& ec, const char* label, Getter&& get) {
    for (int rank = 1; rank < ec.tp; ++rank) {
        require_mtp_shard(get(0), get(rank), label);
    }
}

} // namespace

void Variant::attention_projection(const TpArray<Tensor>& hidden,
                                   const TpArray<const FullAttentionProjectionWeights*>& w,
                                   const TpArray<Tensor>& query,
                                   const TpArray<Tensor>& gate,
                                   const TpArray<Tensor>& key,
                                   const TpArray<Tensor>& value, qwen3_6::TextPhase,
                                   const TpArray<WorkspaceArena*>& workspace,
                                   const ExecutionContext& ec) {
    if (std::holds_alternative<SplitAttentionProjectionPayload>(*w[0])) {
        const auto split = require_same_alternative<SplitAttentionProjectionPayload>(
            w, "attention projection", ec);
        ops::attn_input_proj_column_parallel(
            hidden,
            rank_shards<Weight>(ec, split,
                                [](const SplitAttentionProjectionPayload& p) { return p.query_key; }),
            rank_shards<Weight>(
                ec, split,
                [](const SplitAttentionProjectionPayload& p) { return p.gate_value; }),
            query, gate, key, value, workspace, ec);
        return;
    }
    const auto fused = require_same_alternative<FusedAttentionProjectionPayload>(
        w, "attention projection", ec);
    ops::attn_input_proj_column_parallel(
        hidden,
        rank_shards<Weight>(ec, fused,
                            [](const FusedAttentionProjectionPayload& p) {
                                return p.query_key_gate_value;
                            }),
        query, gate, key, value, text_policy(fused[0]->query_key_gate_value), workspace, ec);
}

void Variant::attention_output_projection(const TpArray<Tensor>& attention,
                                          const TpArray<Weight>& weight,
                                          const TpArray<Tensor>& residual,
                                          const TpArray<Tensor>& staging, qwen3_6::TextPhase,
                                          const TpArray<WorkspaceArena*>& workspace,
                                          const ExecutionContext& ec, const ops::PeerEvents& ev) {
    ops::linear_add_row_parallel(attention, weight, residual, staging, text_policy(weight[0]),
                                 workspace, ec, ev);
}

void Variant::gdn_input_projection(const TpArray<Tensor>& hidden,
                                   const TpArray<const GdnProjectionWeights*>& w,
                                   const TpArray<Tensor>& qkv,
                                   const TpArray<Tensor>& output_gate, qwen3_6::TextPhase,
                                   const TpArray<WorkspaceArena*>& workspace,
                                   const ExecutionContext& ec) {
    // The caller holds `z` as [head_dim, value_heads, T]; the Op wants the flat [value_dim, T],
    // exactly as the tp1 leaf above does. The shard's value_dim is read off the tensor rather than
    // assumed, so this stays correct if the head split ever changes.
    TpArray<Tensor> output_gate_flat;
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot    = static_cast<std::size_t>(rank);
        output_gate_flat[slot] =
            output_gate[slot].view({output_gate[slot].ne[0] * output_gate[slot].ne[1],
                                    hidden[slot].ne[1]});
    }
    const TpArray<const GdnInputProjectionPayload*> input =
        rank_shards<const GdnInputProjectionPayload*>(
            ec, w, [](const GdnProjectionWeights& x) { return &x.input_projection; });
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split = require_same_alternative<SplitGdnInputProjectionPayload>(
            input, "GDN input projection", ec);
        ops::gdn_input_proj_column_parallel(
            hidden,
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnInputProjectionPayload& p) { return p.query_key; }),
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnInputProjectionPayload& p) { return p.value_z; }),
            qkv, output_gate_flat, ec);
        return;
    }
    const auto fused = require_same_alternative<FusedGdnInputProjectionPayload>(
        input, "GDN input projection", ec);
    ops::gdn_input_proj_column_parallel(
        hidden,
        rank_shards<Weight>(ec, fused,
                            [](const FusedGdnInputProjectionPayload& p) {
                                return p.query_key_value_z;
                            }),
        qkv, output_gate_flat, text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::gdn_input_projection_snapshot(
    const TpArray<Tensor>& hidden, const TpArray<const GdnProjectionWeights*>& w,
    const TpArray<Tensor>& conv_weight, const TpArray<Tensor>& conv_states,
    const TpArray<Tensor>& valid_columns, const TpArray<Tensor>& initial_slot,
    const TpArray<Tensor>& snapshot_base_slot, const TpArray<Tensor>& query,
    const TpArray<Tensor>& key, const TpArray<Tensor>& value,
    const TpArray<Tensor>& output_gate, qwen3_6::TextPhase,
    const TpArray<WorkspaceArena*>& workspace, const ExecutionContext& ec) {
    const TpArray<const GdnInputProjectionPayload*> input =
        rank_shards<const GdnInputProjectionPayload*>(
            ec, w, [](const GdnProjectionWeights& x) { return &x.input_projection; });
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split = require_same_alternative<SplitGdnInputProjectionPayload>(
            input, "GDN snapshot projection", ec);
        ops::gdn_input_proj_conv_snapshot_column_parallel(
            hidden,
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnInputProjectionPayload& p) { return p.query_key; }),
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnInputProjectionPayload& p) { return p.value_z; }),
            conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot, query, key,
            value, output_gate, workspace, ec);
        return;
    }
    const auto fused = require_same_alternative<FusedGdnInputProjectionPayload>(
        input, "GDN snapshot projection", ec);
    ops::gdn_input_proj_conv_snapshot_column_parallel(
        hidden,
        rank_shards<Weight>(ec, fused,
                            [](const FusedGdnInputProjectionPayload& p) {
                                return p.query_key_value_z;
                            }),
        conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot, query, key,
        value, output_gate, text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::gdn_output_projection(const TpArray<Tensor>& hidden,
                                    const TpArray<Weight>& weight,
                                    const TpArray<Tensor>& residual,
                                    const TpArray<Tensor>& staging, qwen3_6::TextPhase,
                                    const TpArray<WorkspaceArena*>& workspace,
                                    const ExecutionContext& ec, const ops::PeerEvents& ev) {
    if (weight[0].qtype == QType::GGML_K) {
        ops::ggml_k_gdn_output(hidden, weight, residual, staging, workspace, ec, ev);
        return;
    }
    ops::linear_add_row_parallel(hidden, weight, residual, staging, text_policy(weight[0]),
                                 workspace, ec, ev);
}

void Variant::gdn_control_projection(const TpArray<Tensor>& hidden,
                                     const TpArray<const GdnProjectionWeights*>& w,
                                     const TpArray<Tensor>& g,
                                     const TpArray<Tensor>& beta,
                                     const TpArray<WorkspaceArena*>& workspace,
                                     const ExecutionContext& ec) {
    // The tp1 leaf fuses the input RMSNorm into the gating GEMM. There is no split form of the
    // fused kernel and no reason for one: the norm is replicated elementwise work over the
    // full-width residual, so the caller runs it per rank and this leaf takes the normalized
    // hidden directly.
    const TpArray<const GdnControlProjectionPayload*> control =
        rank_shards<const GdnControlProjectionPayload*>(
            ec, w, [](const GdnProjectionWeights& x) { return &x.control_projection; });
    const TpArray<Tensor> a_log =
        rank_shards<Tensor>(ec, w, [](const GdnProjectionWeights& x) { return x.a_log; });
    const TpArray<Tensor> dt_bias =
        rank_shards<Tensor>(ec, w, [](const GdnProjectionWeights& x) { return x.dt_bias; });
    if (std::holds_alternative<SplitGdnControlProjectionPayload>(*control[0])) {
        const auto split = require_same_alternative<SplitGdnControlProjectionPayload>(
            control, "GDN control projection", ec);
        ops::gdn_gating_proj_column_parallel(
            hidden,
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnControlProjectionPayload& p) {
                                    return p.a_projection;
                                }),
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnControlProjectionPayload& p) {
                                    return p.b_projection;
                                }),
            a_log, dt_bias, workspace, g, beta, ec);
        return;
    }
    const auto fused = require_same_alternative<FusedGdnControlProjectionPayload>(
        control, "GDN control projection", ec);
    ops::gdn_gating_proj_column_parallel(
        hidden,
        rank_shards<Weight>(ec, fused,
                            [](const FusedGdnControlProjectionPayload& p) {
                                return p.a_b_projection;
                            }),
        a_log, dt_bias, workspace, g, beta, ec);
}

void Variant::post_mixer(const TpArray<Tensor>& hidden,
                         const TpArray<const PostMixerWeights*>& w,
                         const TpArray<Tensor>& residual,
                         const TpArray<Tensor>& staging, qwen3_6::TextPhase,
                         const TpArray<WorkspaceArena*>& workspace,
                         const ExecutionContext& ec, const ops::PeerEvents& ev) {
    // The activation width is this rank's own gate/up shard, read off the weight rather than
    // assumed: `gate_up` is [2 * intermediate_shard, hidden], so half its rows is the shard.
    const std::int32_t shard_intermediate = w[0]->gate_up.n / 2;
    for (int rank = 1; rank < ec.tp; ++rank) {
        if (w[static_cast<std::size_t>(rank)]->gate_up.n != w[0]->gate_up.n) {
            throw std::logic_error("post_mixer: ranks' gate/up shards disagree on width");
        }
    }
    TpArray<Tensor> activation{};
    // Scope guards cannot be default-constructed, so a fixed-width per-rank array would
// have to name a guard for slots that do not exist; a vector sized by ec.tp does not.
    std::vector<WorkspaceArena::Scope> scopes;
    scopes.reserve(static_cast<std::size_t>(ec.tp));
    for (int r = 0; r < ec.tp; ++r) scopes.push_back(workspace[r]->scope());
    for (std::size_t rank = 0; rank < static_cast<std::size_t>(ec.tp); ++rank) {
        activation[rank] =
            workspace[rank]->alloc(DType::BF16, {shard_intermediate, hidden[0].ne[1]});
    }
    ops::linear_swiglu_column_parallel(
        hidden, rank_shards<Weight>(ec, w, [](const PostMixerWeights& x) { return x.gate_up; }),
        activation, text_policy(w[0]->gate_up), workspace, ec);
    ops::linear_add_row_parallel(
        activation, rank_shards<Weight>(ec, w, [](const PostMixerWeights& x) { return x.down; }),
        residual, staging, text_policy(w[0]->down), workspace, ec, ev);
}

void Variant::gdn_input_projection_record(
    const TpArray<Tensor>& hidden, const TpArray<const GdnProjectionWeights*>& w,
    const TpArray<Tensor>& conv_weight, const TpArray<Tensor>& conv_states,
    const TpArray<Tensor>& valid_columns, const TpArray<Tensor>& initial_slots,
    const TpArray<Tensor>& conv_record, const TpArray<Tensor>& query,
    const TpArray<Tensor>& key, const TpArray<Tensor>& value,
    const TpArray<Tensor>& output_gate, qwen3_6::TextPhase,
    const TpArray<WorkspaceArena*>& workspace, const ExecutionContext& ec) {
    // The record twin of `gdn_input_projection_snapshot` above, reached only from the speculative
    // verify round: instead of snapshotting the post-round conv state it writes a per-column
    // `conv_record` that the peer's `ops::gdn_replay_fold` later folds at
    // FoldGeometry<48, 8, 24, 5120>. Everything else -- the shard extents, the weight storage
    // form, the per-rank issue -- is the snapshot leaf's.
    const TpArray<const GdnInputProjectionPayload*> input =
        rank_shards<const GdnInputProjectionPayload*>(
            ec, w, [](const GdnProjectionWeights& x) { return &x.input_projection; });
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split = require_same_alternative<SplitGdnInputProjectionPayload>(
            input, "GDN record projection", ec);
        ops::gdn_input_proj_conv_record_column_parallel(
            hidden,
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnInputProjectionPayload& p) { return p.query_key; }),
            rank_shards<Weight>(ec, split,
                                [](const SplitGdnInputProjectionPayload& p) { return p.value_z; }),
            conv_weight, conv_states, valid_columns, initial_slots, conv_record, query, key, value,
            output_gate, workspace, ec);
        return;
    }
    const auto fused = require_same_alternative<FusedGdnInputProjectionPayload>(
        input, "GDN record projection", ec);
    ops::gdn_input_proj_conv_record_column_parallel(
        hidden,
        rank_shards<Weight>(ec, fused,
                            [](const FusedGdnInputProjectionPayload& p) {
                                return p.query_key_value_z;
                            }),
        conv_weight, conv_states, valid_columns, initial_slots, conv_record, query, key, value,
        output_gate, text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::mtp_attention_projection(
    const TpArray<Tensor>& hidden,
    const TpArray<const MtpAttentionProjectionWeights*>& w,
    const TpArray<Tensor>& query, const TpArray<Tensor>& gate,
    const TpArray<Tensor>& key, const TpArray<Tensor>& value,
    const TpArray<WorkspaceArena*>& workspace, const ExecutionContext& ec) {
    require_agreeing_shards(ec, "MTP attention projection",
                            [&](int rank) { return w[static_cast<std::size_t>(rank)]->packed; });
    const std::int32_t columns   = hidden[0].ne[1];
    const std::int32_t attn_rows = w[0]->packed.n;
    for (int rank = 1; rank < ec.tp; ++rank) {
        if (hidden[static_cast<std::size_t>(rank)].ne[1] != columns) {
            throw std::logic_error("MTP attention projection: ranks disagree on token count");
        }
    }
    TpArray<Tensor> packed{};
    // Scope guards cannot be default-constructed, so a fixed-width per-rank array would
// have to name a guard for slots that do not exist; a vector sized by ec.tp does not.
    std::vector<WorkspaceArena::Scope> scopes;
    scopes.reserve(static_cast<std::size_t>(ec.tp));
    for (int r = 0; r < ec.tp; ++r) scopes.push_back(workspace[r]->scope());
    for (std::size_t rank = 0; rank < static_cast<std::size_t>(ec.tp); ++rank) {
        packed[rank] = workspace[rank]->alloc(DType::BF16, {attn_rows, columns});
    }
    ops::linear_column_parallel(
        hidden,
        rank_shards<Weight>(ec, w,
                            [](const MtpAttentionProjectionWeights& x) { return x.packed; }),
        packed, ec);
    // `mtp_split_attn_in` selects its section boundaries from the packed row count alone, so the
    // shard geometry (rows [0,3072) Q | [3072,3584) K | [3584,6656) Gate | [6656,7168) V) needs
    // no rank argument. The resulting head indices are DEVICE-LOCAL, which is exactly what the
    // head-local 12|2 gqa_attention downstream consumes.
    for_each_rank(ec, [&](int rank) {
        const auto r          = static_cast<std::size_t>(rank);
        const std::int32_t qh = query[r].numel() / (TextConfig::head_dim * columns);
        const std::int32_t kh = key[r].numel() / (TextConfig::head_dim * columns);
        Tensor query_heads    = query[r].view({TextConfig::head_dim, qh, columns});
        Tensor gate_heads     = gate[r].view({TextConfig::head_dim, qh, columns});
        Tensor key_heads      = key[r].view({TextConfig::head_dim, kh, columns});
        Tensor value_heads    = value[r].view({TextConfig::head_dim, kh, columns});
        ops::mtp_split_attn_in(packed[r], query_heads, key_heads, gate_heads, value_heads,
                               ec.dev[rank]->stream);
    });
}

void Variant::mtp_kv_projection(const TpArray<Tensor>& hidden,
                                const TpArray<const MtpAttentionProjectionWeights*>& w,
                                const TpArray<Tensor>& key,
                                const TpArray<Tensor>& value,
                                const TpArray<WorkspaceArena*>&, const ExecutionContext& ec) {
    // The tp1 leaf fuses these two into one `linear_pair`; the shard's key and value row views
    // are separate blocks of the same packed shard, so at tp2 they are two column-parallel calls.
    require_agreeing_shards(ec, "MTP key projection",
                            [&](int rank) { return w[static_cast<std::size_t>(rank)]->key; });
    require_agreeing_shards(ec, "MTP value projection",
                            [&](int rank) { return w[static_cast<std::size_t>(rank)]->value; });
    ops::linear_column_parallel(
        hidden,
        rank_shards<Weight>(ec, w, [](const MtpAttentionProjectionWeights& x) { return x.key; }),
        key, ec);
    ops::linear_column_parallel(
        hidden,
        rank_shards<Weight>(ec, w, [](const MtpAttentionProjectionWeights& x) { return x.value; }),
        value, ec);
}

void Variant::mtp_q_gate_projection(const TpArray<Tensor>& hidden,
                                    const TpArray<const MtpAttentionProjectionWeights*>& w,
                                    const TpArray<Tensor>& query,
                                    const TpArray<Tensor>& gate,
                                    const TpArray<WorkspaceArena*>&,
                                    const ExecutionContext& ec) {
    require_agreeing_shards(ec, "MTP query projection",
                            [&](int rank) { return w[static_cast<std::size_t>(rank)]->query; });
    require_agreeing_shards(ec, "MTP gate projection",
                            [&](int rank) { return w[static_cast<std::size_t>(rank)]->output_gate; });
    ops::linear_column_parallel(
        hidden,
        rank_shards<Weight>(ec, w, [](const MtpAttentionProjectionWeights& x) { return x.query; }),
        query, ec);
    ops::linear_column_parallel(
        hidden,
        rank_shards<Weight>(ec, w,
                            [](const MtpAttentionProjectionWeights& x) { return x.output_gate; }),
        gate, ec);
}

// See the call site: per-rank shard dump for the MTP post-mixer. First call only; writes
// <prefix>.<rank> as [gate_up shard | silu shard] in BF16.
void mtp_mlp_dump(const ExecutionContext& ec, const TpArray<Tensor>& gate_up,
                  const TpArray<Tensor>& activation, std::int32_t shard_intermediate) {
    static const char* prefix = std::getenv("NINFER_TP4_MTP_MLP_DUMP");
    if (prefix == nullptr) { return; }
    static bool done = false;
    if (done) { return; }
    done = true;
    const CurrentDevice restore;
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(ec.dev[slot]->device));
        const std::size_t first  = static_cast<std::size_t>(gate_up[slot].bytes());
        const std::size_t second = static_cast<std::size_t>(activation[slot].bytes());
        std::vector<unsigned char> host(first + second);
        CUDA_CHECK(cudaMemcpyAsync(host.data(), gate_up[slot].data, first,
                                   cudaMemcpyDeviceToHost, ec.dev[slot]->stream));
        CUDA_CHECK(cudaMemcpyAsync(host.data() + first, activation[slot].data, second,
                                   cudaMemcpyDeviceToHost, ec.dev[slot]->stream));
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[slot]->stream));
        const std::string path = std::string(prefix) + "." + std::to_string(rank);
        FILE* file             = std::fopen(path.c_str(), "wb");
        if (file == nullptr) { throw std::runtime_error("MTP MLP dump: path unreadable"); }
        const std::int32_t header[3] = {rank, shard_intermediate,
                                        static_cast<std::int32_t>(gate_up[slot].ne[1])};
        std::fwrite(header, sizeof(header), 1, file);
        std::fwrite(host.data(), 1, host.size(), file);
        std::fclose(file);
        std::fprintf(stderr, "[mtp-mlp] rank %d: gate_up rows=%d cols=%d, activation rows=%d\n", rank,
                     gate_up[slot].ne[0], gate_up[slot].ne[1], activation[slot].ne[0]);
        // In-process self-check of the pairing this function itself just computed: row i of the
        // activation must be silu(row i of the shard) * row (si + i) of the shard. Printing the
        // three raw numbers here rules out any doubt about how the dump file is parsed.
        {
            const auto* gu_bits = static_cast<const unsigned short*>(
                static_cast<const void*>(host.data()));
            const auto* act_bits = static_cast<const unsigned short*>(
                static_cast<const void*>(host.data() + first));
            const auto to_float = [](unsigned short bits) {
                const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
                float out                = 0.0F;
                std::memcpy(&out, &wide, sizeof(out));
                return out;
            };
            const std::int32_t cols = gate_up[slot].ne[1];
            for (std::int32_t i : {0, 1, 2}) {
                const float g = to_float(gu_bits[static_cast<std::size_t>(i) * cols]);
                const float u = to_float(gu_bits[static_cast<std::size_t>(shard_intermediate + i) * cols]);
                const float a = to_float(act_bits[static_cast<std::size_t>(i) * cols]);
                const float expect = (g / (1.0F + std::exp(-g))) * u;
                std::fprintf(stderr,
                             "[mtp-mlp]   row %d: gate=%+.5f up=%+.5f act=%+.5f silu(g)*u=%+.5f\n", i,
                             g, u, a, expect);
            }
        }
    }
}

void Variant::mtp_post_mixer(const TpArray<Tensor>& hidden,
                             const TpArray<const MtpPostMixerWeights*>& w,
                             const TpArray<Tensor>& residual,
                             const TpArray<Tensor>& staging,
                             const TpArray<WorkspaceArena*>& workspace,
                             const ExecutionContext& ec, const ops::PeerEvents& ev) {
    // The MTP post-mixer is composed exactly the way the tp1 leaf above composes it -- separate
    // `linear` / `silu_mul` / `linear` / `residual_add`, NOT the fused linear_swiglu + linear_add
    // pair the text post-mixer uses. That is not a stylistic choice: neither
    // `linear_swiglu_column_parallel` nor `linear_add_row_parallel` registers W8G32_F16S, which
    // is the format of every MTP object, and the tp1 MTP leaf already avoids both fused Ops for
    // the same reason. `tests/ops/test_mtp_split.cpp`'s Leg A proves this exact composition at
    // tp2 -- column-parallel gate_up, a shard-local silu_mul over the shard's own gate/up halves,
    // then row-parallel down plus the all-reduce.
    require_agreeing_shards(ec, "MTP post mixer gate/up",
                            [&](int rank) { return w[static_cast<std::size_t>(rank)]->gate_up; });
    require_agreeing_shards(ec, "MTP post mixer down",
                            [&](int rank) { return w[static_cast<std::size_t>(rank)]->down; });
    const std::int32_t shard_intermediate = w[0]->gate_up.n / 2;
    const std::int32_t columns            = hidden[0].ne[1];
    TpArray<Tensor> gate_up{};
    TpArray<Tensor> activation{};
    TpArray<Tensor> delta{};
    // Scope guards cannot be default-constructed, so a fixed-width per-rank array would
// have to name a guard for slots that do not exist; a vector sized by ec.tp does not.
    std::vector<WorkspaceArena::Scope> scopes;
    scopes.reserve(static_cast<std::size_t>(ec.tp));
    for (int r = 0; r < ec.tp; ++r) scopes.push_back(workspace[r]->scope());
    for (std::size_t rank = 0; rank < static_cast<std::size_t>(ec.tp); ++rank) {
        gate_up[rank]    = workspace[rank]->alloc(DType::BF16, {w[rank]->gate_up.n, columns});
        activation[rank] = workspace[rank]->alloc(DType::BF16, {shard_intermediate, columns});
        delta[rank]      = workspace[rank]->alloc(DType::BF16, {TextConfig::hidden, columns});
    }
    ops::linear_column_parallel(
        hidden,
        rank_shards<Weight>(ec, w, [](const MtpPostMixerWeights& x) { return x.gate_up; }), gate_up,
        ec);
    // Each rank's gate and up halves are its OWN shard's halves -- the ShardPlan splits gate_up
    // as two independent column blocks, so rank r holds gate rows [r*I/2 ...] and up rows in the
    // matching block, and the SiLU pairing is rank-local with nothing to communicate.
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::silu_mul(gate_up[r].slice(0, 0, shard_intermediate),
                      gate_up[r].slice(0, shard_intermediate, shard_intermediate), activation[r],
                      ec.dev[rank]->stream);
    });
    // Debug-only (NINFER_TP4_MTP_MLP_DUMP=<prefix>): one file per rank holding that rank's gate_up
    // shard and its silu output, so the global vectors can be stitched back together from the
    // column-parallel ownership (rank r owns gate_up rows [r*n/tp, (r+1)*n/tp), where the shard's
    // own first/second half are the gate and up blocks). The local identity
    // activation[i] == silu(gate[i]) * up[i] is then checkable per rank without any cross-width
    // agreement, which separates a wrong pairing from a wrong GEMM slice.
    mtp_mlp_dump(ec, gate_up, activation, shard_intermediate);
    // Debug-only (NINFER_TP4_MTP_WEIGHT_DUMP=<prefix>): each rank's RAW gate_up payload plus its
    // geometry header. The weight is input-independent, so the same GLOBAL rows can be compared
    // across widths byte for byte: rank r owns gate rows [r*gate/2/tp, ...) and the up rows in the
    // shard's second half, so tp4 ranks 0+1 must equal tp2 rank 0. A mismatch here localizes a
    // wrong per-rank weight image rather than a wrong GEMM.
    if (const char* wprefix = std::getenv("NINFER_TP4_MTP_WEIGHT_DUMP"); wprefix != nullptr) {
        static bool weight_done = false;
        if (!weight_done) {
            weight_done = true;
            for (int r = 0; r < ec.tp; ++r) {
                const auto rslot        = static_cast<std::size_t>(r);
                const Weight& weight    = w[rslot]->gate_up;
                const std::size_t bytes = static_cast<std::size_t>(weight.payload_bytes);
                std::vector<unsigned char> host(bytes);
                const CurrentDevice restore;
                CUDA_CHECK(cudaSetDevice(ec.dev[rslot]->device));
                CUDA_CHECK(cudaMemcpyAsync(host.data(), weight.payload, bytes,
                                           cudaMemcpyDeviceToHost, ec.dev[rslot]->stream));
                CUDA_CHECK(cudaStreamSynchronize(ec.dev[rslot]->stream));
                const std::string path = std::string(wprefix) + "." + std::to_string(r);
                FILE* file             = std::fopen(path.c_str(), "wb");
                if (file == nullptr) { throw std::runtime_error("MTP weight dump: unreadable"); }
                const std::int64_t header[5] = {r, weight.n, weight.k, weight.group,
                                                static_cast<std::int64_t>(weight.payload_bytes)};
                std::fwrite(header, sizeof(header), 1, file);
                std::fwrite(host.data(), 1, host.size(), file);
                std::fclose(file);
                std::fprintf(stderr,
                             "[mtp-weight] rank %d: n=%d k=%d group=%d bytes=%llu qtype=%d\n", r,
                             weight.n, weight.k, weight.group,
                             static_cast<unsigned long long>(weight.payload_bytes),
                             static_cast<int>(weight.qtype));
            }
        }
    }
    ops::linear_row_parallel(
        activation, rank_shards<Weight>(ec, w, [](const MtpPostMixerWeights& x) { return x.down; }),
        delta, staging, ec, ev);
    // `delta` is identical on both ranks after the collective, so the residual fold is replicated
    // elementwise work and keeps the residual bit-identical across devices.
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::residual_add(delta[r], const_cast<Tensor&>(residual[r]), ec.dev[rank]->stream);
    });
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
