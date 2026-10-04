#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Nvfp4AttnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return tokens >= 4 ? Nvfp4AttnInputRoute::W4A4 : Nvfp4AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    // SM70 fast path (ported from upstream e27de9bf, 10/04): the QPN pre-packed-weight kernel
    // writes directly into the four output planes (T < 128); T >= 128 uses cutlass + split.
    // The small_t chunk loop below re-reads the whole weight every 32 tokens and ran 3.4-4.0x
    // slower per launch than the QPN kernel at decode (185-189us vs 43-51us, nsys decode
    // window 10/04), so it is unreachable in VOLTA builds.
    nvfp4_attn_input_sm70_launch(x, weight, q, gate, k, v, workspace, stream);
    return;
#endif
    (void)workspace;
    constexpr std::int32_t kChunk  = kNvfp4LastSmallT;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor query_chunk(query, DType::BF16, {kQRows, active});
        Tensor gate_chunk(output_gate, DType::BF16, {kQRows, active});
        Tensor key_chunk(key, DType::BF16, {kKvRows, active});
        Tensor value_chunk(value, DType::BF16, {kKvRows, active});
        if (active == 1) {
            nvfp4_attn_input_decode_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                           value_chunk, stream);
        } else {
            nvfp4_attn_input_small_t_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                            value_chunk, stream);
        }
    }
}

// The tp2/tp4 column shard -- same chunking discipline, per-rank row counts taken from the
// caller's own output shapes rather than hardcoded to Nvfp4AttnInputTp2ColumnGeometry. The tp2
// constants baked in originally are wrong for the tp4 shard (1536/256): at any token_begin > 0
// the output pointers stride by the tp2 width and write out of bounds while the real token
// region stays unwritten (found 10/04 by the tp4 op-level test: non-finite at the last-chunk
// tokens, T>=33).
void launch_a16_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                      Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    // SM70 fast path (ported from upstream e27de9bf, 10/04): QPN pre-packed-weight kernel for
    // T < 128, cutlass + split for T >= 128 -- replaces the previous T >= 33 cutlass / T <= 32
    // small_t split. The small_t chunk re-reads the whole shard weight every 32 tokens and ran
    // 3.4-4.0x slower per launch than the QPN kernel at decode (185-189us vs 43-51us, nsys
    // decode window 10/04), i.e. it accounted for ~85% of the fork-vs-upstream decode gap.
    // The dedicated cutlass-shard launcher (nvfp4_attn_input_cutlass_sm70_launch_shard) is
    // now unused in VOLTA builds.
    nvfp4_attn_input_sm70_launch(x, weight, q, gate, k, v, workspace, stream);
    return;
#endif
    constexpr std::int32_t kChunk  = kNvfp4LastSmallT;
    const std::int32_t kQRows      = q.ne[0];
    const std::int32_t kKvRows     = k.ne[0];
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor query_chunk(query, DType::BF16, {kQRows, active});
        Tensor gate_chunk(output_gate, DType::BF16, {kQRows, active});
        Tensor key_chunk(key, DType::BF16, {kKvRows, active});
        Tensor value_chunk(value, DType::BF16, {kKvRows, active});
        if (active == 1) {
            nvfp4_attn_input_decode_launch_shard(input_chunk, weight, query_chunk, gate_chunk,
                                                 key_chunk, value_chunk, stream);
        } else {
            nvfp4_attn_input_small_t_launch_shard(input_chunk, weight, query_chunk, gate_chunk,
                                                  key_chunk, value_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    if (resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::W4A4) {
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4AttnInputGeometry::kInputRows);
    }
#ifdef NINFER_VOLTA_BUILD
    // The sm70 fast path (T >= 128 cutlass + split) and the retained dedicated cutlass-shard
    // launcher both stage the projection and the GEMM operands in the caller workspace. Size to
    // the larger of the two, at the parent geometry (the shard needs less; the over-provision
    // is the FP8 rule; the variant's capacity query passes the parent row count).
    constexpr std::int32_t kVoltaCutlassMinT = 33;
    if (max_tokens >= kVoltaCutlassMinT) {
        return std::max(nvfp4_attn_input_cutlass_workspace_bytes(max_tokens),
                        nvfp4_attn_input_sm70_workspace_bytes(
                            Nvfp4AttnInputGeometry::kOutputRows, max_tokens));
    }
#endif
    return 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

// --- TP2 column-shard sibling --------------------------------------------------------------------
// Route selection (resolve_route) is a pure function of (policy, token count), inherited unchanged
// from the tp1 parent; only the underlying kernel Geometry (and its section offsets) differ. The
// W4A4 quantize workspace size is a pure function of (tokens, K) and K=5120 is unchanged by the
// shard (only the output row count N halves), so nvfp4_attn_input_workspace_capacity_bytes above is
// reused as-is -- no separate shard capacity query is needed.
void nvfp4_attn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, LinearPolicy policy,
                                     WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch_a16_shard(x, weight, q, gate, k, v, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj column-parallel requires caller "
                                    "workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch_shard(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
