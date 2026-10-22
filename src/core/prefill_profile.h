#pragma once

// Prefill segment profiler.
//
// Compiled in unconditionally; armed at run time by NINFER_PREFILL_PROFILE=1 and appended to
// NINFER_PREFILL_PROFILE_OUT (default stderr). A build that never sets the variable behaves like
// the uninstrumented one apart from one predictable branch per segment, so a single binary can be
// A/B-ed against itself in one boot.
//
// Every timestamp comes from CUDA events recorded on ONE stream, the profile stream: the caller
// sets it at the start of a prefill chunk (rank 0's compute stream) and each Scope whose stream
// does not match the profile stream is a no-op. That keeps the table attributable to the
// critical-path rank without having to model four streams, and it means the per-rank mirrored
// stages (norms, RoPE, gating) are timed once, on rank 0, while the multi-rank Ops
// (column/row-parallel projections, NCCL reductions) are timed over rank 0's own share of the
// enqueue, which is what rank 0's stream actually waits for.
//
// NESTING. `AllReduce` is recorded inside the row-parallel projection that issues it, so it is a
// sub-segment of GdnOutProj/AttnOutProj/MlpDown/MtpStem and must not be added to their totals.
// Every other segment is disjoint and their sum, plus the row-parallel share each reduction
// accounts for, is what has to reconcile with the chunk wall clock.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::profile {

enum class Seg : int {
    Embed = 0,
    NormIn,
    GdnCtrlProj,
    GdnInProj,
    GdnConv,
    GdnCore,
    GdnNormOut,
    GdnOutProj,
    AttnProj,
    AttnNormRope,
    AttnCore,
    AttnGate,
    AttnOutProj,
    AllReduce,
    NormPost,
    MlpGateUp,
    MlpDown,
    FinalNorm,
    LmHead,
    Sample,
    MtpStem,
    MtpKv,
    MtpTail,
    DFlashFeat,
    DFlashLayers,
    Count
};

[[nodiscard]] const char* seg_name(Seg seg) noexcept;
[[nodiscard]] bool seg_is_nested(Seg seg) noexcept;
[[nodiscard]] bool enabled() noexcept;
[[nodiscard]] cudaStream_t profile_stream() noexcept;
void set_profile_stream(cudaStream_t stream) noexcept;

// RAII segment scope. Records a start event on `stream` when armed and the end event when it goes
// out of scope. A no-op unless profiling is enabled and `stream` is the profile stream.
class Scope {
public:
    Scope(Seg seg, cudaStream_t stream) noexcept;
    ~Scope();
    // Closes the segment before the enclosing lexical block ends. Idempotent.
    void stop() noexcept;
    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Seg seg_;
    std::size_t index_ = 0;
    bool armed_        = false;
};

// Marks a prefill chunk: opens the wall-clock interval and resets the per-chunk table.
void begin_chunk(std::int32_t tokens, std::int32_t base, std::int32_t rank_count);
// Closes the wall-clock interval, drains the recorded events and appends the table.
void end_chunk();

} // namespace ninfer::profile
