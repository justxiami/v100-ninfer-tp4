#pragma once

// One-shot push allreduce for tp == 4, single-process UVA.
//
// WHY THIS EXISTS. The tp > 2 path uses NCCL (src/core/tp_comm.cu), measured at 4.6 ms per
// decode round for the 10 KiB allreduce shape. NCCL's per-collective cost is dominated by its
// transport setup/teardown. The push protocol (transcribed from 1Cat's sm70_cross_device_reduce_
// 1stage_push) keeps the sequencing inside a single kernel: every rank pushes its 16-byte packs
// into the peers' buffers, polls until the peers' packs arrive (sentinel = reserved bf16 NaN),
// reduces in a fixed rank order, and clears its slots. No host-side events, no NCCL channels.
//
// THE ONE THING THAT BROKE THE FIRST ATTEMPT. The kernel dereferences peer device pointers
// directly (raw cross-GPU access), which only works if peer access is EXPLICITLY enabled per
// ordered pair with cudaDeviceEnablePeerAccess -- querying cudaDeviceCanAccessPeer is not enough
// (it says "possible", not "enabled"). The first port never called EnablePeerAccess and never
// printed its return code, so the failure was misdiagnosed as a hardware constraint (it is not:
// the reopen sensor at bench-results/2026-10-22/push-ar-reopen/sensor.cu proves raw cross-GPU
// access works once peer access is enabled). The buffer allocation in tp_comm.cu now enables
// peer access for every ordered pair and checks the return code.
//
// THE PROTOCOL. Per rank, one device buffer of layout
//   [ epoch words: kSignalBytes ][ payload: kEpochs * 4 ranks * kMaxBytes ]
// The epoch words (one per launch block, 0/1) select which of the two payload epochs this launch
// uses, so a launching block never races a still-polling block from the previous launch. The
// payload is pre-filled with the sentinel byte 0x7f (every 16-byte pack = 0x7f7f7f... = 8
// sentinel bf16 NaNs). A real NaN 0x7f7f operand is escaped to 0x7e00 (another NaN, still
// decodes as NaN) before pushing, so the sentinel is never confused with a payload.
//
// IN-PLACE. The push kernel reads `input` and writes `output`; the caller passes the same buffer
// for both (an in-place sum), so each rank ends holding the identical sum. A single thread reads
// input[off] and later writes output[off] at the same address -- a plain read-modify-write, safe.
//
// FALLBACK. push_ar_try returns false when the push path cannot handle the call (push disabled by
// env, message larger than kMaxBytes, not a multiple of 16 bytes, not BF16, or no push buffer),
// and the caller falls back to NCCL. The decode allreduce is 10 KiB BF16, so it always fits.

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer {
namespace ops {

// Try to handle `buffer` (an in-place sum, every rank carrying the same contiguous BF16 shape)
// with the one-shot push AR. Returns true if the push kernel was launched on every rank's stream;
// false if the push path cannot handle this call and the caller must fall back to NCCL.
bool push_ar_try(const ExecutionContext& ec, const TpArray<Tensor>& buffer);

}  // namespace ops
}  // namespace ninfer