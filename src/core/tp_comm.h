#pragma once

// Per-rank NCCL communicators for tp > 2, plus the tp > 2 collective entry points.
//
// WHY THIS EXISTS NEXT TO THE PULL PROTOCOL. The tp == 2 path (src/ops/common/allreduce.cu)
// choreographs cross-rank ordering with host-side events, and its per-collective host cost grows
// with the number of peers: at 4 ranks it is 40 captured operations per collective against NCCL's
// 8, measured as 5.83 ms versus 2.40 ms per 129-collective token eagerly, 3.96 ms versus 3.35 ms as
// a replayed graph, and a 3-5x larger cudaGraphLaunch (which is charged per captured operation).
// NCCL keeps the sequencing inside its kernels, so it is the better transport as soon as there is
// more than one peer. nccl.h appears in exactly one translation unit (tp_comm.cu).
//
// THE ONE THING NCCL CANNOT DO LAZILY. The first collective on a fresh communicator performs the
// channel handshake on the host, and every rank must be inside that call at the same time: issuing
// rank 0..N-1 from one thread deadlocks (reproducible at 2 ranks as well). Once the channels are
// warm a single thread issues collectives without blocking (~4 us per launch), which is exactly
// what lets the decode graph stay single-threaded -- so construction runs one small warmup
// collective from one host thread per rank, and nothing afterwards needs a second thread.
//
// Storage lives in ExecutionContext::comm (non-null only for tp > 2, and only after create()).

#include "core/device.h"
#include "core/tensor.h"

#include <memory>

namespace ninfer {

class TpComm {
public:
    // Brings up one communicator per rank and runs the warmup collective. Requires ec.tp > 2 with
    // one distinct device context per rank. Throws std::runtime_error/std::invalid_argument on
    // failure; on success every rank's stream has been synchronized, so no warmup work is left to
    // leak into a later capture.
    static std::unique_ptr<TpComm> create(const ExecutionContext& ec);

    ~TpComm();
    TpComm(const TpComm&)            = delete;
    TpComm& operator=(const TpComm&) = delete;

    [[nodiscard]] int tp() const noexcept { return tp_; }

    // Opaque `ncclComm_t` for `rank`; only tp_comm.cu interprets it.
    [[nodiscard]] void* comm_handle(int rank) const;

private:
    struct Impl;
    TpComm(int tp, std::unique_ptr<Impl> impl);
    int tp_ = 0;
    std::unique_ptr<Impl> impl_;
};

namespace ops {

// tp > 2 counterparts of the tp == 2 pull collectives, with the same contracts as the ops
// documented in include/ninfer/ops/allreduce.h:
//
//   allreduce_sum  -- in place; every rank ends holding the identical sum. `staging` is the pull
//                     protocol's scratch and is unused here (NCCL reduces in place).
//   allgather_rows -- destination[r] receives the identical [C, R] image with
//                     dest[c * R + row] == part[row][c], where rank r owns rows
//                     [r * R/ranks, (r+1) * R/ranks).
//
// The allgather writes NCCL's rank-major concatenation straight into the destination, which is
// byte-identical to the Op's layout when C == 1 -- and C == 1 is the only shape the model asks for:
// the logits head gathers one vocabulary column at a time, so the piece is [1, shard_vocab] and the
// destination is [1, vocab]. A C > 1 request would need a strided scatter, so it is rejected loudly
// instead of being written wrong.
void nccl_allreduce_sum(const ExecutionContext& ec, const TpArray<Tensor>& buffer,
                        const TpArray<Tensor>& staging);
void nccl_allgather_rows(const ExecutionContext& ec, const TpArray<Tensor>& destination,
                         const TpArray<Tensor>& part);

}  // namespace ops
}  // namespace ninfer
