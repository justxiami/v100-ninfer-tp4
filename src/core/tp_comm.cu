// Implements: src/core/tp_comm.h (read that first: it explains why tp > 2 uses NCCL, and why the
// warmup collective has to run from one thread per rank).
//
// This is the only translation unit that includes nccl.h. Everything else reaches the tp > 2
// collectives through the declarations in tp_comm.h, so a future NCCL change is a change here.

#include "core/tp_comm.h"

#include <nccl.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ninfer {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void check_nccl(ncclResult_t status, const char* what) {
    if (status != ncclSuccess) {
        fail(std::string("tp transport: ") + what + ": " + ncclGetErrorString(status));
    }
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fail(std::string("tp transport: ") + what + ": " + cudaGetErrorName(status));
    }
}

// Restores the calling thread's current device on scope exit. Every collective below switches the
// current device per rank (a NCCL call must run with its own device current), and leaving the last
// rank current would make every LATER launch on rank 0's stream fail with
// cudaErrorInvalidResourceHandle -- which is exactly what the eager tp4 path hit.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { check_cuda(cudaGetDevice(&previous_), "cudaGetDevice"); }
    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "tp transport: cudaSetDevice restore failed: %s\n",
                         cudaGetErrorName(status));
        }
    }
    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

private:
    int previous_ = 0;
};


ncclDataType_t to_nccl_dtype(DType dtype) {
    switch (dtype) {
        case DType::BF16: return ncclBfloat16;
        case DType::FP16: return ncclFloat16;
        case DType::FP32: return ncclFloat32;
        default:
            fail("tp transport: unsupported collective dtype " +
                 std::to_string(static_cast<int>(dtype)));
    }
}

} // namespace

struct TpComm::Impl {
    std::array<ncclComm_t, kMaximumDevices> comms{};
    ~Impl() {
        for (ncclComm_t& comm : comms) {
            if (comm != nullptr) {
                // Destructors must not throw; a destroy failure at teardown is worth reporting but
                // not worth aborting the process.
                const ncclResult_t status = ncclCommDestroy(comm);
                if (status != ncclSuccess) {
                    std::fprintf(stderr, "tp transport: ncclCommDestroy failed: %s\n",
                                 ncclGetErrorString(status));
                }
                comm = nullptr;
            }
        }
    }
};

TpComm::TpComm(int tp, std::unique_ptr<Impl> impl) : tp_(tp), impl_(std::move(impl)) {}
TpComm::~TpComm() = default;

void* TpComm::comm_handle(int rank) const {
    if (rank < 0 || rank >= tp_) { fail("tp transport: comm_handle rank out of range"); }
    return static_cast<void*>(impl_->comms[static_cast<std::size_t>(rank)]);
}

std::unique_ptr<TpComm> TpComm::create(const ExecutionContext& ec) {
    if (ec.tp <= 2) {
        throw std::invalid_argument(
            "TpComm is for tp > 2; tp == 2 uses the hand-written pull protocol");
    }
    std::vector<int> devices;
    devices.reserve(static_cast<std::size_t>(ec.tp));
    for (int rank = 0; rank < ec.tp; ++rank) {
        if (!ec.dev[static_cast<std::size_t>(rank)].has_value()) {
            throw std::invalid_argument("TpComm requires one device context per rank");
        }
        devices.push_back(ec.dev[static_cast<std::size_t>(rank)]->device);
    }

    auto impl = std::make_unique<Impl>();
    check_nccl(ncclCommInitAll(impl->comms.data(), ec.tp, devices.data()), "ncclCommInitAll");
    auto comm = std::unique_ptr<TpComm>(new TpComm(ec.tp, std::move(impl)));

    // Warmup. One host thread per rank because the first collective's channel handshake is a host
    // rendezvous: a single thread issuing rank 0..N-1 blocks forever (see tp_comm.h).
    //
    // The payload size matters as much as the threading. NCCL picks its algorithm and channel
    // count from the message size, so warming 4 bytes leaves every larger size class cold -- and a
    // cold class was measured to produce a *partial* reduction (the tail fifth of a 480 KiB
    // allreduce came back with some ranks missing, non-deterministically, from a single-threaded
    // issue loop, while the 10 KiB case was always exact). 1 MiB covers every collective this model
    // issues: 10 KiB per decode allreduce, 480 KiB for a 48-token prefill chunk, 496 KiB for the
    // vocabulary gather.
    constexpr std::size_t kWarmupBytes = 1u << 20;
    // Every loop below switches the current device per rank (a cudaMalloc or NCCL call must run
    // with its own device current), so the caller's device is restored on return: leaving the last
    // rank current would break the very next launch on rank 0's stream.
    const CurrentDeviceGuard restore_device;
    std::vector<void*> warm(static_cast<std::size_t>(ec.tp), nullptr);
    for (int rank = 0; rank < ec.tp; ++rank) {
        check_cuda(cudaSetDevice(devices[static_cast<std::size_t>(rank)]), "cudaSetDevice");
        check_cuda(cudaMalloc(&warm[static_cast<std::size_t>(rank)], kWarmupBytes),
                   "cudaMalloc warmup");
    }
    std::vector<std::string> errors(static_cast<std::size_t>(ec.tp));
    {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(ec.tp));
        for (int rank = 0; rank < ec.tp; ++rank) {
            threads.emplace_back([&, rank] {
                const std::size_t slot = static_cast<std::size_t>(rank);
                const cudaError_t set = cudaSetDevice(devices[slot]);
                if (set != cudaSuccess) {
                    errors[slot] = std::string("cudaSetDevice: ") + cudaGetErrorName(set);
                    return;
                }
                const ncclResult_t status =
                    ncclAllReduce(warm[slot], warm[slot], kWarmupBytes / sizeof(std::uint32_t),
                                  ncclInt32, ncclSum, comm->impl_->comms[slot],
                                  ec.dev[slot]->stream);
                if (status != ncclSuccess) {
                    errors[slot] = std::string("warmup allreduce: ") + ncclGetErrorString(status);
                    return;
                }
                const cudaError_t sync = cudaStreamSynchronize(ec.dev[slot]->stream);
                if (sync != cudaSuccess) {
                    errors[slot] = std::string("warmup sync: ") + cudaGetErrorName(sync);
                }
            });
        }
        for (std::thread& thread : threads) { thread.join(); }
    }
    for (int rank = 0; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        check_cuda(cudaSetDevice(devices[slot]), "cudaSetDevice");
        check_cuda(cudaFree(warm[slot]), "cudaFree warmup");
        if (!errors[slot].empty()) { fail("tp transport: " + errors[slot]); }
    }
    return comm;
}

namespace ops {
namespace {



// Every rank must agree on the element count and dtype; the ops already guarantee matching shapes,
// but a mismatch here would be a silently wrong collective, so it is checked once per call.
void require_uniform(const ExecutionContext& ec, const TpArray<Tensor>& operands, const char* op) {
    if (ec.comm == nullptr) {
        throw std::logic_error(std::string(op) + ": tp > 2 requires an initialized TpComm");
    }
    for (int rank = 1; rank < ec.tp; ++rank) {
        const Tensor& first = operands[0];
        const Tensor& other = operands[static_cast<std::size_t>(rank)];
        if (first.dtype != other.dtype || first.ne[0] != other.ne[0] ||
            first.ne[1] != other.ne[1] || first.ne[2] != other.ne[2] ||
            first.ne[3] != other.ne[3] || !other.is_contiguous()) {
            throw std::invalid_argument(std::string(op) + ": every rank must carry the same "
                                                            "contiguous shape and dtype");
        }
    }
    if (!operands[0].is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": operands must be contiguous");
    }
}

} // namespace

void nccl_allreduce_sum(const ExecutionContext& ec, const TpArray<Tensor>& buffer,
                        const TpArray<Tensor>& staging) {
    (void)staging; // the pull protocol's scratch; NCCL reduces in place
    require_uniform(ec, buffer, "allreduce_sum (tp > 2)");
    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }
    const std::size_t count = bytes / dtype_size(buffer[0].dtype);
    const ncclDataType_t dtype = to_nccl_dtype(buffer[0].dtype);
    // One thread issues every rank's collective, in rank order. This is safe -- and 3-5x cheaper
    // than spawning a thread per rank per collective (measured: 17.9 us vs 94.6 us for the 10 KiB
    // decode shape, 31.7 us vs 112.5 us for the 480 KiB prefill shape) -- PROVIDED the caller
    // respects the Op's caller obligation: inputs staged with the plain cudaMemcpy/cudaMemset/
    // <<<...>>> forms must be retired before the collective reads them, and for a large payload a
    // per-stream sync is not enough (a 480 KiB allreduce over freshly staged, freshly allocated
    // buffers came back with a wrong tail under stream sync and was bit-exact under a device sync).
    // That obligation is documented in include/ninfer/ops/allreduce.h and holds for the tp2 pull
    // path too; it is not specific to NCCL. Callers that produce their inputs with kernels on
    // DeviceContext::stream -- the normal case in the model -- need no extra step.
    const CurrentDeviceGuard restore;
    for (int rank = 0; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        check_cuda(cudaSetDevice(ec.dev[slot]->device), "cudaSetDevice");
        check_nccl(ncclAllReduce(buffer[slot].data, buffer[slot].data, count, dtype, ncclSum,
                                 static_cast<ncclComm_t>(ec.comm->comm_handle(rank)),
                                 ec.dev[slot]->stream),
                   "ncclAllReduce");
    }
}

void nccl_allgather_rows(const ExecutionContext& ec, const TpArray<Tensor>& destination,
                         const TpArray<Tensor>& part) {
    require_uniform(ec, part, "allgather_rows (tp > 2)");
    for (int rank = 0; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        if (!destination[slot].is_contiguous() ||
            destination[slot].dtype != part[0].dtype ||
            destination[slot].ne[0] != part[0].ne[0] ||
            destination[slot].ne[1] != part[0].ne[1] * ec.tp) {
            throw std::invalid_argument(
                "allgather_rows (tp > 2): destination must be contiguous [C, R * ranks] in the "
                "part's dtype");
        }
    }
    if (part[0].ne[0] != 1) {
        // See tp_comm.h: with C == 1 NCCL's rank-major concatenation IS the Op's [C, R] layout, so
        // no scatter is needed. Anything else would need one, and silently writing the wrong bytes
        // is far worse than refusing.
        throw std::invalid_argument(
            "allgather_rows (tp > 2) supports row length C == 1 (the logits head gathers one "
            "column at a time); C > 1 needs a strided scatter that is not implemented");
    }
    const std::size_t count = static_cast<std::size_t>(part[0].ne[1]);
    const ncclDataType_t dtype = to_nccl_dtype(part[0].dtype);
    const CurrentDeviceGuard restore;
    for (int rank = 0; rank < ec.tp; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank);
        check_cuda(cudaSetDevice(ec.dev[slot]->device), "cudaSetDevice");
        check_nccl(ncclAllGather(part[slot].data, destination[slot].data, count, dtype,
                                 static_cast<ncclComm_t>(ec.comm->comm_handle(rank)),
                                 ec.dev[slot]->stream),
                   "ncclAllGather");
    }
}

} // namespace ops
} // namespace ninfer
