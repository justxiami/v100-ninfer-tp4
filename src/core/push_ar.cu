// Implements: src/core/push_ar.h (read that first: it explains the protocol, the in-place
// contract, and the one thing that broke the first attempt -- peer access must be explicitly
// enabled, which the buffer allocation in tp_comm.cu now does).
//
// This is the only translation unit that launches the push kernel. The kernel is a faithful
// bf16 transcription of 1Cat's sm70_cross_device_reduce_1stage_push
// (~/src/1Cat-vLLM-main/src-d49e32b/csrc/custom_all_reduce.cuh, line 762): the sentinel-epoch
// synchronization, the fixed-rank-order FP32 accumulation, and the 16-byte volatile pack
// loads/stores are all transcribed verbatim; the only change is the data type (fp16 -> bf16) and
// the peer pointers (IPC-opened -> UVA device pointers, since this is a single process with up
// to kMaximumDevices CUDA devices).
//
// Two bugs this file owes a note about, both found by running it inside the engine rather than in
// the standalone test:
//   (1) the launch loop calls cudaSetDevice per rank and, without restoring the caller's device,
//       returns with the LAST rank's device current. Every other device-switching entry point in
//       the transport already restores it (CurrentDeviceGuard, src/core/tp_comm.cu:42); this one
//       did not, so the next op -- launched on rank 0's stream while rank 3 is current -- failed
//       with cudaErrorInvalidResourceHandle at the next error check (reported from scatter.cu:32).
//       The standalone test never saw it because it manages the current device itself.
//   (2) TpComm::create used `impl->` after std::move(impl), i.e. a null pointer (fixed in
//       src/core/tp_comm.cu, not here).

#include "core/push_ar.h"
#include "core/tp_comm.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ninfer {
namespace ops {
namespace {

// Restores the caller's current CUDA device on scope exit. Mirrors CurrentDeviceGuard in
// src/core/tp_comm.cu, which guards every other device-switching entry point.
class CurrentDeviceRestore {
public:
    CurrentDeviceRestore() : ok_(cudaGetDevice(&previous_) == cudaSuccess) {}
    ~CurrentDeviceRestore() {
        if (ok_) { cudaSetDevice(previous_); }
    }
    CurrentDeviceRestore(const CurrentDeviceRestore&)            = delete;
    CurrentDeviceRestore& operator=(const CurrentDeviceRestore&) = delete;

private:
    int  previous_ = 0;
    bool ok_       = false;
};

constexpr int kNgpus      = 4;
constexpr int kThreads    = 128;
constexpr size_t kMaxBytes = 81920;  // M8: 8 x 5120 x 2
constexpr uint16_t kSentinel = 0x7f7f;  // bf16 NaN (exponent 0xFF, mantissa 0x7F)

// One 16-byte pack: 8 bf16 elements. 16-byte alignment is required by the v4.b32 vector
// loads/stores.
struct alignas(16) Pack {
    __nv_bfloat16 v[8];
};

__device__ __forceinline__ void escape_sentinel(Pack& p) {
    uint16_t* b = reinterpret_cast<uint16_t*>(&p);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (b[i] == kSentinel) { b[i] = 0x7e00u; }  // real NaN -> another NaN (still decodes as NaN)
    }
}

__device__ __forceinline__ bool is_sentinel(const Pack& p) {
    const uint16_t* b = reinterpret_cast<const uint16_t*>(&p);
    bool any = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        any |= (b[i] == kSentinel);
    }
    return any;
}

__device__ __forceinline__ void load_pack(Pack& out, const void* addr, int off) {
    const Pack* s = reinterpret_cast<const Pack*>(addr) + off;
    uint4 bits;
    asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(s));
    *reinterpret_cast<uint4*>(&out) = bits;
}

__device__ __forceinline__ void store_pack(const Pack& in, void* addr, int off) {
    Pack* d = reinterpret_cast<Pack*>(addr) + off;
    uint4 bits = *reinterpret_cast<const uint4*>(&in);
    asm volatile("st.volatile.global.v4.b32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(d), "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w));
}

__global__ void __launch_bounds__(1024, 1)
push_ar_1stage(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
               int rank, int packed_size, void* __restrict__ local_storage,
               const void* __restrict__ peer0, const void* __restrict__ peer1,
               const void* __restrict__ peer2, const void* __restrict__ peer3) {
    const void* peers[kNgpus] = {peer0, peer1, peer2, peer3};
    uint32_t* epochs          = reinterpret_cast<uint32_t*>(local_storage);
    const uint32_t epoch      = epochs[blockIdx.x];
    constexpr int stride      = static_cast<int>(kMaxBytes / sizeof(Pack));  // 81920 / 16 = 5120
    const int eoff            = epoch * kNgpus * stride;
    const int first           = blockIdx.x * blockDim.x + threadIdx.x;

    for (int off = first; off < packed_size; off += gridDim.x * blockDim.x) {
        Pack value = *reinterpret_cast<const Pack*>(
            reinterpret_cast<const char*>(input) + off * static_cast<int>(sizeof(Pack)));
        escape_sentinel(value);
#pragma unroll
        for (int d = 0; d < kNgpus; ++d) {
            void* dest = reinterpret_cast<char*>(const_cast<void*>(peers[d])) +
                         static_cast<int>(sizeof(uint32_t) * 80) +
                         (eoff + rank * stride) * static_cast<int>(sizeof(Pack));
            store_pack(value, dest, off);
        }
        Pack peer[kNgpus];
        while (true) {
            bool any = false;
#pragma unroll
            for (int s = 0; s < kNgpus; ++s) {
                const void* src = reinterpret_cast<const char*>(local_storage) +
                                  static_cast<int>(sizeof(uint32_t) * 80) +
                                  (eoff + s * stride) * static_cast<int>(sizeof(Pack));
                load_pack(peer[s], src, off);
                any |= is_sentinel(peer[s]);
            }
            if (!any) { break; }
        }
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[e] = __bfloat162float(peer[0].v[e]); }
#pragma unroll
        for (int r = 1; r < kNgpus; ++r) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { acc[e] += __bfloat162float(peer[r].v[e]); }
        }
        Pack out;
#pragma unroll
        for (int e = 0; e < 8; ++e) { out.v[e] = __float2bfloat16(acc[e]); }
        *reinterpret_cast<Pack*>(reinterpret_cast<char*>(output) + off * static_cast<int>(sizeof(Pack))) =
            out;
        Pack empty;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            uint16_t s = kSentinel;
            empty.v[e] = *reinterpret_cast<__nv_bfloat16*>(&s);
        }
#pragma unroll
        for (int s = 0; s < kNgpus; ++s) {
            void* src = reinterpret_cast<char*>(local_storage) +
                        static_cast<int>(sizeof(uint32_t) * 80) +
                        (eoff + s * stride) * static_cast<int>(sizeof(Pack));
            store_pack(empty, src, off);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) { epochs[blockIdx.x] = (epoch + 1) % 2; }
}

}  // namespace

bool push_ar_try(const ExecutionContext& ec, const TpArray<Tensor>& buffer) {
    const CurrentDeviceRestore restore_device;  // the launch loop below switches devices
    static const bool enabled = [] {
        const char* raw = std::getenv("NINFER_PUSH_AR");
        if (raw == nullptr || raw[0] == '\0') { return true; }
        return std::strcmp(raw, "0") != 0;
    }();
    if (!enabled) { return false; }
    if (ec.comm == nullptr) { return false; }
    if (ec.comm->push_ar_tp() < kNgpus) { return false; }
    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return true; }  // empty, nothing to do
    if (bytes > kMaxBytes) { return false; }
    if (bytes % sizeof(Pack) != 0) { return false; }
    if (buffer[0].dtype != DType::BF16) { return false; }
    for (int rank = 1; rank < ec.tp; ++rank) {
        const Tensor& first = buffer[0];
        const Tensor& other = buffer[static_cast<std::size_t>(rank)];
        if (first.dtype != other.dtype || first.ne[0] != other.ne[0] ||
            first.ne[1] != other.ne[1] || first.ne[2] != other.ne[2] ||
            first.ne[3] != other.ne[3] || !other.is_contiguous()) {
            return false;
        }
    }
    if (!buffer[0].is_contiguous()) { return false; }
    const int packed_size = static_cast<int>(bytes / sizeof(Pack));
    const int blocks      = (packed_size + kThreads - 1) / kThreads;
    void* peers[kNgpus];
    for (int r = 0; r < kNgpus; ++r) { peers[r] = ec.comm->push_ar_buffer(r); }
    const int debug_rank = [] {
    const char* raw = std::getenv("NINFER_PUSH_AR_DEBUG_RANK");
    if (raw == nullptr || raw[0] == '\0') { return -1; }
    return std::atoi(raw);
}();  // -1 = off
    for (int rank = 0; rank < ec.tp; ++rank) {
        cudaError_t se = cudaSetDevice(ec.dev[static_cast<std::size_t>(rank)]->device);
        if (debug_rank >= 0 && se != cudaSuccess) {
            std::fprintf(stderr, "PUSH_AR_DIAG rank=%d cudaSetDevice FAILED %d (%s)\n", rank, (int)se,
                         cudaGetErrorString(se));
        }
        push_ar_1stage<<<blocks, kThreads, 0, ec.dev[static_cast<std::size_t>(rank)]->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(buffer[static_cast<std::size_t>(rank)].data),
            reinterpret_cast<__nv_bfloat16*>(buffer[static_cast<std::size_t>(rank)].data), rank,
            packed_size, peers[rank], peers[0], peers[1], peers[2], peers[3]);
        cudaError_t le = cudaGetLastError();
        if (debug_rank >= 0 && le != cudaSuccess) {
            std::fprintf(stderr, "PUSH_AR_DIAG rank=%d launch FAILED %d (%s)\n", rank, (int)le,
                         cudaGetErrorString(le));
        }
    }
    cudaError_t fe = cudaGetLastError();  // last-error check after the whole loop
    if (debug_rank >= 0 && fe != cudaSuccess) {
        std::fprintf(stderr, "PUSH_AR_DIAG post-loop FAILED %d (%s)\n", (int)fe, cudaGetErrorString(fe));
    }
    return true;
}

}  // namespace ops
}  // namespace ninfer