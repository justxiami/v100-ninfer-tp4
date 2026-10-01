#pragma once

#include <cuda_runtime.h>

#include "ninfer/types.h" // kMaximumDevices

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer {

// src/core/tp_comm.h: per-rank NCCL communicators for tp > 2. Forward declared so this header stays
// free of nccl.h -- every translation unit includes it.
class TpComm;

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

// One process, up to kMaximumDevices CUDA devices. dev[0..tp-1] hold constructed DeviceContext
// instances; the remaining slots stay empty. tp == 1 unless the caller opts into `--tp 2` (or the
// TP4 degree), which runs the tensor-parallel program across that many devices.
struct ExecutionContext {
    std::array<std::optional<DeviceContext>, kMaximumDevices> dev;
    int tp = 1;

    // Non-null only for tp > 2, and only once TpComm::create() has succeeded. tp == 2 keeps using
    // the hand-written pull protocol plus PeerEvents, so this stays null there and the collectives
    // pick their backend from `tp` (see src/ops/common/allreduce.cu).
    //
    // shared_ptr rather than unique_ptr on purpose: TpComm is incomplete in this header, and a
    // shared_ptr erases its deleter at construction (which happens where the type is complete),
    // whereas unique_ptr would need an out-of-line ExecutionContext destructor.
    std::shared_ptr<TpComm> comm;

    // device_ids.size() must be in 1..kMaximumDevices and becomes tp. Every id is validated to
    // exist by DeviceContext's own constructor; when more than one id is given they must
    // additionally share the same compute capability (sm major.minor), since nothing downstream
    // can reconcile mismatched architectures.
    explicit ExecutionContext(const std::vector<int>& device_ids);

    [[nodiscard]] DeviceContext& primary() noexcept { return *dev[0]; }
    [[nodiscard]] const DeviceContext& primary() const noexcept { return *dev[0]; }
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace ninfer
