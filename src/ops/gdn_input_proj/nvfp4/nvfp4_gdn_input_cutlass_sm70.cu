#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
struct CutlassSections;

// Reads the same per-geometry section traits the shard output does, so a newly registered width
// reaches the cutlass split kernel automatically. The fused parent's physical row order is
// Q|K|V|Z, so the first kQkvRows rows are the qkv plane and the remainder is z.
template <class Geometry>
struct CutlassSections {
    static constexpr std::int32_t kQkvRows =
        2 * Nvfp4GdnInputSections<Geometry>::kKeyRows + Nvfp4GdnInputSections<Geometry>::kValueRows;
    static constexpr std::int32_t kZRows = Nvfp4GdnInputSections<Geometry>::kValueRows;
};

template <class Geometry, class Allocator>
Tensor allocate_projected(Allocator& allocator, int tokens) {
    return allocator.alloc(DType::BF16, {Geometry::kOutputRows, tokens});
}

template <class Geometry>
__global__ void split_gdn_output(const __nv_bfloat16* __restrict__ projected,
                                 __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ z,
                                 std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    using Sections = CutlassSections<Geometry>;
    const int row   = static_cast<int>(i % Geometry::kOutputRows);
    const int token = static_cast<int>(i / Geometry::kOutputRows);
    if (row < Sections::kQkvRows) {
        qkv[static_cast<std::int64_t>(token) * Sections::kQkvRows + row] = projected[i];
    } else {
        z[static_cast<std::int64_t>(token) * Sections::kZRows + row - Sections::kQkvRows] =
            projected[i];
    }
}

template <class Geometry>
std::size_t workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected<Geometry>(layout, tokens);
    return layout.peak_bytes(1) +
           nvfp4_cutlass_sm70_workspace_bytes(Geometry::kOutputRows, Geometry::kInputRows, tokens);
}

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected<Geometry>(workspace, x.ne[1]);
    nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * Geometry::kOutputRows;
    split_gdn_output<Geometry><<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(qkv.data),
        static_cast<__nv_bfloat16*>(z.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t nvfp4_gdn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputGeometry>(tokens);
}

// The tp2 column shard over-provisions the tp4 one (8192 vs 4096 projection rows), the same rule
// the FP8 sibling follows.
std::size_t nvfp4_gdn_input_cutlass_shard_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputTp2ColumnGeometry>(tokens);
}

void nvfp4_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Nvfp4GdnInputGeometry>(x, weight, qkv, z, workspace, stream);
}

void nvfp4_gdn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                               Tensor& z, WorkspaceArena& workspace,
                                               cudaStream_t stream) {
    if (weight.n == Nvfp4GdnInputTp4ColumnGeometry::kOutputRows) {
        launch<Nvfp4GdnInputTp4ColumnGeometry>(x, weight, qkv, z, workspace, stream);
        return;
    }
    launch<Nvfp4GdnInputTp2ColumnGeometry>(x, weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail