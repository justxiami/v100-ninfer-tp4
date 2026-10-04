#include "ninfer/ops/dflash_selector.h"
#include "ops/launcher/dflash_selector.h"
#include <stdexcept>
#include <string>
namespace ninfer::ops {
void dflash2_select(const Tensor& logits, const Tensor& gate, const Tensor& predecessor,
                    const Tensor& successor, const Tensor& anchors, const Tensor& partial,
                    Tensor& topk, Tensor& out,
                    cudaStream_t stream) {
    const int V = logits.ne[0], K = logits.ne[1], B = logits.ne[2], R = gate.ne[0];
    if (logits.dtype != DType::BF16 || gate.dtype != DType::BF16 || predecessor.dtype != DType::BF16 ||
        successor.dtype != DType::BF16 || anchors.dtype != DType::I32 || out.dtype != DType::I32 ||
        V <= 0 || K <= 0 || B <= 0 || R <= 0 || predecessor.ne[0] != V || predecessor.ne[1] != R ||
        successor.ne[0] != V || successor.ne[1] != R || gate.ne[1] != K || gate.ne[2] != B ||
        logits.ne[2] != B || anchors.ne[0] != B ||
        partial.dtype != DType::I64 || partial.ne[0] != 16 ||
        partial.ne[1] != (V + kDFlashSelectorTile - 1) / kDFlashSelectorTile ||
        partial.ne[2] != K * B ||
        topk.dtype != DType::I32 || topk.ne[0] != 16 || topk.ne[1] != K ||
        topk.ne[2] != B || out.ne[0] != K || out.ne[1] != B)
        throw std::invalid_argument("dflash2_select: invalid tensor shape");
    if (!logits.is_contiguous() || !gate.is_contiguous() || !predecessor.is_contiguous() ||
        !successor.is_contiguous() || !anchors.is_contiguous() ||
        !partial.is_contiguous() || !topk.is_contiguous() || !out.is_contiguous())
        throw std::invalid_argument("dflash2_select: tensors must be contiguous");
    detail::dflash2_select_launch(logits, gate, predecessor, successor, anchors,
                                  partial, topk, out, stream);
}

void dflash2_local_topk(const Tensor& logits, Tensor& partial, Tensor& keys,
                        std::int32_t global_id_offset, cudaStream_t stream) {
    constexpr const char* op = "dflash2_local_topk";
    const int V = logits.ne[0], C = logits.ne[1];
    if (logits.dtype != DType::BF16 || partial.dtype != DType::I64 || keys.dtype != DType::I64 ||
        V <= 0 || C <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1 || partial.ne[0] != 16 ||
        partial.ne[1] != (V + kDFlashSelectorTile - 1) / kDFlashSelectorTile || partial.ne[2] != C ||
        keys.ne[0] != 16 || keys.ne[1] != C || keys.ne[2] != 1 || keys.ne[3] != 1 ||
        global_id_offset < 0 || !logits.is_contiguous() || !partial.is_contiguous() ||
        !keys.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": invalid tensor shape");
    }
    detail::dflash2_local_topk_launch(logits, partial, keys, global_id_offset, stream);
}

void dflash2_select_sharded(const Tensor& keys, const Tensor& gate, const Tensor& predecessor,
                            const Tensor& successor, const Tensor& anchors, Tensor& global_keys,
                            Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "dflash2_select_sharded";
    const int C = gate.ne[1], B = gate.ne[2], R = gate.ne[0];
    if (keys.dtype != DType::I64 || gate.dtype != DType::BF16 || predecessor.dtype != DType::BF16 ||
        successor.dtype != DType::BF16 || anchors.dtype != DType::I32 ||
        global_keys.dtype != DType::I64 || out.dtype != DType::I32 ||
        keys.ne[0] % 16 != 0 || keys.ne[0] < 2 * 16 ||
        keys.ne[1] != C * B || keys.ne[2] != 1 || keys.ne[3] != 1 || gate.ne[1] != C ||
        gate.ne[2] != B || predecessor.ne[1] != R || successor.ne[1] != R ||
        predecessor.ne[0] != successor.ne[0] || anchors.ne[0] != B || global_keys.ne[0] != 16 ||
        global_keys.ne[1] != C * B || global_keys.ne[2] != 1 || global_keys.ne[3] != 1 ||
        out.ne[0] != C || out.ne[1] != B || !keys.is_contiguous() || !gate.is_contiguous() ||
        !predecessor.is_contiguous() || !successor.is_contiguous() || !anchors.is_contiguous() ||
        !global_keys.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": invalid tensor shape");
    }
    detail::dflash2_select_sharded_launch(keys, gate, predecessor, successor, anchors,
                                          global_keys, out, stream);
}
}
