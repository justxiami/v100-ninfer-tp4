#pragma once
// 运行时开关：SM70 长上下文 decode 注意力路由（上游 861633e855db 的 opt-in 路径）。
//
// 分级模式（2026-10-15 加）：三臂窗显示「整条打开」时每轮 −13% 但接受率 −20% ⇒ 需要分辨**
// 是 target verify 侧变了、还是 MTP 草稿头侧变了**造成的数值分歧，故把开关拆成两侧：
//
//   NINFER_TP4_VOLTA_LONG_ATTN=0  全关（默认；gqa_volta_long_i8_splits 返回 0 ⇒ 与基线逐位同路）
//   =1  两侧都开（= 2026-10-15 首个窗口的臂 C）
//   =2  只给 target verify（与 batch）开，MTP 草稿头 ar[] 保持默认  ← 本次实验臂 C1
//   =3  只给 MTP 草稿头 ar[] 开，target verify 保持默认              ← 本次实验臂 C2
//
// 生效条件仍写在策略函数里（上游原样）：KV 为 I8、window > 65536、tokens ∈ {1,5,6}；
// 几何门已被本树从 `QHeads == 12` 放开到也包括 `QHeads == 6`（TP4）。
#include <cstdlib>

namespace ninfer::targets::qwen3_6::detail {

[[nodiscard]] inline int volta_long_attention_mode() noexcept {
    static const int mode = [] {
        const char* raw = std::getenv("NINFER_TP4_VOLTA_LONG_ATTN");
        if (raw == nullptr || *raw == '\0') { return 0; }
        const int value = std::atoi(raw);
        return (value >= 0 && value <= 3) ? value : 1;   // 非数字或越界按「全开」处理
    }();
    return mode;
}

// target verify（+ batch）侧是否用长档
[[nodiscard]] inline bool volta_long_attention_target() noexcept {
    const int m = volta_long_attention_mode();
    return m == 1 || m == 2;
}

// MTP 草稿头（ar[] 步）侧是否用长档
[[nodiscard]] inline bool volta_long_attention_draft() noexcept {
    const int m = volta_long_attention_mode();
    return m == 1 || m == 3;
}

} // namespace ninfer::targets::qwen3_6::detail
