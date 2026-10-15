# ninfer TP4 decode 内核回迁 + 长窗路由 — 完整记录（2026-10-13 → 10-15 03:51）

## 结论（一句话）

上游 `tuxKOH/ninfer-V100X2 @ 861633e8`（10/13「perf(v100): optimize long decode」）的 decode 侧内核已移植到 tp4；
修掉两处 CUDA graph profile「连带」后按写死判据过线：**pi 形状 120k decode 96.0 → 109.1 tok/s（1.136×，C1 臂）**，
以 **MTP4 + 只开 target 侧路由** 落生产（10/15 03:51 起 :8901 现役）。

---

## 1. 移植内容与文件

上游 861633e8 是 TP2 原生优化；本树把它缺的 decode 侧内核族补进 TP4（9 改 + 3 增，共 12 文件）：

| 文件 | 角色 |
|---|---|
| `src/ops/launcher/gqa_attention_decode.cu` | decode 注意力入口 + 长档内核选择 |
| `src/ops/kernel/gqa_attention_decode_i8_tc_volta_long.cuh` | 新增：长窗 split-K FP32-PV 内核（TP4 分片版） |
| `src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh` / `gqa_attention_decode.cuh` | 短档与共用件对齐 |
| `src/ops/kernel/gqa_attention_kv_quant.cuh` | KV 量化（int8）路径对齐 |
| `src/ops/linear_attention/gated_delta_net/recurrent.cuh` | GDN recurrent 对齐 |
| `include/ninfer/ops/gqa_attention.h` | 接口 |
| `src/targets/qwen3_6/impl/runtime/volta_long_attention_switch.h` | 新增：路由开关（见 §2） |
| `src/targets/qwen3_6/impl/runtime/lookup_draft.h` | 新增：草稿查表件 |
| `src/targets/qwen3_6/impl/runtime/program{,_impl}.h`、`qwen3_6_27b/impl/variant.cpp` | 装配与分派 |

TP2 行为不变；本轮只在 tp4 上测。

## 2. TP4 适配点

- `NINFER_TP4_VOLTA_LONG_ATTN`（env，默认 2）选路由侧：`0` 全关 / `1` 两侧全开 / `2` 只 target（推荐）/ `3` 只草稿头。
  只作用于 target verify；MTP 草稿头保持默认策略。
- 生效条件（硬编码）：KV=int8、window > 65536、验证宽度 tokens ∈ {4,5,6}（MTP4 → 5）。
- 长档门与 KV 分段策略共用同一门槛 65536；launcher 不再用独立的 16390。

## 3. 两处「图 profile 连带」修复（第一个 promote 窗因它不收）

| 修什么 | 原来 | 现在 |
|---|---|---|
| launcher 选长档内核的门槛 | `implementation_window > 16390` ⇒ 16k–65k 也走长档内核（FP32 PV） | `> 65536`（与分段策略同一门槛） |
| MTP 图 profile 边界 | 边界表止于 `{…16390, 32768}` ⇒ 20k–32k 落进「[32761, capacity]」profile，捕获期按 capacity 选内核 | 边界表加 `65536` ⇒ ≤65536 与 >65536 各自成 profile |

机理：某个 profile 的 envelope 一旦超过 65536，该 profile 内**所有窗口（含 32k/短档）都跑长档内核**，
短档跟着吃数值偏移——这就是 promote 窗里 `general 0.960` / `long32k 0.949` 的来源。

## 4. 窗口实测（全部窗前写死判据；客户端 decode tok/s 中位；臂序轮换）

判据（写死）：主判 ≥1.08 收 / <1.03 不收 / 1.03–1.08 中间态；副作用闸 = 各档比值 ≥0.97。

**promote 窗（10/15 01:33–02:03，各臂 3 boot × 2 rep，修复前）**：
M3T/A = 1.065、C1/A = 1.048 ⇒ 中间态；且 M3T `general` 0.960、C1 `long32k` 0.949 挂闸 ⇒ 不 promote，回基线（MTP3、无路由）。

**pi 形状窗（10/15 02:54–03:37，各臂 3 boot × 3 rep，修复后）**：

表 1 客户端 decode tok/s（中位）：

| 档 | A 现役 MTP3 | M3T MTP3+路由 | C1 MTP4+路由 | M3T/A | C1/A |
|---|---:|---:|---:|---:|---:|
| pi20k | 122.6 | 127.3 | 121.6 | 1.038 | 0.992 |
| pi32k | 118.2 | 116.6 | 115.8 | 0.986 | 0.980 |
| pi64k | 105.3 | 101.2 | 113.2 | **0.961** | 1.075 |
| **pi120k** | **96.0** | **105.4** | **109.1** | **1.098** | **1.136** |
| needle64k | 144.3 | 146.6 | 154.2 | 1.016 | 1.069 |

表 2 决策档 pi120k 机理分解（A = 基准）：

| 指标 | A | M3T | C1 | M3T/A | C1/A |
|---|---:|---:|---:|---:|---:|
| tok/轮 | 2.60 | 2.50 | **2.80** | 0.962 | **1.077** |
| ms/轮 | 27.06 | **23.70** | 25.62 | **0.876** | 0.947 |
| decode tok/s | 96.0 | 105.4 | 109.1 | 1.098 | **1.136** |

判定：主判 M3T/A 1.098、并判 C1/A 1.136 都 ≥1.08 过线；副作用闸 **C1 全过**（0.980–1.136）、
M3T 的 pi64k 0.961 不过 ⇒ 按判据第 3 条「两条都成立时按短档选」取 **C1 = MTP4 + 只开 target**。

工件 = QUASAR QAT（qwen3.8-27b nvfp4），TP4 200k×2c、KV int8 409600；A 臂基线二进制 `afce97ba4484`，
M3T/C1 臂二进制 `dce234eafcf1`。

## 5. 生产形态（2026-10-15 起）

- `--draft-tokens 4`（MTP draft 3→4）+ `NINFER_TP4_VOLTA_LONG_ATTN=2`；`--vision` 常开；TP4 200k×2c。
- 现役二进制 = 2026-10-15 03:51 重建，sha256 前缀 `a018c47f681e8`（440,055,440 B）；
  与测量臂 `dce234eafcf1` 同源，落生产时重建。
- 回滚 = `DRAFT_TOKENS=3` + `NINFER_TP4_VOLTA_LONG_ATTN=0`，走切换脚本一条命令，不留空窗。

## 6. 已知边界

- 跨窗噪声 ~5% 量级（promote 窗 C1 ms/轮 −4.7% 与上一窗同配置 −11.8%）⇒ 4–6% 量级的结论要更大样本。
- 短档 0.980–0.992 是「无代价」而非「净亏」；收益集中在 64k 以上（1.075 / 1.136）——路由的设计区间。
- dflash 与 vision 互斥（引擎硬拒）；dflash 路径 batch-1。
- 跨树逐位对拍不作判据：FP 归约顺序 ~1 ULP 差被 greedy+MTP 混沌放大；只验同二进制确定性、质量探针、性能。
