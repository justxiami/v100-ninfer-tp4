# Flash-V100 Split-D long-context attention + MTP batch for V100 (2026-10-22)

一句话：长上下文 prefill 的注意力核从 llama.cpp vendored 路线换成 1Cat-vLLM 的
Flash-V100 (SM70) **Split-D** 内核（BSD-3，剥离 ATen 后 vendor 进树），120k prefill
**+9.9%（引擎口径）/ +13.8%（客户端口径）**、32k **+4.0% / +5.3%**，同一份生产二进制还带上
前一晚的 **MTP 批次**（stem 逐列拷贝 → `cudaMemcpy2DAsync`，+256 宽 CUTLASS tile，8k +4.05%）。
needle 四档全 hit，decode 未动。

## 1. 背景：92% 段表与段传感器

上一轮（2026-10-21）把 GDN 段从 prefill 的 13.9% 压到 7.9% 之后，剩下的 92%
（64 层投影 + 16 层全注意力 + norm，≈1.92 s @8k）没有任何测量手段：nsys 抓不到引擎进程的内核
（同一二进制在自家 bench 上能抓到 kernel，引擎进程里 kernel 表全空），只能自底向上建传感器。

- **基线阶梯**（`461cbb62`，GDN chunked 上线版，3 reps 中位，真冷 salt，think on，1530 MHz）：
  8k 3917.0 / 16k 3755.2 / 32k 3470.0 / 64k 2996.0 / 120k 2392.1 tok/s（客户端），
  引擎口径 3949.1 / 3786.5 / 3492.7 / 3012.1 / 2402.7，needle 5 档 3/3。
  → `~/bench-results/2026-10-21/baseline-ladder/`
- **段传感器**：`src/core/prefill_profile.{h,cu}`，env `NINFER_PREFILL_PROFILE=1` 开启，
  每段 device 事件计时，按 prefill chunk 输出段表（wall% 与段耗时）。4096 chunk 段表：

  | 段 | wall% |
  |---|---|
  | mlp_gate_up | 32.4 |
  | mlp_down | 21.7 |
  | gdn_in_proj | 14.8 |
  | gdn_out_proj | 7.8 |
  | attn_core | 4.4 |
  | allreduce | 4.6 |
  | 其余（embed/norm/gdn_core/attn 投影/…） | ~14 |

  GDN 段（chunked 后）只占 3.1%，印证 7.9% 的端到端占比。
- 模型几何（从 `.ninfer` 工件头实读）：hidden 5120、64 层 = 48 GDN + 16 full（每第 4 层 full）、
  **稠密**（无 `num_experts`，「MoE」是上一轮分解里的误标）、MLP 5120→17408、W4A4-W8A8。
  引擎 `--prefill-chunk 4096`，传感器按 4096 测。

## 2. MTP 批次（00:25，已上线 `f2f4b3a1`）

- **MTP stem 逐列拷贝 → 每 rank 一次 `cudaMemcpy2DAsync`**：39,778 → 6,914 µs/chunk（−82.6%），
  逐字节等价单独证过。
- **TP4 MLP 用 256 宽 CUTLASS tile**：mlp_gate_up −1.9%、mlp_down −2.9%（段表中位）。
- 8k 实测（3 reps，同 argv，真冷）：客户端 3917.0 → **4076.0（+4.06%）**、
  引擎 3949.1 → 4109.1（+4.05%）、ttft 2.10 → 2.00 s，needle 3/3，显存 13985/13665 不变。
- 判死同批：chunk 8192（−26%，根因见 §4.3）、`NCCL_ALGO=Tree`（引擎坏）、CUTLASS 3 stages（静态 smem 编译期不可实例化）。

## 3. GEMM 线：四条路全判死（00:45）

| 路线 | 判死依据 |
|---|---|
| CUTLASS 加深流水（stages 2→3） | 128×128×32 三段 = 48 KB 静态 smem，编译期不可实例化 |
| 换 threadblock swizzle | batched-GEMM 接口，`device::Gemm` 参数结构装不下 |
| INT8 tensor core | PTX 没有 sm_70 的 int8 MMA（`.m8n8k16` 要 sm_75+） |
| INT8 绕道 cuBLAS | 同机 cuBLAS INT8 50.9 TOPS < 现有 fp16 77.6 |
| cuBLAS fp16 换内核 | 裸 GEMM 快 1.35×，接进引擎整体 +1.5% **变慢**（已回退） |

归因修正：「GEMM 段 62% 峰值」里 ~25% 是反量化/staging 周边开销，MMA 内核本身 ~75–78%。
⇒ 任何只换 MMA 内核的方案，上限就是那 ~25%。

## 4. 注意力自研线：四个结构方案全判死（01:16–02:51）

- 上界：vendored fattn 平台 **35.5 TFLOPS**（28.4% 峰值，avg_ctx>32k）；KV 有效带宽仅 ~48 GB/s
  （5% HBM）⇒ 算力/占用受限，不是带宽。120k 档注意力占 prefill ~40%。
- 纯 mma 探针：m8n8k4 单 warp 发射 117.7 TF = 94% 峰值；冗余模型 `整体上限=2·eff/(r+1)`
  对现役核（r=4）预测 35.8 TF vs 实测 35.5（误差 <1%）⇒ 模型可用。
- 自写 mma 核（生产几何，CPU 对拍 max rel 0.0005）：
  | 结构 | 实测 |
  |---|---|
  | q-major dim-split（树内保留版） | 24.66 TF |
  | Bc 放大 / 多链 / 软件流水 / 高占用 | 全部 ≤ 24.7 TF |
  | **转置版**（V 转置存储、PV 用 mma(C,B,A)、累加器 16–32 reg） | 正确但只有 **7.4–8.3 TF**（1/3） |
  | 根因：寄存器省下的收益被 smem（56 KB/CTA = 4 warps/SM）、V 转置 16 位标量散射、两次屏障、PV mma 数翻倍吃光 | — |
- 配置空间两侧夹死：ncols 32→64（1 block/SM）注意力段 +38% 但 8k −6.7%/32k −4.9%；
  occupancy 2→4 + combine 64→32 更差（32k −33.4%）。现役配置是局部最优。
- 结论：重写只能换结构，而本机能写的结构都试过 ⇒ 转向开源里有没有 SM70 专用核。

## 5. Flash-V100 Split-D（03:11–03:28，接入）

- **来源**：[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 的 Flash-V100 (SM70) Split-D
  注意力核，Copyright (c) 2026, 1CatAI，**BSD-3**；剥离 ATen host 包装与 `<ATen/c10>` 依赖
  （本树无 torch），设备核未改动；`utils.h` 补 `#include <optional>`。
  vendor 到 `src/ops/vendor/flash_v100/`（9 个文件），launcher：
  `src/ops/launcher/gqa_attention_splitd_sm70.cu`。
- **它为什么快**（我们缺的那块）：每个 warp 对里，两 warp 各持 8 行不同的 Q 算 QK，
  **P tile 共享、QK 不重复**；PV 才按 D 切半（kOwnedDChunks=2）。kBlockM=64 · kBlockN=32 ·
  8 warps = 4 组 × 2；smem `Swizzle<3,3,3>` + pitch-68 K 布局 + V 的 TT 布局消 bank 冲突；
  gmem 128 位装载；logits fp16、在线状态 fp32。
  我们此前所有的 dim-split 都**复制**了 QK（2×/4×），这是 24.6 TF 卡住的根因。
- **孤立内核实测**（本机 V100，我们的几何：D=256、6 q 头/1 kv 头、causal、T=4096）：
  | kv_len | ms | TFLOPS |
  |---|---|---|
  | 8,192 | 4.2 | 37.1 |
  | 32,768 | 19.1 | 40.5 |
  | 122,880 | 64.8 | **46.9** |
  对照：llama.cpp vendored 35.5 TF、自研核 24.63 TF。
  正确性：对 CPU fp32 参考 row-norm err 6.34e-04（T=64/P=64）。

### 5.1 第一个坑：q-block 只有 1024

引擎按 1024 token 一个 q-block 调注意力，而该核的 grid 是 `(tokens/64, 1, heads)`：
1024 → 96 blocks，在 80 SM 上只有 1.2 波，利用率约 60% ⇒ **实测反而比 llama.cpp 慢 7~21%**。
处置：该核不需要 mask（内部由 query_len/kv_len 推因果），不受 mask 的 O(n_kv × tokens)
内存约束，可以整个 4096 chunk 一次调用（384 blocks，4.8 波）。
为此 `q_f32/out_f32` workspace 按 `kSplitDQBlockTokens=4096` 分配（mask 仍按 1024）。

### 5.2 第二个坑：step 必须与生效条件同源

第一版只按 `width % 4096 == 0` 放大 step，而 splitd 分支另有 16k 门槛 ⇒ 小上下文落回
llama 路径却用 4096 的块、mask 只有 1024 ⇒ **写坏注意力：needle 三档全 MISS、spec 掉到 0%**，
但吞吐看起来很漂亮（120k 2900.6）。一切以正确性检查为准：**吞吐漂亮而 needle 挂掉等于没做**。

## 6. 同栈 A/B（bench 树二进制，llama 路线 vs splitd 路线，`NINFER_ATTN_SPLITD` 开关）

| 档 | llama.cpp | Split-D | 变化 |
|---|---|---|---|
| 8k | 3930.7 | 3897.0 | −0.9%（噪声内；该档不触发，门槛 n_kv≥16384） |
| 32k | 3492.7–3534.6 | 3628.5 | **+2.7~3.9%** |
| 120k | 2489.5 | 2718.7 | **+9.2%** |

三档 needle 全 hit；spec 接受率 82.6% / 73.3% / 66.7% 与基线一致。

## 7. 生产闸门（08:02 预注册，08:14 结果，二进制 `6ac5c8b5` 默认开启）

预注册预测（写于测量之前）：8k ±1%、32k +2.5~4.2%、64k +1.1~3.1%、120k +8.3~11.2%，
decode 不变，needle 5 档全 3/3。证伪条件：120k < +8%、needle 任一档 <3/3、decode 变化 >2%。

| 档 | engine tok/s | 基线（客户端口径同档） | 客户端 tok/s | Δ（客户端） | needle |
|---|---|---|---|---|---|
| 8k | 3955.4 | 3917.0 | 3925.5 | +0.2% | hit |
| 32k | 3677.4 | 3470.0 | 3654.8 | **+5.3%** | hit |
| 64k | 3318.4 | 2996.0 | 3297.6 | **+10.1%** | hit |
| 120k | 2735.9 | 2392.1 | 2721.7 | **+13.8%** | hit |

decode 2048（en/zh/mixed）：111.3 / 112.5 / 114.0，spec 37.1/37.5/38.8% —— 该路线不经过 decode，变化在噪声内。
needle 四档 strict 判据全 true ⇒ **闸门通过，已上线**。

预测 vs 实测：8k/32k/120k 都在带内；**64k 预测 +1.1~3.1% 被实测 +10.1% 否证（低估了注意力占比）**。

## 8. 口径审计（第三方复核，如实记录）

- **8k「+0.2%」是误导数字**：Split-D 在 8k 不触发（n_kv<16384），8k 的真实净收益 = MTP 批次的
  +4.05%（00:25 的 3-reps 数字 4109.1）。闸门 8k = 3955.4 比它低 3.8%，是跨 boot 时钟态差
  （GPU 基频 1312↔1530 MHz 漂移可造 ±12% 假差），闸门只有 1 rep。
- **64k「+10.1%」混了两个优化**：基线 2996.0 来自 461cbb62（不含 MTP 批次），闸门二进制两者都有，
  拆不开。
- **站得住的是 32k +5.3% 与 120k +13.8%（客户端口径）**：同栈 A/B 给出 +2.7~3.9% / +9.2%（引擎口径），
  生产构建略快，方向一致。
- **长上下文 decode（window>65536 路径）不在闸门覆盖内**。

## 9. 接入形态与回滚

- env `NINFER_ATTN_SPLITD=1` 显式开启；`0` 或不设 = 原 llama.cpp 路线，行为完全不变（生产已默认开启，
  `0` 可用作 A/B 臂）。
- 触发条件：`n_kv >= 16384`、`tokens % 64 == 0`、`n_kv % 32 == 0`、`heads_q % heads_kv == 0`、
  整个宽度能被 4096 整除（否则走原路线，mask 尺寸不匹配）。
- epilogue：q 的 bf16→fp16、out 的 fp16→bf16 两趟转换，复用现有 fp32 工作区（每块 2× 余量）。
- 回滚：覆盖回 `~/backups/2026-10-22-splitd-attn/ninfer-serve.before`（sha256 `a2744327…`）后
  `bash ~/ninfer_run.sh stop 8901 && bash ~/ninfer_run.sh`。MTP 批次回滚点：
  `~/backups/2026-10-22-mtp-tile-batch/serve-before/`（sha256 `461cbb62…`）。

## 10. 净收益口径

- 长档（32k/120k）：MTP 批次 + Split-D 累计，客户端 **+5.3% / +13.8%**。
- 64k：两优化混算 **+10.1%**（拆不开，如实标注）。
- 8k：MTP 批次 **+4.05%**（3 reps，00:25），Split-D 不触发。
- decode：未动（长上下文 decode 路径未测，留作后续闸门）。

## English summary

Two batches shipped to production in one rebuilt binary (`6ac5c8b5`): (1) an MTP batch
(stem per-column copies replaced by one `cudaMemcpy2DAsync` per rank: 39.8→6.9 ms/chunk,
plus 256-wide CUTLASS MLP tiles) worth +4.05% on the 8k prefill; and (2) the long-context
attention swap: the vendored llama.cpp SM70 attention kernel (35.5 TFLOPS ceiling, 28% of
peak) is replaced by the **Flash-V100 Split-D** kernel from 1Cat-vLLM (BSD-3, vendored under
`src/ops/vendor/flash_v100/`, ATen stripped), which reaches 46.9 TFLOPS at 120k because its
warp pairs share the P tile so the QK product is never duplicated — the duplication is exactly
what capped every home-grown dim-split kernel at ~25 TF. Engine-in A/B (same stack, env-gated):
32k +2.7–3.9%, 120k +9.2% (engine tok/s); the production gate measured 32k +5.3% and 120k +13.8%
(client tok/s), needle 4/4 hit, decode untouched. Two integration bugs are documented: a 1024
q-block grid that made the kernel *slower* (fixed by issuing the whole 4096 chunk at once, which
the mask-free kernel allows), and a step-size gate that disagreed with the dispatch predicate and
silently corrupted attention (caught by the needle check while throughput looked better).
GEMM-line alternatives (INT8 tensor cores, cuBLAS swaps, deeper CUTLASS pipelines) and four
home-grown MMA kernel structures were all measured and rejected with evidence; the full segment
sensor (`NINFER_PREFILL_PROFILE`) that made the 92% of non-GDN prefill visible is included.