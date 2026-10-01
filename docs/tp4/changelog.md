# M2 变更记录（无 git 树，逐条留档）

回滚网：`~/backups/2026-09-30-ninfer-tp4-m2/src_snapshot_2026-09-30.tar.gz`（21:22 整树快照，先于全部 W1 改动）。
之后每个文件的改动都在这条基线之后，回滚 = 从该 tar 提取单个文件覆盖。

## W1 基建（2026-09-30 21:35–21:45，已编译+冒烟）

| 文件 | 改动 |
|---|---|
| `include/ninfer/types.h` | 新增 `inline constexpr std::size_t kMaximumDevices = 4;`（设备上限**单一真相**）；`LoadSummary::devices` 2→`kMaximumDevices`；`EngineOptions::tp` 契约注释改为 1/2/4 |
| `src/artifact/binder.h` | 删除本地 `kMaximumDevices = 2`；改为 `using ninfer::kMaximumDevices;`（**注**：首版直接删会破坏 `bindings.cpp` 的 `artifact::kMaximumDevices` 非限定查找，已修） |
| `src/artifact/binder.cpp` | 错误文案 1..4 |
| `src/core/device.h` | `ExecutionContext::dev` 2→`kMaximumDevices`；注释 |
| `src/core/device.cu` | sm 一致性校验 `tp == 2` → `tp > 1`（逐设备对 dev[0]）；错误/注释文案 |
| `apps/cli/options.cpp` | `parse_tp` 接受 {1,2,4}（3 缺省因为不整除 hidden/vocab）；`parse_devices` 上限 1..4；usage/help |
| `src/serve/serve_options.cpp` | **serve 侧独立一份选项解析**（现役实例走它，M1/handoff 都没提）：`parse_tp`/`parse_devices`/usage 同步 |

验证：`ninfer-serve --help` 与 `ninfer --help` 显示 `--tp 1|2|4`；`--tp 3` 被拒（"must be 1, 2 or 4"）；
`--tp 4` 解析通过后撞下一道闸门 `src/runtime/engine/engine.cpp:71`（预期，见下）。

## 剩余闸门清单（fail-loud，是 W3/W5 的天然 checklist）

| 位置 | 当前行为 |
|---|---|
| `src/runtime/engine/engine.cpp:71` | `EngineOptions.tp must be 1 or 2` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h:101` | `sequence plan tp must be 1 or 2` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h:669` | `tensor-parallel width must be 1 or 2` |
| `src/targets/qwen3_6_27b/impl/load/bindings.cpp:926` / `:1093` | `qwen3_6_27b: tp must be 1 or 2` |

**性质**：这些断言让迁移可以「开一个闸门 → 跑 → 读下一个报错」推进，不会静默错分片。

## 待办修正（W1 期间发现）

1. **DFlash 在 tp2 是支持的能力**（`--help` 原文：`--tp 2 supports --spec mtp and --spec dflash, but
   not --vision`），而 `gather_columns_rank0`/`broadcast_rank0` 这两个 rank0-centric op 只被
   `dflash_impl.h` 用 ⇒ **TP4 必须显式拒绝 `--spec dflash`**（否则会走 2-rank 假设的通信路径），
   不能只靠「首版不覆盖」。W3 加这道闸门。
2. `src/targets/qwen3_6_35b_a3b/` 也共享同一 runtime（含 `variant.{h,cpp}` 104 处 2-元数组），
   但它在 tp>1 时被上游拒绝；W2 的改动仍必须让**它**编译通过。

## W2.0 类型拓宽（2026-09-30 22:0x–22:4x，整树编译通过）

约定：`TpArray<T> = std::array<T, kMaximumDevices>`（`ninfer/types.h`），配 `kMaximumDevices = 4`。
扫替脚本把 TP 语义的 `std::array<X, 2>` 换成 `TpArray<X>`：**1177 处 / 43 文件**（`Tensor` 867、`Weight` 101、
`WorkspaceArena*` 100、各种 `*Weights*`/`*Payload*` 指针、`DeviceBuffer` 22、`ops::RopeFrequencyOverride`、
`PagedKV*`、`qwen3_6::RoundState*`、`MtpStemRoots`、`void*`、`cudaEvent_t`、`std::size_t` 等）。

### 误判与回退（扫描按类型名，不看语义 —— 这些不是 per-rank）
| 类型 | 真实语义 | 处理 |
|---|---|---|
| `SlicePlane`（storage_layouts.cpp） | 「code plane + scale plane」对 | 回退 `std::array<SlicePlane, 2>` |
| `OutputDelta`（frontend.h `PublishedOutput`） | 定容小向量（有 `size_`/`push_back`/`back`） | 回退 `std::array<OutputDelta, 2>` |
| `RouteSpec`（5 个 plan.cpp） | **形状分档路由目录**（`{1,27}`/`{28,kAnyCols}`，`catalog_is_closed` 静态断言会挂） | 回退 `std::array<RouteSpec, 2>` |
| `W8PairRouteSpec`（w8_pair_plan.cpp） | 同上（非 Volta 分支本就是 3 档） | 回退 |
| `std::uint64_t`（12 处）、`std::string`（1 处） | 二维 shape 对 / `stop_pending` | 未纳入扫替 |

### 其它必要修正
- **`kTensorParallelWidth`（text_context.h:81 原 `= 2`）统一到 `kMaximumDevices`** —— 代码库本来就有这个具名宽度常量（24 处在用），
  它是 per-rank 数组宽度的既有真相；现在两者同源（`TpArray` 是同一个宽度的另一个名字）。
- **`WorkspaceArena::Scope` 不可默认构造** ⇒ `TpArray<Scope> x = {a,b}` 非法（空槽无法 value-init）。
  3 处改成按 rank 构造的 `std::vector<Scope>`（顺带就是 TP4 需要的形式）。
- **tests/ 与 bench/ 回退**（11 个文件）：它们建模固定 2 卡配置，本就该写 2；TP4 测试后面另写。
- 全树补 `#include "ninfer/types.h"`（41 个 src/include 文件）。

### 新增规则（W2.1 必须遵守）
**per-rank / per-device 数组转成 span 或按 `.size()` 遍历时，extent 必须是 rank 数（`ec.tp`），不是数组宽度。**
反例已确认存在过风险但当前实现已合规：`resolve_kv_capacity_symmetric` 收 `std::span` 并用 `min_element`；
`src/targets/registry.cpp:139/158` 已经用 `reserve(tp)` + `rank < tp` 构造 vector 传入 ✓ 无需改。
（若曾按整数组传入，空槽的 0 会成为瓶颈 → KV 池被压到最小。）

### 验证状态
- `ninfer-serve` + `ninfer` 链接通过；7 个测试目标已编出（含 `ninfer_qwen3_8_27b_tp2_real_test`、
  `*_mtp_tp2_real_test`、`*_graph_tp2_test`、`kv_capacity_tp2`、`public_api`、artifact 两个）。
- **待跑（需要 GPU 窗口）**：TP2 实测回归（gate）。窗口计划：`stop 8901` + `stop 8902` → 跑上述测试 →
  `start` 恢复（命令见 `restore_instances.md`）。

## TP2 回归闸门：**通过**（2026-09-30 22:4x–23:0x，四卡窗口）

窗口操作：`stop 8901` + `stop 8902`（按用户授权，**不恢复**）；四卡全空 16140 MiB/卡。

### A/B（老 = `build-v100`，新 = `build-v100-tp4`，同工件同参数）

| 检查 | 结果 |
|---|---|
| 仓库测试 `ninfer_qwen3_8_27b_{tp2_real,mtp_tp2_real,graph_tp2}_test` | 三个都**新旧输出逐字节一致**（都撞硬件显存守卫：这些测试的配置要 19–21 GiB/卡，2×16 GB 装不下；等价性因此成立） |
| host-only `ninfer_kv_capacity_tp2_test` / `ninfer_public_api_test` | 新二进制 **ok**（rc=0） |
| **服务路径 A/B**（`--tp 2 --devices 0,1 --kv-capacity 131072 --spec mtp --draft-tokens 3 --lm-head-draft`，官方 nvfp4 件，贪心 request） | 见下 |

| 项 | OLD | NEW |
|---|---|---|
| 贪心输出 | `'alpha beta gamma'` | `'alpha beta gamma'` |
| completion / prompt tokens | 37 / 59 | 37 / 59 |
| `KV capacity ... resolved` | 131072 tokens, 2048/2048 pages, runtime 3.67 GiB | 完全相同 |
| free-after-weights / startup | 5.00 / 1.40 GiB | 完全相同 |
| **graph-nodes** | **2562** | **2562** |
| 每卡占用 | 14713 / 14711 MiB | 完全相同 |
| P2P | 静默（FORCE=1 走直连） | `[ninfer] direct P2P enabled (peer access qualified, 1024 KiB/rank probe)` ← B 补丁生效 |

⇒ **1177 处类型拓宽 + W1 基建未改变 TP2 行为**（bit 级等价的 KV 尺寸、图节点数、贪心输出、显存）。

### 本轮抓到的真 bug（我引入的，已被 fail-loud 校验逮住）

`kTensorParallelWidth` **不是数组宽度，是并行度**：`kShardQHeads = n_q / width`、`kShardVocab = vocab / width`，
还参数化 `workspace_recipe::*`。首版把它统一到 `kMaximumDevices`(4) ⇒ 分片算术按 4 份算
（`g` 从 24 行变 12 行、显存需求变小），被 `gdn_gating_proj: invalid g` 挡下。
**修正**：程度归 `2`（并在注释里写明「TP4 落地时要变成运行期 `ec.tp`」），**宽度归 `TpArray`**；
`text_context.{h,impl}` 里两处 `std::array<RopeFrequencyOverride, kTensorParallelWidth>` 改 `TpArray<...>`。
**教训（写进本条以免重犯）**：**「宽度」和「程度」必须分开命名**，一个 `= 2` 的常量同时兼两个角色时，
把它改大等于悄悄重分片。

### 顺带的永久改进
- `gdn_gating_proj.cpp` 的两处 tensor 校验错误信息改为**带实测值**（dtype/ne/data/contiguous + want），
  否则「invalid g」无法区分「空槽」和「真形状错」。备份 `~/backups/2026-09-30-ninfer-tp4-m2/gdn_gating_proj.cpp.pre-diag`。

### 供 M2/M4 用的实测数字
- **TP2 现役图：2562 节点**（129 集合通信 × 2 rank 的真实模型）。
  按 M1 的 op 成本模型，这个量级的图 launch ≈ 1–3 ms host 时间/ token —— M4 算 TP2/TP4 预算时要用。
- TP2 官方 nvfp4 件：weights 20.92 GiB、加载 11.4 s、KV 131072 tokens = 3.67 GiB、启动后余 1.40 GiB。

## W2.1 rank 循环通用化（2026-09-30 23:2x–23:5x）

把「遍历 rank」的循环从硬编码 2 改成 `ec.tp` 驱动：**89 处**（脚本 79 + 手工 10）。
涉及 `src/ops/common/{allreduce.cu,split_launch.h}`、`src/ops/wrapper/{gdn_gating_proj,gdn_input_proj,attn_input_proj,linear_swiglu}.cpp`、
`src/ops/linear/linear.cpp`、`src/targets/qwen3_6/impl/runtime/{text_context_impl.h,schedule.h}`、
`src/targets/qwen3_6_27b/impl/variant.cpp`。

- **3 处重复的 `for_each_rank` 实现**（`split_launch.h:107`、`schedule.h:53`、`variant.cpp`）都改到 `ec.tp`；
  `split_launch.h` 那处一处覆盖 62 个调用点。
- 顺带修的两个签名问题：`linear.cpp` 的 `validated_outputs` 原本没有 ExecutionContext（加了参数 + 2 个调用点）；
  `text_context_impl.h` 的 `mtp_prefill_chunk_tp2` 在成员函数里用了不存在的局部 `execution`（改 `ec().tp`）。
- **显式保留的 13 处 `< 2` 循环**（非 rank 语义，绝不能改）：4-bit 打包 `word_index`（4 处）、split-K 的 `group`/`ki`、
  `local_segment`、量化组 `i`、卷积 `tap`、`pass`、mma tile `mi`（2 处）。

### 过程事故（我造成的，值得记）

首个脚本用「整行替换」改写循环头，**吃掉了行首缩进和行尾 ` {`**（79 行受损）。靠**快照里的原始行按变量名配对**确定性修复（52 + 27 行）。
两条教训：
1. 批量改代码的脚本必须用 `re.sub` 只替换匹配段，**绝不能整行替换**（行尾可能有 `{`、注释、单行语句体）。
2. 无 git 树下这类事故只有编译错误能发现，而编译器只报第一处（`allreduce.cu:120`）——
   如果没有 21:22 的整树快照，这批文件就得手工重建。**每批改动前留快照不是形式主义。**
   （反面例子：`text_context_impl.h:2876` 那行原本是单行循环体 `{ rope_position[r] = ...; }`，整行替换会把它整段丢掉，
   而丢掉后**仍能编译通过**——静默错误。是快照配对避免了它。）

### 验证（TP2 闸门）
- `ninfer_kv_capacity_tp2_test` / `ninfer_public_api_test`：**ok**。
- 服务路径 A/B（同工件同参数同贪心请求）：**OLD / NEW2 / NEW3 三方完全一致**
  —— 输出 `'alpha beta gamma'`、completion 37 / prompt 59、`graph-nodes=2562`、KV 131072/2048 pages/3.67 GiB、
  每卡 14713/14711 MiB。

### 仍未通用化（各自归属）
- `1 - rank` 风格 12 处 → W3（通信层，TP4 下没有「另一个 rank」，要遍历 peer 集合）。
- `tp == 2` 断言 70 处 → W3（按后端分派：tp==2 走 pull、tp==4 走 NCCL，其余报错）。
- 分片算术（`kShardVocab`/`kShardQHeads`… 现为编译期常量 `kTensorParallelWidth = 2`）→ **W5**（改成运行期 `ec.tp` 的计划器）。

## W3 第一版完成（NCCL 后端）＋ 一个**方案级发现**（2026-10-01 00:xx–01:xx）

### 落地内容
- `src/core/tp_comm.{h,cu}`：`TpComm`（每 rank 一个 NCCL comm + **多线程 warmup**，落实 M1 的首次集合通信约束）+
  `nccl_allreduce_sum` / `nccl_allgather_rows`；**只有这一个 TU include nccl.h**。
- `ExecutionContext::comm`（`shared_ptr<TpComm>`，前向声明，header 不含 nccl.h）；引擎唯一挂接点
  （`program_impl.h:310` 一带）留待 W5 接。
- `allreduce_sum` / `allgather_rows` 按 `ec.tp` 分派（tp2 → pull；tp>2 → NCCL）；
  `gather_columns_rank0` / `broadcast_rank0` 在 tp>2 **显式拒绝**（rank0-centric，D5）。
- `require_split_context` 放宽到「≥2 rank、每 rank 一个 context、设备互不相同」（单一改点，全 op 生效）。
- `PeerEvents` 泛化到任意 tp（op 签名要求它；tp>2 只构造不使用）。
- **allgather 无需重排 kernel**：真实调用是 C=1（`[1, shard_vocab]` → `[1, vocab]`），NCCL 的 rank-major
  拼接与该布局**逐字节相同**；C>1 会显式拒绝而不是写错。
- CMake：`NINFER_NCCL_ROOT` 发现 NCCL（显式候选路径，因为 conda 只有 `libnccl.so.2` 没有 symlink）+ 链入 ninfer_core。
- 新测试 `tests/ops/test_allreduce_nccl4.cpp`（4 卡 opt-in，skip 77）：decode 形状、prefill 形状 ×2、
  allgather、连打 50×4、4 线程直连对照、计时。**当前全绿（rc=0）**。

### ★ 方案级发现：单线程发射大 payload 会**静默**算错

实测（4 rank，真实形状）：

| 发射方式 | 10 KiB allreduce | 480 KiB allreduce（prefill 的 [5120,48]） | 496 KiB allgather |
|---|---|---|---|
| **单线程**（rank0..3 依次，本运行时的默认形态） | bit-exact | **尾部 ~20% 是错的 partial sum，非确定，随 NCCL chunking 漂**（4 rank 拿到同一错值） | bit-exact |
| 持久线程池（cv 交接） | — | 仍错 | — |
| 持久线程池（原子自旋交接） | — | 仍错 | — |
| **每次调用新建 4 个线程**（各自 setDevice/launch/stream-sync） | bit-exact | **bit-exact** | bit-exact |

- 复现矩阵：Ring / Tree / 4 / 8 / 默认通道数、`NCCL_LAUNCH_MODE=PARALLEL|GROUP` —— **都错**。
- 错误形态：rank3（最高 rank）的贡献在尾部被漏掉或重复计入 ⇒ 部分归约。
- **M1 的结论需要修正**：M1 说「链路热后单线程发射非阻塞」——那只对**小消息**成立；M1 的大 payload 实验是
  496 KiB **allgather**（单相，恰好没问题），**从没做过大 payload 的 allreduce**（双相）。这正是 M1 harness
  与真实模型的差异点。
- **代价**：新建线程的路径每集合通信 ≈ **106 μs（10 KiB）/ 112 μs（480 KiB）**，对比 M1 单线程 NCCL 的
  21–26 μs ⇒ **贵 4–5×**。折算：decode 129 集合 × 106 μs = **13.7 ms/token**（比现役 TP2 全 token 11.5 ms 还慢）；
  prefill 每 48-token chunk 128 × 112 μs = **14.3 ms**（16k 预填 341 chunk ≈ 4.9 s 纯集合通信）。
- 机制**未查明**（NCCL 对调用线程的要求？per-thread 状态？）——已作为 open item 记录，不粉饰。

### 当前状态判定
W3 的「tp>2 走 NCCL」在**正确性**上成立（每次新建线程路径全绿），但在**性能**上不成立：
两条路都不可接受（单线程=静默错；新建线程=慢 4–5×）。⇒ 需要用户在三条路里选：
(a) 给 ninfer 加 per-rank 执行线程（架构级改动，vLLM 形态，能同时解决 pull 的 host-op 扩展性问题）；
(b) 混合传输：小 payload（decode，图内单线程）用 NCCL，大 payload（prefill）用 pull 协议（单线程、无建线程开销，M1.5 已证 4 卡 bit-exact）。需要实测 pull 在 480 KiB 下的成本；
(c) 暂时搁置 B 线（保留 TP2），或限时继续查机制。

## (b) 混合传输的可行性实验 → **(b) 被自己的数据否掉**（2026-10-01 00:15–00:40）

程序：`~/work/ninfer-tp4-m2/tp4_transport_ab.cu`（同一进程、同缓冲、同机器状态下对拍三种发射方式，全部逐位校验）。

**实测 μs/集合通信（4 rank，bit-exact 已验）**

| payload | pull（单线程 mesh） | NCCL 单线程 | NCCL 每调用新建线程 |
|---|---|---|---|
| 10 KiB（decode 形状 [5120,1]） | 97.8 | **17.9** | 94.6 |
| 480 KiB（prefill 形状 [5120,48]） | **148.6** | **31.7** | 112.5 |
| allgather 485 KiB（[1,248320]，C=1） | 105.8 | **26.4** | — |

⇒ **pull 在大 payload 上比「新建线程的 NCCL」还慢（148.6 vs 112.5）**，(b) 的「省线程开销」论据不成立。
pull 的价值只剩「自研协议、无 NCCL 内部依赖」这一条，不构成性能理由。

**同时发现：NCCL 单线程在独立小程序里 120 次调用全对**（8 种 base 偏移 × 12 轮 = 96 次 + 分配 churn 交错 10 KiB/480 KiB × 24 次），
而在 ninfer 树那个 330 MB 二进制的测试进程里**反复错**。已排除的触发因素：**base 对齐**（0/1/2/4/8/16/64/256 元素偏移全对）、
**分配 churn**（0/24 错）、`NCCL_LAUNCH_MODE`、Ring/Tree/4/8/默认通道数。
⇒ **失败是环境/时序相关的**（这也解释了「同一进程内第 1 次对、第 2 次错」），而这正是最危险的形态：静默、偶发。
机制**未查明**（skew 实验因我自己的悬垂指针 bug 未跑成，待重做）。

### 对方案的含义（需要用户决策）
1. tp4 上**没有任何既便宜又可信的传输**：decode 形状单线程 NCCL 便宜（18 μs）但同一代码在别的进程里错过 ⇒ 不可托付；
   prefill 形状只能靠「每 rank 一线程」（112 μs）或 pull（149 μs），折算 128 集合/chunk = 14.3/19 ms，
   16k 预填要多 5–6.5 s（现役 TP2 16k 预填约 8 s）⇒ **prefill 明显退步**。
2. B 线因此**卡在执行模型上**：单线程运行时 + NCCL 大 payload 组合不成立。
3. 两条出路：(a) 给 ninfer 加 per-rank 执行线程（架构级）；或 (b') **先花半天把触发条件找出来**——
   重开单线程路径（env 开关），在小程序里逐步复刻 ninfer 进程的特征（大二进制、PeerEvents 事件创建、
   1-D→2-D 顺序、诊断循环），定位后 tp4 可能重新变便宜且正确。

## ★ 触发条件查明：**是我的测试违反了 op 文档里的调用方义务**（2026-10-01 00:40–01:20）

二分过程（都在 ninfer 二进制内，`tests/ops/test_tp4_issue_probe.cpp` 最小探针）：

| 变体（单线程发射 480 KiB） | 结果 |
|---|---|
| 缓冲一次分配、复用 | bit-exact |
| 每轮 malloc/free（地址抖动）+ 集合通信前 **stream sync** | **WRONG** |
| 每轮 malloc/free + 集合通信前 **device sync** | **bit-exact** |

⇒ 触发条件是 **`allreduce.h` 的 CALLER OBLIGATION**：
用 `cudaMemcpy`/`cudaMemset`/`<<<>>>`（legacy default stream）暂存的输入必须 **retired** 后再让集合通信读；
而且**大 payload 下 per-stream sync 不够，需要 device sync**（文档原文：「omitting the wait makes the suite
intermittently read pre-staging bytes」）。

### 对先前结论的更正（两条）
1. ~~「M1 的『热后单线程发射安全』只对小消息成立，大 payload 需要并发进入」~~ → **错**。
   单线程发射对大 payload 也是对的：`test_allreduce_nccl4` 现在在**纯单线程路径**下 480 KiB 两轮都 bit-exact
   （rc=0）。真实原因不是线程模型，是输入暂存没有被正确 retired。
2. ~~「(b) 混合传输（大 payload 走 pull）」~~ → 不需要了，而且 pull 实测更慢（480 KiB：pull 148.6 vs
   NCCL 单线程 **31.7** μs）。

### 为什么确认是「我的测试」而不是 NCCL：独立小程序从未复现
`~/work/ninfer-tp4-m2/tp4_transport_ab.cu` 的 `fill_main()` 里**恰好**有 `cudaDeviceSynchronize()`（我写它时
是照 tp2 测试的 `retire_staging` 惯例），所以它 120+ 次全对；而大测试的 `check_allreduce` 用的是 stream sync
⇒ 复现。两处矛盾由此完全一致。

### 最终性能（单线程发射，含每集合通信 4 次 launch 的 host 成本）
| payload | NCCL 单线程（现役实现） | 曾误用的「每调用新建线程」 |
|---|---|---|
| 10 KiB（decode 形状） | **16.8 μs** | 94.6 μs |
| 480 KiB（prefill 形状） | **32.6 μs** | 112.5 μs |
| allgather 485 KiB | 26.4 μs | — |

折算：decode 129 集合 ≈ **2.2 ms/token** 通信；prefill 每 48-token chunk 128 集合 ≈ **4.2 ms**（16k 预填 341 chunk ≈ 1.4 s）
⇒ 都比 TP2 现有预算宽裕，B 线不需要动执行模型（(a) 也不必做）。

### 现状与遗留
- `src/core/tp_comm.cu` 的 eager 路径已**改回纯单线程**；`NINFER_TP4_ISSUE` 开关与线程路径、`RankTeam` 全部移除。
- 文档级结论写进了 `nccl_allreduce_sum` 的注释（含「stream sync 不够、要 device sync」的实测依据）。
- **M3/M4 待办（新增）**：审计模型侧是否有把 **allreduce/allgather 输入**用 `cudaMemcpy`/`cudaMemset` 暂存的路径
  （尤其 prefill 的 token/position 上传）；若有，必须 device-sync 退休，否则大 payload 下会静默错。
  测试脚本 `tests/ops/test_tp4_issue_probe.cpp` 保留，作为该义务的回归探针。

## W3 完成：NCCL 传输打通到「W5 边界」（2026-10-01 01:05–01:15）

- 引擎挂接：`program_impl.h` 的 TP 分支改为「tp==2 → pull 机制（peer arena/PeerEvents/bridge）；tp>2 → `TpComm::create`」
  （`execution.comm` 由 op 读取）。守卫也一并泛化：每个 rank 必须有 device context；`tp != plan.tp` 即报不一致。
- `engine.cpp` 的 tp 校验放宽到 {1,2,4}（3 缺省：不整除 hidden/vocab），并**显式拒绝 tp>2 + `--spec dflash`**
  （D5；rank0-centric 协议无 rank>2 推广）。tensor 层的 `gather_columns_rank0`/`broadcast_rank0` 仍会二次拦截。
- 验证：
  - `--tp 4` 现在**干净地停在计划器闸门**：`tensor-parallel width must be 1 or 2`（`layouts_impl.h:669`）
    ⇒ 这就是 W5 的边界，报错清晰、不是崩溃。
  - `--tp 4 --spec dflash` → 明确拒绝；`--tp 3` → 仍被 CLI 拒。
  - **TP2 闸门复验通过**（engine.cpp 改动后）：输出 `'alpha beta gamma'`、tokens 37/59、每卡 14713/14711 MiB，与老基线逐项一致。
- 新增测试（4 卡 opt-in）：`tests/ops/test_allreduce_nccl4.cpp`（全绿）、`tests/ops/test_tp4_issue_probe.cpp`
  （调用方义务的回归探针）。

### 剩余路线
- **W4**：`DecodeGraphPeerBridge` 推广到 N 设备（tp>2 时需要 1 origin + 3 peer 的 fork/join 事件数组）。
  M1 已证「单 capture 装 4 设备」可行，且 tp>2 的集合通信捕获时是单线程记录（无执行）⇒ 机制上无障碍；
  注意 M1 的 W4 待办：**图内含大 payload allreduce 的 replay 未验证**（ninfer 只把 decode 入图，decode payload 是 10 KiB）。
- **W5（大头）**：计划器通用化——`layouts_impl.h` 的两道闸门（sequence plan tp / tensor-parallel width）、
  `bindings.cpp` 的两道、以及 `kShardVocab`/`kShardQHeads` 这类**编译期分片常量改成运行期 `ec.tp`**。
  先产出「TP2 假设清单」再动手。

## W5.2b 第一批：ops wrapper 分片常量运行期化 + GQA/GDN-fold/gating 内核 tp4 实例化（2026-10-01 02:00–02:10）

**背景更正**：交接文档把 W5.2b 说成「ops 层 18 个 kShard* 常量 / ~120 处」，实际上那只是 wrapper 层的
校验常量。真正的第二层是**底层 kernel 几何注册表**——`nvfp4_config.h` / `fp8_config.h` 的
`Nvfp4GemvGeometry<N,K>` / `Fp8Geometry<N,K>` 模板别名、`Nvfp4Problem` / `Fp8Problem` 枚举、
`gqa_attention_geometry.cuh` 的 `GqaGeometry<q,kv,scale>`、`recurrent.cuh` 的 `FoldGeometry<...>`，
全是 tp2 硬编码（全树 `Tp2` 出现 326 次 / ~70 文件）。切 tp4 = 每个几何都要以 N/4、K/4 再实例化一份 +
枚举/分发/X-macro 加一项。这是「5–8 天」的主体。

**本批完成（全部增量，tp2 行为不变）**：
1. 新增 `src/ops/wrapper/shard_extent.h`：27B 各 fused 对象的**全局(tp1)行数**为单一真相 +
   `shard_rows(global, tp)` + `shard_columns(...)` + `tp_array_copy(...)`（把 const TpArray 拷成可写槽数组）。
2. 5 个 wrapper 的分片常量全部运行期化（tp 从 `ec.tp` 取）：
   `gdn_input_proj.cpp`(52 处)、`attn_input_proj.cpp`(19)、`gdn_gating_proj.cpp`(14)、
   `linear_swiglu.cpp`(8)、`linear_add.cpp`。同时把「手工拼 2 元对」`TpArray<T> x{a[0],a[1]}` 改成
   `detail::tp_array_copy(...)`（gdn_input_proj 20 处、attn 8、swiglu 1、linear_add 2），
   以及三处「只比较 rank0/rank1」的 pair 校验改成遍历所有 rank。
   **workspace-capacity 查询函数刻意保持 tp2 数值**（它们只被 tests 调用；模型侧 arena 来自
   workspace_recipe，属于 W5.4），保证 tp2 capacity 测试逐项不变。
3. GQA：`Gqa27Tp4Geometry = GqaGeometry<6,1,4>`（Q 24→6、KV 4→1、DecodeSplitScale 2→4，使
   DecodeSplits×KVHeads 保持 340/2240 不变）；注册进 `NINFER_GQA_GEOMETRIES` 与
   `NINFER_GQA_KV_REPRESENTATIVES`（KVHeads=1 是首个该值），加 static_assert；
   `gqa_geometry_dispatch.cuh` 加一致性断言；`gqa_attention.cpp` 的 `kv_heads_for_q_heads`
   加 6|1、`require_kv_heads` 放行 1、volta flash workspace 的头数改走该函数；
   `gqa_attention_volta_flash.cu` 加 tiling/meta/launch 三分支。
4. GDN fold：`FoldGeometry48x12Tp4 = FoldGeometry<48,4,12,2560>`，进 X-macro；
   `replay.cpp` 的 host 镜像 `is_registered_fold_geometry` 加 48/4/12/2560 一支。
5. BF16 GDN gating 分片内核：`kShardN`(24) + `kShardN4`(12) 两档，gemv 与
   small-T-split10 两个模板按 `a_weight.n` 运行期分派；`bf16_gdn_gating_shard_workspace_bytes`
   加 `heads` 参数。
6. **W5.1 四道闸门打开**：`layouts_impl.h:101`（sequence plan tp）、`:669`（tensor-parallel width，
   并把 vision 拒绝从 `tp == 2` 改成 `tp > 1`）、`bindings.cpp`（load plan 与 LoadedModelData 两处）；
   文案同步改「1, 2, or 4」。

**验收**：全量 `ninja ninfer-serve` 通过；**TP2 回归闸门（build-v100-tp4 新二进制，:8912）逐项一致**：
`content='alpha beta gamma'`、tokens 37/59、`resolved=131072 pages=2048/2048 runtime=3.67 GiB
free-after-weights=5.00 GiB graph-nodes=2562`、卡 0/1 = 14713/14711 MiB。

**已知未做（下一批）**：`LoadedModelData` 仍是「rank0 + 单个 runtime_peer」两视图；
`ProgramImplCore` 的 `std::optional<PeerRuntime> peer` / `TpPeerCore` / ingress-egress 仍是 tp2 单 peer 形态；
NVFP4/FP8/Q4Q5/W8 的 tp4 几何尚未注册（Q4_K_M 走 GGML_K 运行期形状，不需要它们）。

### W5.2b 第二批：LoadedModelData 每 rank 视图 + runtime 层护栏（2026-10-01 02:10–02:55）

- `LoadedModelData`（`src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp}`）的 `std::optional<RuntimeModelView>
  runtime_peer` → `std::array<std::optional<RuntimeModelView>, kMaximumDevices> runtime_peers`，
  构造时按 tp 建 rank 1..tp-1 的视图（纯描述符工作，不复制权重）；`view(rank)` 改按数组取。
  `package.cpp` 取 rank 1 视图当 `peer` 传给 `create_program`。
- **刻意加的 fail-loud 护栏**：`ProgramImplCore` 构造函数里 `if (tp > 2) throw
  "Qwen3.6 program: tp > 2 is not wired up yet (two-rank runtime structure)"`。
  原因：`TextContext` 仍是「rank0 + 单个 peer」结构（68 处 `TpArray<...>` 访问器硬编码 2 元素、
  `*_tp2` 函数族形参 `w0,w1`、`peer_core` 单例、`sequence.text_peer/backend_peer` 单池），
  在 tp4 下若放行会绑一个 2 路视图 → 静默错分片。护栏把边界显式化，等 runtime 推广完成后删除。
- 中途试过把 `create_program` 的 `const ModelView* peer_model` 改成
  `std::span<const ModelView> peer_models`（已回退，`runtime.h`/`api_impl.h`/`program.h` 与快照逐字节一致）：
  这处签名属于 runtime N 卡推广的同一批，不该单独落地留半迁移状态。

**最终态验证**（02:55）：`ninja ninfer-serve` 无待办；TP2 回归闸门（:8912，NEW4）逐项一致；
`--tp 4` 冒烟（Q4_K_M，17.41 GiB 权重全量加载完）停在上述护栏，报错清晰。
四卡各 5 MiB 空闲、无 ninfer 端口监听、两个现役实例仍停。

**改动清单**：`changed_files_vs_w5b_snapshot_2026-10-01.txt`（18 M + 1 A）。

### W5.2c：ops 层 tp4 kernel 几何注册（NVFP4 / FP8 / Q4-Q5）（2026-10-01 03:00–05:30）

**为什么**：上一批只做了 wrapper 层的常量。真正让 tp4 分片 GEMM 能跑的是**每个家族的 kernel 几何注册表**
（`Nvfp4Problem` / `Fp8Problem` 枚举 + `Nvfp4GemvGeometry<N,K>` / `Fp8Geometry<N,K>` 实例 +
各家族 `_shard` launcher 的按形状分派）。tp2 当年正是这么加的，本批照同一先例补 tp4。

**tp4 几何（= tp1 的 1/4；列并行只分 N，行并行只分 K）**
- NVFP4：`AttnInput 3584`、`GdnInput 4096`、`MlpGateUp 8704`、`Residual6144Row K=1536`、
  `Residual17408Row K=4352`；新增激活量化几何 `<1536>` `<4352>`。
- FP8：`Vocabulary 62080`、`AttnInput 3584`、`GdnInput 4096`、`MlpGateUp 8704`、
  `Residual6144Row K=1536`、`Residual17408Row K=4352`；激活几何 `<1536>` `<4352>`。
- 注册方式：alias + 枚举值（追加，不破坏旧编码）+ `Nvfp4ParentGeometry`/X-macro
  （NVFP4）/ 显式 `resolve_fp8_problem` + `is_fp8_linear_problem`（FP8）+ decode/small-T schedule
  继承父几何。

**★ 唯一的 tuning 例外：K = 4352**（17408 的 tp4 四分之一）不是 512 元素相位的整数倍
（4352 = 8×512 + 256）。两条路都要单独处理：
- NVFP4 decode：`Nvfp4LinearDecodeProductionSchedule<Nvfp4Residual17408Tp4RowGeometry>` →
  `Nvfp4GemvSchedule<8,2,8,4,...>`（8 values/lane，相位 256，4352 = 17×256）。
- NVFP4 / FP8 small-T：同样给该几何固定 `kValuesPerLane = 8`（其余几何 T 大时本来就 8）。
- FP8 decode 不用改（本来就是 8 values/lane）。
这是注册表里**唯一** K 不是 512 倍数的几何；其余（5120/6144/17408/3072/8704/1536 及所有列分片）都能继承。

**分派方式**：所有 `_shard` launcher 改成按 `weight.n`（列分片）选中 tp2/tp4 几何，而不是硬编码
tp2。这样 tp4 张量永远不会被 tp2 kernel 读；无法识别的行数直接 fail-loud。
带 `weight` 的入口在函数内分派；没有 `weight` 的（TMA shard、plan 里的 `launch_tma`）加了
`_shard_tp4` 具体函数 + 调用点按几何分派。FP8 的 section traits（`Fp8AttnInputSections` /
`Fp8GdnInputSections`）与 cutlass 的 `CutlassSections` 改成**从同一 traits 读**，
注册一个宽度不会再漏掉某个 kernel。

**Q4/Q5（ET W4A4W8A8 件的 GDN/attention 分组量化权重）**：这些家族的 shard 支持是
「shape 白名单」而非几何模板，按 tp1/4 补条目：
- `q4_q5_gdn_input_plan`：`supported_shard_shape` 加 tp4（qk 1024 / value_z 3072 / qkv 2560 / z 1536）。
- `q4_q5_attn_input_plan`：`q4_q5_attn_input_admits_shard` 加 tp4（query 1536 / kv 256）。
- `q4_dispatch` / `q5_dispatch` 的 shard 表：加 tp4 行数（256/1024/1536/1792/8704/32768，行分片 K 1536/4352）。
- `q5_linear_add_plan`：`kSupports` 加 `{5120,1536}`、`{5120,4352}`。
- **W8 家族没有任何 shard 支持（tp2 也没有）**，故 tp4 同样不支持；不在本批范围。

**验收**：`ninja ninfer-serve` 全绿；**TP2 回归闸门（官方 nvfp4 件，覆盖 NVFP4+FP8 路径）逐项一致**：
`content='alpha beta gamma'`、37/59 tokens、`resolved=131072 pages=2048/2048 runtime=3.67 GiB
free-after-weights=5.00 GiB graph-nodes=2562`、卡 0/1 = 14713/14711 MiB。

**现场**：两个现役实例（:8901/:8902）已拉起并 `/health=200`（跑的是 build-v100 老二进制，不受影响）。

**改动清单**：`changed_files_vs_w52c_2026-10-01.txt`。

### Phase 3a：runtime 层 N 卡推广（第一批：TextContext / TpExecution / ExecutionCore / drivers）（2026-10-01 05:30–08:00）

**目标**：把 tp2 的「rank0 + 单个 peer」结构改成按 rank 索引，使 ops 的 `TpArray` 参数在任意宽度下都完整。
**原则**：每一步都保持可编译 + tp2 行为逐项不变（每步跑 TP2 闸门）。

**已完成**
1. `text_context.h`：
   - 新增 `TextRankBinding`（每 rank 的 execution + 权重绑定 + per-call 控制张量 + proposal head）。
   - 成员 `const TpExecution* tp_` + 8 组 `*_peer_` → `std::array<TextRankBinding, kMaximumDevices> ranks_` + `tp_count_`。
   - `TpPeers = TpArray<std::optional<TpExecution>>`（放在 TpExecution 旁边）。
   - 新增 per-rank 数组访问器：`embed_weights/final_norms/lm_heads/full_layers/gdn_layers/
     attention_projections/attention_output_weights/gdn_projections/gdn_output_weights/
     full_post_mixers/gdn_post_mixers/proposal_heads/proposal_head_id_maps/rank_ios/rank_states/
     rank_scopes`，以及 `rank_embed/rank_final_norm/rank_proposal_head/rank_prefill_hidden/
     rank_batch_kv/rank_batch_mtp_kv/rank_mtp_kv/peer_events`。
   - `workspaces()`/`synchronize_all()`/`stream_for`/`io_for`/`state_for` 改成按 rank 遍历。
2. `text_context_impl.h`：构造器改为吃 `const TpPeers&`（按槽填 `ranks_`、由 lane 数推 `tp_count_`）；
   `bind()` 对每个非零 rank 建绑定；11 处 `auto scope_N = tp_->work->scope()` → `rank_scopes()`；
   8 处 `tp_->work->reset()` → `reset_peer_workspaces()`；`*tp_->events` → `peer_events()`；
   `_tp2` 函数族的显式 2 元列表（`{w0.projection, w1.projection}`、`{a[0],a[1]}`、
   `{io_.mtp->ar_hidden, tp_->io->mtp->ar_hidden}` 等）全部改成 rank 循环；
   `logits_tp2(hidden, logits, peer_logits)` → `logits_tp2(hidden, TpArray<Tensor> logits)`；
   新增 `ScopedRankTensorBinding`（把 per-call 控制张量绑到每个非零 rank，退出时还原；
   原 `ScopedValue<const Tensor*> peer_xxx_binding` 的直接替代）。
3. `schedule.h`：`ExecutionCore::peer`（单指针）→ `TpArray<const TpPeerCore*> peers` + `peer_at(rank)`；
   `tp_execution()` → `tp_executions()`（返回 `TpPeers`）；新增 `peer_lanes` / `peer_mtp_kv_lanes` /
   `peer_ingress_lanes` 三个「单个 rank 1 车道」辅助；`PrefillContext::mtp_kv_peer` →
   `mtp_kv_peers`（数组）；`OrdinaryBatchContext::peer_host_ingress` → 数组。
   **刻意不留 `peer` 别名字段**：同一事实的两份拷贝正是「tp2 路径在 tp4 下继续跑」的成因
   （本轮踩到过：`execution.peer` 没设 → MTP 桥走 tp1 分支 → tp1 的 `gdn_norm_gating_proj`
   拿到分片权重 → `gdn_gating_proj: unsupported ab_weight geometry`）。所有 tp2-only 读取点
   改用 `peer_at(1)`。
4. drivers：`text_prefill_impl.h` / `decode_impl.h` / `mtp_impl.h` / `dflash_impl.h`
   6 个 `TextContext` 构造点改传 `TpPeers`；ordinary decode 的 ingress 上传改成每 rank 循环。

**验收**：`ninja ninfer-serve` 全绿（27B + 35B 两个 variant 都编译）；TP2 回归闸门逐项一致
（共跑 4 次：RTC1 结构改造后 / RTC2 首次 driver 改造后【红】/ RTC3【红】/ RTC4 修好 `peer_at` 后【绿】）。
红→绿的根因见上面第 3 条。

**未完成（下一批）**
- `program.h` / `program_impl.h` 仍是「rank0 + 单个 `std::optional<PeerRuntime> peer` + 单个
  `peer_core`」：需要 `std::vector<std::optional<PeerRuntime>> peers` + `peer_cores` 数组，
  并把 ~40 处 `peer->` 分门别类（每 rank 循环 vs 保持 rank 1）。
- `SequenceKV` 的 `text_peer`/`backend_peer`（每序列单个 peer 池）→ 每 rank 一组；
  这是 tp4 每序列 KV 镜像的前置。
- `DecodeGraphPeerBridge`（两设备 fork/join）→ N 设备（W4）。**首版 tp4 冒烟建议先
  `--no-cuda-graph`**（CLI 已有此逃生门），绕过 W4 把 prefill/decode 主路径先跑通。
- `--spec mtp` 的分片调度仍是两 rank 结构（`mtp_impl.h` 里已注释标明）；tp4 首版用 `--spec off`。
- 上述完成后删 `ProgramImplCore` 的 `tp > 2` 护栏。

### ★ Phase 3b：runtime program 层 N 卡化 + **tp4 四卡跑出 token**（2026-10-01 08:00–12:00）

**里程碑：tp4 在 4×V100 上跑出正确 token。**（`--tp 4 --devices 0,1,2,3 --no-cuda-graph`，Q4_K_M 件）

```
req1: content='alpha beta gamma' finish=stop_token prompt=59 gen=38 ttft=222ms prefill=266 tok/s decode=46.0 tok/s
req2: content='alpha beta gamma' cache=57 reuse=restore_turn_checkpoint ttft=25ms  decode=46.2 tok/s
req3: content='alpha beta gamma' cache=57 reuse=restore_turn_checkpoint ttft=25ms  decode=46.2 tok/s
```
推理链连贯（思考档开着）。

**改动（本轮）**
1. `program.h`/`program_impl.h`：`std::optional<PeerRuntime> peer` → `TpArray<std::optional<PeerRuntime>> peers`
   （每 rank 一个，含 35B 变体）；`peer_core` → `peer_cores` 数组 + `peer_lanes()`；
   `ordinary_peer_host_ingress` → 每 rank 数组 + `publish_peer_ordinary_ingress()` 循环；
   构造器 per-rank 分配（各自 device current）并把 memset/`set_peer_i32`/synchronize 全改循环。
   `peer()` 保留为 slot 1 访问器供 tp2-only 路径（MTP/DFlash/egress-check）读取。
2. `create_program` API：`const ModelView* peer_model` → `std::span<const ModelView* const>`（**按 rank 索引**，
   size == tp）；`package.cpp`（27B）逐 rank 填视图。
3. `--spec mtp/dflash` 在 tp>2 **显式拒绝**（两 rank 调度），`tp>2` 要求 `--no-cuda-graph`（W4 图桥未做）。
4. **27B variant 的 target 层手工 2 元数组全部改 rank 循环**（W5.3 剩余部分，~20 处）：
   `require_same_alternative` 加 ec、新增 `rank_shards<Result>` / `require_agreeing_shards`，
   `pair_of(a,b)` 全部删除。`src/ops/linear/linear.cpp` 的 `validated_outputs` 也有一处
   `TpArray<Tensor>{out[0], out[1]}` → 循环（**这一处直接导致 rank 2/3 的输出张量是 1x1**）。
5. **两个真 bug（tp4 暴露、tp2 也受影响但恰好不触发）**
   - `src/core/tp_comm.cu`：`nccl_allreduce_sum` / `nccl_allgather_rows` 按 rank `cudaSetDevice` 后
     **不恢复当前设备** → 集合通信后停在第 3 卡，之后 rank 0 stream 的任何 launch 报
     `cudaErrorInvalidResourceHandle`（core dump）。加 `CurrentDeviceGuard`。
   - `src/ops/linear/ggml_k/ggml_k.cu`：tiled GDN 的列重排只认 16/8 key heads（6144/3072），
     加 tp4 的 4 heads（1536）分支与其白名单项。
   - `src/ops/gdn_input_proj/gdn_projected_conv.cu`：加 tp4 的 `<2560,512,512,1536>` 几何
     （并把 2560 纳入窄 CTA 特化）。
   - `src/ops/wrapper/gdn_gating_proj.cpp`：`ab_weight` 形状检查的报错补上实际数值（诊断改进）。
6. `--tp 4` 冒烟的报错逐个推进（每步都是 fail-loud，非常有效）：
   `gdn_gating_proj ab_weight`（空张量→variant 2 元数组）→ `GGML K GDN output K=1536` →
   `linear out=1x1`（validated_outputs）→ `GDN projected-conv` → `cudaErrorInvalidResourceHandle` → 出 token。

**验收**：TP2 回归闸门逐项一致（FINAL 次，`graph-nodes=2562`、14713/14711 MiB）；
tp4 三连请求全部精确输出 `'alpha beta gamma'`。

**现场**：两个现役实例已恢复（:8901 / :8902 均 /health=200）。
备份：`~/backups/2026-10-01-ninfer-tp4-token/src_snapshot_tp4_first_token.tar.gz`。

**tp4 现在还不能做的**（下一批）
- CUDA graph（W4：`DecodeGraphPeerBridge` 1 origin + 3 peer）—— 现在必须 `--no-cuda-graph`
- `--spec mtp` / `--spec dflash`（两 rank 调度）
- 生产件（ET W4A4W8A8 / 官方 nvfp4）的 tp4 冒烟未做（Q4_K_M 先跑通；ET 件要走
  NVFP4/FP8/Q4Q5 的 tp4 几何，已注册但未在 tp4 端到端验证）
- 性能基准（当前 decode 46 t/s 是 eager 无 MTP 的数；tp2 参照 decode 87 t/s、TP4 vLLM 160 t/s）

### ★ Phase 4：tp4 CUDA graph（W4）+ 生产件 tp4 端到端（2026-10-01 12:00–12:30）

**结论：tp4 现在默认走 CUDA graph，且 ET W4A4W8A8 生产件在 tp4 上跑通。**

实测（ET W4A4W8A8 件，131072 context，graph 开）：
```
pages=2048/2048 runtime=3.04 GiB free-after-weights=9.57 GiB graph-nodes=4178 graphs=22/36 MiB
'alpha beta gamma' ✅  ttft 124ms  prefill 477 tok/s  decode 49.1 tok/s
```
tp2 对照（同件、同 KV 量、现役 :8901）：`pages=2048/2048 runtime=3.67 GiB free-after-weights=4.49 GiB`。

**★ W5.5 的核心问题当场得到回答**：tp4 的 **page 几何与 tp2 逐字相同**（2048 pages、64 tokens/page），
**没有被再除一次** —— 「page 几何与 block table 故意按设备复制」这一设计前提成立；
而且每卡 KV runtime 反而略小（3.04 vs 3.67 GiB，KV head 数 1 vs 2），
**free-after-weights 从 4.49 GiB 涨到 9.57 GiB**（权重被四等分），tp4 的余量是 tp2 的两倍多。

**改动**
1. `DecodeGraphPeerBridge`（`src/core/decode_graph.{h,cpp}`）两设备 → N 设备：
   `peer_devices` 按 rank 索引，每 peer 一对 fork/join 事件 + **各自一个 gate 事件**
   （gate 必须建在其记录流的设备上 —— 第一版建在 origin 上，`cudaEventRecord` 到 peer 流
   直接 `cudaErrorInvalidResourceHandle`）；`fork_peer`/`join_peer`/`discard_capture` 全改循环；
   `gate_launch` 收 `TpArray<cudaStream_t>`（按 rank）。
2. `DecodeGraphPeerCapture` → `{bridge, TpArray<cudaStream_t> streams}`；`capture()` 要求每个
   peer rank 都有流。
3. `graph_impl.h`：新增 `peer_graph_streams(execution)`；`capture_graph`/`run_prepared` 覆盖全 rank。
4. `program_impl.h`：`graph_bridge` 在 tp>1 就构造（peer 设备表按 rank）；**删掉 tp>2 要求
   `--no-cuda-graph` 的护栏**。
5. **图显存额度按 tp 缩放**（`layouts_impl.h`）：原来 `tp==2 ? 20MiB : 12MiB` 每请求；
   改成 `tp==1 ? 12 : (8*tp + 4)` MiB（tp2=20 不变，tp4=36）。实测 tp4 单批 22 MiB，36 够；
   额度是每卡预算，四个 rank 都要装下自己那份驱动状态。
6. **`TpComm::create` 也犯了「不恢复当前设备」的错**（循环 `cudaSetDevice` 后停在第 3 卡）——
   它是构造函数里最后一个动设备的调用，导致 `prepare_graphs()` 的第一个 rank-0 launch 直接
   `cudaErrorInvalidResourceHandle`。加 `CurrentDeviceGuard`。**这是本阶段第 3 个同类 bug**
   （前两个：两个 NCCL 集合通信函数）。**审计结论：`cudaSetDevice` 处必须配对恢复。**
7. 生产件 tp4 的 shape 白名单补齐（都有 tp4 分支了，只是 plan 层漏了）：
   `fp8_linear_add_plan.cpp` 的 `resolve_route`、`nvfp4_linear_add_plan.cpp` 的
   `is_6144_family`/`is_17408_family` 加 1536/4352；报错带上实际 (n,k)。
   这验证了 **W5.2c 注册的 NVFP4/FP8/Q4Q5 tp4 几何在 tp4 端到端成立**。

**验收**：TP2 回归闸门两次（W4GATE / ETGATE）逐项一致（`graph-nodes=2562`、14713/14711 MiB、tokens 37/59）。
现场：两实例已恢复（:8901 / :8902，`/health=200`）。
备份：`~/backups/2026-10-01-ninfer-tp4-graph-et/src_snapshot_tp4_graph_plus_et.tar.gz`。

**下一块：MTP 在 tp4（未做，已探明障碍）**
`mtp_forward_stem_tp2` 的映射是 tp2 专用的：rank 0 做 **embedding 全量归一化**、rank 1 做
**hidden 全量归一化**，各自再取自己那半。tp4 下 `append_row_parallel` 把 K 四等分
（rank r 拿 packed 行 `[r*K/4, (r+1)*K/4)`），映射应推广为
`half = tp/2`：rank `r < half` 走 embedding 侧、取第 `r` 个 1/half 切片；`r >= half` 走 hidden 侧。
**但更深的障碍在 MTP round 的两卡结构**：`mtp_impl.h` 的 `MtpRoundView`、ingress 发布
（`publish_peer_mtp_ingress`）、`MtpDecodeState` 每 rank 帧、以及 egress 校验都只处理 rank 1；
`--spec mtp` 现在在 tp>2 被显式拒绝。估时：**stem 半天，整条 round 1–2 天**。

### Phase 4 附：MTP-tp4 探底的结论（2026-10-01 12:30）

放开 tp>2 的 MTP 护栏试跑，**在 `prepare_graphs()` 里 SIGSEGV**，栈：
```
ninfer::ops::speculative_prepare_verify_inputs
  ← mtp_decode_batch_body (mtp_impl.h) 的 lambda
  ← ProgramImplCore::prepare_graphs
  ← ProgramImplCore::ProgramImplCore
```
⇒ 确认整条 MTP round 是两卡结构（`MtpRoundView`、`publish_peer_mtp_ingress`、每 rank
`MtpDecodeState` 帧、egress 校验都只处理 rank 1；rank 2/3 的帧不存在 → 空张量解引用）。
**护栏已装回**（tp>2 + `--spec mtp/dflash` 明确拒绝，不再有 segfault 路径），注释里写了崩溃点。
量准了工作量：stem 的 rank↔半区映射（`append_row_parallel` 四等分 K）约半天，
整条 round 1–2 天。**这是 tp4 剩下的唯一性能大项。**

### ★ Phase 5：tp4 prefill 慢的真根因（路由白名单漏了 6q）+ MTP-tp4 推进到「图能捕获、执行 fault」（2026-10-01 12:30–13:20）

#### 5.1 prefill：根因找到并修好，tp4 反超 tp2 1.7 倍

**先纠正口径**：我之前报的「tp4 prefill ≈ 300–500 t/s」是 **59-token prompt** 的数（ttft 主导），
不是吞吐。用同模型（ET W4A4W8A8）、同 prompt、同默认 `--prefill-chunk 1024` 实测：

| prompt | tp2 | tp4（修前） | **tp4（修后）** |
|---|---|---|---|
| 3643 tok | 1877.1 t/s | 1828.1 | **3178.6** |
| 14416 tok | 1836.5 t/s | 1470.6 | **3129.7 / 3141.0** |

**根因**：`src/ops/wrapper/gqa_attention.cpp` 的 `volta_flash_route_possible()` 白名单
`q_heads == 24 || q_heads == 12` —— **tp4 是 6q，被排除**，于是 tp4 的长 prompt prefill 掉进
`ChunkedSmallT` 慢路径（源码注释自己写了 "slower"）。我把 `Gqa27Tp4Geometry` 注册进了 flash
*launcher*，却漏了这个 *路由* 白名单 → 典型的「注册了 kernel 但没注册路由」。

**定量证据（拟合 prefill 时间 = a·T + b·T²）**：
| | 线性项 a（GEMM 类） | 二次项 b（attention 类） |
|---|---|---|
| tp2 | 0.5324 ms/tok | 9.26e-7 |
| tp4 修前 | 0.5288 ms/tok | **1.05e-5（11.4×）** |
| tp4 修后 | — | 应回到 ~1e-6 量级（3129 t/s 已印证） |

线性项两栈几乎相同（说明 GEMM 部分 tp2→tp4 没提速，因为行并行 GEMM 的 K 被四等分、算术强度下降），
但二次项差 11 倍 —— 这个信号直接指向了 attention 路由。

**`--prefill-chunk` 不是杠杆**：tp4/14.4k 下 512→1385、1024→1471（最优）、4096→1129 t/s。
生产脚本用 4096，在 tp4 上偏大。

#### 5.2 MTP-tp4：已推广到「图能捕获」，执行阶段仍 fault（护栏已装回）

**本轮完成（tp2 已被闸门验证通过，因为闸门就开着 `--spec mtp`）**
1. `mtp_decode_batch_body` 的 else 分支：`MtpRoundView* views[2]` → `TpArray<MtpRoundView>`（每 rank 一个，
   来自各自的 `io->mtp_decode`）——**原 `views[2]` 在 tp4 越界就是最初的 segfault 根因**；
   所有 `{v.x, p.x}` 2 元数组改循环。
2. `target_verify_accept` 的 tp2 重载 → `const TpArray<TargetVerifyFrameView>& frames`（每 rank 一帧）；
   调用点（MTP / DFlash）同步改。
3. **MTP stem 的 rank↔半区映射**（`mtp_forward_stem_tp2`）：packed fc 输入是
   `[embedding_norm (hidden) | hidden_norm (hidden)]`，`append_row_parallel` 按行四等分 →
   `half = tp/2`，rank `r < half` 走 embedding 侧、取第 r 个 1/half 切片，`r >= half` 走 hidden 侧；
   tp2 退化为原来的「rank0=embedding、rank1=hidden」。切片有 stride（[K/tp,T] 是行范围）故加一次
   连续拷贝（几 KB）。**踩到的坑**：`rank_binding(0)` 是空的（`ranks_` 只填 1..tp-1），取设备要用 `ec().dev[r]`。
4. MTP 权重数组 7 处 `{mtp_weights_for(0)…, mtp_weights_for(1)…}` → 新增
   `mtp_attention_weights()/mtp_output_weights()/mtp_post_mixer_weights()/attention_flat()` 访问器。
5. **`SequenceKVBundle` 的 `text_peer`/`backend_peer`（单 peer 池）→ 每 rank 一组**
   （`text_peers`/`backend_peers`，`TpArray<std::optional<PagedKVAllocation>>`），
   reserve/resize/bind/unbind/materialize/trim/release 全部改 per-rank 循环；
   `mtp_kv_view_peer` → `mtp_kv_views_peer`（每 rank 一个 MTP KV window）。
6. **第三处同类白名单**：`gated_delta_net_replay_record` 的 head 几何（只有 16|48、8|24）→ 加 **4|12**
   （group size 3 保持，与 `FoldGeometry48x12Tp4` 配对）。
7. **第四处**：`mtp_split_attn_in` 的 attn_in 行几何（14336、7168）→ 加 **3584**
   （1536 query/gate + 256 key/value，6q/1kv），kernel 加 `<3584,1536,256>` 实例。

**卡在哪**：tp4 MTP 的 **CUDA graph 能捕获成功（graph-nodes=4615，额度 22/84 MiB）**，
但 warmup 执行时：
- `--no-cuda-graph` 一次跑给出 `linear: expected [K,T] x [N,K] -> [N,T], got x=1x1 w=5120x1536 out=5120x1`
  （`w` = MTP o_proj 的 tp4 分片 [5120,6144/4=1536]，说明 MTP attention 结果数组 `a[r]` 为空）；
- 同一路径另一次跑直接 `cudaErrorIllegalAddress`（**非确定性** ⇒ 更像内存写越界而非形状错）。

**下一次接着做的第一步**：在 `mtp_forward_tail_tp2` 里打印每个 rank 的
`a[r]`/`q[r]` 形状（我加过一版诊断但那次跑撞上 illegal access 没到打印点）；
重点查 `workspace_recipe::mtp_attention_results(*ws[r], T, ec().tp)` 的 [query_size/tp, T]
与实际 `view({head_dim, shard_q_heads(), T})` 是否逐 rank 一致，以及 MTP KV 池在 tp4（kv_heads=1）
下 `gqa_attention_cached` 的 page 几何。

**护栏**：tp>2 + `--spec mtp/dflash` 已装回显式拒绝（不再有 segfault / fault 路径），
注释里写了「已推广但未验证 + 故障形态 + 下一步」。

**验收**：TP2 回归闸门（MTPGATE，带 `--spec mtp`）逐项一致（`graph-nodes=2562`、14713/14711 MiB、
37/59 tokens）；tp4（无 MTP）`'alpha beta gamma'` ✓、14.4k prefill **3141 t/s**、decode 44.9 t/s。
现场：两实例已恢复（:8901 / :8902，`/health=200`）。
备份：`~/backups/2026-10-01-ninfer-tp4-prefill-fix/src_snapshot_tp4_prefill_fix.tar.gz`。

### Phase 6：MTP-tp4 不再崩（最后那个 2 元 brace），但数值仍不对（2026-10-01 13:20–14:00）

**修掉的最后一处崩溃根因**：`mtp_prefill_chunk_tp2` 的 final-chunk 阶段还有一处手工 2 元对
```cpp
ops::linear_row_parallel({a[0].view({shard_q_size(), 1}), a[1].view({shard_q_size(), 1})},
                         mtp_output_weights(), o, last_staging, execution, peer_events());
```
→ `attention_flat(a, 1)`。**注意我的正则扫描漏了它**：模式是 `, 1)` 而不是 `, T)`，而且
`.view({...})` 的**嵌套花括号**让 `\{[^}]*\[0\][^}]*\[1\]\}` 这类模式匹配不到。
**正确扫法**：按「同一行同时出现 `[0]` 与 `[1]`」筛，再人眼过一遍（全树只剩这一处真的行为性 2 元数组；
其余是成对比较的校验，已一并改成遍历全 rank：`linear.cpp` 的 `validate_split_pair`）。

**另修的校验**：`src/ops/linear/linear.cpp` 的 `validate_split_pair` 原来只比 rank0/rank1，
改成 `for rank in 1..tp` 逐对比，column/row 两种情形分开判。

**现在的 tp4 MTP 状态：能跑完、但结果错**
```
done finish=stop_token prompt=59 gen=48 ... speculative=mtp 1.00tok/round (0.0%)
content: 'We need to reply with a short phrase: "PROMised and the user says 'Dial up the phrase ...'
```
- 接受率 **0.0%**（tp2 是 84.8%）⇒ 草案全被拒
- 输出**乱码** ⇒ 说明不只是草案错，target/verify 侧也被影响（MTP 打开时输出由 target 的验证结果组成）
- decode 34.3 t/s（比无 MTP 的 49 还慢，因为每轮都在做无用的 MTP 前向）

**排查线索（按优先级）**
1. **stem 的 fc K-四分映射没有数值验证过**：实现依据是 `bindings.cpp` 的 `append_row_parallel`
   用 `even_chunk`（连续切）。tp2 的语义是「rank0=整个 embedding_norm、rank1=整个 hidden_norm」，
   tp4 变成「rank0/1 = embedding_norm 的两半，rank2/3 = hidden_norm 的两半」，并各自把自己那
   1/half 切片拷成紧凑缓冲。**最快的验证**：同一 prompt、`--no-cuda-graph`，分别在 tp2/tp4 下
   把 `fc_input[r]` 与 tp2 对应缓冲区逐元素对拍（rank0 的 fc_input 应等于 tp2 rank0 的前一半）。
2. **MTP KV 池在 tp4（kv_heads=1）的 page 几何**：`rank_mtp_kv(rank)` / `pages.layer_view(0)` /
   `io_for(rank).backend_kv_table_row` 与 `rank_backend_kv_table_rows(rank)` 是否指同一个池。
3. **proposal head 的 gather**：`total_rows = proposal_head_n_ * tp_count_`（原来是两片相加），
   与 `draft_head_token_ids`（replicated）的 remap 在 4 片下是否仍成立。
4. **verification 的 target logits/hidden 每 rank 帧**：`MtpRoundView` 的各个 slice 偏移
   （`.slice(1,…)` / `.slice(2,…)`）是 tp2 时代定的，tp4 是否需要随 rank 变。

**护栏**：tp>2 + `--spec mtp/dflash` 仍显式拒绝（注释里写了「已能跑但数值错 + 上列线索」）。

**验收（本批）**：TP2 闸门（MTPFIN，带 `--spec mtp`）逐项一致，且 `speculative=mtp 3.55tok/round (84.8%)`
与基线完全相同 ⇒ **MTP 的 N 卡化在 tp2 上正确**；tp4（无 MTP）`'alpha beta gamma'` ✓、
14.4k prefill **3140.8 t/s**、decode 44.9 t/s。
备份：`~/backups/2026-10-01-ninfer-tp4-mtp-partial/`。现场：两实例已恢复。

## 2026-10-01（续）tp4 MTP：崩溃根因 + 状态污染根因（均修）

- **修 1（崩溃）**：`mtp_forward_stem_tp2` 的 ids 跨 stream 竞态。`half = tp/2`，tp4 下 rank 1 也走
  embedding 分支、去读 rank 0 的 ids 缓冲，而写它的只有 rank 0 的 stream → 无排序 → 垃圾 token id
  → embedding gather 越界 → 间歇 `cudaErrorIllegalAddress`（warmup 期 6 次挂 2 次）。修法：stem 入口
  按 rank record event、embed 侧 peer stream wait（capture 安全）。
  定位钩子：`NINFER_TP4_MTP_STAGE_SYNC=1`（sticky fault 只在下一个 sync 报，裸跑永远指向最后 sync 处）。
- **修 2（状态污染，决定性）**：speculative commit 的 `ops::gdn_replay_fold` 只对 rank 0 + `peer()`(rank 1)
  执行 → tp4 rank 2/3 的 GDN conv/recurrent 状态永远不被 accepted prefix 推进 → 每轮读滞后状态 →
  前几 token 正常、之后退化成自循环。修法：遍历 `peer_lanes()` 的 1..tp-1 各自 fold。
  判据实验：`NINFER_TP4_MTP_ZERO_DRAFTS=1`（draft 恒 0、必被拒）——修前仍自循环，修后与无投机同义。
- **一并推广**：`mtp_forward_batch` 的 columns 2 元 brace；MTP peer ingress per-rank（含各自
  `token_counts` 车道）；`resume_hidden` 广播到全 rank（rank 1 保留原 event 路径）；prefill bridge
  `mtp_bridge_tp2` 全 rank 化（work/io/next_hidden/KV 窗口）；`publish_peer_token_counts` 镜像全 rank。
- **结果**：tp4 MTP 输出正确（4 条 prompt 中 3 条与无投机逐字相同，1 条语义等价的措辞差异，无循环），
  护栏改述为"能跑但几乎不接受"（默认仍拒，`NINFER_ALLOW_TP4_MTP_UNVERIFIED=1` 可开）。
- **遗留**：接受率 tp4 0–12% vs tp2 84.8% → 无加速。两条线索：verify 列 1 间歇算成下一列
  （col0/2/3 与无投机逐 token 一致，col1 不一致，已排除 buffer 别名/位置张量/尾部 clamp）；
  每轮第一份 draft 质量差（k=1 接受率即其正确率）。详见
  `~/work/ninfer-tp4-m2/README_tp4_mtp_rca_2026-10-01.md`。
- **回归**：tp2 闸门 `graph-nodes=2562` + `mtp 3.55tok/round (84.8%)`，与改动前逐字一致（tp2 未受影响）。
- 另：w4a4w8a8 件 + tp4 MTP 在 warmup 报 `w8 linear: unsupported shape or T`（q4_k_m 正常），独立问题，未修。

## 2026-10-01（再续）瓶颈判定：verify 是好的，坏的是 MTP 提案

- 新标尺（数数任务，"继续数列 1 2 3 4 5 6 7 8"，`mtp_proposal_quality.sh`）：
  **输出 tp2 与 tp4 逐字相同（`9 10 11 … 30`），接受率 97.9% vs 22.7%。**
  ⇒ verify + 提交链路正确，MTP 提案头被劣化。这条判据不依赖任何参照前向。
- 23 轮 trace：target（模型自身 argmax）链严格递增且完全正确（拼出 ` 10 11 12 … 22`），
  drafts 却反复同一形态（`' 12'` 出现 8 次）→ MTP 上下文推进不良，不是随机噪声。
- 证伪两条旧假说：k=1（width=2）下列 1 依旧给 ` answer`（与宽度无关）；
  `NINFER_TP4_MTP_DUP_DRAFTS=1`（所有 draft 同一 token、各列只差位置）实测各列 logits 指纹互不相同
  → 列 1 没有取列 2 的位置/输入。第 3 轮里列 1 又是对的 ⇒ "verify 列 1 异常"降级为间发、机制未复现。
- 下一个唯一有判别力的实验：把同一份 dump 出来的 MTP hidden 分别喂给 tp2 / tp4 的 proposal 头，
  判定劣化在"头 + logits gather"还是在"MTP 模块（stem/attention/MLP）"。
- 新增脚本 `mtp_proposal_quality.sh`（tp2 vs tp4 提案质量标尺）；护栏注释已按本轮结论更正。

## 2026-10-01（三续）定位实验：头证明是好的；阶段 A/B 有混淆、已更正

- **已证明（唯一无混淆）**：`NINFER_TP4_MTP_HIDDEN_DUMP/LOAD` 把第一轮 `ar_hidden`（[hidden,1]，
  复制的）钉死喂给两边的 `mtp_propose_batch` → tp4 353、**tp2 也是 353** ⇒ **提案头（LM head +
  logits gather）在 tp2/tp4 上对同一输入一致，不是缺陷点**。嫌疑落到 MTP 模块（stem→attention→
  post-mixer）或 MTP KV 状态。
- **更正**：先前"tail / post-mixer 有问题"的结论**撤回**。stage 级 A/B 有两处混淆：
  ①proposal 的输入是 veriy 选中列（`accepted` 随宽度微小差异而变），②attention 读的 MTP KV
  是各自宽度的 prefill 写的。实测"钉死 stem 输出后两边 attention 段输出差 10210/10240 元素、
  max|Δ|=3.79" 看似有力，但**无法区分**宽度算术错与 KV 内容不同 ⇒ 不能当证据。
- **下一步（干净）**：把 MTP KV 的 row 页内容也 dump/load 钉住，再用同一份 stem 输出 + KV
  逐元素 diff tail 输出（`NINFER_TP4_MTP_TAILOUT_DUMP` 已就位）；或把 `tests/ops/test_mtp_split.cpp`
  的 tp2 Leg A 参数化到 tp4（合成输入 + 真实 MTP 权重）。
- 新增探针：`NINFER_TP4_MTP_{STEM,POSTMIX}_{DUMP,LOAD}`、`NINFER_TP4_MTP_TAILOUT_DUMP`
  （均为"进程内第一次 MTP core 调用生效"，比对时须保持 `--draft-tokens` 与调用类型一致）。
- 回归：tp2 闸门 `graph-nodes=2562` + `84.8%`（第三次复测，仍与基线一致）。

## 2026-10-01（四续）证据重审：重量分片逐字节相同；分片 GEMM 执行是唯一未证之处

- **证明**：`NINFER_TP4_MTP_WEIGHT_DUMP` dump 原始 payload，按 artifact 真实布局
  (`ggml-k256-v1` = [每行 8B 描述符平面 | 行数据]，描述符 bit0=q6、其余位=行内字节偏移) 解码后，
  tp4 (rank0+rank1) 覆盖的全局 gate 行与 tp2 rank0 **逐字节相同**（17408 行 0 行不同），up 半同理
  ⇒ MTP 权重分片与行映射无问题。附：`tensor_row_slice` 对 GgmlK256 是按描述符正确实现的。
- **证明**：pin 探针生效（tp4 未钉那次 dump 出的 (x,mh)，同进程 LOAD 后输出逐位相同 = 自洽）。
- **因此**：在"两边都钉同一 (x,mh) + 权重逐字节相同"下 tail 输出仍差 1095/5120 元素（bf16 逐字节比较），
  唯一剩下的宽度相关变量 = 分片 GEMM 执行路径（列并行 gate_up n=8704/k=5120；行并行 down
  n=5120/k=4352 + 2-vs-4 rank allreduce）。
- **纠正我自己的三处测量错误**（写进 RCA §8）：①BF16 当 FP16 解码（差异"个数"仍有效，数值无效）；
  ②把"未钉输入"与"钉输入"两次运行当可比（先从日志确认两侧 probe 行都在）；③probe 的 done 标志各自独立，
  而 `Variant::mtp_post_mixer` 也被 prefill/tp1 路径调用 ⇒ MTP-MLP 内部 dump 的"第一次调用"未必与
  tail 探针同一次，逐行对比无效。**只有同一函数内、同一次 call 的 dump 才可比。**
- **op 测试补测**：`tests/ops/test_ggml_k.cpp` 原本只覆盖 tp1/tp2 形状；本轮补入全部 tp4 分片形状
  (8704×5120 / 5120×4352 / 5120×1536 / 3584×5120 / 5120×2560, T=1,2,4) —— **全部通过 FP64 oracle**
  (relative_l2 0.0011–0.0020，阈值 0.004) ⇒ 内核没问题；未覆盖的是 `linear_*_parallel` wrapper 的 tp4 调度。
- **下一步**：host 侧按描述符反量化 GGML_K 权重 + 复算 MTP 层逐 stage 对拍（无跨宽度 A/B、无 pin，
  故无上述混淆）。KV 页 dump 需新增。
- 现场：8901/8902 已恢复（/health=200）。

## 2026-10-01（五续）修复加载阻断：W8 分派表缺 tp4 分片 extent

- **根因**：`select_w8_a16_launch` 先查表、查不到直接 `throw "w8 linear: unsupported shape or T"`，
  而表里只有 tp1/tp2 的 shard extent；Volta 的兜底在 throw 之后才轮到 ⇒ 凡 MTP 走 W8G32_F16S 的件
  （官方 nvfp4 / ET w4a4w8a8 / Swift-1.5 NVFP4）在 tp4 + MTP 下**加载即失败**。
- **修法**：`w8_dispatch.cpp` 把 tp2 清单镜像 /4（列并行 n∈{256,1536,3584,8704,62080}；
  行并行 k∈{1536,2560,4352}），函数改名 `select_w8_shard_launch`，`n==512` 特例扩成 `512||256`。
- **验证**：①官方 nvfp4 件 tp4+MTP **现在能跑**，数数任务输出正确、接受率 24.6%（tp2 ~97.9%）；
  ②`tests/ops/test_w8_a16.cpp` 补 5 个 tp4 形状 → `OK W8_A16 Linear`（含参考实现逐样本对拍）；
  ③tp2 闸门 `graph-nodes=2562` + `84.8%` 与基线一致。
- 现场：8901/8902 已恢复（/health=200）。

## 2026-10-02（六续）host 参考复算：工具建成 + 分段验证；RMSNorm 段未复现（不下结论）

- **建成**：`NINFER_TP4_MTP_STEM_REF_DUMP`（自描述 dump，同一次调用内取 ids/hidden/embedding/
  stem 输出 x,ah/三个 norm/每 rank 的 normalized_*/fcInput*/fcShard*）+ `mtp_stem_reference.py`
  （fp32 host 参考）。
- **已验证**：①GGML_K Q4_K/Q6_K 反量化在**真实块**上与引擎自己的 oracle `max|d|=0`（随机字节有 fp16 NaN
  不能校验；Q6_K 的 scale 索引每 16 个 j 换一次，写成每 section 一个会静默错 39–707 倍）；
  ②K 打包约定证实（fc_input[0]≡normalized_embedding 的偏差、fc_input[1]≡normalized_hidden）。
- **卡点**：RMSNorm 段对不上（两个宽度都差 20–500 相对值）。穷举 per-col/per-row/global、gain=1+w / w、
  只乘 gain 等组合均不匹配；"行项+列项分解"检验残差 0.07（行项与 log(1+w) 相关性≈0）⇒
  判断是**我读到的缓冲不是那次运算的(输入,输出,权重)三元组**，而非归一化公式的细节差异。
  段长度均已核对（norm 10240B、矩阵 92160B），排除解析/长度错。
- **因此**：参考暂不可用于判定 tp4 stem；§8 的"分片 GEMM 执行"线索维持未证。
- **下一步**：独立的 `ops::rmsnorm` op 级探针（合成输入+权重，dump 三元组）把 op 语义与
  stem 里的缓冲身份解耦。

## ★★ 2026-10-02（七续）tp4 MTP 根因修复：stem 的 K 切片拷贝用了行主序假设

- **根因**：`mtp_forward_stem_tp2` 在 tp>2 时把「行区间 × 所有列」当连续内存一次性 memcpy，
  但缓冲是**列主序**（`ne[0]=hidden` 最快）⇒ 每个 tp4 rank 的 `fc_input` 是错的激活切片。
  tp2 走 `half==1` 快捷分支、不执行这段 ⇒ **只有 tp4 坏**（值仍是真激活，所以症状是"草案差"而非乱码）。
- **修法**：按列逐次拷贝（T 次），注释写明坑与"tp2 为何躲过"。
- **找到它的方法**：host 侧 fp32 参考（自描述 dump + GGML_K 反量化与引擎 oracle 真实块 max|d|=0 +
  列主序 reshape），并**先在 tp2 把参考验证到 rel_l2 ≤ 0.0037** 再判 tp4；tp4 的 norm 段 0.0017/0.0018 ✓
  而 `fc_input` 1.2–1.36 ✗ ⇒ 一步锁定。
- **修后实测**：`fc_input` 0.0016–0.0019、`stem x/ah` 0.0037；数数任务接受率 22.7%→**100.0%**；
  4 条 prompt 73.9–100%（k=1）/57.4–100%（k=3）且 3 条与无投机逐字相同；tp4 decode 43.5→**111.0 t/s**；
  官方 nvfp4 件 tp4+MTP **100% 接受率、188.8 tok/s**；tp2 闸门 2562/84.8% 不变。
- **护栏撤除**：删除 tp>2 的投机拒绝与 `NINFER_ALLOW_TP4_MTP_UNVERIFIED`（两条根因都已修，无可绕过之物）。
- §8 的"分片 GEMM 执行有问题"线索撤销（是激活拷错，不是 GEMM）；§9 的 W8 表修复独立且仍然必要。

## 2026-10-02（八续）收尾：现役件 tp4 实测、sync/reset 残留清理、batch>1 验证

- **现役件 ET w4a4w8a8（W8 路径）tp4+MTP**：输出正确、**接受率 100.0%、decode 182.1 t/s**
  （此前 tp4 加载即失败）⇒ §9 + §11 两处修复都生效。
- **吞吐对比**（同件同 flags）：tp2 无投机 33.9 → MTP 123.6 t/s（100%）；tp4 49.1 → **180.6 t/s**（95.6%）。
  单实例 tp4 快 1.46×，但吃满 4 卡；现役 4 卡两实例聚合 247 t/s > 180.6 ⇒ 布局切换留给用户决定。
- **清掉 sync/reset 残留**：`program_impl.h` 11 处 rank-1-only 的 `peer()->device.synchronize()` /
  `peer()->work.reset()` 改为 `synchronize_peers()` / `reset_peer_works()`（遍历 `peer_lanes()`）。
  回归：tp2 闸门 2562/84.8% ✓、tp4 数数 100%/128.4 t/s ✓、tp4 4-prompt 3/4 逐字相同 ✓。
- **tp4 batch>1 已测**：`--max-concurrency 4` + 4 并发请求，4 条答案全部正确、接受率 45–100%、
  wall 1.47 s ⇒ 之前悬着的 `next_drafts` 形状/注释不一致在实测下无害。
