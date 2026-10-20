# 2026-10-21 — TCP_NODELAY + prefix checkpoints (TP4)

**EN** — Two changes landed on the TP4 stack: a 5-line TCP_NODELAY fix that removes ~22 ms of
first-token latency on reused connections, and host prefix checkpoints that let a request
resume a long shared prefix instead of re-prefilling it (10.3 s → 1.5–3.0 s on a 32k branch).
This note also records the audit of upstream `tuxKOH/ninfer-V100X2 @ 5c515130` (2026-10-19):
which items are portable to TP4, which are gated to TP2/other geometries, and the measurements
behind each rejection.

**中文** — TP4 栈落地两项：5 行 TCP_NODELAY 修掉复用连接上约 22 ms 的首 token 延迟；
host 前缀 checkpoint 让请求从长公共前缀续播，而不是整段重算（32k 分支 10.3 s → 1.5–3.0 s）。
本文同时记录对上游 `tuxKOH/ninfer-V100X2 @ 5c515130`（2026-10-19）的审计：哪些能移植到 TP4、
哪些被 TP2/几何门槛挡住，以及每项判断背后的实测数据。

---

## 1. Upstream audit / 上游审计

Upstream's 2026-10-19 batch claims 143.46 → 198.10 tok/s (about +38%) at 85K, but that is
TP2 / PCIe (no NVLink) / QUASAR / MTP4 with the experimental long-attention and fast-decode
paths both enabled, and the author notes the output may change. TP4 needs different evidence.

上游 10-19 那批声称 85K 下 143.46 → 198.10 tok/s（约 +38%），但口径是 TP2 / PCIe 无 NVLink /
QUASAR / MTP4，且同时开着长注意力与快速解码两个实验开关，作者也注明输出可能变。TP4 不能直接套用。

| Upstream item / 上游项 | TP4 verdict / 本地判定 | Evidence / 依据 |
|---|---|---|
| `verify5` verify kernel | not used / 不用 | `if (invocation.width == 5)`; we run draft 3 (width 4) |
| GDN conv-record fusion | not used / 不用 | requires `w.n == 8192`; TP4 geometry is `n = 4096` (TP2 is 8192) |
| GDN gating `tp2_exact` | not used / 不用 | `kShardN = 24`; TP4 shard is 48/4 = 12 |
| allreduce two-epoch packet | not used / 不用 | packet storage is allocated only when `device_count == 2` |
| QPN fused row-add + publish | not used / 不用 | `ec.device_count == 2` |
| residual-add fusion | below threshold / 低于判据 | nsys: whole `residual_add` kernel = **0.126 ms/round = 0.6%** |
| precast (GDN output) | estimate, not isolated / 估计 | existing `bf16_to_fp16` = 0.152 ms/round = 0.7% |
| `__ldcs` streaming loads | measured slower / 实测更慢 | A/B/B/A: 21.33 → 21.40 ms/round (**+0.3%**) |
| w8 MTP GEMV | not run / 未跑 | w8 family = 0.810 ms/round (**3.9% upper bound**); changes reduction order |
| 1–8-way concurrency | no observed benefit / 无观测收益 | 2 days of logs: `waiting=0`, `avg_decode_batch=1.00` |
| `--prefix-cache-tokens` default | replaced by our checkpoints / 用自有机制替代 | no paged host-prefix subsystem in this tree |

The nsys profile behind the per-round shares: current TP4 binary, 32,340-token prompt, 128
tokens, MTP3, 49 decode rounds, 21.00 ms/round/card.

每轮占比来自 nsys 剖析：现役 TP4 二进制、32,340 token prompt、128 token、MTP3、49 个 decode 轮，
每卡每轮 21.00 ms。

---

## 2. Change A — TCP_NODELAY

**EN** — cpp-httplib ships with `CPPHTTPLIB_TCP_NODELAY` defaulting to `false` and the server
never overrode it. SSE headers and the first token frame are separate small writes, so Nagle
holds the frame behind the peer's delayed ACK on a reused connection.

**中文** — cpp-httplib 的 `CPPHTTPLIB_TCP_NODELAY` 默认 `false`，服务端也没有覆盖。SSE 的响应头
与首帧是两次小写入，复用连接上 Nagle 会等对端延迟 ACK，把首帧压住约 40 ms。

Patch (`src/serve/http_server.cpp`, 1 call + 4 comment lines):

```cpp
server_.set_tcp_nodelay(true);
```

Baseline / 基线 (current binary, n=10):

| Group / 组 | first frame / 首帧 | first content token / 首个 content token |
|---|---:|---:|
| fresh connection / 新连接 | 0.6 ms | 19.1 ms |
| keep-alive reuse / 复用连接 | **40.9 ms** | **41.0 ms** |

Engine-side `ttft` was a constant 19 ms, so the extra ~22 ms was entirely transport.
引擎侧 `ttft` 恒为 19 ms，多出的约 22 ms 全在传输层。

A/B/B/A, two fresh boots per arm, n=10/arm:
两臂各 2 次 fresh boot、每臂 n=10：

| Arm / 臂 | binary / 二进制 | reuse frame / 复用首帧 | reuse content / 复用首 token |
|---|---|---:|---:|
| A1 / A2 | `e97e3bce…` | 40.9 / 40.8 ms | 40.9 / 40.8 ms |
| B1 / B2 | `5cb21aad…` | 0.5 / 0.5 ms | 18.8 / 18.9 ms |

- Reused content: **40.85 → 18.85 ms = −22.0 ms (−53.8%)**; first frame −40.4 ms.
- Fresh connections unchanged; deterministic text fingerprint identical on both arms
  (`fbc8c395aede8af6`).
- 复用 content：**40.85 → 18.85 ms = −22.0 ms（−53.8%）**，首帧 −40.4 ms；新连接不变；
  两臂确定性文本指纹一致。

---

## 3. Change B — host prefix checkpoints + LCP restore

**EN** — A checkpoint is all ranks' GDN convolution/recurrent slots plus rank 0's tail hidden.
It does **not** copy KV: the lane's paged KV bundle stays resident and is trimmed to the chosen
frontier. During prefill the engine captures images at an interval derived from the host budget;
a later request picks the farthest checkpoint whose frontier is ≤ the longest common prefix
(LCP) and whose prefix identity matches, then continues with the suffix.

**中文** — 一个 checkpoint = 所有 rank 的 GDN conv/recurrent slot + rank0 的 tail hidden，
**不复制 KV**：lane 的 paged KV 仍驻留，只截断到选中的 frontier。prefill 期间按 host 预算推导的
间隔捕获镜像；后续请求选「frontier ≤ 最长公共前缀（LCP）且 identity 匹配」的最远 checkpoint，
然后只跑 suffix。

Budget / 预算 (TP4, 27B):

- one rank's GDN slot = 48 layers × (2,560×3×2 B conv + 12×128×128×4 B recurrent) = 36.7 MiB
- one image = 4 ranks + tail hidden = **146.8 MiB**
- `2 GiB / max_concurrency (2)` = 1 GiB/lane ⇒ **6 images/lane**
- capture interval is rounded up to the 4,096-token prefill chunk

Files changed / 改动文件:

| File | Role |
|---|---|
| `src/core/linear_attention_state.{h,cpp}` | `slot_bytes`, `export_slot[_async]`, `import_slot[_async]` |
| `tests/test_state_store.cpp` | export → import → export byte equality, cross-slot, wrong-size rejection |
| `src/targets/qwen3_6/impl/runtime/program.h` | `PrefixCheckpoint`, sequence/plan/prefill fields, predicates |
| `program_impl.h` | capture/import (pinned host image, batched per-rank sync), prefill hook, restore branch |
| `request_plan_impl.h` | capture enablement, LCP selection, MTP preparation |
| `include/ninfer/types.h`, `src/serve/request_log.cpp` | new reuse path enum + log name |
| `src/runtime/contract/types.h`, `src/runtime/engine/concurrent_executor.h`, `src/serve/generation_service.{h,cpp}` | reuse diagnostics (`lcp/led/fr/why`) on the `done` line |

Measured on a 32k branch (A → different question, same 32k document):
32k 分支实测（A → 同一文档换问题）：

| Scenario | TTFT |
|---|---:|
| full prefilling (baseline) / 全量重算 | 10.26 s |
| branch, restore at 24,576 / 分支恢复 | **2.95 s** (−71%) |
| return to A, restore at 28,672 / 回到 A | **1.51 s** (−85%) |
| same prompt again, existing turn checkpoint / 同 prompt 重发 | 89–93 ms |

Quality / 质量:

- Two needles in the 32k document (position ~16k: `473821`; position ~31.5k: `918273`) were both
  answered correctly: the prefix needle comes from restored state, the suffix needle from the
  recomputed suffix.
- Acceptance at equal seed / 96 tokens: 2.21 tok/round (40.9%) restored vs 1.94 (31.9%) full —
  one sample, no regression.
- 32k 文档里两处针（16k 处 `473821`、31.5k 处 `918273`）都答对：前缀针来自恢复，后缀针来自重算。
  同 seed/96 token 接受率：恢复 2.21 tok/轮（40.9%）vs 全量 1.94（31.9%），单样本，不降。

Non-bit-identical output / 非字节一致:

The first generated token is identical, but later text differs at temperature 1.0. This is not
unique to the new path: the existing `restore_turn_checkpoint` path also differs from full
prefill at the same seed (`c84b…` vs `bdcd…`). The regression gate is therefore quality +
acceptance, not byte equality.

首 token 一致，但后段文本在 t=1.0 下会不同。这不是新路径独有：现有
`restore_turn_checkpoint` 同 seed 也与全量不同（`c84b…` vs `bdcd…`）。因此回归闸门是质量 +
接受率，不是字节一致。

---

## 4. Overhead and regression / 开销与回归

- Steady-state full prefill: 10.00–10.01 s vs 9.89 s without capture ⇒ **+1.1%**
  (three images for a 32k prompt). Pinned host images plus one batched sync per capture cut
  this from +1.6%.
- First capture after a boot pays the host allocation (about +0.3–1 s on the first request).
- 稳态全量 prefill：10.00–10.01 s vs 无捕获 9.89 s ⇒ **+1.1%**（32k 三张图）；pinned host +
  每张一次批量同步把开销从 +1.6% 压下来。启动后第一次捕获含 host 分配成本（首请求约 +0.3–1 s）。
- `ninfer_state_store_test` passes (exit 0): export/import round-trip bytes, cross-slot import,
  wrong-size rejection.
- The existing turn-checkpoint and append paths were re-run: same prompt twice still hits
  `restore_turn_checkpoint` at ~90 ms.
- `ninfer_state_store_test` 通过（exit 0）；现有 turn checkpoint / append 路径复测无回归。

---

## 5. Rollback / 回滚

| Binary | SHA-256 prefix | Notes |
|---|---|---|
| current / 现役 | `4456bd9b888c04ff` | checkpoints + diagnostics + TCP_NODELAY |
| diagnostics only / 仅诊断 | `57d76adb…` | `~/backups/2026-10-21-ninfer-prefix-checkpoint/ninfer-serve.diag` |
| TCP only / 仅 TCP | `5cb21aad…` | `~/backups/2026-10-21-ninfer-prefix-diag/ninfer-serve.tcp_only` |
| pre-TCP / 无 TCP | `e97e3bce…` | `~/backups/2026-10-20-ninfer-tcp-nodelay/ninfer-serve.orig` |

---

## 6. Not done / 未做

- **Stage 2**: cross-conversation archive with shared KV chunks. The diagnostic fields show
  how often a full reset has no common prefix (`why=2`); that ratio decides whether Stage 2 is
  worth it.
- **w8 MTP GEMV**: not measured (3.9% upper bound, changes reduction order). Revisit only if a
  real A/B is justified.
- **precast isolation**: the 0.7% figure is the existing cast kernel, not the in-GEMM conversion
  cost.
- **MTP KV at `base-1`**: the restored and full paths may write that position through different
  code, which is the likely source of the later text divergence; quality and acceptance show no
  regression.
- **Stage 2 未做**：跨会话 archive + 共享 KV chunk；等诊断里 `why=2` 的占比决定是否值得。
- **w8 MTP GEMV 未跑**：上限 3.9%，且改约减顺序；除非有真实 A/B 需求再回来。
- **precast 未隔离**：0.7% 是现有 cast 内核，不是 GEMM 内逐组转换的成本。
- **MTP KV 在 `base-1` 的写入**：恢复与全量可能走不同代码，是后段文本差异的疑似来源；
  质量与接受率无回归。
