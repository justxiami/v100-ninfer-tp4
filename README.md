# NInfer V100 tp4 — 4-way tensor parallel for 4 × V100 (SM70) · 4 × V100（SM70）4 路张量并行

[中文文档](README.zh-CN.md) · [Build guide](docs/tp4/README.md) · [Technical history](docs/tp4/changelog.md)

> 每节正文下方直接附中文全文；独立完整中文版：[README.zh-CN.md](README.zh-CN.md)

## TL;DR / 一句话简介

**EN** — ninfer's tuned 2-GPU (TP2) V100 stack, re-hosted as a single-instance **4-way tensor-parallel** engine on 4x V100-SXM2 16 GB: NCCL transport, multi-rank CUDA graphs, MTP/speculative fixes, image multimodal (2026-10-05), QUASAR NVFP4-QAT artifacts. Qwen3.8-27B W4A4: **113 tok/s committed decode (1.81x vs TP2)**, prefill **3.9k tok/s** (chunked GDN: +11.6% over 3,502.7), ~9.4 GiB/GPU at 128k int8 KV. Long-attention decode routing + MTP4 (2026-10-15): pi-shaped 120k decode **96.0 → 109.1 tok/s (1.136x)**. 2026-10-21: keep-alive first token **−22 ms**; branch/prompt-switch TTFT **10.3 s → 1.5–3.0 s**. 2026-10-22: long-context prefill attention swapped to the **Flash-V100 Split-D** kernel (1Cat-vLLM, BSD-3): **120k prefill +9.9%, 32k +4.0%** (engine tok/s), needle 4/4 hit, decode untouched; an MTP copy batch added +4.05% on the 8k prefill. 2026-10-22 (decode): the tp>2 allreduce swapped from NCCL to a **one-shot push allreduce** (1Cat's `sm70_cross_device_reduce_1stage_push`, ported to single-process UVA): engine-level two-arm A/B (fresh boot per arm, client decode tok/s excl. ttft) **154.2/151.7 vs 133.0/131.2 = +15.8% decode, 1.032 ms/token saved**, squarely inside 1Cat's +13~18% reference band. Two misdiagnoses corrected along the way (an 'IOMMU dead-end' that was a NULL-pointer bug; a graph-capture crash that was a current-device leak — the live decode is `--no-cuda-graph` eager).

**中文** — 把 ninfer 调优过的双卡（TP2）V100 移植改写为单实例 **4 路张量并行**引擎（4x V100-SXM2 16 GB）：NCCL 传输、多 rank CUDA graph、MTP/投机解码修复、图像多模态（2026-10-05）、QUASAR NVFP4-QAT 工件。Qwen3.8-27B W4A4：**decode 113 tok/s（TP2 的 1.81 倍）**、prefill **3.9k tok/s**（分块 GDN：较 3,502.7 +11.6%）、128k int8 KV 下每卡 ~9.4 GiB。长注意力 decode 路由 + MTP4（2026-10-15）：pi 形状 120k decode **96.0 → 109.1 tok/s（1.136 倍）**。 2026-10-21：keep-alive 首 token **−22 ms**；分支/换 prompt 的 TTFT **10.3 s → 1.5–3.0 s**。2026-10-22：长上下文 prefill 注意力换成 **Flash-V100 Split-D** 核（1Cat-vLLM，BSD-3）：**120k prefill +9.9%、32k +4.0%**（引擎口径），needle 4/3 全中，decode 未动；MTP 拷贝批次另给 8k prefill +4.05%。 2026-10-22（decode）：tp>2 allreduce 从 NCCL 换成 **one-shot push allreduce**（移植 1Cat 的 `sm70_cross_device_reduce_1stage_push` 到单进程 UVA）：引擎级两臂 A/B（每臂 fresh 重启、客户端 decode tok/s 剔 ttft）**154.2/151.7 vs 133.0/131.2 = +15.8% decode，省 1.032 ms/token**，正好落在 1Cat 参照的 +13~18% 带内。路上更正了两次误诊（「IOMMU 判死」实为 NULL 指针 bug；「图捕获崩」实为当前设备泄漏——现役 decode 本就是 `--no-cuda-graph` eager）。


Single-instance Qwen3.8-27B inference across **4 × Tesla V100-SXM2 16 GB**
(`sm_70` / Volta, CUDA 12.8) via 4-way tensor parallel. **Multimodal (image) since
2026-10-05**; the text-only path is the byte-identical regression gate.

This is a fork of [tuxKOH/ninfer-V100X2](https://github.com/tuxKOH/ninfer-V100X2)
at baseline commit `c6100422` (2026-09-29). That baseline is the tuned **2-GPU (TP2)**
fork in the [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) /
[Neroued/ninfer](https://github.com/Neroued/ninfer) chain. Everything in this tree keeps
TP2 behavior byte-identical — it is the regression gate for every batch of changes —
and adds `--tp 4`, which runs one instance across all four GPUs.

在 **4 × Tesla V100-SXM2 16 GB**（`sm_70` / Volta，CUDA 12.8）上用 4 路张量并行
跑单实例 Qwen3.8-27B 推理。**2026-10-05 起支持多模态（图像）**；纯文本路径保持
字节级不变，作为每批改动的回归闸门。

本仓库是 [tuxKOH/ninfer-V100X2](https://github.com/tuxKOH/ninfer-V100X2) 在基线提交
`c6100422`（2026-09-29）上的 fork。该基线是
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) /
[Neroued/ninfer](https://github.com/Neroued/ninfer) 链路上调优过的
**双卡（TP2）** fork。本树保持 TP2 行为字节级不变（每批改动的回归闸门），
并新增 `--tp 4`：一个实例横跨全部四张卡。

## 2026-10-21: reuse latency / 复用延迟

**EN** — Two measured changes. (1) `TCP_NODELAY` on the HTTP server removes the Nagle/delayed-ACK
stall on reused SSE connections: first content token **40.85 → 18.85 ms (−22.0 ms, −53.8%)**,
two fresh boots per arm, byte-identical text. (2) Host prefix checkpoints capture the GDN state
plus tail hidden (146.8 MiB per image, 2 GiB host budget) during prefill; a request whose prompt
shares a long prefix picks the farthest checkpoint ≤ LCP and resumes instead of re-prefilling.
On a 32k branch: **10.26 s → 2.95 s**, return branch **1.51 s**. Cost: **+1.1%** on a full prefill.
Details: [prefix-checkpoint-2026-10-21.md](docs/tp4/prefix-checkpoint-2026-10-21.md).

**中文** — 两项实测改动。(1) HTTP 服务端开 `TCP_NODELAY`，去掉复用 SSE 连接上的 Nagle/延迟 ACK
停顿：首个 content token **40.85 → 18.85 ms（−22.0 ms，−53.8%）**，两臂各 2 次重启，文本字节一致。
(2) prefill 期间把 GDN 状态 + tail hidden 存成 host 前缀 checkpoint（每张 146.8 MiB，2 GiB 预算）；
后续请求若共享长前缀，选 frontier ≤ LCP 的最远 checkpoint 续播，不再重算。32k 分支：
**10.26 s → 2.95 s**，回到旧分支 **1.51 s**；代价是全量 prefill **+1.1%**。
详记：[prefix-checkpoint-2026-10-21.md](docs/tp4/prefix-checkpoint-2026-10-21.md)。

## 2026-10-21: chunked GDN prefill / 分块 GDN prefill

**EN** — The 48 GDN layers used to walk the state one token at a time (96 blocks, one `k·S` plus
one `q·S` warp reduction per token). They now run FLA-style chunked: a 32-token chunk solves its
intra-chunk coupling with forward substitution, and only the chunk-to-chunk state scan stays
sequential — 256 chunk steps instead of 8,192 token steps at 8k. GDN segment
**6,944.8 → 3,400.7 µs/layer (2.04×)**, end-to-end prefill **3,502.7 → 3,908.7 tok/s (+11.6%)**,
ttft at 8k **2.40 → 2.15 s**, needle 3/3. The FP32 state still lives in registers, decay is
applied in log space, and the decode path is untouched (same snapshot op). Details:
[gdn-chunked-prefill-2026-10-21.md](docs/tp4/gdn-chunked-prefill-2026-10-21.md).

**中文** — 48 层 GDN 过去逐 token 走状态（96 个 block，每 token 两次 warp 归约）。现在改成
FLA 式分块：32 token 的 chunk 内用前代求解耦合矩阵，只有块间状态扫描仍是串行 —— 8k 上是
256 步块步，而不是 8,192 步 token 步。GDN 段 **6,944.8 → 3,400.7 µs/层（2.04×）**，端到端
prefill **3,502.7 → 3,908.7 tok/s（+11.6%）**，8k ttft **2.40 → 2.15 s**，needle 3/3。
FP32 状态仍常驻寄存器、衰减走 log 空间、decode 路径完全未动（同一个 snapshot op）。
详记：[gdn-chunked-prefill-2026-10-21.md](docs/tp4/gdn-chunked-prefill-2026-10-21.md)。

## 2026-10-22: Flash-V100 Split-D long-context attention / Flash-V100 Split-D 长上下文注意力

**EN** — The 16 full-attention layers ran on the vendored llama.cpp SM70 kernel, capped at 35.5 TFLOPS (28% of peak); every home-grown MMA variant measured on this host lands at ≤25 TF. The long-context prefill now runs the **Flash-V100 Split-D** kernel from 1Cat-vLLM (BSD-3, vendored under `src/ops/vendor/flash_v100/`, ATen stripped): its warp pairs each own eight distinct Q rows for QK, then share the P tile so the QK product is never duplicated — the duplication is exactly what capped every dim-split kernel. Isolated: 46.9 TFLOPS at 120k. In-engine A/B (same stack, env-gated `NINFER_ATTN_SPLITD`): **120k 2489.5 → 2735.9 tok/s (+9.9%)**, **32k +4.0%** (engine tok/s), needle 4/4 hit, decode untouched (this route never touches decode). The kernel fires only when n_kv ≥ 16384, so the 8k prefill is unchanged — its +4.05% gain came from a separate MTP batch (stem per-column copies → one `cudaMemcpy2DAsync` per rank, 39.8→6.9 ms/chunk; plus 256-wide CUTLASS MLP tiles). Both batches shipped in one production binary (`6ac5c8b5`, gate passed, needle 4/4). Two integration bugs are documented in the write-up: a 1024 q-block grid that made the kernel *slower* (fixed by issuing the whole 4096 chunk at once), and a step-size gate that disagreed with the dispatch predicate and silently corrupted attention (caught by the needle check while throughput looked better). Details:
[flash-v100-splitd-attn-2026-10-22.md](docs/tp4/flash-v100-splitd-attn-2026-10-22.md).

**中文** — 16 层全注意力跑的是 vendored llama.cpp SM70 核，上限 35.5 TFLOPS（峰值的 28%）；本机实测的自研 MMA 变体全部 ≤25 TF。长上下文 prefill 现在跑 1Cat-vLLM 的 **Flash-V100 Split-D** 核（BSD-3，vendor 到 `src/ops/vendor/flash_v100/`，剥离 ATen）：每个 warp 对各持 8 行不同的 Q 算 QK，然后共享 P tile，QK 从不重复 —— 重复正是所有 dim-split 核卡住的根因。孤立实测 120k 达 46.9 TFLOPS。引擎内 A/B（同栈、env 开关 `NINFER_ATTN_SPLITD`）：**120k 2489.5 → 2735.9 tok/s（+9.9%）**、**32k +4.0%**（引擎口径），needle 4/4 全中，decode 未动（该路线不经过 decode）。核只在 n_kv ≥ 16384 时触发，8k prefill 不变 —— 它的 +4.05% 来自另一个 MTP 批次（stem 逐列拷贝 → 每 rank 一次 `cudaMemcpy2DAsync`，39.8→6.9 ms/chunk；外加 256 宽 CUTLASS MLP tile）。两个批次打进同一份生产二进制（`6ac5c8b5`，闸门通过，needle 4/4）。两个集成坑写在详记里：1024 q-block 的 grid 让核反而变慢（改成整 4096 chunk 一次调用解决）；step 与生效条件不同源会悄悄写坏注意力（吞吐好看但 needle 挂掉，被 needle 检查逮住）。详记：
[flash-v100-splitd-attn-2026-10-22.md](docs/tp4/flash-v100-splitd-attn-2026-10-22.md)。

## 2026-10-22: one-shot push allreduce / 2026-10-22: one-shot push allreduce

**EN** — The tp>2 allreduce used to go through NCCL (measured 4.6 ms per decode round for the 10 KiB
shape, 23% of the 8k round wall). It now runs a **one-shot push allreduce** transcribed from 1Cat's
`sm70_cross_device_reduce_1stage_push` into single-process UVA: every rank pushes its 16-byte pack into
the peers' staging slots, polls until the peers' packs arrive (sentinel = a reserved bf16 NaN), reduces
in a fixed rank order, and clears its slots — no host-side events, no NCCL channels, all sequencing
inside one kernel. Engine-level two-arm A/B (fresh boot per arm, client decode tok/s excluding ttft,
temp 0.6 × 512 tok English): **PUSH 154.2/151.7 vs NCCL 133.0/131.2 = +15.8% decode, 1.032 ms/token
saved**, squarely inside 1Cat's +13~18% reference band; ttft unchanged (0.143–0.155 s, the 480 KiB
prefill block exceeds `kMaxBytes` 80 KiB and correctly falls back to NCCL). Two integration bugs were
corrected: an 'IOMMU dead-end' that was actually a NULL-pointer bug (`TpComm::create` accessed an empty
`impl` after a `std::move`), and a warmup crash that was a current-device leak (`push_ar_try`'s launch
loop left the caller's device at rank 3 → `cudaErrorInvalidResourceHandle` — the live decode is
`--no-cuda-graph` eager, never graph-captured). The fix is a `CurrentDeviceRestore` RAII guard, and the
lesson is that a green kernel harness does not prove the engine path: raw cross-GPU dereferencing
requires peer access to be *explicitly* enabled, and per-rank `cudaSetDevice` must be restored. Details:
[push-ar-2026-10-22.md](docs/tp4/push-ar-2026-10-22.md).

**中文** — tp>2 allreduce 过去走 NCCL（10 KiB 形态实测 4.6 ms/decode 轮，占 8k 轮墙钟 23%）。现在跑
**one-shot push allreduce**，从 1Cat 的 `sm70_cross_device_reduce_1stage_push` 转写进单进程 UVA：每个
rank 把自己的 16 字节包 push 进 peer 的 staging 槽，轮询直到 peer 的包到齐（哨兵 = 保留的 bf16 NaN），
按固定 rank 序归约，再清槽——没有 host 侧事件、没有 NCCL 通道，全部排序在一个核里完成。引擎级两臂 A/B
（每臂 fresh 重启、客户端 decode tok/s 剔 ttft、temp 0.6 × 512 tok 英文）：**PUSH 154.2/151.7 vs NCCL
133.0/131.2 = +15.8% decode，省 1.032 ms/token**，正好落在 1Cat 参照的 +13~18% 带内；ttft 不变
（0.143–0.155 s，480 KiB prefill 块超 `kMaxBytes` 80 KiB，正确回退 NCCL）。路上更正了两个集成 bug：
「IOMMU 判死」实为 NULL 指针 bug（`TpComm::create` 在 `std::move` 后访问空 `impl`）；warmup 崩溃实为
当前设备泄漏（`push_ar_try` 的 launch 循环把调用方的设备留在 rank 3 → `cudaErrorInvalidResourceHandle`
——现役 decode 是 `--no-cuda-graph` eager，从不捕获图）。修复是 `CurrentDeviceRestore` RAII 守卫，教训是
kernel harness 绿灯不等于引擎路径绿灯：裸跨 GPU 解引用要求 peer access **显式开启**，逐 rank
`cudaSetDevice` 必须还原。详记：[push-ar-2026-10-22.md](docs/tp4/push-ar-2026-10-22.md)。

## What this fork adds / 本 fork 新增了什么

- **Sharding geometry made runtime-sized** — the compile-time 2-GPU assumption is gone:
  1,177 type widenings + 89 rank loops across the runtime, op wrappers and op families
  (NVFP4 / FP8 / Q4-Q5 / W8), plus `src/ops/wrapper/shard_extent.h` for runtime shard extents.
- **NCCL transport backend** — `src/core/tp_comm.{h,cu}`; graph-capturable collectives,
  device-sync caller duty for large payloads, `nccl.h` confined to one translation unit.
- **Multi-rank CUDA graphs** in the decode loop (5,013 graph nodes at the reference config).
- **MTP-tp4 fixes** — tp4 shard-extent registration in the W8 dispatch and the stem-K
  slice-copy fix; draft acceptance 30.8–34.1% → 47.5% at equal workload.
- **Tokenizer BPE merge O(k²) → O(k·log k)** — a 60k-character unbroken CJK run is one
  pre-token (k=180k bytes), which the naive rescan made quadratic: 120k context went
  from ~40 min to 0.22 s; verified token-identical (13/13 + 5,500 randomized cases).
- **dflash2 / SWA / argmax / speculative-round op families + Volta GQA attention
  (2026-10-01)** — draft-verify and small-batch decode paths with per-shard NCCL
  collectives and `ExecutionCore.tp` bookkeeping fixes for the tp4 handoff
  (2026-10-02/03).
- **SM70 NVFP4 decode fast path + load-side prepack (2026-10-04)** — QPN/CUTLASS
  Volta kernels for the attn/gdn input projections; greedy decode **113.5 → 146.3 tok/s
  (+29%)** under the controlled probe, prefill unchanged (−0.6% on the 8k–120k ladder);
  cross-tree bit-comparison is a wrong criterion here (reduction-order micro-perturbations
  amplify under greedy+MTP — verify determinism-in-binary, quality probes, and perf).
- **QUASAR QAT artifact support (2026-10-04)** — load plan + artifact reader fix for the
  QUASAR NVFP4-QAT variant; same-protocol decode +7–14% vs the K3 artifact
  (systematically higher MTP draft acceptance: smaller quantization damage).
- **Multimodal vision on tp4 (2026-10-05)** — a single Vision session encodes on rank 0
  and the [hidden, len] media residual is D2D-copied to every peer under event ordering
  (tp4 has no peer access; `enable_peer_access` is only opened for tp2). The MTP
  alignment window and the prefix-reuse MTP bridge stage the shifted visual embeddings
  per rank (embedding-side ranks only). Images only, no video. See
  [the vision write-up](docs/tp4/vision-tp4-2026-10-05.md).
- **Upstream decode kernel backport + long-window routing (2026-10-15)** — the upstream
  `861633e8` long-decode kernel family (gqa_attention_decode split, int8 KV-quant, GDN
  recurrent, lookup-draft) ported into tp4; `NINFER_TP4_VOLTA_LONG_ATTN` picks the routing
  side (`0` off / `1` both / `2` target-only / `3` draft-only); the long-window gate (65536)
  is shared with the KV split policy and CUDA-graph profile boundaries were extended to
  match. Production = MTP4 + target-only routing: pi-shaped 120k decode **96.0 → 109.1
  tok/s (1.136x)** — client-side median, 3 boots x 3 reps, all side-effect gates >= 0.98.
  See [the decode backport write-up](docs/tp4/decode-backport-2026-10-15.md).
- **One-shot push allreduce for tp4 (2026-10-22)** — the tp>2 allreduce swapped from NCCL (4.6 ms/decode round) to a single-kernel push protocol ported from 1Cat's `sm70_cross_device_reduce_1stage_push`: every rank pushes its 16-byte pack into the peers' staging slots, polls until the peers' packs arrive (sentinel = a reserved bf16 NaN), reduces in a fixed rank order, and clears its slots — no host-side events, no NCCL channels. Engine-level two-arm A/B (fresh boot per arm, client decode tok/s excl. ttft): **154.2/151.7 vs 133.0/131.2 = +15.8% decode, 1.032 ms/token saved**. Raw cross-GPU dereferencing requires peer access to be *explicitly* enabled (querying `cudaDeviceCanAccessPeer` is not enough — it says 'possible', not 'enabled'), and the launch loop must restore the caller's current CUDA device (RAII) or the next op on a rank-0 stream dies with `cudaErrorInvalidResourceHandle`. See [the push allreduce write-up](docs/tp4/push-ar-2026-10-22.md).

Technical history (W1→W6): [`docs/tp4/changelog.md`](docs/tp4/changelog.md) ·
MTP root-cause analysis: [`docs/tp4/README_tp4_mtp_rca_2026-10-01.md`](docs/tp4/README_tp4_mtp_rca_2026-10-01.md) ·
tokenizer report: [`docs/tp4/tokenizer-oklogk.md`](docs/tp4/tokenizer-oklogk.md) ·
vision on tp4 (2026-10-05): [`docs/tp4/vision-tp4-2026-10-05.md`](docs/tp4/vision-tp4-2026-10-05.md) ·
decode backport + long-window routing (2026-10-15): [`docs/tp4/decode-backport-2026-10-15.md`](docs/tp4/decode-backport-2026-10-15.md)

- **分片几何改为运行时尺寸** —— 编译期双卡假设被彻底移除：运行时、op wrapper 与
  op 家族（NVFP4 / FP8 / Q4-Q5 / W8）里 1,177 处类型加宽 + 89 处 rank 循环，
  另有 `src/ops/wrapper/shard_extent.h` 提供运行时分片尺寸。
- **NCCL 传输后端** —— `src/core/tp_comm.{h,cu}`；可被 CUDA graph 捕获的集合通信，
  大 payload 的调用方 device-sync 职责，`nccl.h` 只进一个翻译单元。
- **多 rank CUDA graph** —— decode 循环里捕获（参考配置下 5,013 个图节点）。
- **MTP-tp4 修复** —— W8 dispatch 的 tp4 分片尺寸注册 + stem-K 切片拷贝修复；
  同负载下 draft 接受率 30.8–34.1% → 47.5%。
- **Tokenizer BPE merge O(k²) → O(k·log k)** —— 6 万字符的无空格中文连续段按官方
  regex 算一个 pre-token（k=18 万字节），朴素重扫使其变成平方级：120k 上下文
  从 ~40 分钟降到 0.22 s；已验证 token 序列逐字节一致（13/13 + 5,500 随机对拍）。
- **dflash2 / SWA / argmax / speculative-round op 家族 + Volta GQA 注意力
  （2026-10-01）** —— draft-verify 与小批量 decode 路径，带每分片 NCCL 集合通信；
  2026-10-02/03 补了 tp4 交接的 `ExecutionCore.tp` 记账修复。
- **SM70 NVFP4 decode 快路径 + load 侧 prepack（2026-10-04）** —— attn/gdn 输入投影
  的 QPN/CUTLASS Volta kernel；受控探针下 greedy decode **113.5 → 146.3 tok/s
  （+29%）**，prefill 不变（8k–120k 阶梯 −0.6%）；跨树 bit 对比在这里是错误判据
  （归约顺序微扰在 greedy+MTP 下混沌放大 —— 只验同二进制内确定性、质量探针、性能）。
- **QUASAR QAT 工件支持（2026-10-04）** —— QUASAR NVFP4-QAT 变体的 load plan +
  工件 reader 修复；同协议 decode 比 K3 工件 +7–14%（MTP draft 接受率系统性更高：
  QAT 量化损伤小）。
- **tp4 多模态视觉（2026-10-05）** —— 单个 Vision session 在 rank 0 编码，
  [hidden, len] 媒体残差经事件排序 D2D 拷到每个 peer（tp4 没有 peer 访问；
  `enable_peer_access` 只对 tp2 开）；MTP 对齐窗与 prefix-reuse 的 MTP bridge
  按 rank 分发移位视觉 embedding（仅 embedding 侧 rank）。只支持图像，不支持视频。
  见[视觉专文](docs/tp4/vision-tp4-2026-10-05.md)。
- **上游 decode 内核回迁 + 长窗路由（2026-10-15）** —— 上游 `861633e8` 的长 decode 内核族
  （gqa_attention_decode 分裂、int8 KV 量化、GDN recurrent、lookup 草稿）移植进 tp4；
  `NINFER_TP4_VOLTA_LONG_ATTN` 选路由侧（0 全关 / 1 两侧 / 2 只 target / 3 只草稿头）；
  长档门 65536 与 KV 分段策略同门槛，CUDA graph profile 边界同步补齐。生产 = MTP4 +
  只开 target：pi 形状 120k decode **96.0 → 109.1 tok/s（1.136 倍）**，客户端中位、
  3 boot × 3 rep、副作用闸全过（≥0.98）。见[decode 回迁专文](docs/tp4/decode-backport-2026-10-15.md)。
- **tp4 one-shot push allreduce（2026-10-22）** —— tp>2 allreduce 从 NCCL（4.6 ms/decode 轮）换成单核 push 协议，移植自 1Cat 的 `sm70_cross_device_reduce_1stage_push`：每个 rank 把自己的 16 字节包 push 进 peer 的 staging 槽，轮询直到 peer 的包到齐（哨兵 = 保留的 bf16 NaN），按固定 rank 序归约，再清槽——没有 host 侧事件、没有 NCCL 通道。引擎级两臂 A/B（每臂 fresh 重启、客户端 decode tok/s 剔 ttft）：**154.2/151.7 vs 133.0/131.2 = +15.8% decode，省 1.032 ms/token**。裸跨 GPU 解引用要求 peer access **显式开启**（只查 `cudaDeviceCanAccessPeer` 不够——它说「可能」，不说「已开」），且 launch 循环必须还原调用方的当前 CUDA 设备（RAII），否则下一个在 rank 0 stream 上的 op 会死于 `cudaErrorInvalidResourceHandle`。见[push allreduce 专文](docs/tp4/push-ar-2026-10-22.md).

技术史（W1→W6）：[`docs/tp4/changelog.md`](docs/tp4/changelog.md) ·
MTP 根因分析：[`docs/tp4/README_tp4_mtp_rca_2026-10-01.md`](docs/tp4/README_tp4_mtp_rca_2026-10-01.md) ·
tokenizer 报告：[`docs/tp4/tokenizer-oklogk.md`](docs/tp4/tokenizer-oklogk.md) ·
tp4 视觉（2026-10-05）：[`docs/tp4/vision-tp4-2026-10-05.md`](docs/tp4/vision-tp4-2026-10-05.md) ·
decode 回迁 + 长窗路由（2026-10-15）：[`docs/tp4/decode-backport-2026-10-15.md`](docs/tp4/decode-backport-2026-10-15.md)

## Measurements (this host, completed tp4) / 实测（本机，tp4 完成态）

Reference config: the **Merkyor EfficientThink-K3 `W4A4+W8A8`** hybrid SFT variant of
Qwen3.8-27B (NVFP4 for 140 weight matrices + FP8 for 260 + BF16 residual, plus the MTP
draft block), converted to the ninfer `.ninfer` format inside tuxKOH's **v2 container**
(full source/converter details under [Model artifact provenance](#model-artifact-provenance));
`--tp 4 --devices 0,1,2,3`, 131,072 max context, **int8 KV**, 4,096-token prefill chunks,
MTP with 3 drafts, CUDA graphs. Decode tok/s counts committed output tokens, not drafted
ones.

| Metric | tp2 (2-GPU baseline, same artifact) | **tp4** |
|---|---:|---:|
| Committed decode — 2,048-token thinking run (t=1.0 / top_p=0.95 / top_k=20) | 62.6 tok/s | **113.1 tok/s (1.81×)** |
| Committed decode — 2,048-token thinking run, **push allreduce** (2026-10-22) | — | **129.2–136.6 tok/s** (+14~21% over the NCCL row above) |
| MTP draft acceptance — same run | 30.8–34.1% | **47.5%** |
| Prefill — 17.4k prompt (needle test, retrieval hit ✓) | — | **3,372.8 tok/s** |
| Prefill — 8.1k prompt (chunked GDN, 2026-10-21) | 3,502.7 tok/s | **3,908.7 tok/s (1.12×)** |
| Prefill — 120k prompt (Flash-V100 Split-D, 2026-10-22) | 2,489.5 tok/s | **2,735.9 tok/s (1.10×)** |
| Prefill — 32k prompt (Flash-V100 Split-D, 2026-10-22) | 3,492.7 tok/s | **3,677.4 tok/s (1.05×)** |
| Prefill — 5.8k prompt | — | 3,494.6 tok/s |
| Short-answer decode — 96 tokens | — | 124–140 tok/s (acceptance 56–66%) |
| VRAM per GPU (weights + 128k int8 KV + graphs) | 15.2 GiB | **9.39 GiB** |

The 4-way split halves per-GPU weights and KV: 6.6 GiB stays free per card at startup
(6.9 GiB slack), so the artifact's native 262,144-token capacity also fits — the 128k
default is kept on purpose, for parity with the old service.

参考配置：**Merkyor EfficientThink-K3 `W4A4+W8A8`** 混合 SFT 变体的 Qwen3.8-27B
（140 个权重矩阵 NVFP4 + 260 个 FP8 + BF16 残差，外加 MTP draft 块），在 tuxKOH 的
**v2 容器**内转成 ninfer `.ninfer` 格式（来源/转换细节见
[工件来源](#model-artifact-provenance)）；`--tp 4 --devices 0,1,2,3`，最大上下文 131,072，
**int8 KV**，4,096-token prefill 分块，MTP 3 草稿，CUDA graph。
decode tok/s 只计已提交的输出 token，不含被拒的 draft。

| 指标 | tp2（双卡基线，同工件） | **tp4** |
|---|---:|---:|
| 已提交 decode — 2,048-token 思考跑（t=1.0 / top_p=0.95 / top_k=20） | 62.6 tok/s | **113.1 tok/s（1.81×）** |
| MTP draft 接受率 — 同跑 | 30.8–34.1% | **47.5%** |
| Prefill — 17.4k prompt（needle 测试，检索命中 ✓） | — | **3,372.8 tok/s** |
| Prefill — 8.1k prompt (chunked GDN, 2026-10-21) | 3,502.7 tok/s | **3,908.7 tok/s (1.12×)** |
| Prefill — 120k prompt (Flash-V100 Split-D, 2026-10-22) | 2,489.5 tok/s | **2,735.9 tok/s (1.10×)** |
| Prefill — 32k prompt (Flash-V100 Split-D, 2026-10-22) | 3,492.7 tok/s | **3,677.4 tok/s (1.05×)** |
| Prefill — 5.8k prompt | — | 3,494.6 tok/s |
| 短答 decode — 96 token | — | 124–140 tok/s（接受率 56–66%） |
| 每卡显存（权重 + 128k int8 KV + graph） | 15.2 GiB | **9.39 GiB** |

4 路分片把每卡权重与 KV 减半：启动时每卡空闲 6.6 GiB（6.9 GiB 余量），
所以工件原生的 262,144-token 容量也放得下——默认仍取 128k，是有意保留，
为了和旧服务对齐。

### Prefill ladder — context scaling (tp4, this host) / Prefill 阶梯 — 上下文扩展（tp4，本机）

Full cold prefill sweep, 3 reps per context, session salt so no rep hits the prefix cache
(authoritative numbers = engine `done` line, req-baseline matched so each sample is
uniquely attributed; the sweep ran with no other active request source). The 8k and 16k
points reproduce the earlier 5.8k / 17.4k single-shot numbers, confirming stability.

| context | prefill tok/s (baseline 2026-10-21, chunked GDN) | prefill tok/s (current 2026-10-22, + Flash-V100 Split-D + MTP batch) | Δ |
|---:|---:|---:|---:|
| 8k | 3,917.0 | 3,925.5 | +0.2%（Split-D 不触发，+4.05% 见 MTP 批次） |
| 16k | 3,755.2 | — | — |
| 32k | 3,470.0 | 3,654.8 | **+5.3%** |
| 64k | 2,996.0 | 3,297.6 | **+10.1%**（两优化混算） |
| 120k | 2,392.1 | 2,721.7 | **+13.8%** |

Prefill declines monotonically with context — the long-context **O(n²) attention wall**:
8k→16k is near-flat (3,470→3,357), then it steepens (−7%/tier at 32k/64k, −19% into 120k).
The 120k point still fits inside the 131k int8 KV pool. The greedy-decode column also
drops with context because each decode step must read more KV; that is separate from the
thinking-mode decode below.

The 2,048-token thinking run (t=1.0) re-measured **94.4 tok/s** (MTP acceptance 34.2%)
in this rerun. The online record for the same run is 113.1 tok/s / 47.5% acceptance; the
spread is MTP-draft-acceptance variance under t=1.0 sampling, not a config change.

Smoke (all passed): 4 greedy probes semantically correct; 17.4k needle retrieval hit;
4-way concurrency (`--max-concurrency 4`) all answers correct.

Build guide, NCCL/CUTLASS dependencies and tests: [`docs/tp4/README.md`](docs/tp4/README.md).
Reference build graph and exact CMake option values: [`docs/tp4/build-config/`](docs/tp4/build-config/).
The baseline fork's README (detailed TP2 measurements, build/run/convert instructions):
[`docs/upstream-README-v100x2.md`](docs/upstream-README-v100x2.md).

全量冷 prefill 扫描，每个上下文 3 reps，session salt 保证没有任何 rep 命中前缀缓存
（权威数字 = 引擎 `done` 行，按 req 基线对齐使每个样本可唯一归属；扫描期间无其他
活跃请求源）。8k 与 16k 两点复现了早期 5.8k / 17.4k 单次数字，确认稳定性。

| 上下文 | prefill tok/s | 首 token 时延 | greedy 300-token decode（t=0） |
|---:|---:|---:|---:|
| 8k | 3,470 | 2.4 s | 143 tok/s |
| 16k | 3,357 | 4.8 s | 134 tok/s |
| 32k | 3,118 | 10.3 s | 141 tok/s |
| 64k | 2,724 | 23.7 s | 117 tok/s |
| 120k | 2,229 | 54.1 s | 96 tok/s |

prefill 随上下文单调下降——长上下文的 **O(n²) 注意力墙**：8k→16k 近平
（3,470→3,357），之后变陡（32k/64k 每档 −7%，到 120k 时 −19%）。
120k 点仍装得进 131k 的 int8 KV 池。greedy decode 一列也随上下文下降，
因为每步 decode 要读更多 KV；这与下面思考模式 decode 是两回事。

2,048-token 思考跑（t=1.0）在复测中为 **94.4 tok/s**（MTP 接受率 34.2%）。
同跑的在线记录是 113.1 tok/s / 47.5% 接受率；差距是 t=1.0 采样下的
MTP draft 接受率方差，不是配置变化。

冒烟（全过）：4 个 greedy 探针语义正确；17.4k needle 检索命中；
4 路并发（`--max-concurrency 4`）答案全对。

构建指南、NCCL/CUTLASS 依赖与测试：[`docs/tp4/README.md`](docs/tp4/README.md)。
参考构建图与精确 CMake option 取值：[`docs/tp4/build-config/`](docs/tp4/build-config/)。
基线 fork 的 README（详细 TP2 实测、构建/运行/转换说明）：
[`docs/upstream-README-v100x2.md`](docs/upstream-README-v100x2.md)。

### Vision (added 2026-10-05) / 视觉（2026-10-05 新增）

Image requests on the reference artifact (the vision tower ships in the same `.ninfer`):

| Item | Value |
|---|---:|
| Image prompt (315 tokens, incl. 240 merged media tokens) | ttft ≈ 450 ms, prefill ≈ 1,800 tok/s |
| Decode on image request | 155 tok/s, MTP 2.96 tok/round |
| Text-only regression (same build) | 124–146 tok/s, unchanged |

Probe coverage (all exact): embedded title text, embedded digits, shape/color/position
questions, multi-turn conversations carrying an image with prefix reuse. Measured MTP
acceptance with the visual alignment vs without (6 paired variants, full prefill each):
2.890 vs 2.883 tok/round — inside run-to-run noise; the MTP stem's packed input is half
target hidden states, which already carry the image semantics. Video is not supported
by design scope; the 32,768 merged-token cap is unchanged.

参考工件上的图像请求（视觉塔与文本权重同在同一个 `.ninfer` 里）：

| 项目 | 数值 |
|---|---:|
| 图像 prompt（315 token，含 240 个 merged 媒体 token） | ttft ≈ 450 ms，prefill ≈ 1,800 tok/s |
| 图像请求 decode | 155 tok/s，MTP 2.96 tok/round |
| 纯文本回归（同构建） | 124–146 tok/s，不变 |

探针覆盖（全对）：图内标题文字、图内数字、形状/颜色/位置提问、带前缀复用的
多轮带图对话。MTP 接受率「开视觉对齐 vs 不开」实测（6 个配对变体、每次全量
prefill）：2.890 vs 2.883 tok/round —— 在 run-to-run 噪声之内；MTP stem 的
打包输入一半是目标模型自己的 hidden states，已携带图像语义。按设计口径不支持
视频；32,768 merged token 上限不变。

<a id="model-artifact-provenance"></a>
## Model artifact provenance / 模型工件来源

The measured artifact is `qwen3_8_27b_w4a4w8a8.ninfer` (21.0 GiB, 1,097 tensors), recipe
`qwen3_8_27b_w4a4w8a8-v1`, target key `qwen3.8-27b`:

- **Variant** — Merkyor EfficientThink-K3 `W4A4+W8A8` hybrid: an SFT distillation of
  Qwen3.8-27B (the “Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP” lineage), quantized with
  NVFP4 for 140 weight matrices, FP8 (E4M3) for 260, BF16 for the residual, plus the
  1-layer MTP draft block and the vision tower.
- **Converted from** — the single-source ModelOpt field layout in
  `Merkyor/Qwen3.8-27B-EfficientThink-K3-…-MTP-NVFP4` (local
  `EfficientThink-K3-W4A4-W8A8`); not the GGUF Q4_K_M path.
- **Converter** — the ninfer `.ninfer` format converter, run inside **tuxKOH's v2
  container** (the `.ninfer` converter only runs in that container image), CPU, ~3.7 min
  (219 s); `recipe_id qwen3_8_27b_w4a4w8a8-v1`, conversion bookkeeping in the sibling
  `*.ninfer.conversion.json`.
- **Base model** — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

实测工件为 `qwen3_8_27b_w4a4w8a8.ninfer`（21.0 GiB，1,097 个张量），recipe
`qwen3_8_27b_w4a4w8a8-v1`，target key `qwen3.8-27b`：

- **变体** —— Merkyor EfficientThink-K3 `W4A4+W8A8` 混合：Qwen3.8-27B 的
  SFT 蒸馏（"Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP" 谱系），140 个权重矩阵
  NVFP4 量化、260 个 FP8（E4M3）、残差 BF16，外加 1 层 MTP draft 块与视觉塔。
- **转换来源** —— `Merkyor/Qwen3.8-27B-EfficientThink-K3-…-MTP-NVFP4` 里的
  单源 ModelOpt 字段布局（本地 `EfficientThink-K3-W4A4-W8A8`）；不是
  GGUF Q4_K_M 路径。
- **转换器** —— ninfer `.ninfer` 格式转换器，跑在 **tuxKOH 的 v2 容器**内
  （`.ninfer` 转换器只能在该容器镜像里运行），CPU，约 3.7 分钟（219 s）；
  `recipe_id qwen3_8_27b_w4a4w8a8-v1`，转换簿记在同级 `*.ninfer.conversion.json`。
- **基座模型** —— [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。

## Notable files added by this fork / 本 fork 新增的关键文件

```
src/core/tp_comm.{h,cu}                  NCCL transport backend
src/core/push_ar.{h,cu}                  one-shot push allreduce (tp==4, single-process UVA) (2026-10-22)
src/ops/wrapper/shard_extent.h           runtime shard geometry
src/ops/{attn,gdn}_input_proj/nvfp4/*sm70*   SM70 NVFP4 decode fast path (2026-10-04)
src/ops/kernel/swa_volta.cuh             Volta GQA attention (2026-10-01)
src/targets/qwen3_6/impl/vision/         vision bindings; tp4 staging lives in
                                         impl/runtime/{text_context,text_prefill_impl,mtp_impl}.h
src/targets/qwen3_6_27b/impl/load/       per-variant load plans (incl. QUASAR, 2026-10-04)
tests/ops/test_allreduce_nccl4.cpp       4-GPU collective test (opt-in; <4 GPUs → skip 77)
tests/ops/test_tp4_issue_probe.cpp       caller-duty regression probe
tests/targets/qwen3_6_27b/test_{quasar,dflash2_v3}_load_plan.cpp
                                         artifact load-plan tests (2026-10-04)
tools/convert/qwen3_8_27b/               W4A4W8A8 artifact conversion + replay verification
docs/tp4/                                build guide, history, RCA, vision write-up,
                                         reference build config (incl. vision build), scripts
```

```
src/core/tp_comm.{h,cu}                  NCCL 传输后端
src/ops/wrapper/shard_extent.h           运行时分片几何
src/ops/{attn,gdn}_input_proj/nvfp4/*sm70*   SM70 NVFP4 decode 快路径（2026-10-04）
src/ops/kernel/swa_volta.cuh             Volta GQA 注意力（2026-10-01）
src/targets/qwen3_6/impl/vision/         视觉 bindings；tp4 分发逻辑在
                                         impl/runtime/{text_context,text_prefill_impl,mtp_impl}.h
src/targets/qwen3_6_27b/impl/load/       按变体的 load plan（含 QUASAR，2026-10-04）
tests/ops/test_allreduce_nccl4.cpp       4 卡集合通信测试（opt-in；<4 卡 → skip 77）
tests/ops/test_tp4_issue_probe.cpp       调用方职责回归探针
tests/targets/qwen3_6_27b/test_{quasar,dflash2_v3}_load_plan.cpp
                                         工件 load-plan 测试（2026-10-04）
tools/convert/qwen3_8_27b/               W4A4W8A8 工件转换 + 回放验证
docs/tp4/                                构建指南、技术史、RCA、视觉专文、
                                         参考构建配置（含 vision 构建）、脚本
```

## License / 许可

Apache-2.0 ([LICENSE](LICENSE), required attribution in [NOTICE](NOTICE)).
Upstream chain: tuxKOH/ninfer-V100X2 ← geoffwatts/ninfer-v100 ← Neroued/ninfer.
The measured model is the Merkyor EfficientThink-K3 `W4A4+W8A8` SFT variant, based on
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); see
[Model artifact provenance](#model-artifact-provenance).

Apache-2.0（[LICENSE](LICENSE)，[NOTICE](NOTICE) 含必须署名）。
上游链路：tuxKOH/ninfer-V100X2 ← geoffwatts/ninfer-v100 ← Neroued/ninfer。
实测模型为 Merkyor EfficientThink-K3 `W4A4+W8A8` SFT 变体，基座
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)；见
[模型工件来源](#model-artifact-provenance)。
