# NInfer V100 tp4 — 4 × V100（SM70）4 路张量并行

[English](README.md) · [构建指南](docs/tp4/README.md) · [技术史](docs/tp4/changelog.md)

## TL;DR / 一句话简介

**中文** — 把 ninfer 调优过的双卡（TP2）V100 移植改写为单实例 **4 路张量并行**引擎（4x V100-SXM2 16 GB）：NCCL 传输、多 rank CUDA graph、MTP/投机解码修复、图像多模态（2026-10-05）、QUASAR NVFP4-QAT 工件。Qwen3.8-27B W4A4：**decode 113 tok/s（TP2 的 1.81 倍）**、prefill **3.9k tok/s**（分块 GDN：较 3,502.7 +11.6%）、128k int8 KV 下每卡 ~9.4 GiB。
长注意力 decode 路由 + MTP4（2026-10-15）：pi 形状 120k decode **96.0 → 109.1 tok/s（1.136 倍）**。2026-10-21：复用连接首 token **−22 ms**；分支/换题 TTFT **10.3 s → 1.5–3.0 s**。 2026-10-22：长上下文 prefill 注意力换成 **Flash-V100 Split-D** 核（1Cat-vLLM，BSD-3）：**120k prefill +9.9%、32k +4.0%**（引擎口径），needle 4/3 全中，decode 未动；MTP 拷贝批次另给 8k prefill +4.05%。

**EN** — ninfer's tuned 2-GPU (TP2) V100 stack, re-hosted as a single-instance **4-way tensor-parallel** engine on 4x V100-SXM2 16 GB: NCCL transport, multi-rank CUDA graphs, MTP/speculative fixes, image multimodal (2026-10-05), QUASAR NVFP4-QAT artifacts. Qwen3.8-27B W4A4: **113 tok/s committed decode (1.81x vs TP2)**, prefill **3.9k tok/s** (chunked GDN: +11.6% over 3,502.7), ~9.4 GiB/GPU at 128k int8 KV. Long-attention decode routing + MTP4 (2026-10-15): pi-shaped 120k decode **96.0 → 109.1 tok/s (1.136x)**.


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

## 2026-10-21: 分块 GDN prefill

48 层 GDN 过去逐 token 走状态（96 个 block，每 token 一次 `k·S` + 一次 `q·S` 的 warp 归约）。
现在改成 FLA 式分块：32 token 的 chunk 内用前代求解耦合矩阵 `(I+L)^-1`，只有块间状态扫描
仍是串行 —— 8k 上是 256 步块步，而不是 8,192 步 token 步。

- GDN 段：**6,944.8 → 3,400.7 µs/层（2.04×）**（T=8192，TP4，冷 L2，CUDA Graph，40 samples 中位）
- 端到端 prefill（llm-ladder，thinking on）：**3,502.7 → 3,908.7 tok/s（+11.6%）**
- 8k ttft：**2.40 → 2.15 s**；16k：4.60 → 4.40 s；needle 全中
- decode 路径未动（同一个 snapshot op，字节级回归闸门）

实现要点：FP32 状态按 value 维切块常驻寄存器、喂 WMMA 时才转 fp16；衰减用 log 空间差值
`exp(gcum_t − gcum_s)`，不做小量除法；块内 Gram 与 qk 走 tensor core，前代求解在线程内完成；
`T/32*32 ≥ 128` 走分块路径，余下 tail 用顺序核续算（state 串接）。

详记（算法推导、选型扫描、两个真 bug、验收数据、复现命令）：
[gdn-chunked-prefill-2026-10-21.md](docs/tp4/gdn-chunked-prefill-2026-10-21.md)

## 2026-10-22: Flash-V100 Split-D 长上下文注意力

16 层全注意力跑的是 vendored llama.cpp SM70 核，上限 35.5 TFLOPS（峰值的 28%）；本机实测的自研 MMA 变体全部 ≤25 TF。
长上下文 prefill 现在跑 1Cat-vLLM 的 **Flash-V100 Split-D** 核（BSD-3，vendor 到 `src/ops/vendor/flash_v100/`，剥离 ATen）：
每个 warp 对各持 8 行不同的 Q 算 QK，然后共享 P tile，QK 从不重复 —— 重复正是所有 dim-split 核卡住的根因。
孤立实测 120k 达 46.9 TFLOPS。引擎内 A/B（同栈、env 开关 `NINFER_ATTN_SPLITD`）：**120k 2489.5 → 2735.9 tok/s（+9.9%）**、
**32k +4.0%**（引擎口径），needle 4/4 全中，decode 未动（该路线不经过 decode）。核只在 n_kv ≥ 16384 时触发，8k prefill 不变 ——
它的 +4.05% 来自另一个 MTP 批次（stem 逐列拷贝 → 每 rank 一次 `cudaMemcpy2DAsync`，39.8→6.9 ms/chunk；外加 256 宽 CUTLASS MLP tile）。
两个批次打进同一份生产二进制（`6ac5c8b5`，闸门通过，needle 4/4）。两个集成坑写在详记里：1024 q-block 的 grid 让核反而变慢
（改成整 4096 chunk 一次调用解决）；step 与生效条件不同源会悄悄写坏注意力（吞吐好看但 needle 挂掉，被 needle 检查逮住）。
详记：[flash-v100-splitd-attn-2026-10-22.md](docs/tp4/flash-v100-splitd-attn-2026-10-22.md)。

## 本 fork 新增了什么

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

技术史（W1→W6）：[`docs/tp4/changelog.md`](docs/tp4/changelog.md) ·
MTP 根因分析：[`docs/tp4/README_tp4_mtp_rca_2026-10-01.md`](docs/tp4/README_tp4_mtp_rca_2026-10-01.md) ·
tokenizer 报告：[`docs/tp4/tokenizer-oklogk.md`](docs/tp4/tokenizer-oklogk.md) ·
tp4 视觉（2026-10-05）：[`docs/tp4/vision-tp4-2026-10-05.md`](docs/tp4/vision-tp4-2026-10-05.md) ·
decode 回迁 + 长窗路由（2026-10-15）：[`docs/tp4/decode-backport-2026-10-15.md`](docs/tp4/decode-backport-2026-10-15.md)

## 实测（本机，tp4 完成态）

参考配置：**Merkyor EfficientThink-K3 `W4A4+W8A8`** 混合 SFT 变体的 Qwen3.8-27B
（140 个权重矩阵 NVFP4 + 260 个 FP8 + BF16 残差，外加 MTP draft 块），在 tuxKOH 的
**v2 容器**内转成 ninfer `.ninfer` 格式（来源/转换细节见
[工件来源](#模型工件来源)）；`--tp 4 --devices 0,1,2,3`，最大上下文 131,072，
**int8 KV**，4,096-token prefill 分块，MTP 3 草稿，CUDA graph。
decode tok/s 只计已提交的输出 token，不含被拒的 draft。

| 指标 | tp2（双卡基线，同工件） | **tp4** |
|---|---:|---:|
| 已提交 decode — 2,048-token 思考跑（t=1.0 / top_p=0.95 / top_k=20） | 62.6 tok/s | **113.1 tok/s（1.81×）** |
| MTP draft 接受率 — 同跑 | 30.8–34.1% | **47.5%** |
| Prefill — 17.4k prompt（needle 测试，检索命中 ✓） | — | **3,372.8 tok/s** |
| Prefill — 5.8k prompt | — | 3,494.6 tok/s |
| 短答 decode — 96 token | — | 124–140 tok/s（接受率 56–66%） |
| 每卡显存（权重 + 128k int8 KV + graph） | 15.2 GiB | **9.39 GiB** |

4 路分片把每卡权重与 KV 减半：启动时每卡空闲 6.6 GiB（6.9 GiB 余量），
所以工件原生的 262,144-token 容量也放得下——默认仍取 128k，是有意保留，
为了和旧服务对齐。

### Prefill 阶梯 — 上下文扩展（tp4，本机）

全量冷 prefill 扫描，每个上下文 3 reps，session salt 保证没有任何 rep 命中前缀缓存
（权威数字 = 引擎 `done` 行，按 req 基线对齐使每个样本可唯一归属；扫描期间无其他
活跃请求源）。8k 与 16k 两点复现了早期 5.8k / 17.4k 单次数字，确认稳定性。

| 上下文 | prefill tok/s（基线 2026-10-21，分块 GDN） | prefill tok/s（现值 2026-10-22，+ Flash-V100 Split-D + MTP 批次） | Δ |
|---:|---:|---:|---:|
| 8k | 3,917.0 | 3,925.5 | +0.2%（Split-D 不触发，+4.05% 见 MTP 批次） |
| 16k | 3,755.2 | — | — |
| 32k | 3,470.0 | 3,654.8 | **+5.3%** |
| 64k | 2,996.0 | 3,297.6 | **+10.1%**（两优化混算） |
| 120k | 2,392.1 | 2,721.7 | **+13.8%** |

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

### 视觉（2026-10-05 新增）

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

## 模型工件来源

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

## 本 fork 新增的关键文件

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

## 许可

Apache-2.0（[LICENSE](LICENSE)，[NOTICE](NOTICE) 含必须署名）。
上游链路：tuxKOH/ninfer-V100X2 ← geoffwatts/ninfer-v100 ← Neroued/ninfer。
实测模型为 Merkyor EfficientThink-K3 `W4A4+W8A8` SFT 变体，基座
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)；见
[模型工件来源](#模型工件来源)。