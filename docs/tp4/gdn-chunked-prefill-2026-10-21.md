# Chunked GDN prefill for V100 (2026-10-21)

一句话：把 GDN（gated delta net）线性注意力层的 prefill 从「每 token 一次顺序状态转移」
改成「分块并行 + 块内前代求解 + 块间状态扫描」，GDN 段 **2.04×**，整段 prefill **+11.6%**
（8k：3,502.7 → 3,908.7 tok/s），ttft 2.40 s → 2.15 s。decode 路径未动、零回归。

## 1. 为什么做

Qwen3.8 的 64 层里有 48 层是 GDN。原实现在 VOLTA（sm_70）上走 `recurrent_direct_kernel`：
12 个 value head × 8 个状态瓦片 = 96 个 block，每个 block 顺序走完 T 个 token，
每 token 一次 `k·S` + 一次 `o = q·S`，各要 5 步 warp 归约。

实测（`bench/ops/gdn_layer_bench.cu --mode prefill`，CUDA Graph + 256 MiB L2 flush，
5 warmup + 40 samples，中位）：

| T | 顺序核 µs/层 | 48 层 ms | 占整段 prefill |
|---|---|---|---|
| 2048 | 1751.0 | 84.0 | 14.0% |
| 4096 | 3480.6 | 167.1 | 13.9% |
| 8192 | 6944.8 | 333.3 | 13.9% |

分母用 llm-ladder 的客户端 ttft（同机同引擎，needle 全中）：8k = 2.40 s / 3,502.7 tok/s。
⇒ 段内就算翻倍，整体也只有 +7%；要 +10% 必须段内 3.5×。

顺序路线能挖的已经挖完（另一条提交：状态布局改成行×lane 组，归约从 5 步降到 2 步，
段内 +26.3%），再往上是算法天花板。

## 2. 算法

每 (value head, token)：
```
alpha = exp(g), beta = beta
delta[r] = beta * (v[r] - alpha * sum_c k[c] S[r][c])
S[r][c] = alpha * S[r][c] + delta[r] * k[c]
o[r]     = sum_c q[c] S[r][c]
```
chunk（宽 CS）内令 `gcum_t = sum_{m<=t} g_m`（log 空间累加，避免除小 P）：
```
L[t][s] = beta_t * exp(gcum_t - gcum_s) * (k_t . k_s)          (s < t)
kcd     = (I+L)^-1 (beta * exp(gcum) * k)                      前代求解
vcorr   = (I+L)^-1 (beta * v)
v_new   = vcorr - kcd @ H_in                                   H 是块起始状态 [dv][qk]
H_out   = exp(gcum_last) * H_in + Kch^T @ v_new,  Kch = k * exp(gcum_last - gcum_t)
o       = exp(gcum_t) * (q*scale) @ H_in + qk @ v_new,  qk[t][s] = scale*(q_t.k_s)*exp(gcum_t-gcum_s)
```
`H` 始终 FP32 常驻寄存器，喂 WMMA 时才转 fp16；`L` 的 Gram 与 `qk` 走 tensor core，
前代求解在线程内完成。

## 3. 参数与选型（全部实测）

形状探针 `bench/ops/gdn_chunked_shape_bench.cu`：单个 chunk-head 的四种 matmul 形状。
FP16 WMMA + FP32 累加、冷 L2、median：

| 实现 | ks 形状 (C×D · D×D) | ku 形状 (D×C · C×D) |
|---|---|---|
| 16×16 warp tile | 4.61 TFLOPS | 3.19 |
| 32×32 warp tile | 10.38 | 6.64 |
| 64×32 warp tile | **11.69** | 6.88 |
| smem 整块暂存（无流水线） | 5.13 | 7.47 |

⇒ 0.82 TFLOP 的预算在 11 TFLOPS 下是 1.56 ms/层 = 4.4×，说明天花板够；
瓶颈在装载与计算的重叠，不在算力。

最终选型：**CS=32、BV=16（每 block 16 个 dv 列）、NT=64、每 head 一块**。
扫描（T=8192，ms/层）：

| 配置 | 时间 |
|---|---|
| CS=16（llama.cpp 口径） | 8.37 |
| CS=32 / BV=32 / NT=256 | 3.42 |
| **CS=32 / BV=16 / NT=64** | **3.11** |
| CS=64 / BV=32 | 4.01 |
| CS=32 / BV=64 | 9.06 |
| NT=128 / 256 / 512（拆收缩维 + 确定性归约） | 3.28 / 3.87 / 3.54 |
| 收缩维拆 warp + smem **原子**归约 | 256 ms + NaN（generic 指针走全局原子，弃） |

最有效的四刀：① prepare 的前代加 `#pragma unroll`（`x[32]` 从栈溢出拉回寄存器，1.75→0.90 ms）；
② 去掉每 chunk 的 global→smem 暂存，WMMA 直接从 global 取片段（6.27→4.55）；
③ 融合 (v_new, o_inter) 与 (ΔH, o)：两对 matmul 共用同一 B 算子（4.16→3.42）；
④ prepare 的两处前代并行（一半线程算 kcd、一半算 vcorr，1.13→0.70）。

与 llama.cpp PR #26001（已合并的 chunked GDN CUDA kernel）的关系：借鉴了「状态按 v 维切块
常驻 FP32 寄存器 + fp16 镜像」「log 空间衰减」两条结构性经验，但**没有**沿用它的 CS=16 ——
16 token 的 chunk 意味着 8192 要 512 次串行块步，实测就是 8.37 ms 对 3.11 ms 的差距。

## 4. 接入 op（`src/ops/linear_attention/gated_delta_net/chunked_volta.{cu,cuh}`）

- `chunked_prepare_kernel`：每 (value head, chunk) 一块，256 线程 —— 可选 L2 归一化、
  `gcum` 前缀和、Gram→L（含掩码与衰减）、两处前代、qk、以及 fp16 的 kcd/qs/ks 输出。
- `chunked_state_kernel`：每 (value head, v tile) 一块，64 线程 —— 块间顺序扫描 + 输出。
- 分派：`gated_delta_net.cpp` 的 VOLTA 分支，`T/32*32 >= 128` 走 chunked；剩余 tail 用
  顺序核 `launch_recurrent_inout` 续算（state 串接）。`kChunkedEnabled` 是总闸。
- workspace：容量查询与执行必须**重放同样的 6 次分配**（op 测试要求查询峰值 == 执行峰值
  精确相等），因此 `chunked_volta_allocate` 是模板函数、两处共用；且按 `body = T/32*32`
  而不是 T 计算。

## 5. 两个真 bug（都不是算法）

1. **BF16/FP16 混用**：op 的 q/k/v/out 是 BF16，移植时按 FP16 读写 → fp16 位型写进 bf16 缓冲，
   解码≈0。症状极具误导性：「输出全零但 state 正常」（state 是 FP32，故不受影响）。
   定位方法：launcher 里直接读 `out[0]` 得 −0.0123、探针读**同一地址**得 −0.0000
   → 同地址不同解读 = dtype 错。
2. **GQA 映射**：op 的 `head_map::qk_head` 是**块状** `h_v / (H_v/H_qk)`，移植时写成了取模。
   只在 24/8、32/16 这类 group_size>1 的用例暴露（8/8 时两种写法等价，小用例看不出来）。

## 6. 验收

`ninfer_gated_delta_net_test`（FP64 oracle，覆盖 T=1/7/63/64/65/128/3072、GQA 24/8、
distinct-state、raw-qk、in-place、snapshot 全部用例）：**OK gated_delta_net correctness**。

段内（op 路径，1530 MHz，冷 L2）：

| T | 顺序核 µs/层 | chunked µs/层 | 加速 |
|---|---|---|---|
| 2048 | 1751.0 | 833.5 | 2.10× |
| 4096 | 3480.6 | 1643.5 | 2.12× |
| 8192 | 6944.8 | 3400.7 | 2.04× |

端到端（llm-ladder，thinking on）：

| 档 | 换前 tok/s | 换后 tok/s | 变化 | ttft |
|---|---|---|---|---|
| 2048 | 3364.3 | 3698.9 | +10.0% | 0.60 → 0.57 s |
| 8000 | 3502.7 | **3908.7**（3 reps） | **+11.6%** | **2.40 → 2.15 s** |
| 16000 | 3487.2 | 3684.7（2 reps） | +5.7% | 4.60 → 4.40 s |

needle 全部命中；decode（snapshot 路径）94.2/137.2/179.2/263.2/417.8 µs，与改前同区间。

## 7. 代价与限制

- workspace 中间缓冲（3×FP16 + FP32 vcorr + FP16 qk + gcum）：prefill-chunk=4096 时约 66 MB
  （8192 算是 132 MB、32768 是 530 MB）。现役配置显存 13.99/13.67 GiB，实占只多 7 MiB。
- 未达预注册的 3.5× 门（2.04×）。端到端反而超过原定 +10% 线，是因为引擎里 GDN 段的实际占比
  高于单层传感器估的 13.9%（还连带省掉 normalized-q/k 的预计算与读写）。
- 下一档（未做）：state kernel 仍是 global 片段装载延迟受限（有效 5.5 TFLOPS vs 形状天花板 11），
  需要双缓冲预取；或改用仿射扫描 `S' = (P_C I - C_c) S + B_c` 的树形并行，用 2× FLOPs 换并行度。

## 8. 复现

```bash
# 段内（op 路径）
ninja -C build-v100-tp4-vision ninfer-serve
./build-gdn-bench-*/bench/ninfer_gdn_layer_bench --mode prefill --geometry tp4 \
    --t-sweep 2048,4096,8192 --warmup 5 --repeat 40

# 数值
./build-gdn-bench-*/tests/ninfer_gated_delta_net_test

# 端到端（引擎在跑时）
python3 ~/.pi/agent/skills/llm-ladder/scripts/llm_ladder.py --backend auto --thinking on \
    --tiers 8000 --reps 3 --no-decode
```

探针与脚本：`bench/ops/gdn_chunked_{shape_bench,v3_probe,op_probe}.cu`（另有 v2/core 两个中间版本）、
`bench/ops/gdn_layer_bench.cu`（`--mode prefill` 段传感器）。
离线校验：纯 numpy 的代数 + FP16 精度模拟（`gdn_chunk_precision_sim.py`）、
对 FP64 逐 token 递推的校验器（`gdn_chunk_v2_validate.py`）。
