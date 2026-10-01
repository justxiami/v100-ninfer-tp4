# tp4 MTP：从「崩溃 + 乱码」到「正确但几乎不接受」的根因链（2026-10-01）

承接 `~/work/docs/ninfer_tp4_handoff_2026-10-01.md` §14.3 那三条悬案（MTP stem 的 fc 对拍、
MTP KV 池几何、proposal head gather、MtpRoundView 偏移）。本轮把 tp4 MTP 从
「warmup 随机崩 + 输出乱码 + 0% 接受」推进到：

- **不再崩**（warmup 与请求路径都稳）
- **输出正确**：4 条固定 prompt 里 3 条与 tp4 无投机**逐字节相同**，第 4 条是语义等价的
  另一种措辞（贪心下 batched verify 与单列 decode 的 GEMM 形状不同 → 近似并列翻转，属正常）
- **接受率仍低**：tp4 实测 0–12%，tp2 基线 84.8%（`mtp 3.55tok/round (84.8%)`，tp2 闸门实测）
  → 功能对、**没有加速**。护栏默认仍拒（`NINFER_ALLOW_TP4_MTP_UNVERIFIED=1` 可开）

## 一、修掉的两个真缺陷（都已落地）

### 1. stem 的 ids 跨 stream 竞态 → 间歇 `cudaErrorIllegalAddress`（已修）

**症状**：tp4 + MTP 在 load 期 warmup 就 `cudaErrorIllegalAddress`，**间歇**（6 次里挂 2 次），
偶尔又能起来；起来之后输出乱码。

**根因**：`mtp_forward_stem_tp2` 里 **每个 `rank < half` 的 embedding 侧 rank 都去读 rank 0 的
ids 缓冲**（`ops::embedding(flat_ids, ...)`，`flat_ids` 来自调用方传进来的**单个** Tensor）。
tp2 时 `half = tp/2 = 1`，只有 rank 0 读、且写它的是 rank 0 自己的 stream → 天然有序；
**tp4 时 `half = 2`，rank 1 也 embed → 跨 stream 读 rank 0 的缓冲，无任何排序** →
读到未初始化内容 → token id 是垃圾 → embedding gather 越界读 → sticky illegal address。
"间歇"来自那块缓冲在被写入前的内容不定。

**修法**：在 stem 入口按 rank 记录一个 event，让每个 embed 侧 peer stream 排在 rank 0 的 stream
之后（capture 安全：event wait 变成 graph edge，不是 synchronize）。选这个而不是"每 rank 各拷
一份 ids"，是因为真实请求走的是**已捕获的 graph**，而这里需要覆盖的调用方不止一个
（prefill chunk 的 shifted ids、decode round 的 alignment_ids、AR step 的 draft token、bridge 的
token），一处 handshake 全部覆盖，且不引入新的 per-rank 拷贝。

**定位手段（可复用）**：`NINFER_TP4_MTP_STAGE_SYNC=1` 打开阶段屏障
（`schedule::mtp_stage_barrier`，schedule.h）。设备错误是 sticky 的，只在**下一次 synchronize**
报出来，所以裸跑时错误指向的永远是"最后 sync 的地方"（prefill chunk 的结尾），不是出错的 kernel。
加屏障后错误消失 = 竞态而非纯越界，这一条把方向从"算地址错"转到"缺跨 stream 边"。

### 2. speculative commit 的 GDN replay fold 只 fold rank 1（已修）

**症状**：tp4 MTP 输出**前几个 token 与无投机完全一致，之后逐渐退化成自循环**
（如 `在计算机中，0.1 和 0.2 在计算机中，0.1 和 0.2 在计算机中，…`）。

**根因**：`ProgramImplCore` 的 speculative pending commit 里，GDN 状态的 accepted-prefix
fold 是手写的两 rank 形态：

```cpp
ops::gdn_replay_fold(*replay_records, ..., device.stream);   // rank 0
if (peer()) { ops::gdn_replay_fold(*peer()->replay_records, ...); }   // 只有 rank 1
```

`peer()` = `peers[1]`。**tp4 下 rank 2、3 的 conv/recurrent 状态永远不被推进**，一直停在
record 时刻的快照 → 每轮 linear attention 都读一个滞后的状态，误差随轮数累积 →
前几 token 看不出、之后发散成重复循环。这也解释了为什么"verify 会拒掉本来正确的 draft"
（状态错 → verify 的 logits 错）。

**判据实验（决定性，值得复用）**：`NINFER_TP4_MTP_ZERO_DRAFTS=1` 把每轮 ingress 的
`current_drafts` 强制成 token 0（正常永不接受）。此时每轮只可能提交 target 自己的 argmax，
输出**必须**与无投机逐字相同。修 fold 之前：输出仍是那段自循环文本；修完：输出通顺且与无投机同义。
这一步把"draft 质量差"和"verify 路径自身污染状态"彻底分开，是本轮最有价值的一刀。

**修法**：把 fold 改成遍历 `peer_lanes()` 的 rank 1..tp-1（各自 device / workspace / records）。
顺带保留的未做项：同段里 `if (peer()) { peer()->device.synchronize(); peer()->work.reset(); }`
仍是 rank-1-only；tp4 下 rank 2/3 的 stream 在本轮结束时没有被 retire。当前不影响正确性
（它们的 work arena 由各 round body 自己 reset），但属于同类残留，列在下面的待办里。

## 二、本轮一并推广的 tp2 假设（都是 tp4 之前跑不到的分支）

| 位置 | 原状 | 现状 |
|---|---|---|
| `mtp_forward_batch` 里 proposal 的 `columns` | 手工 2 元 brace `{h[0], h[1]}` | 按 `ec().tp` 逐 rank 组装 |
| MTP peer ingress（`mtp_peer_host_ingress`） | 单个标量，只 rank 1 一份，且**所有 rank 共用 rank 1 的记录** | per-rank 数组，各自带自己的 `sampling[row].token_counts` 车道 |
| `resume_hidden` | 只广播到 rank 1（pull 协议 event dance） | rank 1 保持原路径（tp2 逐字不变），rank 2.. 用事件无关的 peer 拷贝 + 退役 |
| `mtp_bridge_tp2`（prefill bridge） | `work`/`io` 是 2 元 brace，`tp = peers[1]`，`next_hidden` 2 元 | 按 `ec.tp` 全 rank 组装；`state.mtp_kv_peers[r]` 每 rank 各自的 KV 窗口 |
| `publish_peer_token_counts` | 只镜像到 rank 1 | 镜像到每个 rank 的计数车道 |

`mtp_prefill_chunk_tp2` / `mtp_forward_stem_tp2` / `proposal_argmax_tp2` / `logits_tp2` /
`target_verify_batch` / `run_layers_tp2` / GDN 的 verify 分支此前**已经是** N-rank 形态
（W5.2 那批改动的成果），本轮核实过，未再改动。

## 三、残余缺陷（功能对、性能不对，两条线索）

实测（`diff-1001-0740-…`、`qwen3_8_27b_q4_k_m.ninfer`、4 条固定 prompt、greedy、无思考档）：

| prompt | 无投机 vs MTP(k=1) | vs MTP(k=3) |
|---|---|---|
| 0（猫的花色） | 差异，语义正确、无循环 | 同左 |
| 1（算术） | 逐字相同 | 逐字相同 |
| 2（翻译） | 逐字相同 | 逐字相同 |
| 3（浮点） | 逐字相同 | 逐字相同 |

接受率：k=1 → 0.0 / 5.4 / 18.8 / 28.3%，k=3 → 0.0 / 4.1 / 8.8 / 11.9%（tp2 同件 75–100%）。

**(a) verify 的列 1 间歇算成"下一列"的 logits。**
用 `NINFER_TP4_MTP_TRACE=1`（`mtp_trace_round`）打某轮真实请求（思考档开）：

```
anchor=4087 accept=1 extent=3 valid=4
drafts =[1144, 310, 1683]        # 1144=" need" 310=" to" 1683=" think"
verify =[1596, 1144, 310, 1683]  # 列位置 tpos=[69,70,71,72]
target =[1144, 4087, 4087, 303]  # 1144=" need" 4087=" answer" 303=" in"
col 探针: col0 atArgmax(1144)=31.75 | col1 atArgmax(4087)=23.5 | col2 atArgmax(4087)=26.75 | col3 atArgmax(303)=19.88
```

同 prompt 同配置的**无投机**参照文本 = `We need to answer user's request:`
= `[1596, 1144, 310, 4087, 1156, 579, 1622]`。逐列对：
- col0（输入 `We`@69）→ ` need` ✓
- **col1（输入 ` need`@70）→ 应为 ` to`(310)，实得 ` answer`(4087) ✗**
- col2（输入 ` to`@71）→ ` answer` ✓
- col3（输入 draft ` think`@72）→ ` in` ✓

且 col 探针显示四列**原始 logits 互不相同**（无别名），col1 只是"像后一列"（数值相近但不等，符合
RoPE 位置差 1 的特征）。⇒ 不是 buffer 别名，是**列 1 的位置/上下文按后一位算**。
排除项：`tpos`/`trope` 打印为 `[69,70,71,72]` 正确；`verify_ids` 布局 `[width,batch]` 逐列正确；
两 rank 的 frame 打印逐字段一致；`extent=3=width-1` 所以不是"尾部填列 clamp 到同一位置"那条
（那条会让 target[2]==target[3]，本例是 col1 独错）。**未定位**。

**(b) 每轮第一份 draft 的正确率远低于 tp2。** k=1 时接受率直接等于"第一份 draft 与
verify 列 0 argmax 相等"的比率 —— 列 0 已被证明是对的，所以这是 **MTP 提案质量**问题。
第一份 draft 来自 `mtp_forward_decode_batch` 之后 `speculative_select_accepted_hidden` 选中的
那一列 target hidden → `mtp_propose_batch`。(a) 若也污染了 per-column hidden，则 (a)(b) 可能同源。

**下一步建议顺序**（从便宜的判别做起）：
1. 在 trace 里同时 dump 每列 `target_hidden` 的若干行 + `alignment_hidden`/`ar_hidden`
   选中列，看 hidden 是否与 logits 同步只在列 1 偏 —— 若是，问题在共享的前向，不必分开查。
2. 单独对拍 `mtp_forward_decode_batch` 的 `alignment_ids` 语义：实测里 AR 链的第 i 步输入是
   `verify[i]` 还是 `verify[i+1]`，用 tokenizer 解码验证（本轮已用此法证明 drafts 是对的、
   verify 是错的，见上）。
3. `--spec mtp --draft-tokens 1` 下把 verify 列 1 的 `cache_positions`/`rope_positions` 换成
   显式常量再跑，看列 1 是否恢复正常（区分"位置张量错"与"注意力核错"）。
4. 若指向核：查 tp4 下 kv_heads/rank = 1 时 prefill/verify 注意力核的配置选择
   （`gqa_attention_geometry.cuh` + volt flash 的 config 表，SM70 上按 head_dim/ncols 选表，
   本机在 llama.cpp 侧就有过"缺 config → 回退 Ampere 表"的先例）。

## 四、工程资产（本轮新增，全部 env 开关、默认关闭）

| 开关 | 作用 | 位置 |
|---|---|---|
| `NINFER_ALLOW_TP4_MTP_UNVERIFIED=1` | 绕过 tp>2 的投机护栏 | `program_impl.h` 构造器 |
| `NINFER_TP4_MTP_STAGE_SYNC=1` | 每阶段全 rank 同步 + 打印 `[mtp-stage]`（找 sticky fault 的真凶） | `schedule.h: mtp_stage_barrier` |
| `NINFER_TP4_MTP_TRACE=1` | 每轮打印各 rank 的 anchor/extent/valid/drafts/verify/target/licensed/位置 + 列 logits 探针 | `mtp_impl.h: mtp_trace_round` |
| `NINFER_TP4_MTP_ZERO_DRAFTS=1` | 强制 draft=0（永不接受），用于把 verify 路径与提案质量分开 | `program_impl.h` ingress 构造 |

脚本（`~/work/ninfer-tp4-m2/`）：`tp4_mtp_diff.sh`（无投机 vs MTP 的严格贪心对拍，自动停/恢复
8901+8902）、`tp4_mtp_quality.sh`（多配置问答冒烟）、`tp4_mtp_repro.sh`（warmup 崩溃复现）、
`tp4_determinism.sh`（graph/eager 两路径各自确定性 + 互相对拍）。

**验收用的两条硬判据**（今后每批改动后都跑）：
1. `bash tp2_gate_serve_ab.sh build-v100-tp4/apps/ninfer-serve 8912 <TAG>` → 必须
   `graph-nodes=2562` 且 `speculative=mtp 3.55tok/round (84.8%)`（本轮实测值）。
2. `bash tp4_mtp_diff.sh`（tP4）→ 4 条 prompt 中不得出现自循环 / 重复段；允许近似并列导致的措辞差异。

## 五、未做 / 未测

- **w4a4w8a8 件 + tp4 MTP 起不来**：warmup 报 `w8 linear: unsupported shape or T`
  （q4_k_m 件正常）。与本轮改动无关，是独立问题（MTP 的 W8 线性层在 tp4 的 shard 形状）。
  复现：`MODEL=~/models/ninfer-V100X2/qwen3_8_27b_w4a4w8a8.ninfer bash tp4_mtp_quality.sh`。
- 同 commit 段里 rank-1-only 的 `peer()->device.synchronize()` / `peer()->work.reset()`
  （正确性目前无影响，属同类残留）。
- `next_drafts` 的声明形状 `{batch, drafts}` 与 egress 注释所述"step-major"不一致
  （batch=1 时两者等价，batch>1 时可疑）；`current_drafts` 是 `{drafts, batch}`。未在 tp4 下测
  batch>1 的 MTP。
- `MtpDecodeIngress.current_drafts` 宿主侧按 `row * draft_window + j` 写，设备视图是
  `{drafts, batch}`（step-major）—— batch=1 等价，batch>1 需核对。

## 六、2026-10-01 20:30 更新：**verify 不是瓶颈，MTP 提案本身才是**

§三把残余写成"两条线索"（verify 列 1 + 第一份 draft 质量）。继续查之后被推翻了一半：
**verify 基本是好的，真正的主导缺陷是 MTP 提案质量**。三条新增证据：

1. **数数任务标尺**（新脚本 `mtp_proposal_quality.sh`，prompt 为"继续数列 1 2 3 4 5 6 7 8"）：
   | | 输出 | 接受率 |
   |---|---|---|
   | tp2 | `9 10 11 … 30` | **97.9%** |
   | tp4 | `9 10 11 … 30`（**与 tp2 逐字相同**） | **22.7%** |
   输出逐字相同 → verify + 提交链路是对的；接受率差 4 倍 → 提案头被劣化。这条比"verify 列 1"
   干净得多，因为它不需要任何参照前向。

2. **drafts 序列显示上下文推进不良**：同一任务的 23 轮 trace（`NINFER_TP4_MTP_TRACE=1`）里，
   target（模型自己的 argmax 链）严格递增且完全正确（拼出 ` 10 11 12 … 22`），
   而 drafts 反复出现同一形态（`' 12'` 出现 8 次；另有 `'0 0'`、`'123'` 等），
   即 MTP 的提案在上下文上"卡住/滞后"，不是随机噪声。tp2 同任务 97.9% 说明这不是任务难度问题。

3. **列 1 的"异常"站不住**：换了 k=1（width=2）之后，列 1 依旧给出 ` answer`，
   即与 width 无关；且 `NINFER_TP4_MTP_DUP_DRAFTS=1`（所有 draft 设成同一 token，
   于是各列只差位置）实测各列 logits 指纹**互不相同** → 列 1 **没有**取列 2 的位置/输入，
   位置偏移 +1 与"读到列 2 的 id"两个假说都被证伪。第 3 轮里列 1 又是正确的（` request`）。
   ⇒ §三(a) 应降级为"间发、未复现出稳定机制"，真正该查的是 §三(b)。

**下一步（便宜且有判别力）**：
- 把 `proposal_argmax_tp2` 的输入固定成**同一份 MTP hidden**（例如从 trace 里 dump 出某一轮的
  `ar_hidden`，再喂给 tp2/tp4 的 proposal 头）→ 直接判定劣化发生在**提案头 + logits gather**
  还是在 **MTP 模块（stem/attention/MLP）**。这是唯一能把"头"与"模块"分开的实验，其余都是绕。
- 若指向模块：MTP 层在 tp4 下 `shard_kv_heads()=1`（24 q / 4 kv，每 rank 6 q + 1 kv），
  优先核对 MTP 注意力的 GQA 分组与 KV 窗口；MTP 的 KV 池是独立的，别和文本 KV 混。
- 已知 tp2 完全正常 ⇒ 每条 shard 几何都有一份 tp2 的参照实现可对拍，不要盲改。

## 七、2026-10-01 21:30 更正 + 头已被证明是好的

### 已证明（干净，无混淆）

**提案头（shared LM head + logits gather）在 tp2/tp4 上对同一输入给出同一结果。**
做法：`NINFER_TP4_MTP_HIDDEN_DUMP/LOAD` 把第一轮 `ar_hidden`（[hidden,1]，复制的）钉死，
再喂给两边的 `mtp_propose_batch`：tp4 自比 353=353 ✓，**tp2 头也是 353** ✓。
⇒ 头/gather 不是缺陷点，嫌疑落在 **MTP 模块（stem→attention→post-mixer）或 MTP KV 状态**。

### 更正：§三/§四里"tail / post-mixer 有问题"的结论**不成立**（我的 A/B 有两处混淆）

我把 stage 级 A/B 的判据从"proposal argmax"换成了"proposal argmax"（错），原因是：
1. **选列混淆**：proposal 的输入并不是 tail 的输出本身，而是
   `speculative_select_accepted_hidden(tail_out, accepted)` 选出的那一列，而 `accepted`
   由 **verify** 决定 —— verify 的 batched logits 在两个宽度下本来就有微小差异，
   所以"选中的列不同"完全可能是 proposta 不同的原因，与 tail 无关。
2. **KV 状态混淆**：attention 段读 **MTP KV**，而 MTP KV 的内容是各自宽度下的 prefill 写的
   （写它的是 MTP 自己的 tail）。所以即使把 stem 输出钉死，两边的 attention 输入仍不同。
   实测：把 tp4 的 stem 输出钉给 tp2 后，两边 attention 段输出差 10210/10240 个元素、
   max|Δ|=3.79（均值量级 1.77）—— 差异巨大，但**无法区分**"宽度导致的 attention 算术错"
   与"两边 MTP KV 内容不同"。**这条不能当证据。**

### 干净实验（下一步，二选一）

- **首选：把 MTP KV 也钉住**。`mtp_cache` 是 paged，dump/load 该 row 的页内容（含 KV 量化格式）
  后在两个宽度下用**同一份** stem 输出 + 同一份 KV 跑 attention 段，再逐元素 diff tail 输出
  （`NINFER_TP4_MTP_TAILOUT_DUMP` 已就位，dump 的是复制后的 [hidden,T]，可直接逐元素比）。
  这样两处混淆都被消掉。
- **备选：op 级单测**。`tests/ops/test_mtp_split.cpp` 的 tp2 Leg A 已经存在（注释里提到它
  "proves this exact composition at tp2"）；把它参数化到 tp4，用**合成输入 + 真实 MTP 权重**
  单跑 `mtp_attention_projection` / `mtp_post_mixer`，与 tp2 的逐元素对拍。
  但注意本机 engine 级 `*_tp2_real` 测试因单卡显存跑不了（19–21 GiB/卡），所以要走
  "只加载 MTP 权重"的路径，或干脆在服务进程里加一个 env 触发的自检。

### 现有工具（全 env 开关、默认关）

| 开关 | 作用 |
|---|---|
| `NINFER_TP4_MTP_STEM_DUMP/LOAD` | 钉 MTP stem 输出 (x,ah)（tail 的输入） |
| `NINFER_TP4_MTP_POSTMIX_DUMP/LOAD` | 钉 post-mixer 输入 (x,mh)（attention 段的输出） |
| `NINFER_TP4_MTP_TAILOUT_DUMP` | dump tail 输出（复制后的 [hidden,T]，用于逐元素 diff） |
| `NINFER_TP4_MTP_HIDDEN_DUMP/LOAD` | 钉 proposal 头输入（**唯一已证明无混淆的 A/B**） |
| `NINFER_TP4_MTP_ZERO_DRAFTS` / `_DUP_DRAFTS` | draft 恒 0 / 恒等于 anchor（判"verify 是否污染状态"、"列间是否别名"） |

注意各 probe 只在**进程内第一次** MTP core 调用生效（T 由那次调用决定），
比对时必须保证两次运行的 `--draft-tokens` 与调用类型一致，否则文件大小/内容不可比。

## 八、2026-10-01 23:00：证据重审 —— 权重分片逐字节相同；分片 GEMM 执行仍是唯一未证之处

本轮把探针逐个复核，发现并纠正了**我自己**的若干测量错误，结论如下。

### 已证明

1. **提案头无缺陷**（唯一无混淆的 A/B，见 §7）✓
2. **MTP 的 `mlp/gate_up` 权重分片在两个宽度下逐字节相同**。
   做法：`NINFER_TP4_MTP_WEIGHT_DUMP` dump 每 rank 的**原始 payload**，按 artifact 的真实布局解码
   （`ggml-k256-v1` = **[每行 8 字节描述符平面 | 行数据]**，描述符 bit0=q6 标志、其余位=行内字节偏移；
   `tensor_row_slice` 是按描述符正确实现的）。tp4 的 (rank0+rank1) 覆盖的全局 gate 行与 tp2 的 rank0
   **完全一致**（17408 行中 0 行不同），up 半同理 ⇒ 权重与行映射都没问题。
3. **pin 探针确实生效**：tp4 未钉输入那次 dump 出的 (x,mh)，被同一 tp4 进程 LOAD 后输出**逐位相同**
   （这是自洽检验：被钉的值就是它自己的自然输入 ⇒ pin 通路正常）。
4. **在"两边都钉同一 (x,mh)、权重逐字节相同"的条件下，tail 输出仍差 1095/5120 个元素**
   （bf16 逐字节比较；tp4 自比 0 差异）。
   ⇒ 剩下的**唯一宽度相关变量**是分片 GEMM 的**执行路径**（n=8704/k=5120 的列并行 gate_up、
   n=5120/k=4352 的行并行 down + 2-vs-4 rank allreduce）。

### 纠正（我自己的测量错误，避免后人重蹈）

- **BF16 被当 FP16 解码**：早期用 `struct.unpack('<e')` 读 bf16 dump ⇒ 数值全错。**差异的*个数***仍有效
  （16 位模式一一映射），但任何 `max|Δ|`/均值都不能引用。
- **"未钉输入"与"钉输入"的两次运行被当成可比**：`mlp` 系列 dump 里 tp4 那次没有 LOAD，对比无效。
  教训：**每次对比前先从日志确认两侧的 probe 行都在**（"dumped"/"loaded" 都要有）。
- **probe 的 `done` 标志各自独立**：`Variant::mtp_post_mixer` 还被 prefill/tp1 路径调用
  （text_context_impl.h:559/686），所以 MTP-MLP 内部 dump 的"第一次调用"**未必**与 tail 探针的
  "第一次调用"是同一次 ⇒ 逐行对比无效。**只在同一函数内、同一个 call 上取 dump 才可比**
  （tail 输出探针满足这点，MTP-MLP 内部 dump 不满足）。

### op 级测试的边界（本轮补测）

`tests/ops/test_ggml_k.cpp` 原本只覆盖 tp1 与 tp2 分片形状（7168×5120、5120×8704、17408×5120、k=3072）。
本轮补入**全部 tp4 分片形状**（8704×5120 / 5120×4352 / 5120×1536 / 3584×5120 / 5120×2560，T=1,2,4）——
**全部通过 FP64 oracle（relative_l2≈0.0011–0.0020，阈值 0.004）**。
⇒ ggml_k **内核**在这些形状上没问题；未覆盖的是 `linear_column_parallel`/`linear_row_parallel`
这两层 wrapper 在 tp4 下的分片调度（列并行无归约、行并行靠 allreduce）。

### 下一步（唯一无混淆的方法）

**在 host 上复算 MTP 层**（把 GGML_K 权重按描述符反量化成 fp32，连同该 row 的 MTP KV 一起，
用 numpy 逐 stage 复算 stem / attention / post-mixer），与进程内 dump 的逐元素对比 ——
没有跨宽度 A/B、没有 pin，因而不可能有上面那类混淆。前置条件：
`NINFER_TP4_MTP_TAILOUT_DUMP`（tail 输出，复制张量）与 `NINFER_TP4_MTP_STEM_DUMP`（stem 输出）已就位，
KV 页 dump 需新增（paged + 量化）。

## 九、2026-10-01 23:40：修掉一个**加载阻断**级缺陷 —— W8 分派表缺 tp4 分片 extent

### 症状与根因

官方 nvfp4 件（以及 ET w4a4w8a8 件）在 **tp4 + MTP** 下**根本起不来**：

```
[error] ninfer-serve: w8 linear: unsupported shape or T
```

根因：`src/ops/linear/w8/w8_dispatch.cpp` 的 `select_w8_a16_launch` **先查表、查不到直接 throw**，
而表里只有 **tp1（整模型）与 tp2 分片**的 extent，**没有任何 tp4 分片 extent**。
而 Volta 的兜底拦截（`w8_uses_volta_*` / `launch_w8_small_t`）在这条 throw **之后**才轮到 ⇒
tp4 下 MTP 的 W8 GEMM（8 个对象里 5 个形状变了）第一次 warmup 就炸。

### 修法（`w8_dispatch.cpp`）

把 tp2 的清单镜像成 /4：列并行 `n ∈ {256,1536,3584,8704,62080}`、行并行 `k ∈ {1536,2560,4352}`；
函数改名 `select_w8_shard_launch`（它现在同时认 tp2/tp4 两类 extent）；`n==512→r32_c128` 的特例
扩成 `n==512 || n==256`。注释里写明**为什么必须放在 Volta 拦截之前**、以及这是"解锁加载"而非调优。

### 验证

1. **官方 nvfp4 件 tp4+MTP 现在能起、能出正确结果**：数数任务输出 `9 10 11 … 22` ✓，
   接受率 **24.6%**（tp2 同任务 ~97.9%）—— 与 q4_k_m 件在同一水平，说明剩余的接受率问题与宽度无关。
2. **新增回归测试**：`tests/ops/test_w8_a16.cpp` 补入 5 个 tp4 分片形状（8704×5120 / 5120×4352 /
   5120×1536 / 3584×5120 / 5120×2560，T=1,2,4,5,16,17）→ **`OK W8_A16 Linear`** ✓
   （除了覆盖分派，这些形状还与参考实现逐样本对拍 ✓ 顺带证明 tp4 的 W8 数值在 Volta 通用 SIMT 路径上正确）。
3. **tp2 回归**：闸门 `graph-nodes=2562` + `mtp 3.55tok/round (84.8%)` —— 与基线逐字一致 ✓。

### 影响面

`official nvfp4`、`ET W4A4W8A8`、`Swift-1.5 NVFP4`（凡 MTP 走 W8G32_F16S 的件）在 tp4 下从
"加载即失败"变为"可跑（输出正确、接受率低）"。q4_k_m 件（MTP = GGML_K）不受此缺陷影响，
它的 tp4 MTP 从一开始就能跑（其低接受率是 §7/§8 里那个仍未定位的分片 GEMM 执行问题）。

## 十、2026-10-02 00:30：host 参考复算工具已建好并**分段验证**；RMSNorm 段仍未复现（不据此下结论）

按 §8 的建议做了"host 侧复算"这条路，成果与**当前卡点**都写清楚，避免后人重复劳动。

### 已完成并已验证的部分 ✅

1. **dump 探针升级为自描述格式**（`NINFER_TP4_MTP_STEM_REF_DUMP`）：header + ids + 每段 8 字节 tag
   （空格填充）+ 长度 + 数据，另有每 rank 的 `fcShard<r>` 记录（几何 + 原始 payload）。
   段包括：`hidden/embed/stemX/stemAH/normE/normH/normI/nEmb<r>/nHid<r>/fcIn<r>`。
   **全部来自同一次 MTP 调用**，所以不存在"跨宽度 A/B"或"pin"那两类混淆（§8 的教训）。
   踩过的坑：早期依赖写入顺序解析、把引擎在该宽度下**根本没写**的缓冲（如 tp2 rank0 的
   `normalized_hidden`）当有效值比对 ⇒ 现改为按 tag 取，且每 rank 单独 dump。
2. **GGML_K 反量化器（python）与引擎自己的 oracle 逐位一致**：`tests/ops/test_ggml_k.cpp` 的
   `oracle_block` 被逐行移植（Q4_K/Q6_K），并在**真实 payload 块**上实测 `max|d| = 0`
   （随机字节会因 fp16 出现 NaN，不能用来校验）。向量化版还额外注意了一个坑：
   Q6_K 的 scale 索引是 `p[192 + at/16]`，**每 16 个 j 换一次 scale**，写成"每 section 一个 scale"
   会静默错 39–707 倍。
3. **K 打包约定证实**：`fc_input[0]` 与 `normalized_embedding` 的偏差**完全相同**、
   `fc_input[1]` 与 `normalized_hidden` 相同 ⇒ 打包就是 `[norm_emb(5120) | norm_hid(5120)]`，
   rank r 取 packed 行 `[r*K/tp, (r+1)*K/tp)` ✓（与代码注释一致）。

### 当前卡点：RMSNorm 段对不上（**因此尚未对 tp4 下任何结论**）

用同一份 dump 比对"我的 rmsnorm(raw)" 与引擎的 `normalized_*`，**两个宽度都差到 20–500 相对值**：
- 试过 `gain = 1+w`（op 头文件明写 `unit_offset ? 1+w : w`，调用点传 `true`）与 `gain = w`；
- 试过 per-column / per-row / global 三种 rms、以及"只乘 gain 不归一"；
- 试过按"行项+列项可分解"的检验（`log|eng|-log|raw|` 应为行+列之和）：残差 rms 0.07、最大 0.55，
  且行项与 `log(1+w)` 相关性 ≈ 0、列项与 `-log(rms)` 相关性 ≈ -0.25。
⇒ **不是"归一化方向/单位偏移"这类小差异，而是"我读到的缓冲不是那次运算的(输入,输出,权重)三元组"**
（段长度都核对过：norm 段 10240B = 5120×bf16 ✓，矩阵段 92160B = 5120×9×bf16 ✓，排除长度/解析错）。
⇒ 因此在解开这一步之前，本参考**不能**用来判定 tp4 的 stem 是否正确；§8 的"分片 GEMM 执行"线索
维持在"未证"状态。

### 下一步（把 RMSNorm 段单独钉死，最省力）

写一个**独立的 op 级探针**：构造一个已知的合成输入与权重，直接调 `ops::rmsnorm`
（`unit_offset=true` 与 `false` 各一次），dump 输入/权重/输出三个缓冲，然后与 python 公式逐元素对拍。
这样就把"op 语义"与"stem 里哪个缓冲是谁"彻底解耦；op 语义一旦确定，回到 stem 参考只是几分钟的事。
（本机 4 卡随时可腾；该探针不需要模型权重。）

### 补记（同日晚）：`ops::rmsnorm` 的**权威公式**已核对，与我的实现一致

`tests/ops/test_rmsnorm.cpp` 的 `rmsnorm_oracle` 明写：

```cpp
const double inverse = 1.0 / std::sqrt(sum_squares / d + kEps);           // 逐"行"归约，d = ne[0] = 5120
const double gain    = weight[column] + (unit_offset ? 1.0 : 0.0);        // unit_offset 语义
output = input * inverse * gain;
```

我的 python 实现与此**逐字一致** ⇒ **公式不是差异来源**，问题在"我 dump 到的三个缓冲不是同一次运算的
(输入, 权重, 输出) 三元组"。（也核对过：`roots[].embedding` 的 shape 是 `{hidden, tokens}`，
归约维就是 hidden ✓，与 python 里 axis=0 一致 ✓。）

⇒ 因此上一条的"下一步"更精确了：**用合成输入直接调 `ops::rmsnorm` 并 dump 三元组**（不涉模型权重、
不涉 stem 的缓冲身份），先证明"我理解这个 op 的 I/O 方式"是对的，再回 stem 参考。

# ★★ 十一、2026-10-02 01:10：根因找到并修复 —— **MTP stem 的 K 切片拷贝假设了行主序**

## 根因（tp4 专有，tp2 结构上跑不到）

`text_context_impl.h` 的 `mtp_forward_stem_tp2` 里，tp > 2 的"把该 rank 的 K 切片压实"这一步：

```cpp
// 旧代码（错）
cudaMemcpyAsync(slice.data,
                source.data + within * shard_rows * T * kElementBytes,   // ← 当作行连续
                shard_rows * T * kElementBytes, ...);
```

它把「行区间 `[within*shard_rows, (within+1)*shard_rows)` 的**所有列**」当成一段连续内存。
但该缓冲是**列主序**（`ne[0] = hidden` 是最快维）：这些行的同一元素在内存里**跨 `hidden` 步长**，
连续的一段 `shard_rows*T` 个元素其实是**跨列的一团**。⇒ 每个 tp4 rank 的 `fc_input` 都不是它该
contract 的 K 行，MTP stem 输出因此系统性错误（值都是"真的激活值"，所以症状温和：草案质量差、
接受率低，而不是崩或乱码）。

**为什么只有 tp4**：tp2 时 `half == 1`，走 `fc_input[slot] = source;` 快捷分支，**这段拷贝根本不执行**
——这是"tp2 正常、tp4 坏"的结构性原因。

**修法**：按列逐次拷贝（目标列主序、源按 `column * hidden + within * shard_rows` 定位），T 次小拷贝。
位置：`mtp_forward_stem_tp2`，注释里写明了这个坑与"为什么 tp2 躲过"。

## 怎么找到的（方法论，值得复述）

前几轮的跨宽度 A/B 与 pin 全被证伪（§8 的三处测量错误）。这一轮走 **host 侧 fp32 参考复算**，
并在**已知正确的 tp2** 上先把参考本身验证到 `rel_l2 ≤ 0.0037`（引擎自家 op 测试阈值 0.004）才用它判 tp4：

1. `NINFER_TP4_MTP_STEM_REF_DUMP`：**自描述** dump（tag+长度），同一次调用内取
   ids / target hidden / 查表得到的 embedding / stem 输出 x,ah / 三个 norm / 每 rank 的
   `normalized_*`、`fcInput*`、`fcShard*`（原始量化 payload）。
2. `mtp_stem_reference.py`：GGML_K Q4_K/Q6_K 反量化（与引擎自己的 oracle **真实块上 max|d|=0**）+
   fp32 复算；**列主序 reshape**（这是我自己踩的第二坑：用 C order reshape 会把每个张量转置，
   导致 norm 段怎么调都差 20–500 倍）。
3. 逐段比对（tp2 基线）：norm 段 `rel_l2` 0.0017 / 0.0018 ✓、`fc_input` 0.0018 ✓、`stem x` 0.0015 ✓、
   `stem ah` 0.0031 ✓ ⇒ 参考可信。**tp4 同项**：norm 段 0.0017/0.0018 ✓（说明 embedding 查表与三个
   RMSNorm 在 tp4 都对）**但 `fc_input` 1.20–1.36 ✗**、`stem x` 1.32 ✗ ⇒ 缺陷精确落在"fc 输入"这一步 ✓。

## 修复后的验证（全部实测）

| 验证项 | 修前 | 修后 |
|---|---|---|
| tp4 `fc_input[r]` vs 参考（4 个 rank） | rel_l2 1.20–1.36 | **0.0016–0.0019** |
| tp4 `stem x` / `stem ah` | 1.32 / 1.19 | **0.0037 / 0.0037** |
| 数数任务接受率（tp2: 97.9%） | 22.7% | **100.0%**（`3.94tok/round`） |
| 4 条固定 prompt（k=1 / k=3） | 0–12% | **73.9/74.1/100/100%** 与 **57.4/69.4/100/100%** |
| 4 条 prompt 与无投机的一致性 | 自循环 3 条 | **3 条逐字相同**，第 4 条仅措辞差异 |
| tp4 decode（同一次运行） | — | 无投机 43.5 → **MTP 111.0 t/s**（高接受率那条）；FAQ 类 86.3 t/s |
| 官方 nvfp4 件 tp4+MTP | **加载即失败**（w8 表缺 tp4 extent） | **100.0% 接受率、188.8 tok/s**、输出正确 |
| tp2 闸门 | 2562 / 84.8% | **2562 / 84.8%**（不变） |

## 护栏已撤

`program_impl.h` 里对 `tp > 2` + 投机的拒绝已删除（连同 `NINFER_ALLOW_TP4_MTP_UNVERIFIED`
覆盖开关）：两条根因都已修复并验证，没有东西可绕过了。裸 `--spec mtp --tp 4` 现状 =
直接可用（实测 health 200、答案正确、86.3 t/s）。注释里保留了这两条根因与实测数字，防止将来回归。

## 与之前几节的关系

- §7/§8 的"头是好的、权重分片逐字节相同"仍然成立，且与本根因**不矛盾**：坏的是 *fc 输入*（激活侧），
  不是权重、不是头、不是 verify。
- §8 最后留下的"分片 GEMM 执行"线索**可以撤销**：差异来自喂给 GEMM 的激活被拷错，而不是 GEMM 本身。
- §9 的 W8 分派表修复是**独立且仍然必要**的（它决定 W8G32_F16S 件能否在 tp4 加载）。

# 十二、2026-10-02 01:40：收尾三条（现役件实测、sync/reset 残留、batch>1）

## 1. 现役件 ET w4a4w8a8（:8901 的 tp2 配置）在 tp4 下也满级 ✅

```
tp4 --spec mtp --draft-tokens 3 --lm-head-draft（官方件 = ET W4A4W8A8）
out: 9 10 11 12 13 14 15 16 17 18 19 20 21 22      decode=182.1tok/s   mtp 3.90tok/round (100.0%)
```
它的 MTP 走 **W8** 路径（§9 才解锁）+ §11 的 stem 修复 ⇒ 两处修复都对它生效。
此前该件在 tp4 是"加载即失败"。

## 2. tp2 / tp4 干净吞吐对比（同件、同 flags、greedy、无思考档、32768 ctx、200 tok）

| 配置 | 无投机 | MTP | 接受率 |
|---|---|---|---|
| tp2（2 卡） | 33.9 t/s | **123.6 t/s** | 100% |
| tp4（4 卡） | 49.1 t/s | **180.6 t/s** | 95.6% |

- MTP 在两边都是 **≈3.6×**（此前 tp4 是 0–12% 接受率、几乎无收益）。
- 单实例：tp4 比 tp2 快 **1.46×**；但 tp4 吃满 4 卡（只能跑 1 个实例），
  现役布局是 4 卡跑两个 tp2 实例（聚合 ≈2×123.6 = 247 t/s > 180.6）⇒
  **按聚合吞吐现役布局更优、按单请求延迟 tp4 更优**；是否切换由用户决定，未改动现役布局。

## 3. rank-1-only 的 sync/reset 残留（已清）

`program_impl.h` 里 11 处 `if (peer()) { peer()->device.synchronize(); }` /
`if (peer()) { peer()->work.reset(); }`（分布在 `resolve_pending_batch` 投机提交、`advance_prefill`
6 处、`prepare_graphs`）**只认 `peer()` = rank 1** ⇒ tp4 下 rank 2/3 既不同步也不重置工作区
（其流正在跑同样的 round，而它们的 arena 即将被复用）。已改为两个 helper
`ProgramImplCore::synchronize_peers()` / `reset_peer_works()`（遍历 `peer_lanes()`，即所有非 0 rank），
注释写明这条历史假设与为什么多卡下必须遍历。

**回归**（改动后实测）：tp2 闸门 `graph-nodes=2562` + `84.8%` 不变；tp4 数数标尺仍 **100.0%**、
decode 128.4 t/s；tp4 4-prompt 仍 3/4 逐字相同（第 4 条措辞差异）、接受率 57.4–100%。

## 4. tp4 + batch>1（此前未测，现已测）✅

`--max-concurrency 4` + 4 个并发请求（tp4 MTP）：**4 条答案全部正确**
（`1` / 英文翻译 / KV cache 解释 / 浮点解释），接受率 45.1–100%，4 请求 wall 1.47 s。
⇒ 之前标记的 `next_drafts` 声明形状（`{batch, drafts}`）与注释"step-major"不一致的问题
**在 batch=4 实测下未表现出错误**（代码自洽；该条从"未测"降为"已测通过，注释与声明形状仍不一致但无害"）。
