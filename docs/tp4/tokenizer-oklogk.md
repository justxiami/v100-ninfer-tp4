# ninfer tp4 长上下文"卡死"根因定位（2026-10-01）

## 一句话结论

不是死锁、不是并发、不是 KV、不是网络——是 **self-hosted C++ BPE 分词器在超长 word 上的 O(k²) 退化**。
`doc_full.txt` 后半段有 ~6 万字符、无空格的连续中文，被 `qwen_split_words` 当成 **一个 word**（k=179,964 字节），
`append_bpe_ids` 对这个 word 做朴素 BPE → 3.2×10¹⁰ 次 pair 查找 → **~40 分钟**。表现为"卡死"。

## 触发条件（精确）

> 输入里出现**超长无空白连续串**（CJK 最典型：无空格 + `is_letter` 为真 → 不被切分）。
> word 长度 k 一大，O(k²) 就爆。

## 复现（最小、干净）

单发一个请求即可复现，无并发干扰：

```
夹具 = ladder.py 的静态切片 build_doc(120000, needle=True, offset=100000)
单发 → 引擎日志不出现 `submitted` 行，客户端 ttfb 无限等，GPU 0%，CPU 100%（一个线程 R 状态自旋）
```

现场栈（35 线程全量，`backtrace_A-STUCK_125607.txt`）：

```
#0  std::unordered_map<string,int>::find(...)
#1  ninfer::...::frontend_internal::append_bpe_ids(...)
#2  ninfer::...::frontend_internal::Tokenizer::encode(...)
#3  encode_rendered_chat
#4  Frontend::prepare
#5  Engine::prepare
#6  GenerationService::prepare
#7  HttpServer::handle_chat_completions
#8  httplib::dispatch_request → process_request → process_and_close_socket
#11 httplib::ThreadPool::worker
```

## 证据 1：严格 O(k²) 标度（引擎实测，单发串行）

`k` = 该档最长 word 的 UTF-8 字节数（`max_word_bytes`），`ttfb` = 客户端首字节。

| tier | local_tok | k (bytes) | k² | 实测 ttfb | 归一化 |
|---|---|---|---|---|---|
| 1,000 | 888 | 1,203 | 1.4×10⁶ | 0.10 s | — |
| 8,000 | 7,878 | 11,718 | 1.4×10⁸ | 10.06 s | 71.8 s/10⁹ |
| 16,000 | 15,828 | 23,736 | 5.6×10⁸ | 42.06 s | 74.7 s/10⁹ |
| 32,000 | 31,899 | 47,769 | 2.3×10⁹ | 170.04 s | 74.5 s/10⁹ |
| 120,000 | ~119,800 | 179,964 | 3.2×10¹⁰ | **外推 2,413 s ≈ 40 min** | 74.6 s/10⁹ |

- 8k→16k：k² ×4.10，ttfb ×4.18
- 16k→32k：k² ×4.05，ttfb ×4.04

**归一化常数稳定在 ~74.5 s/10⁹ ops → 二次标度确凿。**

## 证据 2：因果实验（只改 word 长度，内容一字未删）

同一份 8k 文本，把长 word 按固定间隔插空格打散：

| 变体 | max_word | ttfb |
|---|---|---|
| 原样 | 11,718 B | **10.20 s** |
| 每 200 字插空格 | 600 B | 0.39 s |
| 每 50 字插空格 | 150 B | 0.09 s |
| 每 5 字插空格 | 44 B | 0.02 s |

→ **word 长度是唯一自变量。**

## 证据 3：HF tokenizers 对照（同段文本）

| | 处理 59,988 字纯中文 | 平均 word |
|---|---|---|
| HF tokenizers 0.23.1 (Rust) | **59,896 tokens / 44.4 ms** | 1.00 字符 |
| ninfer 自研 C++ | 一个 word，k=179,964 B | 59,988 字符 |

HF 对 CJK **逐字符切分** + 线性算法；ninfer 不切分 + O(k²)。

## 为什么 10/01 才出现（关键背景）

会话里 agent 在 11:58 把 `ladder.py` 的 `build_doc` 从**旧夹具**（运行时逐句生成**英文**句子，`sentence(rng)`）
改成**新夹具**（静态 `doc_full.txt` 切片）：

- 11:36 跑的那批 1k–64k + 120k needle/cache 用**旧夹具**（英文，无长 word）→ 正常
- 11:58 后用**新夹具**补跑 120k longdec → 切到 `doc_full.txt` 的中文段上 → 卡死

所以不是引擎"变坏了"，是**测试数据换了内容**。同一天同一个引擎，两套夹具表现天差地别。

## 为什么 vLLM / llama.cpp 不这样

同一个 120k 输入它们也要 tokenize：
- HF tokenizers：CJK 按字符切（word 长度恒定 1），算法近似线性 → 44 ms
- llama.cpp：自己的 split 规则同样把 CJK 切开

**没有任何一个 word 会到 6 万字符**，O(k²) 的前提不成立。

## 旧现象全部得到解释

| 现象 | 真因 |
|---|---|
| 引擎无 `submitted` 行 | 卡在 `prepare`（分词），`log_request_start` 在 prepare 之后 |
| GPU 0% | 没走到 GPU |
| CPU 一个核 100% 自旋 | O(k²) 纯 CPU 计算 |
| 客户端 CLOSE-WAIT | worker 卡在分词不返回，客户端先断 |
| "新连接不 dispatch" | 不是不 dispatch，是 worker 被分词占死，后续请求排队 |
| 4.5 分钟"日志静默" | 空闲间隔本来就不打吞吐行（`report_has_activity` 过滤），不是线程死 |
| 连退都退不出 | 退出要 `join` 卡在分词里的 worker；SIGTERM 走 `server->stop()` → 卡在 shutdown 的 join；**SIGKILL 有效**（无 D 状态） |

## 代码位置

- `src/targets/qwen3_6/impl/frontend/tokenizer.cpp:570-598` `append_bpe_ids`（O(k²) 主循环）
- 同文件 `:319-326` `merge_pair_key`（每轮新建 `std::string`，堆分配）
- 同文件 `:416-500` `qwen_split_words`（CJK 不切分 → 长 word 的来源）
- 报错文本 "requires embedded merges.txt" → tokenizer 资源是**嵌入二进制的**，无外部依赖

## 修复方向（代价递增）

1. **`qwen_split_words` 对 CJK 逐字符切分**（一行级）→ word 长度有上界，O(k²) 立即无害
2. **`append_bpe_ids` 换标准优先队列 BPE**（O(k log k)），或至少避免 `merge_pair_key` 每轮分配
3. **接 HF tokenizers 的 Rust/C++ 绑定**

## 文件清单（本目录）

| 文件 | 内容 |
|---|---|
| `repro.py` | 复现脚本（A/B/C 三种变体） |
| `evidence.sh` | 现场取证（线程状态 + 全栈 + 连接表 + GPU） |
| `sweep.py` / `sweep.jsonl` | 长度扫描（O(k²) 标度数据） |
| `segtest2.py` / `segtest2.jsonl` | 因果实验（打散长 word） |
| `measure_hf_tok.py` | HF tokenizers 对照 |
| `backtrace_A-STUCK_125607.txt` | **根因栈证据**（35 线程全量） |
| `backtrace_REPRO_124757.txt` | 第一次复现的栈 |
| `evidence_A-STUCK_125607.txt` | 线程/连接/GPU 快照 |
| `TIMELINE.txt` | 关键事件时间线 |
| `engine.log.baseline-124603` | 旧日志残留（老实例日志已被覆盖，见 TIMELINE） |

## 复现步骤（照抄即可）

```bash
# 1. 起实例（脚本已改日志分文件）
bash ~/scripts/ninfer-v3/serve_tp4_8901.sh start 8901

# 2. 单发 120k（新夹具）
cd ~/bench-results/2026-10-01/ninfer-120k-newconn-repro
/usr/bin/python3 repro.py A          # 会卡；测短档用 sweep.py

# 3. 卡住时取证（另开 shell）
bash evidence.sh . TAG "note"        # 出 backtrace_*.txt

# 4. 清场（SIGTERM 无效）
kill -9 <pid>
```

---

# 修复记录（2026-10-01 13:0x–13:1x）

## 改动

`src/targets/qwen3_6/impl/frontend/tokenizer.cpp` — `append_bpe_ids`：
O(k²) 朴素扫描 → **双向链表 + 优先队列（O(k log k)）**。语义完全不变
（每轮合并全局最小 rank 的相邻 pair，rank 相同取最左）。

堆条目 = `(rank, left, left_epoch, right, right_epoch)`，**两端 epoch 都要带**。

## 中途踩的坑（重要，别重犯）

第一版只用 `(rank, left, right)` + `left.epoch` 做过期检测 —— **对拍直接抓到 token 序列不一致**（13 个 case 系统性差 23–93 个 token）：

- `left` 变长时我们会主动 push 新条目，能靠 `left.epoch` 判过期；
- 但 **`right` 被合并变长时**，拓扑不变、`left.epoch` 也不变 → 堆里的旧 rank 被当成有效，
  拿旧 rank 去合并已经变了的 pair。

修法：堆条目带上 **`right.epoch`**，pop 时两端 epoch + 拓扑三项全查。

验证：Python 复刻新旧两版算法随机对拍 **5500 例（ASCII+CJK）0 不一致**；
C++ 编译后引擎端口 13 case 对拍**全一致**。

## 效果（同夹具，单发）

| tier | k (bytes) | 修复前 ttfb | 修复后 ttfb | 倍速 |
|---|---|---|---|---|
| 8k | 11,718 | 10.06 s | **0.03 s** | 333× |
| 16k | 23,736 | 42.06 s | **0.03 s** | 1319× |
| 32k | 47,769 | 170.04 s | **0.05 s** | 3458× |
| 64k | 95,841 | (外推 ~680 s) | **0.10 s** | ~6800× |
| 120k | 179,964 | **(外推 2413 s)** | **0.22 s** | **~11000×** |

修复后 `total_s`（2.39/4.86/10.34/23.55/53.79 s）就是纯 prefill，恢复正常。

## 语义对拍（旧二进制 vs FIXED）

13/13 完全一致：

```
ascii-short 54 | ascii-sentence 62 | cjk-short 59 | cjk-long-1000 664 | mixed 66
punct 71 | numbers 93 | repeat 115 | whitespace 60 | english-4k 4053
english-16k 16023 | cjk-tail-longdec 431 | doc-cjk-2k 2055
```

## 资产

| 项 | 位置 |
|---|---|
| 源码备份（原始） | `~/backups/2026-10-01-ninfer-tokenizer-bpe/tokenizer.cpp.orig` |
| 第一版(有bug) | 同上 `tokenizer.cpp.new_Oklogk` |
| epoch 单端(仍错) | 同上 `tokenizer.cpp.new_Oklogk_epoch` |
| **FIXED（现役）** | 同上 `tokenizer.cpp.FIXED_Oklogk` |
| 对拍数据 | `probe_old.jsonl` / `probe_fixed.jsonl` |
| 扫描数据 | `sweep_prefix.jsonl`(修前) / `sweep.jsonl`(修后) |

## ⚠️ 未覆盖

**只重编了 `build-v100-tp4`**（tp4 引擎）。`build-v100`（tp2，`serve_v100x2_v2.sh` 用）
还是旧代码 —— 起 tp2 实例前需要同样重编，否则会撞同一个 O(k²)：

```bash
cd ~/src/ninfer-V100X2-remote && cmake --build build-v100 --target ninfer-serve -j6
```
