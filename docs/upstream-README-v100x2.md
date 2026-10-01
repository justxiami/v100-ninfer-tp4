> Baseline fork README (tuxKOH/ninfer-V100X2 @ c6100422), kept verbatim: detailed TP2
> measurements and the baseline build/run/convert instructions. The repository overview and
> tp4 results are in the top-level [README](../README.md).

# NInfer V100X2

This fork is tuned for one-request Qwen3.8-27B inference on **2 × Tesla V100-SXM2 16 GB**
(`sm_70`, CUDA 12.8). Its default profile uses the LM Studio Q4_K_M-derived `.ninfer` artifact,
180,000-token context capacity, complete INT8 group-64 KV, TP2, CUDA Graphs, and MTP with up to
three drafts (zero accepted drafts is valid). Context capacity is an allocation limit, not the
number of prompt tokens in a benchmark.

The starting point combines the RTX 3060 TP2 work and the Volta implementation from
[`geoffwatts/ninfer-v100`](https://github.com/geoffwatts/ninfer-v100), based on
[Neroued/ninfer](https://github.com/Neroued/ninfer). This README contains V100X2-specific changes
and measurements only; inherited RTX 5090/Ampere/Ada results and general upstream capabilities are
intentionally omitted.

## V100X2 changes

- **Q4_K_M prefill:** cooperative SM70 GGML-K block decoding materializes rows into caller-owned
  FP16 workspace and feeds the Volta CUTLASS Tensor-Core GEMM. GDN control projection has an FP32
  output path. The original GGUF Q4_K/Q6_K codes and scales are kept unchanged.
- **Volta FP8/NVFP4 execution:** wide prefill projections decode weights once per call and use
  SM70 CUTLASS, including the prepacked TP2 matrices. NVFP4 decoding reads complete K16 tuples
  cooperatively. Decode uses QPN Tensor-Core kernels. Both wide prefill and decode keep gate/up
  projections in FP32 through SwiGLU, rounding only the final activation to BF16.
  The prepacked FP8 output head now uses its matching kernel for single-token calls and chunk
  tails too; reading that layout through row-major GEMV was a correctness bug. NVFP4 A4 is not
  supported on V100.
- **Long-context decode:** the split-KV attention reducer shares softmax weights and reads eight
  split vectors cooperatively, then accumulates in the original FP32 order. It retains every
  history token, the same KV format and the same partial-output precision. The INT8 attention
  kernel also computes each QK score tile once and shares it across the four output-dimension
  warps, preserving their softmax and PV accumulation order.
- **GDN prefill:** long normalized inputs prepare each Q/K row once in FP32, avoiding repeated
  normalization across state tiles. The sequential FP32 state transition is unchanged. The
  3,072-token TP2 GDN scratch allocation is 24 MiB.
- **TP2 transfers:** collectives use graph-capturable UVA device-to-device copies and two event
  pairs. Startup checks exact transfer bytes in both directions. On this host's translated
  `DMA-FQ` IOMMU domain, direct P2P is unavailable and CUDA selects its driver-managed staging
  path; NInfer does not maintain a second explicit pinned-host copy route.
- **Volta TP2 residuals:** Q5 row shards now select executable SM70 SIMT, fused MMA, or CUTLASS
  routes with the actual peak workspace reserved. BF16 row shards use the SM70 CUTLASS route.
  Both paths pass their independent FP64 operator checks and the two-card composition tests.
- **MTP and DFlash:** the Q4_K_M target uses the native MTP route. An optional five-layer BF16
  DFlash2 route is integrated for experiments, with TP2 draft-column gathering on rank 0. It is
  much slower than MTP on the measured long-code workload and does not fit the 180K allocation on
  the current two cards; see the measurements below.

## Measurements

All NInfer results below are from this V100X2 host. Decode tok/s counts committed output tokens,
not drafted tokens; occupied prompt length is reported separately from maximum capacity.

### Q4_K_M prefill

TP2, INT8 KV, `prefill_chunk=4096`, and 180,000-token capacity:

| Occupied prompt | Prefill throughput | Measurement |
|---:|---:|---|
| 8,192 tokens | 1,672.9 tok/s | one cold run |
| 85,000 tokens | 1,251.44 ± 2.87 tok/s | two cold runs |

The 85K prompt exceeds the requested 1,000 tok/s target. On the 8K probe, chunk sizes
1,024/2,048/3,072/4,096/5,120/8,192 measured 1,454.4/1,599.3/1,636.1/1,672.9/1,672.3/1,195.2
tok/s. For the 85K corpus, chunks 1,024/2,048/4,096 measured 1,133.0/1,213.4/1,251.0 tok/s.
The 4,096-token chunk reserves 1.49 GiB per device. These are prefill measurements, not decode
rates.

The active TP2 collective path was also run independently on the two V100s: a 10 KiB BF16
all-reduce (the decode-shaped hidden block) measured 48.16 µs mean, 47.09 µs p50 and 64.92 µs p99
over 500 host-synchronized iterations. The test passed all exact-value, guard, uneven-shape and
64-consecutive-round checks. This fixed PCIe/event schedule is why changing Q4/Q6/Q1 weight codes
cannot remove the decode ceiling by itself.

### Q4_K_M decode

With exactly 85,000 occupied prompt tokens, 180,000 capacity, TP2, INT8 KV, MTP3, optimized draft
head and CUDA Graphs, the active UVA-D2D transport path measured two 128-token windows:

| Repetitions | Prefill rate | Committed decode rate |
|---:|---:|---:|
| 2 | **1,251.44 ± 2.87 tok/s** | **50.68 ± 0.04 tok/s** |

Both windows used the same 85K raw token corpus and produced no EOS/EOG; aggregate MTP acceptance
was 78.07%. A separate
512-token occupied prompt at the same capacity
measured **60.0291 ± 0.0571 tok/s** over three 256-token decode windows; this is not an 85K
occupied-context result.

The superseded explicit pinned-host experiment reported **53.4075 ± 0.0639 tok/s** over three
512-token windows; it is not the active transport result and is retained only as historical context.
An earlier matched diagnostic used LM Studio CUDA 2.33.0's automatic GPU split with the same
85,000 prompt IDs, source GGUF, greedy sampling, Q8 KV, maximum context 180,000 (backend rounded
to 180,224), and max-three/min-zero MTP. It measured 35.4977 tok/s in one run. LM Studio's
single-run result is a diagnostic, not a repeated comparison.
Earlier user-reported 45/57 tok/s figures lacked complete workload metadata and are not used as
measured acceptance results.

### Q4_K_M maximum-context capacity sweep

This earlier Q4_K_M sweep changes only the configured maximum context. It uses a 512-token code prompt,
a 256-token decode window, greedy sampling, Q8/INT8 KV and MTP3. Each cell is the mean ± sample
standard deviation of three measured runs after one warmup; prompt occupancy is only 512 tokens.
NInfer uses TP2 and its optimized draft head. LM Studio CUDA 2.33.0 uses automatic two-GPU
splitting with maximum-three/minimum-zero drafts.

| Maximum context | NInfer decode tok/s | LM Studio decode tok/s |
|---:|---:|---:|
| 1,024 | 60.06 ± 0.02 | 62.67 ± 0.24 |
| 2,048 | 60.12 ± 0.02 | 62.78 ± 0.05 |
| 4,096 | 60.11 ± 0.03 | 62.69 ± 0.10 |
| 8,192 | 60.13 ± 0.06 | 62.78 ± 0.07 |
| 16,384 | 60.14 ± 0.04 | 62.63 ± 0.06 |
| 32,768 | 60.16 ± 0.05 | 62.58 ± 0.004 |
| 65,536 | 60.06 ± 0.05 | 62.49 ± 0.16 |

Both engines honored all seven capacities. The output IDs were identical within each engine and all
windows were EOS/EOG-free. With this short prompt neither engine slowed materially as the capacity
increased; LM Studio was about 4% faster. This table is a capacity-setting comparison, not an
85K-filled-context benchmark.

### NVFP4 at 512-token input

The corrected NVFP4 implementation uses a 512-token code prompt and 256 timed decode tokens, with one
warmup and three measured requests per capacity. TP2, INT8 KV, greedy MTP3, optimized draft head,
CUDA Graphs and 1,024-token prefill chunks are fixed. Decode rates are mean ± sample standard deviation;
decode uses first-token-to-completion wall time.

| Maximum context | NVFP4 decode tok/s | Prefill tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 8,192 | 97.65 ± 0.08 | 1,369.9 | 70.04% |
| 16,384 | 97.67 ± 0.09 | 1,370.9 | 70.04% |
| 32,768 | 97.76 ± 0.11 | 1,370.8 | 70.04% |
| 65,536 | 97.55 ± 0.15 | 1,369.9 | 70.04% |

All four requested capacities were allocated exactly. All 12 measured runs produced the same
257 output IDs, without EOS/EOG; each accepted 173/247 drafts over 83 rounds. These are short-input
capacity checks, not 8K–64K occupied-context decode. The LM Studio table above uses Q4_K_M;
it is not a matched NVFP4 comparison.

### NVFP4 at 85K occupied context

Qwen3.8-27B NVFP4, exactly **85,000 prompt tokens**, **180,000 capacity**, TP2, complete INT8
group-64 KV, greedy sampling, MTP3, optimized draft head, CUDA Graphs and 3,072-token prefill chunks:

| Implementation | Runs | Prefill tok/s | Committed decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| FP32 SwiGLU, shared attention scores/reducer weights, prepared GDN Q/K | 2 | **1,279.44 ± 2.94** | **78.36 ± 0.04** | **79.12%** |

Each run generates 513 tokens: one from prefill and **512 timed decode tokens**. The optimized
runs produce identical IDs to each other, with 360/455 accepted drafts over 152 rounds per run.
These exceed the requested 1,000 prefill / 70 committed decode tok/s targets at 85K occupancy.
The 512-token output limit is a throughput window, not a completed-program quality evaluation.
The complete attention operator passes its independent FP64 oracle, including all 24 heads and
four queries at 85K with a 180K envelope. This verifies the measured numerical and generation
behavior; it is not a general coding-quality evaluation.
The GDN path also passes the independent FP64 recurrence oracle for its outputs and final state,
including 3,072 tokens with the real TP2 head geometry and nonzero initial state.

Earlier NVFP4 results, including **78.90 tok/s**, used a wide SwiGLU path that rounded gate/up to
BF16 before SiLU and multiplication. Expanded independent FP64 tests exposed excessive error in
both FP8 and NVFP4 routes. Keeping those intermediates in FP32 fixes the failing tests, including
both TP2 shards at 3,072 tokens, without widening tolerances. The generated sequence changes;
the old figures are real measurements but are **not the current numerical baseline**. Measurements
taken with the earlier mismatched FP8 output-head path are also excluded. A faster experiment
that reassociated the FP32 sum changed this corpus's generated sequence; the delivered reducer
preserves the original order and introduces no additional quantization or approximate attention.
The 3,072-token chunk reserves 1.40 GiB of workspace per device; 4,096 does not fit this NVFP4
artifact together with 180K capacity on the measured host.

Reproduce the measurement with the local artifact and fixed corpus:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --corpus /tmp/v100-code-85000.ids -pg 85000,512 \
  --max-ctx 180000 --prefill-chunk 3072 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --tp 2 --devices 0,1 \
  -r 2 --warmup 0 --capture-generation -o json \
  --output-file profiles/bench/nvfp4_85k_fp32_swiglu.json
```

The measured request averages 1.57 ms of prompt preparation, 66.435 s of prefill and 6.534 s of
decode, or 72.974 s total. The same invocation loads the resident model once in 18.65 s, including
16.25 s of upload. These are raw token-ID inputs, so preparation does not include text tokenization.
The complete per-stage GPU breakdown, MTP verify/proposal costs, copy activity, timing gaps and
remaining optimization decisions are in [the performance ledger](docs/performance.md#nvfp4-full-request-ledger).

An earlier NVFP4 TP2 comparison against the `plus1998/Ninfer-V100-Duo` code path at a 3K prompt,
98,304 capacity and MTP3 measured 976 tok/s prefill / 69.22 tok/s decode in this fork versus
973.5 / 68.96 tok/s upstream. This is a short-input cross-check only, not an 85K result. The
upstream repository and its reported numbers should not be treated as measurements of this fork.

### DFlash2 experiment

On the fixed 85K code corpus at 98,304 capacity, DFlash3 measured **25.19 tok/s** with 81.21%
acceptance; CUDA Graph and eager runs produced identical 513-token IDs. DFlash7 measured
**20.52 tok/s** with 41.56% acceptance and entered repeated special-message output after token 444.
The matched MTP3 control measured **52.77 tok/s**. The 180K DFlash allocation is rejected at
startup because the five-layer draft graph reservation does not fit the remaining memory. The
rank-0 gather change preserves output IDs but did not materially improve end-to-end DFlash speed;
the draft forward and full Q4_K vocabulary head dominate proposal cost.

## Build and run

Requirements for this profile: 64-bit Linux, NVIDIA driver, CUDA 12.8, CMake 3.28+, C++20 host
compiler, Ninja, pkg-config, FFmpeg development libraries (`libavformat >= 60`, `libavcodec >= 60`,
`libavutil >= 58`, `libswscale >= 7`) and libcurl >= 7.85. Build dependencies locally when the
system versions do not meet those requirements:

```bash
tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j2
```

Convert the local LM Studio source model once (Python 3.11 with NumPy):

```bash
python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

Run a request with the V100X2 defaults (devices `0,1`, 180K capacity, 4K prefill chunks, INT8 KV,
MTP3 and optimized draft head):

```bash
tools/v100/ninfer-v100x2.sh \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 128 --greedy --no-thinking
```

Set `NINFER_V100X2_ARTIFACT`, `NINFER_V100X2_DEVICES`, `NINFER_V100X2_MAX_CONTEXT`,
`NINFER_V100X2_PREFILL_CHUNK`, `NINFER_V100X2_KV_DTYPE`, or
`NINFER_V100X2_DRAFT_TOKENS` to override the launcher defaults. The launcher does not constrain
host CPU affinity; benchmark CPU use should remain below the operator's 85% ceiling.

The fixed 85K token corpus used by the measurements is local at `/tmp/v100-code-85000.ids` and is
not included in the repository. See [performance methodology](docs/performance.md) for benchmark
commands and additional qualifications.

## Verification

The CUDA 12.8 `sm_70` build was completed with `-j4`. The all-reduce integration test passed,
including exact cross-device byte probes and the 500-iteration 10 KiB microbenchmark. The focused
`nvfp4_a16`, `fp8_a16`, `swiglu_nvfp4`, and `swiglu_fp8` suites passed. Both SwiGLU suites check
full and TP2 shard outputs directly against the same FP64 mathematical oracle, including wide
prefill and caller-workspace sizing. FP8 A8 and NVFP4 A4 suites intentionally
report unsupported on V100 because those tensor-core routes require newer architectures; they are
not part of the V100X2 execution profile.

The decode update also passes `ninfer_gqa_attention_test` and `ninfer_attention_headlocal_test`,
covering BF16/INT8 caches, TP1/TP2, masks, fragmented pages, cache mutation and the full 85K FP64
oracle. FP8 regression cases cover native and prepacked full/TP2 vocabulary heads, single-token
calls and chunk tails. Both FP8 and NVFP4 prefill tests check native/prepacked real matrix shapes
at T=128/1024/4096 against the original represented weights. Q5 LinearAdd passes the FP64 oracle
at both TP2 row-shard extents, including its interior workspace maximum; BF16 Linear passes at
both row and column shards. The two-card LinearAdd and SwiGLU composition suites pass as well.
CLI, server and benchmark are rebuilt with these changes.

## Acknowledgements and license

V100X2 implementation work builds on [Neroued/ninfer](https://github.com/Neroued/ninfer), the
RTX 3060 TP2 fork, and [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).
Volta NVFP4 and TP2 behavior was cross-checked against
[plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo). The prefill investigation
consulted [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM). DFlash2 follows
[Inco AI's implementation](https://inco.ai/blog/dflash2/) and its
[Qwen3.8-27B draft checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). KVMem was
reviewed but not integrated: selective-history retrieval changes attention semantics, while this
profile retains full-context attention.

The measured models derive from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
The NVFP4 artifact uses the mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), packaged by
[Neroued](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).

NInfer and this fork are licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for required
attribution; third-party dependencies retain their own license files.
