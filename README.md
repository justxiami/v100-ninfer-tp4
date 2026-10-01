# NInfer V100 tp4 — 4-way tensor parallel for 4 × V100 (SM70)

Single-instance Qwen3.8-27B inference across **4 × Tesla V100-SXM2 16 GB**
(`sm_70` / Volta, CUDA 12.8) via 4-way tensor parallel.

This is a fork of [tuxKOH/ninfer-V100X2](https://github.com/tuxKOH/ninfer-V100X2)
at baseline commit `c6100422` (2026-09-29). That baseline is the tuned **2-GPU (TP2)**
fork in the [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) /
[Neroued/ninfer](https://github.com/Neroued/ninfer) chain. Everything in this tree keeps
TP2 behavior byte-identical — it is the regression gate for every batch of changes —
and adds `--tp 4`, which runs one instance across all four GPUs.

## What this fork adds

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

Technical history (W1→W6): [`docs/tp4/changelog.md`](docs/tp4/changelog.md) ·
MTP root-cause analysis: [`docs/tp4/README_tp4_mtp_rca_2026-10-01.md`](docs/tp4/README_tp4_mtp_rca_2026-10-01.md) ·
tokenizer report: [`docs/tp4/tokenizer-oklogk.md`](docs/tp4/tokenizer-oklogk.md)

## Measurements (this host, completed tp4)

Reference config: Qwen3.8-27B W4A4/W8A8 (ModelOpt NVFP4+FP8 mixed) `.ninfer` artifact,
`--tp 4 --devices 0,1,2,3`, 131,072 max context, **int8 KV**, 4,096-token prefill chunks,
MTP with 3 drafts, CUDA graphs. Decode tok/s counts committed output tokens, not drafted
ones.

| Metric | tp2 (2-GPU baseline, same artifact) | **tp4** |
|---|---:|---:|
| Committed decode — 2,048-token thinking run (t=1.0 / top_p=0.95 / top_k=20) | 62.6 tok/s | **113.1 tok/s (1.81×)** |
| MTP draft acceptance — same run | 30.8–34.1% | **47.5%** |
| Prefill — 17.4k prompt (needle test, retrieval hit ✓) | — | **3,372.8 tok/s** |
| Prefill — 5.8k prompt | — | 3,494.6 tok/s |
| Short-answer decode — 96 tokens | — | 124–140 tok/s (acceptance 56–66%) |
| VRAM per GPU (weights + 128k int8 KV + graphs) | 15.2 GiB | **9.39 GiB** |

The 4-way split halves per-GPU weights and KV: 6.6 GiB stays free per card at startup
(6.9 GiB slack), so the artifact's native 262,144-token capacity also fits — the 128k
default is kept on purpose, for parity with the old service.

Smoke (all passed): 4 greedy probes semantically correct; 17.4k needle retrieval hit;
4-way concurrency (`--max-concurrency 4`) all answers correct.

Build guide, NCCL/CUTLASS dependencies and tests: [`docs/tp4/README.md`](docs/tp4/README.md).
Reference build graph and exact CMake option values: [`docs/tp4/build-config/`](docs/tp4/build-config/).
The baseline fork's README (detailed TP2 measurements, build/run/convert instructions):
[`docs/upstream-README-v100x2.md`](docs/upstream-README-v100x2.md).

## Notable files added by this fork

```
src/core/tp_comm.{h,cu}                  NCCL transport backend
src/ops/wrapper/shard_extent.h           runtime shard geometry
tests/ops/test_allreduce_nccl4.cpp       4-GPU collective test (opt-in; <4 GPUs → skip 77)
tests/ops/test_tp4_issue_probe.cpp       caller-duty regression probe
tools/convert/qwen3_8_27b/               W4A4W8A8 artifact conversion + replay verification
docs/tp4/                                build guide, history, RCA, reference build config, scripts
```

## License

Apache-2.0 ([LICENSE](LICENSE), required attribution in [NOTICE](NOTICE)).
Upstream chain: tuxKOH/ninfer-V100X2 ← geoffwatts/ninfer-v100 ← Neroued/ninfer.
The measured model derives from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).