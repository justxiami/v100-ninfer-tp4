# NInfer V100 tp4 — 4-way tensor parallel for 4 × V100 (SM70)

[中文文档](README.zh-CN.md) · [Build guide](docs/tp4/README.md) · [Technical history](docs/tp4/changelog.md)

Single-instance Qwen3.8-27B inference across **4 × Tesla V100-SXM2 16 GB**
(`sm_70` / Volta, CUDA 12.8) via 4-way tensor parallel. **Multimodal (image) since
2026-10-05**; the text-only path is the byte-identical regression gate.

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

Technical history (W1→W6): [`docs/tp4/changelog.md`](docs/tp4/changelog.md) ·
MTP root-cause analysis: [`docs/tp4/README_tp4_mtp_rca_2026-10-01.md`](docs/tp4/README_tp4_mtp_rca_2026-10-01.md) ·
tokenizer report: [`docs/tp4/tokenizer-oklogk.md`](docs/tp4/tokenizer-oklogk.md) ·
vision on tp4 (2026-10-05): [`docs/tp4/vision-tp4-2026-10-05.md`](docs/tp4/vision-tp4-2026-10-05.md)

## Measurements (this host, completed tp4)

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
| MTP draft acceptance — same run | 30.8–34.1% | **47.5%** |
| Prefill — 17.4k prompt (needle test, retrieval hit ✓) | — | **3,372.8 tok/s** |
| Prefill — 5.8k prompt | — | 3,494.6 tok/s |
| Short-answer decode — 96 tokens | — | 124–140 tok/s (acceptance 56–66%) |
| VRAM per GPU (weights + 128k int8 KV + graphs) | 15.2 GiB | **9.39 GiB** |

The 4-way split halves per-GPU weights and KV: 6.6 GiB stays free per card at startup
(6.9 GiB slack), so the artifact's native 262,144-token capacity also fits — the 128k
default is kept on purpose, for parity with the old service.

### Prefill ladder — context scaling (tp4, this host)

Full cold prefill sweep, 3 reps per context, session salt so no rep hits the prefix cache
(authoritative numbers = engine `done` line, req-baseline matched so each sample is
uniquely attributed; the sweep ran with no other active request source). The 8k and 16k
points reproduce the earlier 5.8k / 17.4k single-shot numbers, confirming stability.

| context | prefill tok/s | time-to-first-token | greedy 300-token decode (t=0) |
|---:|---:|---:|---:|
| 8k | 3,470 | 2.4 s | 143 tok/s |
| 16k | 3,357 | 4.8 s | 134 tok/s |
| 32k | 3,118 | 10.3 s | 141 tok/s |
| 64k | 2,724 | 23.7 s | 117 tok/s |
| 120k | 2,229 | 54.1 s | 96 tok/s |

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

### Vision (added 2026-10-05)

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

## Model artifact provenance

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

## Notable files added by this fork

```
src/core/tp_comm.{h,cu}                  NCCL transport backend
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

## License

Apache-2.0 ([LICENSE](LICENSE), required attribution in [NOTICE](NOTICE)).
Upstream chain: tuxKOH/ninfer-V100X2 ← geoffwatts/ninfer-v100 ← Neroued/ninfer.
The measured model is the Merkyor EfficientThink-K3 `W4A4+W8A8` SFT variant, based on
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); see
[Model artifact provenance](#model-artifact-provenance).