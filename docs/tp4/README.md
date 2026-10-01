# V100 (SM70) 4-way tensor parallel — build guide

This tree adds `--tp 4` (4-way tensor parallel) for NVIDIA V100 (SM70 / Volta) on top
of the `ninfer-V100X2` baseline (`c6100422`, 2026-09-29). Baseline behaviour
(`--tp 2`, two GPUs) is unchanged; `--tp 4` shards weights and activations across four
GPUs, uses NCCL as the collective-communication transport, and runs multi-rank CUDA
graphs in the decode loop. A tp4 instance occupies all four GPUs and cannot share them
with a second instance.

## Files in this directory

| File | Content |
|---|---|
| `README.md` | This build guide |
| `changelog.md` | Full technical history (W1 → W6): design decisions, failures, root causes |
| `README_tp4_mtp_rca_2026-10-01.md` | Root-cause analysis of the tp4 MTP (draft) acceptance problem and its fixes |
| `tokenizer-oklogk.md` | Tokenizer BPE O(k²) → O(k·log k) fix: root cause + semantic verification |
| `build-config/build.ninja` | Complete build graph + exact compiler flags of the reference build |
| `build-config/CMakeCache.txt` | Actual values of all CMake options of the reference build |
| `scripts/` | Regression gate + tp4 diagnostics: tp2 A/B serve gate, MTP diff/quality/repro, determinism, transport A/B (`tp4_transport_ab.cu`) |

> Paths inside `build-config/` are scrubbed: `$HOME` marks the original author's home
> directory. Adapt the NCCL / CUTLASS locations to your machine; the system toolchain
> paths, flags, and option values are as-is.

## Dependencies

- CUDA 12.8 toolchain (`nvcc`; reference build: `/usr/local/cuda-12.8/bin/nvcc`)
- CMake + Ninja
- C++20 host compiler (reference: gcc 13)
- **NCCL — hard dependency** (introduced by the tp4 transport backend, `src/core/tp_comm.{h,cu}`):
  point `-DNINFER_NCCL_ROOT=<dir>` at a directory containing `libnccl.so.2` (conda
  `nvidia/nccl` layout; note it ships *no* `libnccl.so` symlink, so the CMake discovery
  uses explicit candidate paths instead of `find_library`). Missing NCCL fails the
  configure deliberately — a silent fallback would push "tp>2 unsupported" to link time.
- CUTLASS v4.4.2: the reference build took it from a local source directory
  (`-DFETCHCONTENT_SOURCE_DIR_CUTLASS=<dir>`), i.e. no network access needed.

## Configure + build (reference)

```bash
cd <tree>
cmake -S . -B build-v100-tp4 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DNINFER_NCCL_ROOT=$HOME/.../site-packages/nvidia/nccl \
  -DFETCHCONTENT_SOURCE_DIR_CUTLASS=$HOME/.../cutlass-v4.4.2 \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=ON

ninja -C build-v100-tp4 ninfer-serve -j4        # full build ≈ 5–15 min
```

Device-code flags of the reference build (visible in `build-config/build.ninja`):
`-O3 -DNDEBUG -std=c++20 --generate-code=arch=compute_70,code=[compute_70,sm_70] -lineinfo`

⚠️ **After (re)configuring a fresh build directory, verify the CUDA release flags are
not empty**: `grep -m1 -- '-O3 -DNDEBUG' build-v100-tp4/build.ninja` must hit. An empty
`CMAKE_CUDA_FLAGS_RELEASE` (e.g. when a CMakeCache is copied between trees) silently
loses `-O3 -DNDEBUG` for device code → prefill ~2× slower.

## Tests

```bash
ninja -C build-v100-tp4 ninfer-serve ninfer_allreduce_nccl4_test ninfer_tp4_issue_probe -j4

./build-v100-tp4/tests/ninfer_allreduce_nccl4_test  # 4-GPU opt-in; <4 GPUs → exit 77 (skip)
./build-v100-tp4/tests/ninfer_tp4_issue_probe       # caller-duty regression probe (churn vs churn+devsync)
./build-v100-tp4/tests/ninfer_allreduce_test        # tp2 pull path (reference behaviour)
```

Per-batch TP2 regression gate (serve A/B; pass criteria include `graph-nodes=2562`):
`scripts/tp2_gate_serve_ab.sh`.

## tp4 serve (smoke)

With four free V100s: `ninfer-serve --tp 4 --devices 0,1,2,3`. Reference runs used int8
KV and 131072 max context; exact reference commands and acceptance data are in
`changelog.md` (W6) and `scripts/`.