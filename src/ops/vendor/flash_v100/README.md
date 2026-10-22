# flash_v100 (vendored, BSD-3)

`flash_fwd_d256_splitd_sm70.cu` and its headers are vendored from
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)'s Flash-V100 (SM70) Split-D attention
kernel, Copyright (c) 2026, 1CatAI, BSD-3-Clause (see the file header).

Deviation: the ATen host wrapper (`sm70_d256_splitd_dense_fwd`) and the
`<ATen/...>`/`<c10/...>` includes are stripped -- ninfer supplies its own raw-pointer
launcher (`src/ops/launcher/gqa_attention_splitd_sm70.cu`). `utils.h` gains an
`#include <optional>`. Nothing in the device kernel changed.

Measured on 4x V100-SXM2, head_dim 256, 6 q heads / 1 kv head, causal, T=4096 chunk
(bench-results/2026-10-22/attn-ceiling, `1cat-splitd`):
  8k ctx 37.1 TF | 32k 40.5 TF | 120k 46.9 TF  (vendored llama.cpp: 35.5 TF)
