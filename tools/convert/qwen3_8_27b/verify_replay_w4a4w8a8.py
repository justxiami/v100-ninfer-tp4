"""Replay verifier for qwen3_8_27b_w4a4w8a8.ninfer (9/30, new tree V100X2-remote).

Recomputes every object payload from the source directory with the same
sha-verified 9/20 converter code (byte-identical, see SHA256SUMS in
~/backups/2026-09-20-ninfer-w4a4-switch/) and compares byte-for-byte with the
on-disk artifact payload. Catches write corruption (9/20 NaN incident class)
and any environment drift; systematic mapping errors were already validated
on 9/20 with the same scripts and the same source layout (preflight-pinned
140 NVFP4 / 260 FP8).

Usage (from tree root, 1cat-vllm-sm70 env):
  python3 tools/convert/qwen3_8_27b/verify_replay_w4a4w8a8.py \
      --model ~/models/EfficientThink-K3-W4A4-W8A8 \
      --artifact ~/models/ninfer-V100X2/qwen3_8_27b_w4a4w8a8.ninfer
"""

from __future__ import annotations

import argparse
import hashlib
import time

import torch

from tools.artifact import layouts as layouts_mod
from tools.artifact.container import Artifact
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_8_27b import convert_efficientthink as ce
from tools.convert.qwen3_8_27b import fp8_embedding
from tools.convert.qwen3_8_27b import inventory_efficientthink as inventory
from tools.convert.qwen3_8_27b import recipe_efficientthink as recipe


def replay_payload(spec, reader, resources, derived):
    """Recompute one object's payload exactly as convert() does (minus write)."""
    from tools.artifact.layouts import encode_direct

    if isinstance(spec, inventory.ResourceSpec):
        return bytes(resources[spec.name])
    if spec.name == "text/token_embedding":
        return b"".join(
            fp8_embedding.iter_reader_payload(
                reader, recipe.OFFICIAL_EMBEDDING_SOURCE.name, spec.shape
            )
        )
    if spec.name == "text/output_head":
        return b"".join(
            fp8_embedding.iter_reader_payload(
                reader, recipe.OUTPUT_HEAD_SOURCE_NAME, spec.shape
            )
        )
    if spec.name in recipe.FP8_WEIGHTS_BY_NAME:
        return ce._encode_fp8_weight(spec, reader)
    if spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
        return ce._encode_nvfp4_weight(spec, reader)
    if spec.name in recipe.INPUT_DIVISORS_BY_NAME:
        scalar = recipe.materialize_input_divisor(
            recipe.INPUT_DIVISORS_BY_NAME[spec.name], reader
        )
        return encode_direct(scalar, inventory.FP32)
    if spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
        tensor = ce._materialize_direct(spec, reader)
        return encode_direct(tensor, spec.format)
    tensor = ce._materialize_official(spec, reader, derived)
    return family_conversion.encode_tensor_payload(tensor, spec, "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    preflight = ce.preflight_conversion(args.model)
    resources = {r.name: r.data for r in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}

    specs = list(inventory.OBJECT_SPECS)
    print(f"objects: {len(specs)}", flush=True)

    failures = []
    checked = 0
    with ShardReader(preflight.model_dir) as reader:
        with Artifact.open(args.artifact) as artifact:
            for index, spec in enumerate(specs, start=1):
                expected = replay_payload(spec, reader, resources, derived)
                actual = bytes(artifact.payload(spec.name))
                checked += 1
                if len(expected) != len(actual) or expected != actual:
                    failures.append(
                        f"{spec.name}: replay {len(expected)}B vs artifact {len(actual)}B"
                    )
                    print(f"[{index}/{len(specs)}] MISMATCH {spec.name}", flush=True)
                elif index % 200 == 0 or index == len(specs):
                    print(f"[{index}/{len(specs)}] ok", flush=True)
                del expected

    elapsed = time.perf_counter() - t0
    print(f"checked {checked} objects in {elapsed:.1f}s")
    if failures:
        print(f"FAIL: {len(failures)} objects differ")
        for line in failures[:20]:
            print("  " + line)
        raise SystemExit(1)
    print("PASS: all objects byte-identical to replay")


if __name__ == "__main__":
    main()