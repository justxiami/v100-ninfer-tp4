"""Build the Qwen3.8-27B W4A4W8A8 artifact from its single source checkpoint.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_efficientthink \
      --model /home/lu/models/Qwen3.8-27B-EfficientThink-W4A4W8A8 \
      --out ~/models/ninfer-V100X2/qwen3_8_27b_w4a4w8a8.ninfer

The one ModelOpt-style checkpoint plays both source roles: the quantized
matrices (ModelOpt field names) and the retained BF16 control tensors
(embedding, norms, GDN control, vision, MTP, lm_head) live in the same
shards. The per-layer MLP schedule (49 NVFP4 / 15 FP8) is carried by the
artifact itself and dispatched by the engine at bind time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import convert_nvfp4 as stock_convert
from . import fp8_embedding
from . import inventory_efficientthink as inventory
from . import recipe_efficientthink as recipe


RECIPE_ID = "qwen3_8_27b_w4a4w8a8-v1"
OUTPUT_BASENAME = "qwen3_8_27b_w4a4w8a8.ninfer"

# Five of the six frontend resources are byte-identical to the official
# Qwen3.8 profile; generation_config.json carries the author's SFT tuning and
# is pinned to this checkpoint's exact bytes.
RESOURCE_SHA256 = {
    "frontend/tokenizer.json": base_convert.OFFICIAL_RESOURCE_SHA256[
        "frontend/tokenizer.json"
    ],
    "frontend/tokenizer_config.json": base_convert.OFFICIAL_RESOURCE_SHA256[
        "frontend/tokenizer_config.json"
    ],
    "frontend/chat_template.jinja": base_convert.OFFICIAL_RESOURCE_SHA256[
        "frontend/chat_template.jinja"
    ],
    "frontend/generation_config.json": (
        "b8eb74d15e0a56623d00ccd14950a4bb87fabbf84b5cc030dcc904b899fb1eb5"
    ),
    "frontend/preprocessor_config.json": base_convert.OFFICIAL_RESOURCE_SHA256[
        "frontend/preprocessor_config.json"
    ],
    "frontend/video_preprocessor_config.json": base_convert.OFFICIAL_RESOURCE_SHA256[
        "frontend/video_preprocessor_config.json"
    ],
}


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: family_recipe.SourcePreflight
    quantized_source: family_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_source_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    summary = family_config.validate_config(config)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        raise ValueError("source config is missing quantization_config")
    family_conversion.check_members(
        "quantization_config",
        quantization,
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
        },
    )
    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping) or tuple(groups) != (
        "group_0",
        "group_1",
    ):
        raise ValueError(
            "source quantization_config must contain exactly group_0 then "
            "group_1"
        )
    float_group = groups["group_0"]
    nvfp4_group = groups["group_1"]
    if not isinstance(float_group, Mapping) or not isinstance(
        nvfp4_group, Mapping
    ):
        raise ValueError("source quantization_config groups must be objects")
    family_conversion.check_members(
        "quantization_config.config_groups.group_0.weights",
        float_group.get("weights", {}),
        {"num_bits": 8, "type": "float", "dynamic": False},
    )
    family_conversion.check_members(
        "quantization_config.config_groups.group_1.weights",
        nvfp4_group.get("weights", {}),
        {"num_bits": 4, "type": "float", "dynamic": False},
    )

    # The author's per-module table pins the exact quantization algorithm of
    # every quantized projection. Cross-check it against the artifact
    # schedule: any divergence means the source no longer matches this
    # converter's inventory and must fail loudly, not silently.
    table = quantization.get("quantized_layers")
    if not isinstance(table, Mapping) or not table:
        raise ValueError("quantization_config.quantized_layers is missing")

    def layer_of(module_name: str) -> int | None:
        parts = module_name.split(".")
        if (
            len(parts) == 6
            and parts[0] == "model"
            and parts[1] == "language_model"
            and parts[2] == "layers"
            and parts[3].isdigit()
            and parts[4] in ("self_attn", "linear_attn", "mlp")
        ):
            return int(parts[3])
        return None

    def suffix_of(module_name: str) -> str:
        return module_name.split(".")[-1]

    gate_up_nvfp4: set[int] = set()
    down_nvfp4: set[int] = set()
    for module_name, entry in table.items():
        layer = layer_of(module_name)
        if layer is None:
            raise ValueError(
                f"quantized_layers entry outside the text stack: {module_name}"
            )
        if not isinstance(entry, Mapping) or entry.get("quant_algo") not in (
            "FP8",
            "NVFP4",
        ):
            raise ValueError(
                f"quantized_layers[{module_name}]: unsupported quant_algo"
            )
        suffix = suffix_of(module_name)
        if suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"):
            if entry["quant_algo"] != "FP8":
                raise ValueError(
                    f"quantized_layers[{module_name}]: attention/GDN "
                    "projections must stay FP8"
                )
        elif suffix in ("gate_proj", "up_proj"):
            partner = table.get(
                module_name.replace(
                    suffix, "gate_proj" if suffix == "up_proj" else "up_proj"
                )
            )
            if not isinstance(partner, Mapping) or entry[
                "quant_algo"
            ] != partner.get("quant_algo"):
                raise ValueError(
                    f"quantized_layers[{module_name}]: gate and up of one "
                    "layer must share a quant_algo"
                )
            if entry["quant_algo"] == "NVFP4":
                gate_up_nvfp4.add(layer)
        elif suffix == "down_proj":
            if entry["quant_algo"] == "NVFP4":
                down_nvfp4.add(layer)
        else:
            raise ValueError(f"quantized_layers[{module_name}]: unknown module")

    if gate_up_nvfp4 != set(inventory.NVFP4_GATE_UP_LAYERS):
        raise ValueError(
            "quantized_layers gate/up schedule does not match the W4A4W8A8 "
            f"inventory: {sorted(gate_up_nvfp4 ^ set(inventory.NVFP4_GATE_UP_LAYERS))}"
        )
    if down_nvfp4 != set(inventory.NVFP4_DOWN_LAYERS):
        raise ValueError(
            "quantized_layers down schedule does not match the W4A4W8A8 "
            f"inventory: {sorted(down_nvfp4 ^ set(inventory.NVFP4_DOWN_LAYERS))}"
        )
    return summary


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def load_resources(model_dir: Path) -> tuple[family_conversion.ResourcePayload, ...]:
    expected_names = tuple(RESOURCE_SHA256)
    spec_names = tuple(spec.name for spec in inventory.RESOURCE_SPECS)
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the resource pins: "
            f"expected {expected_names!r}, got {spec_names!r}"
        )
    resources = family_conversion.load_resources(
        model_dir, inventory.RESOURCE_SPECS
    )
    actual_names = tuple(resource.name for resource in resources)
    if actual_names != expected_names:
        raise ValueError(
            "W4A4W8A8 frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for resource in resources:
        actual = hashlib.sha256(resource.data).hexdigest()
        expected = RESOURCE_SHA256[resource.name]
        if actual != expected:
            filename = resource.name.removeprefix("frontend/")
            raise ValueError(
                f"resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
    return resources


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    stock_convert._validate_index(model)

    config = family_conversion.load_json(model / "config.json")
    config_summary = _validate_source_config(config)
    preflight_inventory()

    with ShardReader(model) as official_reader:
        source = recipe.preflight_official_sources(official_reader)
    with ShardReader(model) as quantized_reader:
        quantized_source = recipe.preflight_quantized_metadata(quantized_reader)

    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        quantized_source=quantized_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _encode_fp8_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.FP8_WEIGHTS_BY_NAME[spec.name]
    codes, scales = recipe.materialize_fp8_weight(selected, reader)
    return encode_fp8_row_scaled(codes, scales, spec.shape)


def _encode_nvfp4_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_nvfp4_weight(selected, reader)
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def _materialize_direct(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> torch.Tensor:
    tensor = recipe.materialize_quantized_direct(spec.name, reader)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _materialize_official(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    tensor = recipe.materialize_official(spec.name, reader, dict(derived))
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.model_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "single_source": {
            "repository": recipe.BASE_REPOSITORY,
            "revision": recipe.BASE_REVISION,
            "model_path": str(preflight.model_dir.resolve()),
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["source_preflight"] = {
        "official_role": {
            "recipes": preflight.source.recipe_count,
            "tensors": preflight.source.source_tensor_count,
            "shards": preflight.source.source_shard_count,
            "dtypes": dict(preflight.source.source_dtype_counts),
        },
        "quantized_role": {
            "recipes": preflight.quantized_source.recipe_count,
            "tensors": preflight.quantized_source.source_tensor_count,
            "shards": preflight.quantized_source.source_shard_count,
            "dtypes": dict(preflight.quantized_source.source_dtype_counts),
            "source_fp8_matrices": len(recipe.FP8_SOURCES),
            "source_nvfp4_matrices": len(recipe.NVFP4_SOURCES),
        },
    }
    report["embedding_encoder"] = fp8_embedding.ENCODER_PROFILE
    return report


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the single-source W4A4W8A8 conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"W4A4W8A8 converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.FP8_SOURCES)} FP8 and "
        f"{len(recipe.NVFP4_SOURCES)} NVFP4 source matrices, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    with ShardReader(preflight.model_dir) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name == "text/token_embedding":
                    payload = fp8_embedding.iter_reader_payload(
                        reader,
                        recipe.OFFICIAL_EMBEDDING_SOURCE.name,
                        spec.shape,
                    )
                elif spec.name == "text/output_head":
                    payload = fp8_embedding.iter_reader_payload(
                        reader,
                        recipe.OUTPUT_HEAD_SOURCE_NAME,
                        spec.shape,
                    )
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = _encode_fp8_weight(spec, reader)
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = _encode_nvfp4_weight(spec, reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name],
                        reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = _materialize_direct(spec, reader)
                    payload = encode_direct(tensor, spec.format)
                    del tensor
                else:
                    tensor = _materialize_official(spec, reader, derived)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
