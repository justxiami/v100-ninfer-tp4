"""Single-source recipe for the Qwen3.8-27B W4A4W8A8 artifact.

One ModelOpt-style checkpoint plays both source roles:

* the quantized role carries the FP8 and NVFP4 matrices with ModelOpt field
  names (``weight`` / ``weight_scale`` / ``weight_scale_2`` / ``input_scale``),
  where FP8 weight scales are per-tensor FP32 scalars and must be expanded to
  the artifact's per-row BF16 plane;
* the official role carries the retained BF16 control tensors (embedding,
  norms, GDN control, vision, MTP, lm_head), byte-identical to the author's
  pre-quantization BF16, from which the FP8 embedding/output heads are
  encoded exactly like the closed NVFP4 converter does.

The per-matrix MLP schedule (91 NVFP4 / 167 FP8, split independently per
gate_up and down) comes from inventory_efficientthink; object names and
shapes are unchanged from the registered Qwen3.8 profile.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable

import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import inventory_efficientthink as inventory
from . import recipe_nvfp4 as stock


BASE_REPOSITORY = (
    "Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4"
)
BASE_REVISION = "W4A4+W8A8 single-source (ModelOpt field layout)"

RowRange = stock.RowRange
MatrixSource = stock.MatrixSource
MatrixPart = stock.MatrixPart
Fp8WeightRecipe = stock.Fp8WeightRecipe
Nvfp4WeightRecipe = stock.Nvfp4WeightRecipe
InputDivisorRecipe = stock.InputDivisorRecipe


def _source(name: str, n: int, k: int) -> MatrixSource:
    return stock._source(name, n, k)


def _all(source: MatrixSource) -> MatrixPart:
    return stock._all(source)


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    return stock._q_part(source, gate)


def _build_quantized_matrix_recipes() -> tuple[
    tuple[Fp8WeightRecipe, ...],
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputDivisorRecipe, ...],
    tuple[tuple[MatrixSource, ...], ...],
]:
    fp8_weights: list[Fp8WeightRecipe] = []
    nvfp4_weights: list[Nvfp4WeightRecipe] = []
    input_divisors: list[InputDivisorRecipe] = []
    divisor_groups: list[tuple[MatrixSource, ...]] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        (
                            _q_part(query, False),
                            _all(key),
                            _q_part(query, True),
                            _all(value),
                        ),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "attention/output",
                        output.shape,
                        (_all(output),),
                    ),
                )
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(
                source_prefix + "linear_attn.out_proj", 5120, 6144
            )
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        (16384, 5120),
                        (_all(query_key_value), _all(z)),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "gdn/output",
                        output.shape,
                        (_all(output),),
                    ),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        # gate/up and down are quantized independently in the source
        # checkpoint: seven middle layers keep gate/up at NVFP4 while down
        # stays FP8, so each object branches on its own matrix schedule.
        if layer in inventory.NVFP4_GATE_UP_LAYERS:
            gate_up_sources = (gate, up)
            divisor_groups.append(gate_up_sources)
            nvfp4_weights.append(
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/gate_up",
                    (34816, 5120),
                    (_all(gate), _all(up)),
                    gate_up_sources,
                )
            )
            input_divisors.append(
                InputDivisorRecipe(
                    object_prefix
                    + "mlp/gate_up_projection/input_scale_divisor",
                    gate_up_sources,
                    (object_prefix + "mlp/gate_up",),
                )
            )
        else:
            fp8_weights.append(
                Fp8WeightRecipe(
                    object_prefix + "mlp/gate_up",
                    (34816, 5120),
                    (_all(gate), _all(up)),
                )
            )
        if layer in inventory.NVFP4_DOWN_LAYERS:
            nvfp4_weights.append(
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/down",
                    down.shape,
                    (_all(down),),
                    (down,),
                )
            )
            input_divisors.append(
                InputDivisorRecipe(
                    object_prefix + "mlp/down_projection/input_scale_divisor",
                    (down,),
                    (object_prefix + "mlp/down",),
                )
            )
        else:
            fp8_weights.append(
                Fp8WeightRecipe(
                    object_prefix + "mlp/down",
                    down.shape,
                    (_all(down),),
                )
            )

    return (
        tuple(fp8_weights),
        tuple(nvfp4_weights),
        tuple(input_divisors),
        tuple(divisor_groups),
    )


(
    FP8_WEIGHT_RECIPES,
    NVFP4_WEIGHT_RECIPES,
    INPUT_DIVISOR_RECIPES,
    WEIGHT_DIVISOR_GROUPS,
) = _build_quantized_matrix_recipes()
FP8_WEIGHTS_BY_NAME = {item.object_name: item for item in FP8_WEIGHT_RECIPES}
NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in NVFP4_WEIGHT_RECIPES
}
INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in INPUT_DIVISOR_RECIPES
}

FP8_SOURCES = tuple(
    dict.fromkeys(
        part.source for recipe in FP8_WEIGHT_RECIPES for part in recipe.parts
    )
)
NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source
        for recipe in NVFP4_WEIGHT_RECIPES
        for part in recipe.parts
    )
)

# The retained BF16/FP32 control tensors are materialized by the same direct
# recipes as the closed converter (identical object names and source names).
QUANTIZED_DIRECT_RECIPES = stock.QUANTIZED_DIRECT_RECIPES
QUANTIZED_DIRECT_BY_NAME = stock.QUANTIZED_DIRECT_BY_NAME
QUANTIZED_DIRECT_SPECS = tuple(
    spec
    for spec in inventory.TEXT_CORE_TENSOR_SPECS
    if spec.name in QUANTIZED_DIRECT_BY_NAME
)

OFFICIAL_TENSOR_SPECS = stock.OFFICIAL_TENSOR_SPECS
OFFICIAL_RECIPES = stock.OFFICIAL_RECIPES
OFFICIAL_RECIPES_BY_NAME = stock.OFFICIAL_RECIPES_BY_NAME
OFFICIAL_EMBEDDING_SOURCE = stock.OFFICIAL_EMBEDDING_SOURCE
# output_head is NOT an FP8 matrix source here: the single-source checkpoint
# keeps lm_head in BF16, so the head is encoded from BF16 exactly like the
# token embedding (see convert_efficientthink).
OUTPUT_HEAD_SOURCE_NAME = "lm_head.weight"


def _merge_requirement(
    result: dict[str, tuple[tuple[int, ...], str]],
    name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    signature = (shape, dtype)
    previous = result.setdefault(name, signature)
    if previous != signature:
        raise ValueError(f"inconsistent source declaration for {name}")


def _source_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in FP8_SOURCES:
        n, k = source.shape
        _merge_requirement(result, source.field("weight"), (n, k), "F8_E4M3")
        _merge_requirement(result, source.field("weight_scale"), (), "F32")
        _merge_requirement(result, source.field("input_scale"), (), "F32")
    for source in NVFP4_SOURCES:
        n, k = source.shape
        _merge_requirement(result, source.field("weight"), (n, k // 2), "U8")
        _merge_requirement(
            result, source.field("weight_scale"), (n, k // 16), "F8_E4M3"
        )
        _merge_requirement(
            result, source.field("weight_scale_2"), (), "F32"
        )
        _merge_requirement(result, source.field("input_scale"), (), "F32")
    for source in family_recipe.source_requirements(
        QUANTIZED_DIRECT_RECIPES
    ).values():
        _merge_requirement(result, source.name, source.shape, source.dtype)
    return result


SOURCE_REQUIREMENTS = _source_requirements()
EXPECTED_QUANTIZED_FIELDS = frozenset(
    name
    for name, (_, dtype) in SOURCE_REQUIREMENTS.items()
    if dtype in ("F8_E4M3", "F32", "U8")
)


def preflight_quantized_metadata(
    reader: ShardReader,
) -> family_recipe.SourcePreflight:
    missing = set(SOURCE_REQUIREMENTS).difference(reader.names)
    if missing:
        raise ValueError(f"source is missing {sorted(missing)[0]}")

    metadata = reader.metadata(reader.names)
    actual_quantized_fields = frozenset(
        name
        for name, item in metadata.items()
        if item.dtype in ("F8_E4M3", "F32", "U8")
    )
    if actual_quantized_fields != EXPECTED_QUANTIZED_FIELDS:
        unexpected = actual_quantized_fields.difference(
            EXPECTED_QUANTIZED_FIELDS
        )
        missing_fields = EXPECTED_QUANTIZED_FIELDS.difference(
            actual_quantized_fields
        )
        detail = (
            sorted(unexpected)[0]
            if unexpected
            else sorted(missing_fields)[0]
        )
        raise ValueError(f"quantized field allocation is not closed: {detail}")

    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, (shape, dtype) in SOURCE_REQUIREMENTS.items():
        actual = metadata[name]
        if actual.shape != shape or actual.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        shards.add(actual.shard)
    return family_recipe.SourcePreflight(
        recipe_count=(
            len(FP8_WEIGHT_RECIPES)
            + len(NVFP4_WEIGHT_RECIPES)
            + len(INPUT_DIVISOR_RECIPES)
            + len(QUANTIZED_DIRECT_RECIPES)
        ),
        source_tensor_count=len(SOURCE_REQUIREMENTS),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def preflight_official_sources(
    reader: ShardReader,
) -> family_recipe.SourcePreflight:
    return family_recipe.preflight_source_reader(reader, OFFICIAL_RECIPES)


def _word(tensor: torch.Tensor, name: str) -> int:
    return stock._word(tensor, name)


def _same_divisor(
    reader: ShardReader,
    sources: Iterable[MatrixSource],
    suffix: str,
) -> int:
    return stock._same_divisor(reader, sources, suffix)


def _select_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    return stock._select_rows(tensor, part)


def _engine_divisor(
    reader: ShardReader,
    sources: Iterable[MatrixSource],
    suffix: str,
) -> bytes:
    """Translate one ModelOpt global multiplier into the engine's divisor word.

    The checkpoint's `weight_scale_2` / `input_scale` are MULTIPLIERS: the reference
    dequantization is `codes * block_scale * scale`, which the source's own magnitudes confirm
    (`6 * max(weight_scale) * weight_scale_2` lands at 0.19 for an MLP gate, while dividing lands
    near 4e7). The artifact's fields are DIVISORS -- `ninfer::ops` dequantizes NVFP4 as
    `codes * e4m3_scale / weight_scale_divisor` (see tests/ops/quantized_weight.h and the Volta
    QPN prepack, which passes `1 / weight_scale_divisor` to its dequant kernel) and scales
    activations by `input_scale_divisor` (nvfp4_codec.cuh) -- so the stored value is the
    reciprocal of the source's. Storing the multiplier verbatim inverts the effective weight
    scale by ~1e4 and the model degenerates without any shape error.

    The stock qwen3_8_27b recipe needs no such conversion because its source stores NVIDIA's
    `weight_global_scale` / `input_global_scale`, which are already divisors.
    """
    items = tuple(sources)
    values: list[float] = []
    for source in items:
        tensor = reader.get(source.field(suffix))
        if tensor.dtype != torch.float32 or tensor.numel() != 1:
            raise ValueError(f"{source.field(suffix)}: expected one FP32 word")
        value = float(tensor.reshape(-1)[0].item())
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{source.field(suffix)}: global multiplier must be finite and positive"
            )
        values.append(value)
    if len(set(values)) != 1:
        raise ValueError(f"{items[0].name}: fused {suffix} multipliers differ")
    divisor = 1.0 / values[0]
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(f"{items[0].name}: {suffix} reciprocal is not a positive FP32")
    return struct.pack("<f", divisor)


def materialize_fp8_weight(
    recipe: Fp8WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble exact FP8 codes plus the per-row BF16 scale plane.

    The source stores one per-tensor FP32 weight scale per matrix; the
    artifact wants a per-row BF16 plane, so the scalar is cast to BF16 and
    replicated to every row (at most one rounding step of the scale).
    """
    code_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    source_words: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe.parts:
        words = source_words.get(part.source)
        if words is None:
            source_codes = reader.get(part.source.field("weight"))
            source_scale = reader.get(part.source.field("weight_scale"))
            if (
                source_codes.dtype != torch.float8_e4m3fn
                or tuple(source_codes.shape) != part.source.shape
                or source_scale.dtype != torch.float32
                or source_scale.numel() != 1
            ):
                raise ValueError(
                    f"{part.source.name}: materialized FP8 source signature "
                    "mismatch"
                )
            row_scales = (
                source_scale.reshape(-1).to(torch.bfloat16).expand(part.source.shape[0])
            )
            words = (source_codes.view(torch.uint8), row_scales)
            source_words[part.source] = words
        code_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_rows(words[1], part))
    codes = (
        code_parts[0].contiguous()
        if len(code_parts) == 1
        else torch.cat(code_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    if tuple(codes.shape) != recipe.shape or tuple(scales.shape) != (
        recipe.shape[0],
    ):
        raise ValueError(f"{recipe.object_name}: materialized FP8 shape mismatch")
    return codes, scales


def materialize_nvfp4_weight(
    recipe: Nvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    source_words: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe.parts:
        words = source_words.get(part.source)
        if words is None:
            n, k = part.source.shape
            source_packed = reader.get(part.source.field("weight"))
            source_scales = reader.get(part.source.field("weight_scale"))
            if (
                source_packed.dtype != torch.uint8
                or tuple(source_packed.shape) != (n, k // 2)
                or source_scales.dtype != torch.float8_e4m3fn
                or tuple(source_scales.shape) != (n, k // 16)
            ):
                raise ValueError(
                    f"{part.source.name}: materialized NVFP4 source signature "
                    "mismatch"
                )
            words = (source_packed, source_scales.view(torch.uint8))
            source_words[part.source] = words
        packed_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_rows(words[1], part))
    packed = (
        packed_parts[0].contiguous()
        if len(packed_parts) == 1
        else torch.cat(packed_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    divisor = _engine_divisor(reader, recipe.divisor_sources, "weight_scale_2")
    if tuple(packed.shape) != (recipe.shape[0], recipe.shape[1] // 2) or tuple(
        scales.shape
    ) != (recipe.shape[0], recipe.shape[1] // 16):
        raise ValueError(
            f"{recipe.object_name}: materialized NVFP4 shape mismatch"
        )
    return packed, scales, divisor


def materialize_input_divisor(
    recipe: InputDivisorRecipe,
    reader: ShardReader,
) -> torch.Tensor:
    divisor = _engine_divisor(reader, recipe.sources, "input_scale")
    return torch.frombuffer(
        bytearray(divisor), dtype=torch.float32
    ).reshape(())


def materialize_quantized_direct(
    object_name: str,
    reader: ShardReader,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        QUANTIZED_DIRECT_BY_NAME[object_name], reader
    )


def materialize_official(
    object_name: str,
    reader: ShardReader,
    derived_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        OFFICIAL_RECIPES_BY_NAME[object_name], reader, derived_tensors
    )


def validate_recipe() -> None:
    family_recipe.validate_recipe_coverage(
        QUANTIZED_DIRECT_RECIPES, QUANTIZED_DIRECT_SPECS
    )
    family_recipe.validate_recipe_coverage(
        OFFICIAL_RECIPES, OFFICIAL_TENSOR_SPECS
    )
    if (
        len(FP8_WEIGHT_RECIPES),
        len(NVFP4_WEIGHT_RECIPES),
        len(INPUT_DIVISOR_RECIPES),
        len(WEIGHT_DIVISOR_GROUPS),
        len(FP8_SOURCES),
        len(NVFP4_SOURCES),
        len(QUANTIZED_DIRECT_RECIPES),
        len(OFFICIAL_RECIPES),
    ) != (165, 91, 91, 49, 260, 140, 401, 348):
        raise ValueError("W4A4W8A8 source recipe is incomplete")
    ownership = (
        {"text/token_embedding", "text/output_head"},
        set(FP8_WEIGHTS_BY_NAME),
        set(NVFP4_WEIGHTS_BY_NAME),
        set(INPUT_DIVISORS_BY_NAME),
        set(QUANTIZED_DIRECT_BY_NAME),
        set(OFFICIAL_RECIPES_BY_NAME).difference(
            {"text/token_embedding"}
        ),
    )
    all_names: set[str] = set()
    for names in ownership:
        if all_names.intersection(names):
            raise ValueError("more than one source route owns an artifact tensor")
        all_names.update(names)
    if all_names != {spec.name for spec in inventory.TENSOR_SPECS}:
        raise ValueError("source routes do not cover the complete tensor inventory")
    if tuple(FP8_WEIGHTS_BY_NAME) != tuple(
        spec.name
        for spec in inventory.FP8_TENSOR_SPECS
        if spec.name not in ("text/token_embedding", "text/output_head")
    ):
        raise ValueError("FP8 recipe order does not match inventory")
    if tuple(NVFP4_WEIGHTS_BY_NAME) != tuple(
        spec.name for spec in inventory.NVFP4_TENSOR_SPECS
    ):
        raise ValueError("NVFP4 recipe order does not match inventory")
    if tuple(INPUT_DIVISORS_BY_NAME) != tuple(
        spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS
    ):
        raise ValueError("input-divisor recipe order does not match inventory")
    for recipe in FP8_WEIGHT_RECIPES:
        stock._validate_matrix_recipe(recipe.object_name, recipe.shape, recipe.parts)
    for recipe in NVFP4_WEIGHT_RECIPES:
        stock._validate_matrix_recipe(recipe.object_name, recipe.shape, recipe.parts)
    bound_weights = tuple(
        name for site in INPUT_DIVISOR_RECIPES for name in site.weight_names
    )
    if (
        len(bound_weights) != 91
        or len(set(bound_weights)) != 91
        or set(bound_weights) != set(NVFP4_WEIGHTS_BY_NAME)
    ):
        raise ValueError("input-divisor sites do not cover NVFP4 parents once")


validate_recipe()


__all__ = [
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "FP8_SOURCES",
    "FP8_WEIGHT_RECIPES",
    "FP8_WEIGHTS_BY_NAME",
    "INPUT_DIVISOR_RECIPES",
    "INPUT_DIVISORS_BY_NAME",
    "NVFP4_SOURCES",
    "NVFP4_WEIGHT_RECIPES",
    "NVFP4_WEIGHTS_BY_NAME",
    "OFFICIAL_EMBEDDING_SOURCE",
    "OFFICIAL_RECIPES",
    "OFFICIAL_RECIPES_BY_NAME",
    "OFFICIAL_TENSOR_SPECS",
    "OUTPUT_HEAD_SOURCE_NAME",
    "QUANTIZED_DIRECT_BY_NAME",
    "QUANTIZED_DIRECT_RECIPES",
    "QUANTIZED_DIRECT_SPECS",
    "SOURCE_REQUIREMENTS",
    "WEIGHT_DIVISOR_GROUPS",
    "materialize_fp8_weight",
    "materialize_input_divisor",
    "materialize_nvfp4_weight",
    "materialize_official",
    "materialize_quantized_direct",
    "preflight_official_sources",
    "preflight_quantized_metadata",
    "validate_recipe",
]
