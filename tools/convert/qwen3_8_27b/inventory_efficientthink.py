"""Persistent-object contract for the Qwen3.8-27B W4A4W8A8 (mixed-precision) artifact.

Identical to inventory_nvfp4 in every way except the per-MATRIX MLP numeric
format: the W4A4W8A8 source checkpoint keeps 91 MLP matrices at NVFP4
(49 gate_up + 42 down) and 167 at row-scaled FP8 (15 gate_up + 22 down),
instead of the closed profile's 112/16 split. The split is per matrix, not
per layer: seven middle layers (21, 42, 44, 46, 48, 49, 53) keep gate/up at
NVFP4 while down stays FP8. Object names, shapes, and every non-MLP format
are unchanged, so the same target package (qwen3_8_27b / Qwen38Nvfp4) binds
the artifact; the binding dispatches each MLP object on the format the
artifact itself declares (see bind_qwen38_fused_text_layers).
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    I32,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
)

from . import inventory_nvfp4 as base

BLOCK_SCALE_LAYOUT = base.BLOCK_SCALE_LAYOUT
ROW_SCALE_LAYOUT = base.ROW_SCALE_LAYOUT

MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "qwen3_8_27b"

NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_BF16S"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
ROW_SCALE_LAYOUT = "row-scale-v1"

FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4, FP8)
LAYOUT_NAMES = (
    CONTIGUOUS_LAYOUT,
    ROW_SPLIT_LAYOUT,
    BLOCK_SCALE_LAYOUT,
    ROW_SCALE_LAYOUT,
)

# W4A4W8A8 source assignment. Ground truth is per-MATRIX, verified two ways:
# the stored tensor dtypes on disk and the author's own
# quantization_config.quantized_layers table (260 FP8 / 140 NVFP4 targets).
# The stale hf_quant_config.json mislabels seven layers wholesale and is not
# used. The split is NOT per-layer: seven middle layers keep gate/up at NVFP4
# while down stays FP8.
#   gate/up NVFP4: 4-49, 51, 53, 55  (49 layers)
#   down    NVFP4: 4-20, 22-41, 43, 45, 47, 51, 55  (42 layers)
#   everything else: FP8 (gate/up in 15 layers, down in 22)
NVFP4_GATE_UP_LAYERS = tuple(
    [layer for layer in range(4, 50)] + [51, 53, 55]
)
NVFP4_DOWN_LAYERS = tuple(
    [layer for layer in range(4, 21)]
    + [layer for layer in range(22, 42)]
    + [43, 45, 47, 51, 55]
)

FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GDN_LAYERS = tuple(
    layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS
)


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format in (Q4, Q5, Q6, W8):
        layout = ROW_SPLIT_LAYOUT
    elif numeric_format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    elif numeric_format == FP8:
        layout = ROW_SCALE_LAYOUT
    else:
        raise ValueError(f"unsupported Qwen3.8 W4A4W8A8 format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (248320, 5120), FP8),
    ]
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        specs.append(tensor_spec(prefix + "input_norm", (5120,), BF16))
        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        FP8,
                    ),
                    tensor_spec(prefix + "attention/query_norm", (256,), BF16),
                    tensor_spec(prefix + "attention/key_norm", (256,), BF16),
                    tensor_spec(
                        prefix + "attention/output", (5120, 6144), FP8
                    ),
                )
            )
        else:
            specs.extend(
                (
                    tensor_spec(prefix + "gdn/a_log", (48,), FP32),
                    tensor_spec(prefix + "gdn/dt_bias", (48,), FP32),
                    tensor_spec(
                        prefix + "gdn/convolution", (4, 10240), BF16
                    ),
                    tensor_spec(
                        prefix + "gdn/a_b_projection", (96, 5120), BF16
                    ),
                    tensor_spec(
                        prefix + "gdn/query_key_value_z",
                        (16384, 5120),
                        FP8,
                    ),
                    tensor_spec(prefix + "gdn/norm", (128,), BF16),
                    tensor_spec(prefix + "gdn/output", (5120, 6144), FP8),
                )
            )

        specs.append(
            tensor_spec(prefix + "post_attention_norm", (5120,), BF16)
        )
        if layer in NVFP4_GATE_UP_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "mlp/gate_up", (34816, 5120), NVFP4
                    ),
                    tensor_spec(
                        prefix
                        + "mlp/gate_up_projection/input_scale_divisor",
                        (),
                        FP32,
                    ),
                )
            )
        else:
            specs.append(
                tensor_spec(prefix + "mlp/gate_up", (34816, 5120), FP8)
            )
        if layer in NVFP4_DOWN_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "mlp/down", (5120, 17408), NVFP4
                    ),
                    tensor_spec(
                        prefix + "mlp/down_projection/input_scale_divisor",
                        (),
                        FP32,
                    ),
                )
            )
        else:
            specs.append(
                tensor_spec(prefix + "mlp/down", (5120, 17408), FP8)
            )

    specs.extend(
        (
            tensor_spec("text/final_norm", (5120,), BF16),
            tensor_spec("text/output_head", (248320, 5120), FP8),
        )
    )
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = base.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = base.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = base.VISION_TENSOR_SPECS

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

# Format-independent logical views and aliases are shared with the base profile.
LOGICAL_ROW_VIEW_SPECS = base.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = base.ALIAS_SPECS

NVFP4_TENSOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.format == NVFP4
)
FP8_TENSOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.format == FP8
)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec
    for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("W4A4W8A8 inventory contains duplicate names")
    if (
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(NVFP4_TENSOR_SPECS),
        len(FP8_TENSOR_SPECS),
        len(INPUT_SCALE_DIVISOR_SPECS),
    ) != (750, 2, 12, 333, 1097, 1103, 91, 167, 91):
        raise ValueError("registered W4A4W8A8 inventory is incomplete")
    if FORMAT_COUNTS != {
        BF16: 534,
        FP32: 187,
        I32: 1,
        Q4: 55,
        Q5: 54,
        Q6: 1,
        W8: 7,
        NVFP4: 91,
        FP8: 167,
    }:
        raise ValueError(f"unexpected numeric allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {
        CONTIGUOUS_LAYOUT: 722,
        ROW_SPLIT_LAYOUT: 117,
        BLOCK_SCALE_LAYOUT: 91,
        ROW_SCALE_LAYOUT: 167,
    }:
        raise ValueError(f"unexpected layout allocation: {LAYOUT_COUNTS}")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BF16",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FP8",
    "FP8_TENSOR_SPECS",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "INPUT_SCALE_DIVISOR_SPECS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "NVFP4_DOWN_LAYERS",
    "NVFP4_GATE_UP_LAYERS",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SCALE_LAYOUT",
    "ROW_SPLIT_LAYOUT",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
    "W8",
    "tensor_spec",
    "validate_inventory",
]
