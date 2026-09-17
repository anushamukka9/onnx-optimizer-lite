"""Dtype and quantization guidance.

These are *estimates and recommendations*, not a quantization implementation:
they tell you how much memory a dtype conversion or dynamic INT8 quantization
would save, and which ops are friendly (or sensitive) to quantization, so you
can decide whether quantizing is worth the effort for a given model.
"""

from __future__ import annotations

from typing import Dict, List

from .analyzer import op_counts, total_size_bytes
from .graph import Graph

DTYPE_ITEMSIZE = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "bool": 1,
}

#: Ops whose weights quantize cleanly to INT8 with minimal accuracy impact.
QUANT_FRIENDLY_OPS = {"MatMul", "Gemm", "Conv"}

#: Ops where reduced precision is most likely to hurt accuracy.
QUANT_SENSITIVE_OPS = {
    "Softmax",
    "LayerNorm",
    "Sigmoid",
    "Tanh",
    "BatchNormalization",
    "InstanceNormalization",
}


def dtype_report(graph: Graph) -> Dict[str, Dict[str, int]]:
    """Per-dtype parameter and byte totals over initializers."""
    report: Dict[str, Dict[str, int]] = {}
    for spec in graph.initializers.values():
        entry = report.setdefault(spec.dtype, {"params": 0, "bytes": 0})
        if spec.numel is not None:
            entry["params"] += spec.numel
        if spec.size_bytes is not None:
            entry["bytes"] += spec.size_bytes
    return report


def size_after_dtype_conversion(graph: Graph, target: str = "float16") -> Dict:
    """Estimate weight size if float32/float64 tensors moved to ``target``.

    Only floating-point tensors are converted; integer tensors (indices,
    masks, lookup tables) are left as-is, mirroring what mixed-precision
    conversion tools do in practice.
    """
    if target not in DTYPE_ITEMSIZE:
        raise ValueError(f"Unknown target dtype {target!r}")
    before = total_size_bytes(graph)
    after = 0
    converted = 0
    for spec in graph.initializers.values():
        nbytes = spec.size_bytes
        if nbytes is None:
            continue
        if spec.dtype in ("float32", "float64"):
            nbytes = (spec.numel or 0) * DTYPE_ITEMSIZE[target]
            converted += 1
        after += nbytes
    return {
        "target_dtype": target,
        "converted_tensors": converted,
        "before_bytes": before,
        "after_bytes": after,
        "saved_bytes": before - after,
        "saved_pct": round(100.0 * (before - after) / before, 2) if before else 0.0,
    }


def quantization_guidance(graph: Graph) -> Dict:
    """Estimate INT8 dynamic-quantization wins and flag sensitive ops.

    Dynamic quantization keeps activations in floating point and quantizes
    weights to INT8, so the estimated saving applies to initializer bytes.
    """
    counts = op_counts(graph)
    friendly = {op: counts[op] for op in QUANT_FRIENDLY_OPS if op in counts}
    sensitive = {op: counts[op] for op in QUANT_SENSITIVE_OPS if op in counts}

    weights_bytes = total_size_bytes(graph)
    weights_mb = weights_bytes / (1024 * 1024)
    est_int8_mb = weights_mb / 4.0  # INT8 weights vs FP32
    fp16 = size_after_dtype_conversion(graph, "float16")

    recommendations: List[str] = []
    if not friendly:
        recommendations.append(
            "No MatMul/Gemm/Conv nodes found: quantization would yield little "
            "latency benefit for this graph. Focus on graph cleanup "
            "(dead nodes, constant folding) instead."
        )
    else:
        total_friendly = sum(friendly.values())
        recommendations.append(
            f"{total_friendly} quantization-friendly node(s) "
            f"({', '.join(f'{op} x{n}' for op, n in sorted(friendly.items()))}): "
            "good candidates for dynamic INT8 quantization."
        )
        recommendations.append(
            f"Dynamic INT8 quantization would shrink weights from "
            f"{weights_mb:.2f} MB to ~{est_int8_mb:.2f} MB (~4x), with "
            "activations kept in floating point."
        )
    if sensitive:
        total_sensitive = sum(sensitive.values())
        recommendations.append(
            f"{total_sensitive} precision-sensitive node(s) "
            f"({', '.join(f'{op} x{n}' for op, n in sorted(sensitive.items()))}) "
            "detected: validate accuracy on a held-out set after quantizing, "
            "and consider excluding these ops' inputs from quantization."
        )
    if fp16["saved_pct"] > 1:
        recommendations.append(
            f"Converting float weights to float16 would cut weight memory "
            f"~{fp16['saved_pct']:.1f}% "
            f"({fp16['before_bytes'] / 1024 / 1024:.2f} MB -> "
            f"{fp16['after_bytes'] / 1024 / 1024:.2f} MB) with no graph "
            "changes; verify numerics on your validation set."
        )

    largest = None
    for spec in sorted(
        graph.initializers.values(),
        key=lambda s: s.size_bytes or 0,
        reverse=True,
    ):
        if spec.size_bytes:
            largest = {
                "name": spec.name,
                "mb": round(spec.size_bytes / (1024 * 1024), 3),
                "pct_of_weights": round(100.0 * spec.size_bytes / weights_bytes, 1)
                if weights_bytes
                else 0.0,
            }
            break
    if largest and largest["pct_of_weights"] >= 20:
        recommendations.append(
            f"'{largest['name']}' alone is {largest['mb']} MB "
            f"({largest['pct_of_weights']}% of weights): the single biggest "
            "quantization win."
        )

    return {
        "weights_bytes": weights_bytes,
        "weights_mb": weights_mb,
        "est_int8_dynamic_weights_mb": est_int8_mb,
        "float16_conversion": fp16,
        "quantization_friendly_ops": friendly,
        "quantization_sensitive_ops": sensitive,
        "largest_initializer": largest,
        "recommendations": recommendations,
    }
