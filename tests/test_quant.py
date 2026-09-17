import numpy as np
import pytest

from onnx_optimizer_lite import GraphBuilder
from onnx_optimizer_lite.quant import (
    dtype_report,
    quantization_guidance,
    size_after_dtype_conversion,
)


def test_dtype_report(mlp_graph):
    report = dtype_report(mlp_graph)
    assert report == {"float32": {"params": 58, "bytes": 232}}


def test_dtype_report_mixed_dtypes():
    g = (
        GraphBuilder("mixed")
        .add_initializer("w", np.ones((4,), dtype=np.float32))
        .add_initializer("idx", np.ones((4,), dtype=np.int64))
        .build()
    )
    report = dtype_report(g)
    assert report["float32"] == {"params": 4, "bytes": 16}
    assert report["int64"] == {"params": 4, "bytes": 32}


def test_float16_conversion_halves_float_weights(mlp_graph):
    conv = size_after_dtype_conversion(mlp_graph, "float16")
    assert conv["before_bytes"] == 232
    assert conv["after_bytes"] == 116
    assert conv["saved_bytes"] == 116
    assert conv["saved_pct"] == pytest.approx(50.0)
    assert conv["converted_tensors"] == 4


def test_conversion_leaves_int_tensors_alone():
    g = (
        GraphBuilder("mixed2")
        .add_initializer("w", np.ones((4,), dtype=np.float32))
        .add_initializer("idx", np.ones((4,), dtype=np.int64))
        .build()
    )
    conv = size_after_dtype_conversion(g, "float16")
    # only the 16 float bytes halve; the 32 int bytes stay
    assert conv["after_bytes"] == 8 + 32
    assert conv["converted_tensors"] == 1


def test_conversion_unknown_dtype_raises(mlp_graph):
    with pytest.raises(ValueError, match="Unknown target dtype"):
        size_after_dtype_conversion(mlp_graph, "float8")


def test_guidance_flags_friendly_and_sensitive_ops(mlp_graph):
    g = quantization_guidance(mlp_graph)
    assert g["quantization_friendly_ops"] == {"MatMul": 2}
    assert g["quantization_sensitive_ops"] == {"Softmax": 1}
    assert g["weights_mb"] == pytest.approx(232 / (1024 * 1024))
    assert g["est_int8_dynamic_weights_mb"] == pytest.approx(g["weights_mb"] / 4)
    assert g["largest_initializer"]["name"] == "W1"
    assert any("MatMul" in r for r in g["recommendations"])
    assert any("Softmax" in r for r in g["recommendations"])


def test_guidance_without_friendly_ops_suggests_cleanup():
    g = (
        GraphBuilder("nomatmul")
        .add_input("X", "float32", (3,))
        .add_node("Relu", ["X"], ["Y"])
        .add_output("Y", "float32", (3,))
        .build()
    )
    guidance = quantization_guidance(g)
    assert guidance["quantization_friendly_ops"] == {}
    assert any("little latency benefit" in r for r in guidance["recommendations"])
