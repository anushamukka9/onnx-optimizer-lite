"""Round-trip tests on a small generated ONNX model.

The whole module is skipped when the optional ``onnx`` package is not
installed (``pytest.importorskip``), so the core suite stays numpy-only.
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from onnx_optimizer_lite import infer_shapes, optimize  # noqa: E402
from onnx_optimizer_lite.onnx_adapter import (  # noqa: E402
    from_onnx,
    load_onnx,
    save_onnx,
    to_onnx,
)


def _tiny_model():
    """X -> Identity -> MatMul -> Add -> Relu -> Y, plus a dead Sigmoid."""
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 2], np.arange(8, dtype=np.float32).tolist()
    )
    b = helper.make_tensor("b", TensorProto.FLOAT, [2], [0.5, -0.5])
    nodes = [
        helper.make_node("Identity", ["X"], ["x2"]),
        helper.make_node("MatMul", ["x2", "W"], ["t1"]),
        helper.make_node("Add", ["t1", "b"], ["t2"]),
        helper.make_node("Relu", ["t2"], ["Y"]),
        helper.make_node("Sigmoid", ["X"], ["dead"]),
    ]
    graph = helper.make_graph(nodes, "tiny", [X], [Y], [W, b])
    model = helper.make_model(graph, producer_name="onnx-optimizer-lite-test")
    onnx.checker.check_model(model)
    return model


def test_from_onnx_structure():
    graph = from_onnx(_tiny_model())
    assert graph.name == "tiny"
    assert len(graph.nodes) == 5
    assert set(graph.initializers) == {"W", "b"}
    assert graph.initializers["W"].data.shape == (4, 2)
    assert [t.name for t in graph.inputs] == ["X"]
    assert [t.name for t in graph.outputs] == ["Y"]
    graph.validate()


def test_adapter_optimize_roundtrip_validates():
    graph = from_onnx(_tiny_model())
    optimized, results = optimize(graph)
    ops = [n.op for n in optimized.nodes]
    assert "Identity" not in ops  # bypassed
    assert "Sigmoid" not in ops   # dead branch eliminated
    assert [n.op for n in optimized.nodes] == ["MatMul", "Add", "Relu"]
    # shape inference still works on the converted graph
    shapes = infer_shapes(optimized)["tensors"]
    assert shapes["t1"]["shape"] == [1, 2]
    # and the optimized IR exports back to a valid ONNX model
    model = to_onnx(optimized)
    onnx.checker.check_model(model)


def test_save_load_onnx_roundtrip(tmp_path):
    graph = from_onnx(_tiny_model())
    path = str(tmp_path / "tiny.onnx")
    save_onnx(graph, path)
    reloaded = load_onnx(path)
    assert len(reloaded.nodes) == len(graph.nodes)
    assert set(reloaded.initializers) == set(graph.initializers)
    np.testing.assert_array_equal(
        reloaded.initializers["W"].data, graph.initializers["W"].data
    )


def test_folded_constant_exports_to_onnx():
    """A Constant subgraph folded in the IR must survive a to_onnx export."""
    from onnx import helper, TensorProto, numpy_helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    c1 = helper.make_node(
        "Constant", [], ["c1"],
        value=numpy_helper.from_array(np.ones((2, 2), dtype=np.float32)),
    )
    c2 = helper.make_node(
        "Constant", [], ["c2"],
        value=numpy_helper.from_array(2 * np.ones((2, 2), dtype=np.float32)),
    )
    nodes = [
        c1, c2,
        helper.make_node("Add", ["c1", "c2"], ["t"]),
        helper.make_node("MatMul", ["X", "t"], ["Y"]),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "folded", [X], [Y]),
        producer_name="onnx-optimizer-lite-test",
    )
    onnx.checker.check_model(model)

    graph = from_onnx(model)
    from onnx_optimizer_lite.optimizer import fold_constants  # noqa: E402

    folded_graph, result = fold_constants(graph)
    assert result.details["folded_nodes"], "expected the Add of two constants to fold"
    folded_node = next(
        n for n in folded_graph.nodes if n.name in result.details["folded_nodes"]
    )
    assert folded_node.op == "Constant"
    np.testing.assert_array_equal(
        folded_node.attributes["value"], 3 * np.ones((2, 2), dtype=np.float32)
    )
    onnx.checker.check_model(to_onnx(folded_graph))
