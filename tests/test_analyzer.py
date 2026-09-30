import json

import numpy as np
import pytest

from onnx_optimizer_lite import GraphBuilder
from onnx_optimizer_lite.analyzer import (
    analyze,
    compare_analysis,
    count_edges,
    dtype_histogram,
    graph_depth,
    infer_shapes,
    initializer_report,
    op_counts,
    topological_order,
    total_parameters,
    total_size_bytes,
)


def test_op_counts(mlp_graph):
    assert op_counts(mlp_graph) == {
        "MatMul": 2,
        "Add": 2,
        "Relu": 1,
        "Softmax": 1,
    }


def test_total_parameters_and_size(mlp_graph):
    # W1: 4*8=32, b1: 8, W2: 8*2=16, b2: 2  -> 58 params, 232 bytes (float32)
    assert total_parameters(mlp_graph) == 58
    assert total_size_bytes(mlp_graph) == 58 * 4


def test_count_edges(mlp_graph):
    # 2+2+1+2+2+1 input connections
    assert count_edges(mlp_graph) == 10


def test_initializer_report_ordering(mlp_graph):
    rows = initializer_report(mlp_graph)
    assert rows[0]["name"] == "W1"
    assert rows[0]["bytes"] == 128
    assert rows[0]["pct_of_total"] == pytest.approx(100 * 128 / 232, abs=0.01)
    names = [r["name"] for r in rows]
    assert names == ["W1", "W2", "b1", "b2"]
    assert sum(r["pct_of_total"] for r in rows) == pytest.approx(100.0, abs=0.01)


def test_graph_depth(mlp_graph):
    # MatMul -> Add -> Relu -> MatMul -> Add -> Softmax : 6 nodes deep
    assert graph_depth(mlp_graph) == 6


def test_topological_order_respects_dependencies(mlp_graph):
    order = topological_order(mlp_graph)
    position = {n.name: i for i, n in enumerate(order)}
    producer = {}
    for n in mlp_graph.nodes:
        for out in n.outputs:
            producer[out] = n.name
    for n in mlp_graph.nodes:
        for inp in n.inputs:
            if inp in producer:
                assert position[producer[inp]] < position[n.name]


def test_cycle_raises():
    g = (
        GraphBuilder("cyclic")
        .add_input("X", "float32", (2,))
        .add_node("Relu", ["b_out"], ["a_out"])
        .add_node("Sigmoid", ["a_out"], ["b_out"])
        .add_output("b_out", "float32", (2,))
        .build()
    )
    with pytest.raises(ValueError, match="cycle"):
        graph_depth(g)


def test_analyze_returns_json_serializable_dict(mlp_graph):
    result = analyze(mlp_graph)
    assert result["graph_name"] == "mlp"
    assert result["num_nodes"] == 6
    assert result["num_inputs"] == 1
    assert result["num_outputs"] == 1
    assert result["total_parameters"] == 58
    assert result["dtype_histogram"] == {"float32": {"params": 58, "bytes": 232}}
    json.dumps(result)  # must not raise


def test_graph_json_roundtrip(mlp_graph):
    data = mlp_graph.to_dict()
    restored = mlp_graph.from_dict(json.loads(json.dumps(data)))
    assert restored.name == mlp_graph.name
    assert [n.op for n in restored.nodes] == [n.op for n in mlp_graph.nodes]
    assert set(restored.initializers) == set(mlp_graph.initializers)
    np.testing.assert_array_equal(
        restored.initializers["W1"].data, mlp_graph.initializers["W1"].data
    )
    # dynamic dims survive the round-trip
    g2 = GraphBuilder("dyn").add_input("X", "float32", (None, 4)).build()
    r2 = g2.from_dict(json.loads(json.dumps(g2.to_dict())))
    assert r2.inputs[0].shape == (None, 4)


def test_validate_catches_unknown_tensor():
    g = (
        GraphBuilder("bad")
        .add_input("X", "float32", (2,))
        .add_node("Relu", ["nope"], ["Y"])
        .add_output("Y", "float32", (2,))
        .build()
    )
    with pytest.raises(ValueError, match="unknown tensor"):
        g.validate()


# ------------------------------------------------------------ shape inference

def test_infer_shapes_mlp(mlp_graph):
    shapes = infer_shapes(mlp_graph)["tensors"]
    assert shapes["t1"]["shape"] == [1, 8]
    assert shapes["t2"]["shape"] == [1, 8]      # Add broadcasts (1,8) + (8,)
    assert shapes["t3"]["shape"] == [1, 8]      # Relu preserves
    assert shapes["t5"]["shape"] == [1, 2]      # MatMul (1,8) x (8,2)
    assert shapes["Y"]["shape"] == [1, 2]
    assert shapes["Y"]["kind"] == "output"
    assert all(info["known"] for info in shapes.values())


def test_infer_shapes_dynamic_dim():
    g = (
        GraphBuilder("dyn")
        .add_input("X", "float32", (None, 4))
        .add_initializer("W", np.ones((4, 8), dtype=np.float32))
        .add_node("MatMul", ["X", "W"], ["t1"])
        .add_node("Relu", ["t1"], ["Y"])
        .add_output("Y", "float32", (None, 8))
        .build()
    )
    shapes = infer_shapes(g)["tensors"]
    assert shapes["t1"]["shape"] == [None, 8]
    assert shapes["t1"]["known"] is False
    assert shapes["Y"]["shape"] == [None, 8]


def test_infer_shapes_transpose_concat_reshape():
    g = (
        GraphBuilder("layout")
        .add_input("X", "float32", (2, 6))
        .add_initializer("shape", np.array([3, -1], dtype=np.int64))
        .add_node("Transpose", ["X"], ["t1"], perm=[1, 0])
        .add_node("Reshape", ["t1", "shape"], ["t2"])
        .add_node("Concat", ["t2", "t2"], ["Y"], axis=0)
        .add_output("Y", "float32", (6, 4))
        .build()
    )
    shapes = infer_shapes(g)["tensors"]
    assert shapes["t1"]["shape"] == [6, 2]
    assert shapes["t2"]["shape"] == [3, 4]   # -1 inferred from 12 elements
    assert shapes["Y"]["shape"] == [6, 4]


def test_infer_shapes_unknown_op_stays_unknown():
    g = (
        GraphBuilder("unk")
        .add_input("X", "float32", (2, 3))
        .add_node("Gather", ["X"], ["t1"])   # unsupported: shape unknown
        .add_node("Relu", ["t1"], ["Y"])     # ... and stays unknown downstream
        .add_output("Y", "float32", (2, 3))
        .build()
    )
    rep = infer_shapes(g)
    assert rep["tensors"]["t1"]["shape"] is None
    assert rep["tensors"]["t1"]["known"] is False
    assert rep["coverage"] < 1.0


def test_infer_shapes_constant_node():
    g = (
        GraphBuilder("const")
        .add_constant("c", np.zeros((2, 2), dtype=np.float32))
        .add_node("Identity", ["c"], ["Y"])
        .add_output("Y", "float32", (2, 2))
        .build()
    )
    shapes = infer_shapes(g)["tensors"]
    assert shapes["c"]["shape"] == [2, 2]
    assert shapes["Y"]["shape"] == [2, 2]


# ------------------------------------------------------- before/after compare

def test_compare_analysis_reports_deltas(graph_with_dead_branch):
    from onnx_optimizer_lite.optimizer import eliminate_dead_nodes

    before = analyze(graph_with_dead_branch)
    pruned, _ = eliminate_dead_nodes(graph_with_dead_branch)
    comp = compare_analysis(before, analyze(pruned))
    assert comp["metrics"]["num_nodes"] == {"before": 4, "after": 2, "delta": -2}
    assert comp["metrics"]["num_initializers"]["delta"] == -1
    assert comp["metrics"]["total_parameters"]["delta"] == -4
    assert comp["op_delta"] == {"Add": -1, "Sigmoid": -1}


def test_compare_analysis_noop_has_zero_deltas(mlp_graph):
    comp = compare_analysis(analyze(mlp_graph), analyze(mlp_graph))
    assert all(m["delta"] == 0 for m in comp["metrics"].values())
    assert comp["op_delta"] == {}
