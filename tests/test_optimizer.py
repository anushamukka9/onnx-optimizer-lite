import numpy as np
import pytest

from onnx_optimizer_lite import GraphBuilder
from onnx_optimizer_lite.optimizer import (
    eliminate_dead_nodes,
    fold_constants,
    optimize,
    remove_identity_nodes,
)


def test_dead_node_elimination(graph_with_dead_branch):
    pruned, result = eliminate_dead_nodes(graph_with_dead_branch)
    assert result.changed
    assert set(result.details["removed_nodes"]) == {"Sigmoid_2", "Add_3"}
    assert "orphan" in result.details["pruned_initializers"]
    # live part of the graph is intact
    assert {n.op for n in pruned.nodes} == {"MatMul", "Relu"}
    assert set(pruned.initializers) == {"W"}
    # original untouched
    assert len(graph_with_dead_branch.nodes) == 4


def test_dead_node_elimination_noop_on_clean_graph(mlp_graph):
    pruned, result = eliminate_dead_nodes(mlp_graph)
    assert not result.changed
    assert len(pruned.nodes) == len(mlp_graph.nodes)


def test_identity_removal():
    g = (
        GraphBuilder("ident")
        .add_input("X", "float32", (3,))
        .add_node("Relu", ["X"], ["r"])
        .add_node("Identity", ["r"], ["r2"])
        .add_node("Sigmoid", ["r2"], ["Y"])
        .add_output("Y", "float32", (3,))
        .build()
    )
    out, result = remove_identity_nodes(g)
    assert result.changed
    assert [n.op for n in out.nodes] == ["Relu", "Sigmoid"]
    sigmoid = next(n for n in out.nodes if n.op == "Sigmoid")
    assert sigmoid.inputs == ["r"]  # rewired past the identity


def test_identity_feeding_output_is_kept():
    g = (
        GraphBuilder("ident-out")
        .add_input("X", "float32", (3,))
        .add_node("Relu", ["X"], ["r"])
        .add_node("Identity", ["r"], ["Y"])
        .add_output("Y", "float32", (3,))
        .build()
    )
    out, result = remove_identity_nodes(g)
    assert not result.changed  # kept: removing it would rename the model output
    assert any(n.op == "Identity" for n in out.nodes)


def test_constant_folding_add():
    g = (
        GraphBuilder("fold")
        .add_input("X", "float32", (3,))
        .add_constant("c1", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        .add_constant("c2", np.array([4.0, 5.0, 6.0], dtype=np.float32))
        .add_node("Add", ["c1", "c2"], ["s"])
        .add_node("Mul", ["X", "s"], ["Y"])
        .add_output("Y", "float32", (3,))
        .build()
    )
    folded, result = fold_constants(g)
    assert result.changed
    assert result.details["folded_nodes"] == ["Add_2"]
    add_node = next(n for n in folded.nodes if n.name == "Add_2")
    assert add_node.op == "Constant"
    assert add_node.inputs == []
    np.testing.assert_array_equal(
        add_node.attributes["value"], np.array([5.0, 7.0, 9.0], dtype=np.float32)
    )


def test_constant_folding_gemm_with_attrs():
    g = (
        GraphBuilder("gemm")
        .add_initializer("A", np.array([[1.0, 2.0]], dtype=np.float32))
        .add_initializer("B", np.array([[3.0], [4.0]], dtype=np.float32))
        .add_initializer("C", np.array([[10.0]], dtype=np.float32))
        .add_node("Gemm", ["A", "B", "C"], ["Y"], alpha=2.0, beta=0.5)
        .add_output("Y", "float32", (1, 1))
        .build()
    )
    folded, result = fold_constants(g)
    assert result.changed
    node = folded.nodes[0]
    assert node.op == "Constant"
    # 2.0 * ([[1,2]] @ [[3],[4]]) + 0.5 * [[10]] = 2*11 + 5 = 27
    np.testing.assert_allclose(node.attributes["value"], np.array([[27.0]]))


def test_constant_folding_skips_non_constant_inputs(mlp_graph):
    _, result = fold_constants(mlp_graph)
    assert not result.changed
    assert result.details["folded_nodes"] == []


def test_optimize_pipeline_folds_then_prunes():
    g = (
        GraphBuilder("pipe")
        .add_input("X", "float32", (2,))
        .add_constant("c1", np.array([1.0, 1.0], dtype=np.float32))
        .add_constant("c2", np.array([2.0, 3.0], dtype=np.float32))
        .add_node("Add", ["c1", "c2"], ["s"])  # folds to [3,4], then dead
        .add_node("Relu", ["X"], ["Y"])
        .add_output("Y", "float32", (2,))
        .build()
    )
    out, results = optimize(g)
    by_name = {r.pass_name: r for r in results}
    assert by_name["fold_constants"].changed
    assert by_name["eliminate_dead_nodes"].changed
    assert [n.op for n in out.nodes] == ["Relu"]
    # input graph not mutated
    assert len(g.nodes) == 4


def test_optimize_unknown_pass_raises(mlp_graph):
    with pytest.raises(ValueError, match="Unknown passes"):
        optimize(mlp_graph, passes=("nope",))
