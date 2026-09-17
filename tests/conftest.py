"""Shared fixtures for the onnx-optimizer-lite test suite.

All fixtures build graphs with GraphBuilder, so the entire suite runs with
only numpy installed — no ``onnx`` or ``onnxruntime`` required.
"""

import numpy as np
import pytest

from onnx_optimizer_lite import GraphBuilder


@pytest.fixture
def mlp_graph():
    """A small 2-layer MLP: X -> MatMul -> Add -> Relu -> MatMul -> Add -> Softmax -> Y."""
    rng = np.random.default_rng(42)
    return (
        GraphBuilder("mlp")
        .add_input("X", "float32", (1, 4))
        .add_initializer("W1", rng.standard_normal((4, 8), dtype=np.float32))
        .add_initializer("b1", rng.standard_normal((8,), dtype=np.float32))
        .add_initializer("W2", rng.standard_normal((8, 2), dtype=np.float32))
        .add_initializer("b2", rng.standard_normal((2,), dtype=np.float32))
        .add_node("MatMul", ["X", "W1"], ["t1"])
        .add_node("Add", ["t1", "b1"], ["t2"])
        .add_node("Relu", ["t2"], ["t3"])
        .add_node("MatMul", ["t3", "W2"], ["t4"])
        .add_node("Add", ["t4", "b2"], ["t5"])
        .add_node("Softmax", ["t5"], ["Y"], axis=1)
        .add_output("Y", "float32", (1, 2))
        .build()
    )


@pytest.fixture
def graph_with_dead_branch():
    """MLP-ish graph plus a dead branch and an orphaned initializer."""
    rng = np.random.default_rng(7)
    return (
        GraphBuilder("dead")
        .add_input("X", "float32", (1, 4))
        .add_initializer("W", rng.standard_normal((4, 4), dtype=np.float32))
        .add_initializer("orphan", rng.standard_normal((4,), dtype=np.float32))
        .add_node("MatMul", ["X", "W"], ["t1"])
        .add_node("Relu", ["t1"], ["Y"])
        .add_node("Sigmoid", ["X"], ["dead1"])  # dead: feeds nothing
        .add_node("Add", ["dead1", "orphan"], ["dead2"])  # dead
        .add_output("Y", "float32", (1, 4))
        .build()
    )
