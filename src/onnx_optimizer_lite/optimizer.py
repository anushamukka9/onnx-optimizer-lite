"""Graph optimization passes.

Each pass takes a :class:`~onnx_optimizer_lite.graph.Graph`, returns a new
graph plus a :class:`PassResult` describing what changed. Passes never mutate
their input. :func:`optimize` runs a standard pipeline of passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .analyzer import topological_order
from .graph import (
    Graph,
    Node,
    numpy_dtype,
    onnx_dtype_to_str,
)


@dataclass
class PassResult:
    """Outcome of a single optimization pass."""

    pass_name: str
    changed: bool
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dead-node elimination
# ---------------------------------------------------------------------------


def eliminate_dead_nodes(graph: Graph) -> Tuple[Graph, PassResult]:
    """Remove nodes that cannot influence any graph output.

    A node is dead when none of its outputs is consumed — transitively — by a
    graph output. Initializers that only fed dead nodes are pruned as well.
    """
    g = graph.copy()
    producer: Dict[str, Node] = {}
    for node in g.nodes:
        for out in node.outputs:
            producer[out] = node

    live_nodes: set = set()
    stack = [o.name for o in g.outputs]
    seen: set = set()
    while stack:
        tensor = stack.pop()
        if tensor in seen:
            continue
        seen.add(tensor)
        node = producer.get(tensor)
        if node is None:
            continue
        live_nodes.add(node.name)
        stack.extend(node.inputs)

    removed = [n.name for n in g.nodes if n.name not in live_nodes]
    g.nodes = [n for n in g.nodes if n.name in live_nodes]

    used = {o.name for o in g.outputs}
    for node in g.nodes:
        used.update(node.inputs)
    pruned = [k for k in g.initializers if k not in used]
    for k in pruned:
        del g.initializers[k]

    return g, PassResult(
        "eliminate_dead_nodes",
        changed=bool(removed or pruned),
        details={"removed_nodes": removed, "pruned_initializers": pruned},
    )


# ---------------------------------------------------------------------------
# Identity removal
# ---------------------------------------------------------------------------


def remove_identity_nodes(graph: Graph) -> Tuple[Graph, PassResult]:
    """Bypass ``Identity`` nodes by rewiring their consumers directly.

    Identities that feed a graph output are left alone: removing them would
    rename the model's outputs, silently changing its interface.
    """
    g = graph.copy()
    output_names = {o.name for o in g.outputs}
    removed: List[str] = []

    for _ in range(len(g.nodes) + 1):  # bounded fixpoint for chained identities
        progressed = False
        for node in list(g.nodes):
            if (
                node.op != "Identity"
                or len(node.inputs) != 1
                or len(node.outputs) != 1
            ):
                continue
            src, dst = node.inputs[0], node.outputs[0]
            if src == dst or dst in output_names:
                continue
            for other in g.nodes:
                other.inputs = [src if i == dst else i for i in other.inputs]
            g.nodes.remove(node)
            removed.append(node.name)
            progressed = True
        if not progressed:
            break

    return g, PassResult(
        "remove_identity_nodes",
        changed=bool(removed),
        details={"removed_nodes": removed},
    )


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


def _fold_add(a, attrs): return np.add(a[0], a[1])
def _fold_sub(a, attrs): return np.subtract(a[0], a[1])
def _fold_mul(a, attrs): return np.multiply(a[0], a[1])
def _fold_div(a, attrs): return np.divide(a[0], a[1])
def _fold_matmul(a, attrs): return np.matmul(a[0], a[1])
def _fold_neg(a, attrs): return np.negative(a[0])
def _fold_sqrt(a, attrs): return np.sqrt(a[0])
def _fold_relu(a, attrs): return np.maximum(a[0], 0)
def _fold_sigmoid(a, attrs): return 1.0 / (1.0 + np.exp(-a[0]))
def _fold_tanh(a, attrs): return np.tanh(a[0])
def _fold_pow(a, attrs): return np.power(a[0], a[1])


def _fold_gemm(a, attrs):
    A, B = a[0], a[1]
    if attrs.get("transA", 0):
        A = A.T
    if attrs.get("transB", 0):
        B = B.T
    y = float(attrs.get("alpha", 1.0)) * (A @ B)
    if len(a) > 2:
        y = y + float(attrs.get("beta", 1.0)) * a[2]
    return y


def _fold_concat(a, attrs):
    return np.concatenate(a, axis=int(attrs.get("axis", 0)))


def _fold_transpose(a, attrs):
    perm = attrs.get("perm")
    return np.transpose(a[0], axes=tuple(int(p) for p in perm) if perm else None)


def _fold_reshape(a, attrs):
    return np.reshape(a[0], [int(d) for d in a[1].ravel().tolist()])


def _fold_cast(a, attrs):
    if "to" not in attrs:
        raise ValueError("Cast without 'to' attribute")
    return a[0].astype(numpy_dtype(onnx_dtype_to_str(int(attrs["to"]))))


_FOLDABLE: Dict[str, Callable] = {
    "Add": _fold_add,
    "Sub": _fold_sub,
    "Mul": _fold_mul,
    "Div": _fold_div,
    "MatMul": _fold_matmul,
    "Gemm": _fold_gemm,
    "Neg": _fold_neg,
    "Sqrt": _fold_sqrt,
    "Relu": _fold_relu,
    "Sigmoid": _fold_sigmoid,
    "Tanh": _fold_tanh,
    "Pow": _fold_pow,
    "Concat": _fold_concat,
    "Transpose": _fold_transpose,
    "Reshape": _fold_reshape,
    "Cast": _fold_cast,
}


def fold_constants(graph: Graph) -> Tuple[Graph, PassResult]:
    """Evaluate subgraphs whose inputs are all constants at build time.

    Folded nodes become ``Constant`` nodes carrying the computed array, so a
    follow-up :func:`eliminate_dead_nodes` pass can drop the (now unused)
    producer nodes. Only single-output nodes with a supported op are folded;
    anything else is left untouched.
    """
    g = graph.copy()

    known: Dict[str, np.ndarray] = {}
    for name, spec in g.initializers.items():
        if spec.data is not None:
            known[name] = spec.data
    for node in g.nodes:
        if node.op == "Constant" and not node.inputs and node.outputs:
            value = node.attributes.get("value")
            if isinstance(value, np.ndarray):
                known[node.outputs[0]] = value

    folded: List[str] = []
    for node in topological_order(g):
        if node.op == "Constant" or node.op not in _FOLDABLE or len(node.outputs) != 1:
            continue
        try:
            arrays = [known[inp] for inp in node.inputs]
        except KeyError:
            continue  # not all inputs are constant
        try:
            result = np.ascontiguousarray(_FOLDABLE[node.op](arrays, node.attributes))
        except Exception:
            continue  # e.g. bad shapes/dtypes: leave the node alone
        node.op = "Constant"
        node.inputs = []
        node.attributes = {"value": result}
        known[node.outputs[0]] = result
        folded.append(node.name)

    return g, PassResult(
        "fold_constants",
        changed=bool(folded),
        details={"folded_nodes": folded, "foldable_ops": sorted(_FOLDABLE)},
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

PASSES: Dict[str, Callable[[Graph], Tuple[Graph, PassResult]]] = {
    "remove_identity": remove_identity_nodes,
    "fold_constants": fold_constants,
    "eliminate_dead_nodes": eliminate_dead_nodes,
}

DEFAULT_PASSES = ("remove_identity", "fold_constants", "eliminate_dead_nodes")


def optimize(
    graph: Graph, passes: Tuple[str, ...] = DEFAULT_PASSES
) -> Tuple[Graph, List[PassResult]]:
    """Run optimization passes in order; returns (graph, per-pass results)."""
    unknown = [p for p in passes if p not in PASSES]
    if unknown:
        raise ValueError(f"Unknown passes: {unknown}. Available: {sorted(PASSES)}")
    g = graph.copy()
    results: List[PassResult] = []
    for name in passes:
        g, result = PASSES[name](g)
        results.append(result)
    return g, results
