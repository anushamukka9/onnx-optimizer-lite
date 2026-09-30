"""Static analysis of a :class:`~onnx_optimizer_lite.graph.Graph`.

All functions here are pure: they inspect the graph and return plain Python
data (dicts/lists/ints/floats), never mutating it.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .graph import Graph, Node


def op_counts(graph: Graph) -> Dict[str, int]:
    """Number of nodes per operator type, e.g. ``{"MatMul": 4, "Relu": 3}``."""
    return dict(Counter(node.op for node in graph.nodes))


def count_nodes(graph: Graph) -> int:
    return len(graph.nodes)


def count_edges(graph: Graph) -> int:
    """Data-flow edges, counting one edge per node input connection."""
    return sum(len(node.inputs) for node in graph.nodes)


def total_parameters(graph: Graph) -> int:
    """Total parameter count across initializers with known shapes."""
    total = 0
    for spec in graph.initializers.values():
        if spec.numel is not None:
            total += spec.numel
    return total


def total_size_bytes(graph: Graph) -> int:
    """Total initializer storage in bytes (known sizes only)."""
    total = 0
    for spec in graph.initializers.values():
        if spec.size_bytes is not None:
            total += spec.size_bytes
    return total


def initializer_report(graph: Graph, top_n: int = 10) -> List[Dict]:
    """Largest initializers first, each with size and share of the total."""
    total = total_size_bytes(graph) or 1
    rows = []
    for spec in graph.initializers.values():
        rows.append(
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "numel": spec.numel,
                "bytes": spec.size_bytes,
                "pct_of_total": round(100.0 * (spec.size_bytes or 0) / total, 2),
            }
        )
    rows.sort(key=lambda r: (r["bytes"] is None, -(r["bytes"] or 0)))
    return rows[:top_n]


def dtype_histogram(graph: Graph) -> Dict[str, Dict[str, int]]:
    """Per-dtype parameter and byte totals over initializers."""
    hist: Dict[str, Dict[str, int]] = {}
    for spec in graph.initializers.values():
        entry = hist.setdefault(spec.dtype, {"params": 0, "bytes": 0})
        if spec.numel is not None:
            entry["params"] += spec.numel
        if spec.size_bytes is not None:
            entry["bytes"] += spec.size_bytes
    return hist


def topological_order(graph: Graph) -> List[Node]:
    """Nodes in topological order. Raises ValueError if the graph has a cycle."""
    producer: Dict[str, Node] = {}
    for node in graph.nodes:
        for out in node.outputs:
            producer[out] = node

    deps: Dict[str, set] = {node.name: set() for node in graph.nodes}
    by_name = {node.name: node for node in graph.nodes}
    for node in graph.nodes:
        for inp in node.inputs:
            pred = producer.get(inp)
            if pred is not None and pred.name != node.name:
                deps[node.name].add(pred.name)

    order: List[Node] = []
    ready = sorted(name for name, d in deps.items() if not d)
    while ready:
        name = ready.pop()
        order.append(by_name[name])
        for other, d in deps.items():
            if name in d:
                d.discard(name)
                if not d:
                    ready.append(other)
        ready.sort()
    if len(order) != len(graph.nodes):
        leftover = [n for n in deps if n not in {nd.name for nd in order}]
        raise ValueError(f"Graph has a cycle involving nodes: {leftover}")
    return order


def graph_depth(graph: Graph) -> int:
    """Length (in nodes) of the longest data-flow path.

    A rough proxy for the sequential work an inference pass must do; wider
    graphs with the same node count parallelize better.
    """
    producer: Dict[str, Node] = {}
    for node in graph.nodes:
        for out in node.outputs:
            producer[out] = node
    level: Dict[str, int] = {}
    for node in topological_order(graph):
        pred_levels = [
            level[producer[inp].name]
            for inp in node.inputs
            if inp in producer
        ]
        level[node.name] = (max(pred_levels) if pred_levels else 0) + 1
    return max(level.values(), default=0)


def analyze(graph: Graph, top_n_initializers: int = 10) -> Dict:
    """Run the full analysis suite and return a JSON-serializable dict."""
    return {
        "graph_name": graph.name,
        "num_nodes": count_nodes(graph),
        "num_edges": count_edges(graph),
        "num_initializers": len(graph.initializers),
        "num_inputs": len(graph.inputs),
        "num_outputs": len(graph.outputs),
        "graph_depth": graph_depth(graph),
        "op_counts": op_counts(graph),
        "total_parameters": total_parameters(graph),
        "total_size_bytes": total_size_bytes(graph),
        "total_size_mb": round(total_size_bytes(graph) / (1024 * 1024), 3),
        "dtype_histogram": dtype_histogram(graph),
        "largest_initializers": initializer_report(graph, top_n=top_n_initializers),
    }


# ---------------------------------------------------------------------------
# Before/after comparison
# ---------------------------------------------------------------------------

_COMPARE_SCALARS = (
    "num_nodes",
    "num_edges",
    "num_initializers",
    "num_inputs",
    "num_outputs",
    "graph_depth",
    "total_parameters",
    "total_size_bytes",
)


def compare_analysis(before: Dict, after: Dict) -> Dict:
    """Diff two :func:`analyze` dicts: per-metric before/after values + delta.

    Returns ``{"metrics": {name: {"before", "after", "delta"}}, "op_delta":
    {op: change}}`` where ``delta`` is ``after - before`` and ``op_delta``
    only lists operator counts that changed.
    """
    metrics = {}
    for key in _COMPARE_SCALARS:
        b, a = before.get(key), after.get(key)
        delta = None
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
        metrics[key] = {"before": b, "after": a, "delta": delta}
    b_ops = before.get("op_counts", {})
    a_ops = after.get("op_counts", {})
    op_delta = {}
    for op in sorted(set(b_ops) | set(a_ops)):
        d = a_ops.get(op, 0) - b_ops.get(op, 0)
        if d:
            op_delta[op] = d
    return {"metrics": metrics, "op_delta": op_delta}


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------

_SAME_SHAPE_UNARY = {
    "Relu", "Sigmoid", "Tanh", "Sqrt", "Neg", "Exp", "Log", "Abs", "Floor",
    "Ceil", "Round", "Not", "Reciprocal", "Identity", "Cast", "Dropout",
    "Elu", "Selu", "LeakyRelu", "Softmax", "Clip", "Erf", "Sign",
}

_ELEMENTWISE_BINARY = {
    "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Mean", "And", "Or",
    "Xor", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
}


def _broadcast_dim(d1, d2):
    """Broadcast one dimension; None means dynamic/unknown."""
    if d1 == d2:
        return d1
    if d1 == 1:
        return d2
    if d2 == 1:
        return d1
    if d1 is None or d2 is None:
        return None
    return "incompatible"


def _broadcast_shapes(s1, s2):
    if s1 is None or s2 is None:
        return None
    out = []
    for d1, d2 in zip(reversed(s1), reversed(s2)):
        d = _broadcast_dim(d1, d2)
        if d == "incompatible":
            return None
        out.append(d)
    longer = s1 if len(s1) > len(s2) else s2
    out.extend(reversed(longer[: abs(len(s1) - len(s2))]))
    return list(reversed(out))


def _matmul_shape(s1, s2):
    if s1 is None or s2 is None:
        return None
    a, b = list(s1), list(s2)
    # Promote 1-D operands to 2-D per numpy matmul rules, then drop the axis.
    drop_a = drop_b = False
    if len(a) == 1:
        a = [1] + a
        drop_a = True
    if len(b) == 1:
        b = b + [1]
        drop_b = True
    if len(a) < 2 or len(b) < 2:
        return None
    if a[-1] is not None and b[-2] is not None and a[-1] != b[-2]:
        return None
    batch = _broadcast_shapes(a[:-2], b[:-2])
    if batch is None:
        return None
    out = batch + [a[-2], b[-1]]
    if drop_a:
        out = out[:-2] + out[-1:]
    if drop_b:
        out = out[:-1]
    return out


def _transpose_shape(shape, perm):
    if shape is None:
        return None
    rank = len(shape)
    if perm is None:
        perm = list(reversed(range(rank)))
    if sorted(perm) != list(range(rank)):
        return None
    return [shape[p] for p in perm]


def _reshape_shape(in_shape, shape_vals):
    """Apply a Reshape given the target shape values (may hold 0 and -1)."""
    if in_shape is None or shape_vals is None:
        return None
    out = []
    infer_idx = None
    for i, dim in enumerate(shape_vals):
        if dim == 0:
            if i >= len(in_shape) or in_shape[i] is None:
                return None
            out.append(in_shape[i])
        elif dim == -1:
            if infer_idx is not None:
                return None  # two -1s: ambiguous
            infer_idx = i
            out.append(None)
        elif dim < -1:
            return None
        else:
            out.append(dim)
    if infer_idx is not None:
        if any(d is None for d in in_shape):
            return None
        total = 1
        for d in in_shape:
            total *= d
        known = 1
        for d in out:
            if d is not None:
                known *= d
        if known == 0 or total % known != 0:
            return None
        out[infer_idx] = total // known
    return out


def _concat_shape(in_shapes, axis):
    if any(s is None for s in in_shapes):
        return None
    rank = len(in_shapes[0])
    if any(len(s) != rank for s in in_shapes):
        return None
    if axis < 0:
        axis += rank
    if not 0 <= axis < rank:
        return None
    out = []
    for i in range(rank):
        dims = [s[i] for s in in_shapes]
        if i == axis:
            out.append(sum(dims) if all(d is not None for d in dims) else None)
        else:
            out.append(dims[0] if all(d == dims[0] for d in dims) else None)
    return out


def _infer_node_shape(node, in_shapes, const_values):
    """Output shape for one node, or None when it cannot be determined."""
    op = node.op
    if op == "Constant":
        value = node.attributes.get("value")
        if isinstance(value, np.ndarray):
            return list(value.shape)
        return None
    if op in _SAME_SHAPE_UNARY:
        return list(in_shapes[0]) if in_shapes and in_shapes[0] is not None else None
    if op in _ELEMENTWISE_BINARY:
        if len(in_shapes) >= 2:
            return _broadcast_shapes(in_shapes[0], in_shapes[1])
        return None
    if op == "MatMul":
        if len(in_shapes) >= 2:
            return _matmul_shape(in_shapes[0], in_shapes[1])
        return None
    if op == "Gemm":
        if len(in_shapes) >= 2 and in_shapes[0] and in_shapes[1]:
            a, b = list(in_shapes[0]), list(in_shapes[1])
            if len(a) == 2 and len(b) == 2:
                if node.attributes.get("transA", 0):
                    a = [a[1], a[0]]
                if node.attributes.get("transB", 0):
                    b = [b[1], b[0]]
                if a[1] is not None and b[0] is not None and a[1] != b[0]:
                    return None
                return [a[0], b[1]]
        return None
    if op == "Transpose":
        perm = node.attributes.get("perm")
        if perm is not None:
            perm = [int(p) for p in perm]
        return _transpose_shape(in_shapes[0] if in_shapes else None, perm)
    if op == "Reshape":
        shape_vals = None
        if len(node.inputs) >= 2:
            arr = const_values.get(node.inputs[1])
            if isinstance(arr, np.ndarray) and arr.size:
                shape_vals = [int(x) for x in arr.ravel().tolist()]
        return _reshape_shape(in_shapes[0] if in_shapes else None, shape_vals)
    if op == "Concat":
        axis = int(node.attributes.get("axis", 0))
        return _concat_shape(in_shapes, axis)
    if op == "Squeeze":
        shape = in_shapes[0] if in_shapes else None
        if shape is None:
            return None
        axes = node.attributes.get("axes")
        axes = [int(a) for a in axes] if axes is not None else None
        out = []
        for i, d in enumerate(shape):
            if d == 1 and (axes is None or i in axes or i - len(shape) in axes):
                continue
            out.append(d)
        return out
    if op == "Unsqueeze":
        shape = in_shapes[0] if in_shapes else None
        if shape is None:
            return None
        axes = [int(a) for a in node.attributes.get("axes", [])]
        out = list(shape)
        rank = len(shape) + len(axes)
        for a in sorted(axes):
            idx = a if a >= 0 else a + rank
            out.insert(idx, 1)
        return out
    if op == "Flatten":
        shape = in_shapes[0] if in_shapes else None
        if shape is None or any(d is None for d in shape):
            return None
        axis = int(node.attributes.get("axis", 1))
        if axis < 0:
            axis += len(shape)
        head, tail = shape[:axis], shape[axis:]
        prod = 1
        for d in head:
            prod *= d
        prod2 = 1
        for d in tail:
            prod2 *= d
        return [prod, prod2]
    return None


def infer_shapes(graph: Graph) -> Dict:
    """Best-effort static shape inference over supported ops.

    Propagates shapes from graph inputs, initializers, and constant nodes
    through shape-preserving and layout ops (elementwise with broadcasting,
    MatMul, Gemm, Transpose, Reshape, Concat, Squeeze, Unsqueeze, Flatten,
    unary activations). Anything else leaves the output shape unknown.

    Returns ``{"tensors": {name: {"shape", "known", "kind"}}, "n_tensors",
    "n_known", "coverage"}`` where ``shape`` is a list with ``None`` for
    dynamic dimensions (or ``None`` when nothing is known), ``known`` is True
    only when every dimension is concrete, and ``kind`` is one of
    ``input`` / ``initializer`` / ``intermediate`` / ``output`` /
    ``value_info``.
    """
    shapes: Dict[str, Optional[list]] = {}
    kinds: Dict[str, str] = {}
    const_values: Dict[str, np.ndarray] = {}

    for t in graph.inputs:
        shapes[t.name] = list(t.shape) if t.shape else None
        kinds[t.name] = "input"
    for name, spec in graph.initializers.items():
        if spec.data is not None:
            shapes[name] = list(spec.data.shape)
            const_values[name] = spec.data
        elif spec.shape:
            shapes[name] = list(spec.shape)
        else:
            shapes[name] = None
        kinds[name] = "initializer"
    for t in graph.value_info:
        shapes.setdefault(t.name, list(t.shape) if t.shape else None)
        kinds.setdefault(t.name, "value_info")

    for node in topological_order(graph):
        in_shapes = [shapes.get(i) for i in node.inputs]
        out_shape = _infer_node_shape(node, in_shapes, const_values)
        if node.op == "Constant":
            value = node.attributes.get("value")
            if isinstance(value, np.ndarray):
                for o in node.outputs:
                    const_values[o] = value
        for o in node.outputs:
            shapes[o] = out_shape
            kinds[o] = "intermediate"

    for t in graph.outputs:
        kinds[t.name] = "output"

    tensors = {}
    n_known = 0
    for name, shape in shapes.items():
        known = shape is not None and all(d is not None for d in shape)
        n_known += 1 if known else 0
        tensors[name] = {
            "shape": shape,
            "known": known,
            "kind": kinds.get(name, "intermediate"),
        }
    return {
        "tensors": tensors,
        "n_tensors": len(tensors),
        "n_known": n_known,
        "coverage": round(n_known / len(tensors), 4) if tensors else 1.0,
    }
