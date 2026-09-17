"""Static analysis of a :class:`~onnx_optimizer_lite.graph.Graph`.

All functions here are pure: they inspect the graph and return plain Python
data (dicts/lists/ints/floats), never mutating it.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

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
