"""Optimize a small MLP and show the before/after comparison.

Builds a 2-layer MLP with cruft a real export might carry (an Identity node,
a dead branch, a foldable constant subgraph), runs the optimizer, and prints
a before/after comparison plus a shape-inference report.

Runs from the repo root (only numpy needed):
    python examples/optimize_mlp.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from onnx_optimizer_lite import (  # noqa: E402
    GraphBuilder,
    infer_shapes,
    optimize_and_compare,
)


def build_mlp():
    rng = np.random.default_rng(42)
    return (
        GraphBuilder("mlp")
        .add_input("X", "float32", (1, 4))
        .add_initializer("W1", rng.standard_normal((4, 8), dtype=np.float32))
        .add_initializer("b1", rng.standard_normal((8,), dtype=np.float32))
        .add_initializer("W2", rng.standard_normal((8, 2), dtype=np.float32))
        .add_initializer("b2", rng.standard_normal((2,), dtype=np.float32))
        .add_initializer("unused", rng.standard_normal((3,), dtype=np.float32))
        .add_constant("c1", np.ones((2,), dtype=np.float32))
        .add_constant("c2", np.full((2,), 2.0, dtype=np.float32))
        .add_node("MatMul", ["X", "W1"], ["t1"])
        .add_node("Identity", ["t1"], ["t1b"])          # bypassed
        .add_node("Add", ["t1b", "b1"], ["t2"])
        .add_node("Relu", ["t2"], ["t3"])
        .add_node("MatMul", ["t3", "W2"], ["t4"])
        .add_node("Add", ["t4", "b2"], ["t5"])
        .add_node("Softmax", ["t5"], ["Y"], axis=1)
        .add_node("Add", ["c1", "c2"], ["s"])           # folds, then dead
        .add_node("Sigmoid", ["X"], ["dead"])           # dead branch
        .add_output("Y", "float32", (1, 2))
        .build()
    )


def main():
    graph = build_mlp()
    result = optimize_and_compare(graph)
    comp = result["comparison"]

    print("=== before -> after ===")
    for key in ("num_nodes", "num_edges", "num_initializers", "graph_depth",
                "total_parameters", "total_size_bytes"):
        m = comp["metrics"][key]
        delta = m["delta"]
        suffix = f" ({delta:+d})" if delta else " (no change)"
        print(f"  {key:<18} {m['before']} -> {m['after']}{suffix}")
    if comp["op_delta"]:
        changes = ", ".join(f"{op} {d:+d}" for op, d in comp["op_delta"].items())
        print(f"  op changes: {changes}")

    print("\n=== passes ===")
    for r in result["pass_results"]:
        print(f"  {r.pass_name:<22} {'changed' if r.changed else 'no change'}")

    print("\n=== shape inference on optimized graph ===")
    shapes = infer_shapes(result["graph"])
    for name, info in sorted(shapes["tensors"].items()):
        print(f"  {name:<6} {str(info['shape']):10} known={info['known']}")
    print(f"coverage: {shapes['coverage']:.0%} of tensors have known shapes")


if __name__ == "__main__":
    main()
