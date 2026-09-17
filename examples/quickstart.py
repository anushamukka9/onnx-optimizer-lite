"""Quickstart example: build a small graph, analyze it, optimize it.

Run with:  python examples/quickstart.py
(Works with only numpy installed — no onnx/onnxruntime needed.)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from onnx_optimizer_lite import GraphBuilder, optimize
from onnx_optimizer_lite.report import build_report, render_text


def build_demo_graph():
    """A tiny classifier with a dead branch, an identity, and foldable constants."""
    rng = np.random.default_rng(0)
    return (
        GraphBuilder("demo-classifier")
        .add_input("X", "float32", (1, 8))
        .add_initializer("W1", rng.standard_normal((8, 16), dtype=np.float32))
        .add_initializer("b1", rng.standard_normal((16,), dtype=np.float32))
        .add_initializer("W2", rng.standard_normal((16, 4), dtype=np.float32))
        .add_node("MatMul", ["X", "W1"], ["h1"])
        .add_node("Add", ["h1", "b1"], ["h2"])
        .add_node("Relu", ["h2"], ["h3"])
        .add_node("Identity", ["h3"], ["h4"])          # will be bypassed
        .add_node("MatMul", ["h4", "W2"], ["logits"])
        .add_node("Softmax", ["logits"], ["probs"])
        # dead branch: never reaches an output
        .add_node("Tanh", ["X"], ["dead1"])
        .add_constant("scale", np.array([0.5], dtype=np.float32))
        .add_node("Mul", ["dead1", "scale"], ["dead2"])
        # foldable: both inputs constant, but dead -> folded then pruned
        .add_constant("k1", np.array([2.0, 3.0], dtype=np.float32))
        .add_constant("k2", np.array([4.0, 5.0], dtype=np.float32))
        .add_node("Add", ["k1", "k2"], ["ksum"])
        .add_output("probs", "float32", (1, 4))
        .build()
    )


def main():
    graph = build_demo_graph()

    print("=== BEFORE OPTIMIZATION ===")
    print(render_text(build_report(graph)))

    optimized, results = optimize(graph)

    print("=== AFTER OPTIMIZATION ===")
    print(render_text(build_report(optimized, pass_results=results)))

    out_path = os.path.join(os.path.dirname(__file__), "demo_graph.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(optimized.to_dict(), fh, indent=2)
    print(f"Saved optimized graph JSON to {out_path}")
    print("Try: onnx-opt analyze examples/demo_graph.json --format markdown")


if __name__ == "__main__":
    main()
