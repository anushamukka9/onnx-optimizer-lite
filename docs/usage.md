# Usage guide

## Concepts

`onnx-optimizer-lite` works on a small internal **graph IR** (`Graph`,
`Node`, `TensorSpec` in `onnx_optimizer_lite.graph`). The IR mirrors the
parts of ONNX that matter for analysis and optimization — operators, tensor
shapes/dtypes, initializers — without depending on the `onnx` package.

There are two ways to get a graph:

1. **Build one in Python** with `GraphBuilder` (great for tests, synthetic
   models, and quick experiments).
2. **Load a real model** with `onnx_optimizer_lite.onnx_adapter.load_onnx`
   (requires `pip install onnx-optimizer-lite[onnx]`).

Graphs serialize to plain JSON (`Graph.to_dict()` / `Graph.from_dict()`),
which the CLI accepts even when `onnx` is not installed.

## Analyzing a graph

```python
from onnx_optimizer_lite import GraphBuilder
from onnx_optimizer_lite.analyzer import analyze, initializer_report, graph_depth
import numpy as np

g = (GraphBuilder("m")
     .add_input("x", "float32", (1, 16))
     .add_initializer("w", np.ones((16, 8), dtype=np.float32))
     .add_node("MatMul", ["x", "w"], ["y"])
     .add_output("y", "float32", (1, 8))
     .build())

info = analyze(g)
print(info["op_counts"])          # {'MatMul': 1}
print(info["total_parameters"])   # 128
print(graph_depth(g))             # 1
for row in initializer_report(g):
    print(row["name"], row["bytes"], f'{row["pct_of_total"]}%')
```

`analyze()` returns a JSON-serializable dict with node/edge counts, graph
depth (longest path in nodes — a rough proxy for sequential inference work),
operator histogram, total parameters, total byte size, dtype histogram, and
the largest initializers with their share of total size.

## Optimizing a graph

```python
from onnx_optimizer_lite import optimize

optimized, results = optimize(g)          # default: identity -> fold -> dce
for r in results:
    print(r.pass_name, "changed:" , r.changed, r.details)
```

Available passes (`onnx_optimizer_lite.optimizer.PASSES`):

- `remove_identity` — bypasses `Identity` nodes by rewiring consumers.
  Identities feeding a graph output are kept, since removing them would
  rename the model's outputs.
- `fold_constants` — evaluates subgraphs whose inputs are all constants
  (initializers or `Constant` nodes) and replaces them with a single
  `Constant`. Supported ops: `Add Sub Mul Div MatMul Gemm Neg Sqrt Relu
  Sigmoid Tanh Pow Concat Transpose Reshape Cast`. Unsupported ops, dynamic
  shapes, or missing attributes are skipped, never crashed on.
- `eliminate_dead_nodes` — removes nodes that cannot reach a graph output,
  and prunes initializers that only fed dead nodes.

Passes are pure: the input graph is never mutated. Run a custom pipeline
with `optimize(g, passes=("fold_constants", "eliminate_dead_nodes"))`.

## Quantization guidance

```python
from onnx_optimizer_lite import quantization_guidance

g2 = quantization_guidance(g)
print(g2["est_int8_dynamic_weights_mb"])   # ~4x smaller weights estimate
for rec in g2["recommendations"]:
    print("-", rec)
```

This is guidance, not a quantizer: it estimates the weight-memory win from
dynamic INT8 quantization (weights → INT8, activations stay float) and from
float16 conversion, flags quantization-friendly ops (`MatMul`, `Gemm`,
`Conv`) versus precision-sensitive ones (`Softmax`, `LayerNorm`, …), and
emits concrete, model-specific recommendations.

## Reports

```python
from onnx_optimizer_lite.report import build_report, render_text, render_markdown, render_json

report = build_report(optimized, pass_results=results)
print(render_text(report))       # human-readable CLI-style report
open("report.md", "w").write(render_markdown(report))
open("report.json", "w").write(render_json(report))
```

## Benchmarking latency

```python
from onnx_optimizer_lite import benchmark_onnx

res = benchmark_onnx("model.onnx", runs=100, warmup=10)
print(res.summary())
```

Requires `pip install onnx-optimizer-lite[bench]` (`onnxruntime`). The
harness builds random float32 inputs from the model's input specs (dynamic
axes become 1), warms up, then reports mean / p50 / p95 / min / max latency
in milliseconds.

## CLI reference

```
onnx-opt analyze  model.onnx|graph.json [--format text|markdown|json] [--output FILE]
onnx-opt optimize  model.onnx|graph.json [--passes p1,p2] [--output optimized.onnx|json]
onnx-opt quant     model.onnx|graph.json [--format text|json]
onnx-opt bench     model.onnx [--runs N] [--warmup N] [--provider PROVIDER]
```

`.onnx` inputs need the `onnx` extra; `.json` graphs (produced by
`Graph.to_dict()` or the `optimize` command) always work.

## FAQ

**Do I need `onnx` installed?**
Only to read/write real `.onnx` files. Analysis, optimization, reports, and
the JSON workflow run on numpy alone, and every function that needs an
optional dependency raises a clear error naming the exact `pip install`
command.

**Is constant folding safe?**
Folding only fires when *every* input is a known constant and the op is in
the supported list. Anything ambiguous is left in place, and the pass
returns exactly which nodes were folded so you can audit it.

**Why is my optimized graph identical?**
Clean graphs often are — `optimize()` is a no-op when there is nothing to
fold, bypass, or prune. Each `PassResult.changed` flag tells you whether a
pass did anything.
