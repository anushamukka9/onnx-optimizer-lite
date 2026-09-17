# onnx-optimizer-lite

Analyze and optimize ONNX-style computation graphs: operator statistics,
initializer size analysis, dead-node elimination, constant folding, dtype /
quantization guidance, and an `onnxruntime` latency benchmark.

The core library works on a lightweight internal graph IR and needs **only
numpy**. Reading real `.onnx` files needs the optional `onnx` extra; the
latency benchmark needs the optional `onnxruntime` extra. Everything else —
analysis, optimization, reports, the CLI — works without them.

## Install

```bash
pip install onnx-optimizer-lite            # core only (numpy)
pip install "onnx-optimizer-lite[onnx]"    # + read/write real .onnx files
pip install "onnx-optimizer-lite[bench]"   # + onnxruntime latency benchmark
pip install "onnx-optimizer-lite[all]"     # everything
```

## Quickstart

```python
from onnx_optimizer_lite import GraphBuilder, optimize
from onnx_optimizer_lite.report import build_report, render_text

graph = (
    GraphBuilder("tiny")
    .add_input("X", "float32", (1, 8))
    .add_initializer("W", __import__("numpy").ones((8, 4), dtype="float32"))
    .add_node("MatMul", ["X", "W"], ["Y"])
    .add_output("Y", "float32", (1, 4))
    .build()
)

optimized, passes = optimize(graph)
print(render_text(build_report(optimized, pass_results=passes)))
```

Or run the bundled example (no `onnx` needed):

```bash
python examples/quickstart.py
```

## CLI

```bash
# Analyze a model (needs onnx extra) or a JSON graph (always works)
onnx-opt analyze model.onnx
onnx-opt analyze graph.json --format markdown --output report.md

# Optimize: identity removal + constant folding + dead-node elimination
onnx-opt optimize model.onnx --output optimized.onnx
onnx-opt optimize graph.json --passes fold_constants,eliminate_dead_nodes

# Quantization guidance: INT8 estimates, sensitive ops, recommendations
onnx-opt quant model.onnx
onnx-opt quant model.onnx --format json

# Latency benchmark (needs onnxruntime extra)
onnx-opt bench model.onnx --runs 100 --warmup 10
```

## Python API

| Module | What it does |
| --- | --- |
| `onnx_optimizer_lite.graph` | `Graph` IR, `TensorSpec`, `Node`, `GraphBuilder`, JSON round-trip |
| `onnx_optimizer_lite.analyzer` | `analyze()`, `op_counts()`, `initializer_report()`, `graph_depth()`, `topological_order()` |
| `onnx_optimizer_lite.optimizer` | `optimize()`, `eliminate_dead_nodes()`, `fold_constants()`, `remove_identity_nodes()` |
| `onnx_optimizer_lite.quant` | `quantization_guidance()`, `dtype_report()`, `size_after_dtype_conversion()` |
| `onnx_optimizer_lite.report` | `build_report()`, `render_text/markdown/json()` |
| `onnx_optimizer_lite.benchmark` | `benchmark_onnx()` → mean/p50/p95 latency (needs `onnxruntime`) |
| `onnx_optimizer_lite.onnx_adapter` | `from_onnx()` / `to_onnx()` / `load_onnx()` / `save_onnx()` (needs `onnx`) |

See [docs/usage.md](docs/usage.md) for the full usage guide.

## Architecture

```
                    +------------------+
                    |  .onnx file      |  (optional: needs `onnx` extra)
                    +--------+---------+
                             | from_onnx / to_onnx
                    +--------v---------+
                    |  Graph IR        |  graph.py — the single data model
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
 +------v------+     +-------v-------+    +-------v-------+
 | analyzer    |     | optimizer     |    | quant         |
 | op counts,  |     | dead-node     |    | dtype report, |
 | sizes,      |     | elimination,  |    | INT8 guidance |
 | depth       |     | const folding |    +----------------+
 +------+------+     +-------+-------+
        |                    |
        +---------+----------+
                  |
           +------v------+
           | report      |  text / markdown / JSON
           +-------------+
```

All passes are pure functions: they take a `Graph` and return a new one plus
a result object, never mutating the input.

## Optional dependencies

| Feature | Requirement | Without it |
| --- | --- | --- |
| `analyze` / `optimize` / `quant` on `.onnx` | `pip install onnx-optimizer-lite[onnx]` | clear error message; use `.json` graphs instead |
| `bench` | `pip install onnx-optimizer-lite[bench]` | clear error message |
| Everything else | just `numpy` | works |

## Development

```bash
pip install -e .[dev]
pytest -q
```

## License

MIT — Copyright (c) 2026 Anusha Mukka. See [LICENSE](LICENSE).
