"""onnx-optimizer-lite: analyze and optimize ONNX-style computation graphs.

The core works on a lightweight internal graph IR and needs only numpy.
Reading real ``.onnx`` files needs the optional ``onnx`` extra, and the
latency benchmark needs the optional ``onnxruntime`` extra.
"""

from .analyzer import (
    analyze,
    compare_analysis,
    graph_depth,
    infer_shapes,
    initializer_report,
    op_counts,
)
from .benchmark import BenchmarkResult, benchmark_onnx
from .graph import Graph, GraphBuilder, Node, TensorSpec
from .optimizer import (
    PassResult,
    eliminate_dead_nodes,
    fold_constants,
    optimize,
    optimize_and_compare,
    remove_identity_nodes,
)
from .quant import dtype_report, quantization_guidance, size_after_dtype_conversion
from .report import build_report, render_json, render_markdown, render_text

__version__ = "0.1.0"

__all__ = [
    "Graph",
    "GraphBuilder",
    "Node",
    "TensorSpec",
    "PassResult",
    "BenchmarkResult",
    "analyze",
    "compare_analysis",
    "graph_depth",
    "infer_shapes",
    "initializer_report",
    "op_counts",
    "eliminate_dead_nodes",
    "fold_constants",
    "remove_identity_nodes",
    "optimize",
    "optimize_and_compare",
    "dtype_report",
    "quantization_guidance",
    "size_after_dtype_conversion",
    "build_report",
    "render_json",
    "render_markdown",
    "render_text",
    "benchmark_onnx",
    "__version__",
]
