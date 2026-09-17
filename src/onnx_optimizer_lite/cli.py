"""Command-line interface: ``onnx-opt analyze|optimize|quant|bench ...``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import __version__
from .graph import Graph
from .optimizer import DEFAULT_PASSES, PASSES, optimize
from .report import (
    build_report,
    render_json,
    render_markdown,
    render_text,
)


def load_graph(path: str) -> Graph:
    """Load a graph from ``.onnx`` (needs the ``onnx`` extra) or ``.json``."""
    lowered = path.lower()
    if lowered.endswith(".onnx"):
        from . import onnx_adapter

        return onnx_adapter.load_onnx(path)
    if lowered.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            return Graph.from_dict(json.load(fh))
    # Best effort: sniff JSON content for anything else.
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return Graph.from_dict(json.load(fh))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError(
            f"cannot read {path!r}: unsupported extension. Use a .onnx "
            "file (requires: pip install onnx-optimizer-lite[onnx]) or a "
            ".json graph exported by this tool."
        )


def save_graph(graph: Graph, path: str) -> None:
    """Write a graph to ``.onnx`` (needs the ``onnx`` extra) or ``.json``."""
    if path.lower().endswith(".onnx"):
        from . import onnx_adapter

        onnx_adapter.save_onnx(graph, path)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(graph.to_dict(), fh, indent=2)


def cmd_analyze(args: argparse.Namespace) -> int:
    graph = load_graph(args.model)
    graph.validate()
    report = build_report(graph)
    text = {"text": render_text, "markdown": render_markdown, "json": render_json}[
        args.format
    ](report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {args.format} report to {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    graph = load_graph(args.model)
    graph.validate()
    passes = tuple(p.strip() for p in args.passes.split(",") if p.strip())
    optimized, results = optimize(graph, passes=passes)
    report = build_report(optimized, pass_results=results)
    if args.output:
        save_graph(optimized, args.output)
        print(f"Wrote optimized graph to {args.output}")
    print(render_text(report))
    total_removed = sum(
        len(r.details.get("removed_nodes", [])) for r in results
    )
    print(
        f"Optimization: {len(optimized.nodes)} nodes "
        f"({total_removed} removed), "
        f"{len(optimized.initializers)} initializers."
    )
    return 0


def cmd_quant(args: argparse.Namespace) -> int:
    from .quant import quantization_guidance

    graph = load_graph(args.model)
    graph.validate()
    guidance = quantization_guidance(graph)
    if args.format == "json":
        print(render_json({"quantization": guidance}))
        return 0
    print(f"Quantization guidance: {graph.name}")
    print(f"  weights: {guidance['weights_mb']:.2f} MB")
    print(
        f"  dynamic INT8 estimate: ~{guidance['est_int8_dynamic_weights_mb']:.2f} MB"
    )
    fp16 = guidance["float16_conversion"]
    print(
        f"  float16 conversion: saves {fp16['saved_pct']:.1f}% "
        f"({fp16['converted_tensors']} tensors)"
    )
    print("  recommendations:")
    for rec in guidance["recommendations"]:
        print(f"    - {rec}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .benchmark import benchmark_onnx

    if not args.model.lower().endswith(".onnx"):
        raise SystemExit("error: bench needs a real .onnx model file.")
    result = benchmark_onnx(
        args.model, runs=args.runs, warmup=args.warmup, providers=args.provider
    )
    print(result.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-opt",
        description="Analyze and optimize ONNX-style computation graphs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Print a graph statistics report.")
    p.add_argument("model", help="Model file (.onnx needs the onnx extra; .json always works)")
    p.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    p.add_argument("--output", help="Write the report to a file instead of stdout")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("optimize", help="Run optimization passes on a graph.")
    p.add_argument("model", help="Model file (.onnx or .json)")
    p.add_argument(
        "--passes",
        default=",".join(DEFAULT_PASSES),
        help=f"Comma-separated passes (available: {', '.join(sorted(PASSES))})",
    )
    p.add_argument("--output", help="Write the optimized graph (.onnx or .json)")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("quant", help="Show dtype/quantization guidance.")
    p.add_argument("model", help="Model file (.onnx or .json)")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_quant)

    p = sub.add_parser("bench", help="Benchmark .onnx latency via onnxruntime.")
    p.add_argument("model", help="Model file (.onnx)")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--provider", action="append", default=None,
                   help="Execution provider (repeatable)")
    p.set_defaults(func=cmd_bench)

    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:  # missing optional dependency -> clean message
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
