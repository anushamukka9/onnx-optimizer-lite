"""Render analysis/optimization/quantization results as text, Markdown, or JSON."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from .analyzer import analyze
from .graph import Graph
from .optimizer import PassResult
from .quant import quantization_guidance


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def build_report(
    graph: Graph,
    analysis: Optional[Dict] = None,
    quant: Optional[Dict] = None,
    pass_results: Optional[List[PassResult]] = None,
) -> Dict:
    """Assemble the full report dict; missing pieces are computed on demand."""
    if analysis is None:
        analysis = analyze(graph)
    if quant is None:
        quant = quantization_guidance(graph)
    report = {"analysis": analysis, "quantization": quant}
    if pass_results is not None:
        report["optimization"] = [
            {
                "pass": r.pass_name,
                "changed": r.changed,
                "details": r.details,
            }
            for r in pass_results
        ]
    return report


def render_json(report: Dict) -> str:
    return json.dumps(report, indent=2, default=_json_default)


def _fmt_mb(nbytes: Optional[int]) -> str:
    if nbytes is None:
        return "unknown"
    return f"{nbytes / (1024 * 1024):.2f} MB"


def render_text(report: Dict) -> str:
    a = report["analysis"]
    q = report["quantization"]
    lines = [
        f"ONNX Optimizer Lite report: {a['graph_name']}",
        "=" * 60,
        "",
        "Graph structure",
        f"  nodes:        {a['num_nodes']}",
        f"  edges:        {a['num_edges']}",
        f"  depth:        {a['graph_depth']} (longest path in nodes)",
        f"  inputs:       {a['num_inputs']}",
        f"  outputs:      {a['num_outputs']}",
        f"  initializers: {a['num_initializers']}",
        "",
        "Operator counts",
    ]
    for op, n in sorted(a["op_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {op:<22} {n}")
    lines += [
        "",
        "Parameters & size",
        f"  total params: {a['total_parameters']:,}",
        f"  total size:   {_fmt_mb(a['total_size_bytes'])}",
        "",
        "Largest initializers",
    ]
    for row in a["largest_initializers"]:
        shape = "x".join(str(d) for d in row["shape"])
        lines.append(
            f"  {row['name']:<24} {row['dtype']:<8} [{shape}] "
            f"{_fmt_mb(row['bytes'])} ({row['pct_of_total']}%)"
        )
    lines += [
        "",
        "Quantization guidance",
        f"  weights: {_fmt_mb(q['weights_bytes'])} -> "
        f"~{q['est_int8_dynamic_weights_mb']:.2f} MB with dynamic INT8",
    ]
    for rec in q["recommendations"]:
        lines.append(f"  - {rec}")

    if "optimization" in report:
        lines += ["", "Optimization passes"]
        for p in report["optimization"]:
            status = "changed" if p["changed"] else "no change"
            lines.append(f"  {p['pass']:<22} {status}")
            for key, value in p["details"].items():
                if isinstance(value, list) and value and key != "foldable_ops":
                    shown = ", ".join(value[:5])
                    extra = f" (+{len(value) - 5} more)" if len(value) > 5 else ""
                    lines.append(f"    {key}: {shown}{extra}")
    return "\n".join(lines) + "\n"


def render_markdown(report: Dict) -> str:
    a = report["analysis"]
    q = report["quantization"]
    lines = [
        f"# ONNX Optimizer Lite report: `{a['graph_name']}`",
        "",
        "## Graph structure",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| nodes | {a['num_nodes']} |",
        f"| edges | {a['num_edges']} |",
        f"| depth (longest path) | {a['graph_depth']} |",
        f"| inputs | {a['num_inputs']} |",
        f"| outputs | {a['num_outputs']} |",
        f"| initializers | {a['num_initializers']} |",
        f"| total params | {a['total_parameters']:,} |",
        f"| total size | {_fmt_mb(a['total_size_bytes'])} |",
        "",
        "## Operator counts",
        "",
        "| op | count |",
        "| --- | --- |",
    ]
    for op, n in sorted(a["op_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {op} | {n} |")
    lines += [
        "",
        "## Largest initializers",
        "",
        "| name | dtype | shape | size | share |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in a["largest_initializers"]:
        shape = "x".join(str(d) for d in row["shape"])
        lines.append(
            f"| `{row['name']}` | {row['dtype']} | {shape} | "
            f"{_fmt_mb(row['bytes'])} | {row['pct_of_total']}% |"
        )
    lines += [
        "",
        "## Quantization guidance",
        "",
        f"Weights: {_fmt_mb(q['weights_bytes'])} -> "
        f"~{q['est_int8_dynamic_weights_mb']:.2f} MB with dynamic INT8.",
        "",
    ]
    for rec in q["recommendations"]:
        lines.append(f"- {rec}")

    if "optimization" in report:
        lines += ["", "## Optimization passes", ""]
        for p in report["optimization"]:
            status = "changed the graph" if p["changed"] else "no change"
            lines.append(f"- **{p['pass']}**: {status}")
    return "\n".join(lines) + "\n"
