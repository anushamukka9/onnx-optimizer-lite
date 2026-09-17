import json

from onnx_optimizer_lite.optimizer import optimize
from onnx_optimizer_lite.report import (
    build_report,
    render_json,
    render_markdown,
    render_text,
)


def test_build_report_sections(mlp_graph):
    report = build_report(mlp_graph)
    assert set(report) == {"analysis", "quantization"}
    assert report["analysis"]["num_nodes"] == 6
    assert "recommendations" in report["quantization"]


def test_build_report_with_optimization(graph_with_dead_branch):
    optimized, results = optimize(graph_with_dead_branch)
    report = build_report(optimized, pass_results=results)
    assert "optimization" in report
    dce = next(p for p in report["optimization"]
               if p["pass"] == "eliminate_dead_nodes")
    assert dce["changed"] is True
    assert set(dce["details"]["removed_nodes"]) == {"Sigmoid_2", "Add_3"}


def test_render_json_roundtrip(mlp_graph):
    text = render_json(build_report(mlp_graph))
    data = json.loads(text)
    assert data["analysis"]["op_counts"]["MatMul"] == 2
    assert data["quantization"]["weights_bytes"] == 232


def test_render_text_contains_key_sections(mlp_graph):
    text = render_text(build_report(mlp_graph))
    for section in (
        "Graph structure",
        "Operator counts",
        "Parameters & size",
        "Largest initializers",
        "Quantization guidance",
        "MatMul",
        "W1",
    ):
        assert section in text


def test_render_markdown_tables(mlp_graph):
    md = render_markdown(build_report(mlp_graph))
    assert md.startswith("# ONNX Optimizer Lite report: `mlp`")
    assert "| op | count |" in md
    assert "| MatMul | 2 |" in md
    assert "`W1`" in md
