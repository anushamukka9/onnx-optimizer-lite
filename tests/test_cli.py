"""CLI tests. Everything here runs without onnx/onnxruntime installed."""

import json

import pytest

from onnx_optimizer_lite.cli import main

try:
    import onnx  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def _write_graph_json(graph, tmp_path, name="model.json"):
    path = tmp_path / name
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    return str(path)


def test_analyze_text_to_stdout(mlp_graph, tmp_path, capsys):
    path = _write_graph_json(mlp_graph, tmp_path)
    assert main(["analyze", path]) == 0
    out = capsys.readouterr().out
    assert "ONNX Optimizer Lite report: mlp" in out
    assert "Operator counts" in out
    assert "MatMul" in out


def test_analyze_markdown_format(mlp_graph, tmp_path, capsys):
    path = _write_graph_json(mlp_graph, tmp_path)
    assert main(["analyze", path, "--format", "markdown"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ONNX Optimizer Lite report: `mlp`")
    assert "| MatMul | 2 |" in out


def test_analyze_json_format_is_parseable(mlp_graph, tmp_path, capsys):
    path = _write_graph_json(mlp_graph, tmp_path)
    assert main(["analyze", path, "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["analysis"]["total_parameters"] == 58


def test_analyze_writes_output_file(mlp_graph, tmp_path):
    src = _write_graph_json(mlp_graph, tmp_path)
    dest = str(tmp_path / "report.md")
    assert main(["analyze", src, "--format", "markdown", "--output", dest]) == 0
    assert "Operator counts" in open(dest, encoding="utf-8").read()


def test_optimize_writes_json_and_reports(mlp_graph, tmp_path, capsys):
    src = _write_graph_json(mlp_graph, tmp_path)
    dest = str(tmp_path / "optimized.json")
    assert main(["optimize", src, "--output", dest]) == 0
    out = capsys.readouterr().out
    assert "Optimization:" in out
    saved = json.loads(open(dest, encoding="utf-8").read())
    assert len(saved["nodes"]) == 6


def test_optimize_dead_branch_graph(tmp_path, capsys, graph_with_dead_branch):
    src = _write_graph_json(graph_with_dead_branch, tmp_path)
    dest = str(tmp_path / "optimized.json")
    assert main(["optimize", src, "--output", dest]) == 0
    saved = json.loads(open(dest, encoding="utf-8").read())
    assert {n["op"] for n in saved["nodes"]} == {"MatMul", "Relu"}
    assert "orphan" not in saved["initializers"]


def test_quant_command(mlp_graph, tmp_path, capsys):
    path = _write_graph_json(mlp_graph, tmp_path)
    assert main(["quant", path]) == 0
    out = capsys.readouterr().out
    assert "dynamic INT8 estimate" in out
    assert "recommendations:" in out


def test_quant_json(mlp_graph, tmp_path, capsys):
    path = _write_graph_json(mlp_graph, tmp_path)
    assert main(["quant", path, "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["quantization"]["quantization_friendly_ops"] == {"MatMul": 2}


def test_unsupported_file_errors_cleanly(tmp_path, capsys):
    bad = tmp_path / "model.txt"
    bad.write_text("not a graph", encoding="utf-8")
    assert main(["analyze", str(bad)]) == 1
    assert "unsupported extension" in capsys.readouterr().err


@pytest.mark.skipif(HAS_ONNX, reason="requires onnx to be absent")
def test_onnx_file_without_onnx_installed_errors_cleanly(capsys):
    assert main(["analyze", "model.onnx"]) == 2
    err = capsys.readouterr().err
    assert "pip install onnx-optimizer-lite[onnx]" in err


@pytest.mark.skipif(HAS_ONNX, reason="requires onnx to be absent")
def test_bench_without_runtime_errors_cleanly(capsys):
    assert main(["bench", "model.onnx"]) == 2
    assert "onnxruntime" in capsys.readouterr().err
