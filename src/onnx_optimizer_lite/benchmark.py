"""Latency benchmark harness for ONNX models.

Requires the optional ``onnxruntime`` extra (``pip install
onnx-optimizer-lite[bench]``). Without it, every public function raises a
clear error telling you exactly what to install — the rest of the package
keeps working.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def require_onnxruntime():
    """Import onnxruntime or raise a helpful error if it is missing."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Benchmarking needs the 'onnxruntime' package, which is not "
            "installed. Install it with: pip install onnx-optimizer-lite[bench]"
        ) from exc
    import onnxruntime

    return onnxruntime


@dataclass
class BenchmarkResult:
    """Latency statistics for one model / configuration."""

    model_path: str
    runs: int
    warmup: int
    providers: List[str] = field(default_factory=list)
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def summary(self) -> str:
        prov = ", ".join(self.providers) if self.providers else "default"
        return (
            f"Benchmark: {self.model_path} [{prov}]\n"
            f"  runs: {self.runs} (+{self.warmup} warmup)\n"
            f"  mean: {self.mean_ms:.3f} ms\n"
            f"  p50:  {self.p50_ms:.3f} ms\n"
            f"  p95:  {self.p95_ms:.3f} ms\n"
            f"  min:  {self.min_ms:.3f} ms / max: {self.max_ms:.3f} ms"
        )


def benchmark_onnx(
    model_path: str,
    runs: int = 50,
    warmup: int = 10,
    providers: Optional[List[str]] = None,
    seed: int = 0,
) -> BenchmarkResult:
    """Measure per-run inference latency of an ONNX file with onnxruntime.

    Runs ``warmup`` untimed iterations, then ``runs`` timed iterations over
    random float32 inputs, and returns mean/p50/p95/min/max latency.
    """
    ort = require_onnxruntime()
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    providers = providers or ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    rng = np.random.default_rng(seed)
    feed = {}
    for spec in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
        feed[spec.name] = rng.standard_normal(shape, dtype=np.float32)
    output_names = [o.name for o in session.get_outputs()]

    for _ in range(warmup):
        session.run(output_names, feed)

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(output_names, feed)
        latencies.append((time.perf_counter() - start) * 1000.0)

    lat = np.array(latencies)
    return BenchmarkResult(
        model_path=model_path,
        runs=runs,
        warmup=warmup,
        providers=list(session.get_providers()),
        mean_ms=float(np.mean(lat)),
        p50_ms=float(np.percentile(lat, 50)),
        p95_ms=float(np.percentile(lat, 95)),
        min_ms=float(np.min(lat)),
        max_ms=float(np.max(lat)),
    )
