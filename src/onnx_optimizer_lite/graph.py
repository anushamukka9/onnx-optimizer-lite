"""Lightweight ONNX-like intermediate representation (IR).

Every analysis and optimization pass in this package operates on this IR, so
the core library never needs the ``onnx`` package installed. There are two
ways to obtain a :class:`Graph`:

* build one programmatically with :class:`GraphBuilder` (handy for tests,
  synthetic models, and environments without ``onnx``), or
* convert a real model with
  :func:`onnx_optimizer_lite.onnx_adapter.from_onnx` (requires the optional
  ``onnx`` extra).

Graphs round-trip through plain JSON via :meth:`Graph.to_dict` /
:meth:`Graph.from_dict`, which is also the format the CLI accepts when
``onnx`` is unavailable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Subset of onnx.TensorProto.DataType values. These are stable public
# constants from the ONNX spec, used here so dtype metadata survives a
# round-trip without importing onnx.
DTYPE_FLOAT = 1
DTYPE_UINT8 = 2
DTYPE_INT8 = 3
DTYPE_INT16 = 5
DTYPE_INT32 = 6
DTYPE_INT64 = 7
DTYPE_BOOL = 9
DTYPE_FLOAT16 = 10
DTYPE_DOUBLE = 11
DTYPE_UINT32 = 12
DTYPE_UINT64 = 13
DTYPE_BFLOAT16 = 16

_ONNX_DTYPE_TO_STR = {
    DTYPE_FLOAT: "float32",
    DTYPE_UINT8: "uint8",
    DTYPE_INT8: "int8",
    DTYPE_INT16: "int16",
    DTYPE_INT32: "int32",
    DTYPE_INT64: "int64",
    DTYPE_BOOL: "bool",
    DTYPE_FLOAT16: "float16",
    DTYPE_DOUBLE: "float64",
    DTYPE_UINT32: "uint32",
    DTYPE_UINT64: "uint64",
    DTYPE_BFLOAT16: "bfloat16",
}
_STR_TO_ONNX_DTYPE = {v: k for k, v in _ONNX_DTYPE_TO_STR.items()}


def onnx_dtype_to_str(dtype_int: int) -> str:
    """Map an ONNX TensorProto data-type int to a canonical dtype string."""
    try:
        return _ONNX_DTYPE_TO_STR[int(dtype_int)]
    except KeyError:
        raise ValueError(f"Unsupported ONNX data type: {dtype_int!r}")


def str_to_onnx_dtype(dtype: str) -> int:
    """Map a canonical dtype string back to an ONNX TensorProto data-type int."""
    try:
        return _STR_TO_ONNX_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype!r}")


_STR_TO_NUMPY = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
}


def numpy_dtype(dtype: str):
    """Return the numpy dtype for a canonical dtype string."""
    try:
        return _STR_TO_NUMPY[dtype]
    except KeyError:
        raise ValueError(f"No numpy equivalent for dtype {dtype!r}")


def dtype_str_of(array: np.ndarray) -> str:
    """Canonical dtype string for a numpy array."""
    return str(np.dtype(array.dtype).name)


# ---------------------------------------------------------------------------
# JSON helpers for numpy payloads
# ---------------------------------------------------------------------------

_NDARRAY_MARKER = "__ndarray__"


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            _NDARRAY_MARKER: True,
            "dtype": dtype_str_of(value),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, tuple):
        return [_encode_value(v) for v in value]
    if isinstance(value, list):
        return [_encode_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get(_NDARRAY_MARKER):
        return np.array(value["data"], dtype=numpy_dtype(value["dtype"])).reshape(
            value["shape"]
        )
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _decode_value(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# IR dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    """A named tensor: graph input/output, intermediate value, or initializer."""

    name: str
    dtype: str = "float32"
    shape: Tuple[Optional[int], ...] = ()
    data: Optional[np.ndarray] = None  # set for initializers / folded constants

    @property
    def numel(self) -> Optional[int]:
        """Number of elements, or None when the shape is dynamic/unknown."""
        if not self.shape or any(d is None for d in self.shape):
            return None
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def size_bytes(self) -> Optional[int]:
        """Storage size in bytes, or None when it cannot be determined."""
        if self.data is not None:
            return int(self.data.nbytes)
        from .quant import DTYPE_ITEMSIZE  # deferred to avoid a cycle

        itemsize = DTYPE_ITEMSIZE.get(self.dtype)
        n = self.numel
        if itemsize is None or n is None:
            return None
        return n * itemsize

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data": _encode_value(self.data),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorSpec":
        return cls(
            name=d["name"],
            dtype=d.get("dtype", "float32"),
            shape=tuple(d.get("shape", ())),
            data=_decode_value(d.get("data")),
        )


@dataclass
class Node:
    """A single operator invocation in the graph."""

    name: str
    op: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": _encode_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        return cls(
            name=d["name"],
            op=d["op"],
            inputs=list(d.get("inputs", [])),
            outputs=list(d.get("outputs", [])),
            attributes=_decode_value(d.get("attributes", {})),
        )


@dataclass
class Graph:
    """A whole model: inputs, outputs, nodes, and initializers."""

    name: str = "model"
    inputs: List[TensorSpec] = field(default_factory=list)
    outputs: List[TensorSpec] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    initializers: Dict[str, TensorSpec] = field(default_factory=dict)
    value_info: List[TensorSpec] = field(default_factory=list)

    def copy(self) -> "Graph":
        """Deep copy (arrays are duplicated so passes never alias the input)."""
        return copy.deepcopy(self)

    def producer_of(self, tensor: str) -> Optional[Node]:
        for node in self.nodes:
            if tensor in node.outputs:
                return node
        return None

    def validate(self) -> None:
        """Basic structural sanity checks; raises ValueError on problems."""
        known = {t.name for t in self.inputs} | set(self.initializers)
        seen_outputs: Dict[str, str] = {}
        for node in self.nodes:
            for out in node.outputs:
                if out in seen_outputs:
                    raise ValueError(
                        f"Tensor {out!r} produced by both "
                        f"{seen_outputs[out]!r} and {node.name!r}"
                    )
                seen_outputs[out] = node.name
            for inp in node.inputs:
                if inp not in known and inp not in seen_outputs:
                    raise ValueError(
                        f"Node {node.name!r} ({node.op}) consumes unknown "
                        f"tensor {inp!r}"
                    )
            known.update(node.outputs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "nodes": [n.to_dict() for n in self.nodes],
            "initializers": {k: v.to_dict() for k, v in self.initializers.items()},
            "value_info": [t.to_dict() for t in self.value_info],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Graph":
        return cls(
            name=d.get("name", "model"),
            inputs=[TensorSpec.from_dict(t) for t in d.get("inputs", [])],
            outputs=[TensorSpec.from_dict(t) for t in d.get("outputs", [])],
            nodes=[Node.from_dict(n) for n in d.get("nodes", [])],
            initializers={
                k: TensorSpec.from_dict(v)
                for k, v in d.get("initializers", {}).items()
            },
            value_info=[TensorSpec.from_dict(t) for t in d.get("value_info", [])],
        )


class GraphBuilder:
    """Fluent builder for :class:`Graph`. All methods return ``self``."""

    def __init__(self, name: str = "model"):
        self._graph = Graph(name=name)
        self._counter = 0

    def add_input(
        self, name: str, dtype: str = "float32", shape: Tuple[Optional[int], ...] = ()
    ) -> "GraphBuilder":
        self._graph.inputs.append(TensorSpec(name, dtype, tuple(shape)))
        return self

    def add_output(
        self, name: str, dtype: str = "float32", shape: Tuple[Optional[int], ...] = ()
    ) -> "GraphBuilder":
        self._graph.outputs.append(TensorSpec(name, dtype, tuple(shape)))
        return self

    def add_initializer(self, name: str, array: np.ndarray) -> "GraphBuilder":
        array = np.ascontiguousarray(array)
        self._graph.initializers[name] = TensorSpec(
            name, dtype_str_of(array), tuple(array.shape), array
        )
        return self

    def add_node(
        self,
        op: str,
        inputs: List[str],
        outputs: List[str],
        name: Optional[str] = None,
        **attrs: Any,
    ) -> "GraphBuilder":
        if name is None:
            name = f"{op}_{self._counter}"
            self._counter += 1
        self._graph.nodes.append(
            Node(name=name, op=op, inputs=list(inputs), outputs=list(outputs),
                 attributes=dict(attrs))
        )
        return self

    def add_constant(self, output: str, array: np.ndarray,
                     name: Optional[str] = None) -> "GraphBuilder":
        """Add a Constant node producing ``output`` with the given value."""
        return self.add_node("Constant", [], [output], name=name,
                             value=np.ascontiguousarray(array))

    def build(self) -> Graph:
        graph = self._graph
        self._graph = Graph(name=graph.name)
        return graph
