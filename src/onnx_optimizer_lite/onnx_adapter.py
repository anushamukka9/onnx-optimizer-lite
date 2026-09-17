"""Bridge between real ONNX models and the internal IR.

Everything in this module requires the optional ``onnx`` extra
(``pip install onnx-optimizer-lite[onnx]``). Importing this module is safe
without it — the error is raised only when you actually call a function that
needs ``onnx`` — so ``import onnx_optimizer_lite.onnx_adapter`` never breaks
a minimal install.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .graph import (
    Graph,
    TensorSpec,
    dtype_str_of,
    numpy_dtype,
    onnx_dtype_to_str,
    str_to_onnx_dtype,
)


def require_onnx():
    """Import onnx or raise a helpful error if it is missing."""
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Reading/writing .onnx files needs the 'onnx' package, which is "
            "not installed. Install it with: pip install onnx-optimizer-lite[onnx] "
            "(or analyze a JSON graph instead — see docs/usage.md)."
        ) from exc
    import onnx

    return onnx


def _value_info_to_spec(onnx, vi) -> TensorSpec:
    from onnx import TensorProto

    ttype = vi.type.tensor_type
    dtype = onnx_dtype_to_str(ttype.elem_type)
    shape = []
    for dim in ttype.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(None)  # dynamic / symbolic dimension
    return TensorSpec(name=vi.name, dtype=dtype, shape=tuple(shape))


def _attribute_to_python(onnx, attr) -> Any:
    from onnx import AttributeProto

    if attr.type == AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == AttributeProto.INT:
        return int(attr.i)
    if attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if attr.type == AttributeProto.TENSOR:
        return _tensor_to_array(onnx, attr.t)
    if attr.type == AttributeProto.FLOATS:
        return [float(x) for x in attr.floats]
    if attr.type == AttributeProto.INTS:
        return [int(x) for x in attr.ints]
    if attr.type == AttributeProto.STRINGS:
        return [s.decode("utf-8", errors="replace") for s in attr.strings]
    # Graphs / sparse tensors / tensor lists are out of scope for the IR.
    raise ValueError(f"Unsupported ONNX attribute type: {attr.type}")


def _tensor_to_array(onnx, tensor) -> np.ndarray:
    from onnx import numpy_helper

    return np.ascontiguousarray(numpy_helper.to_array(tensor))


def from_onnx(model) -> Graph:
    """Convert an ``onnx.ModelProto`` into the internal :class:`Graph` IR."""
    onnx = require_onnx()
    from .graph import Node

    og = model.graph
    graph = Graph(name=og.name or "model")

    for vi in og.input:
        graph.inputs.append(_value_info_to_spec(onnx, vi))
    for vi in og.output:
        graph.outputs.append(_value_info_to_spec(onnx, vi))
    for vi in og.value_info:
        graph.value_info.append(_value_info_to_spec(onnx, vi))

    for init in og.initializer:
        array = _tensor_to_array(onnx, init)
        graph.initializers[init.name] = TensorSpec(
            name=init.name,
            dtype=dtype_str_of(array),
            shape=tuple(array.shape),
            data=array,
        )

    used_names: set = set()

    def unique(name: str, op: str, idx: int) -> str:
        base = name or f"{op}_{idx}"
        candidate, n = base, 1
        while candidate in used_names:
            n += 1
            candidate = f"{base}_{n}"
        used_names.add(candidate)
        return candidate

    for i, node in enumerate(og.node):
        attrs: Dict[str, Any] = {}
        for attr in node.attribute:
            try:
                attrs[attr.name] = _attribute_to_python(onnx, attr)
            except ValueError:
                continue  # skip attributes the IR cannot represent
        graph.nodes.append(
            Node(
                name=unique(node.name, node.op_type, i),
                op=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attrs,
            )
        )
    return graph


def to_onnx(graph: Graph):
    """Convert the internal :class:`Graph` IR back to an ``onnx.ModelProto``."""
    onnx = require_onnx()
    from onnx import helper, numpy_helper

    def make_vi(spec: TensorSpec):
        return helper.make_tensor_value_info(
            spec.name,
            str_to_onnx_dtype(spec.dtype),
            [d if d is not None else 1 for d in spec.shape] or None,
        )

    initializers = []
    for spec in graph.initializers.values():
        if spec.data is None:
            raise ValueError(
                f"Initializer {spec.name!r} has no data; cannot export to ONNX"
            )
        initializers.append(numpy_helper.from_array(spec.data, name=spec.name))

    nodes = []
    for node in graph.nodes:
        attrs: Dict[str, Any] = {}
        for key, value in node.attributes.items():
            if (
                node.op == "Constant"
                and key == "value"
                and isinstance(value, np.ndarray)
            ):
                attrs[key] = numpy_helper.from_array(
                    np.ascontiguousarray(value), name=f"{node.name}_value"
                )
            elif isinstance(value, np.ndarray):
                attrs[key] = numpy_helper.from_array(
                    np.ascontiguousarray(value), name=f"{node.name}_{key}"
                )
            elif isinstance(value, bool):
                attrs[key] = int(value)
            else:
                attrs[key] = value
        nodes.append(
            helper.make_node(
                node.op, node.inputs, node.outputs, name=node.name, **attrs
            )
        )

    onnx_graph = helper.make_graph(
        nodes,
        graph.name,
        [make_vi(t) for t in graph.inputs],
        [make_vi(t) for t in graph.outputs],
        initializer=initializers,
        value_info=[make_vi(t) for t in graph.value_info],
    )
    model = helper.make_model(onnx_graph, producer_name="onnx-optimizer-lite")
    onnx.checker.check_model(model)
    return model


def load_onnx(path: str) -> Graph:
    """Load a ``.onnx`` file into the IR (requires the ``onnx`` extra)."""
    onnx = require_onnx()
    return from_onnx(onnx.load(path))


def save_onnx(graph: Graph, path: str) -> None:
    """Write the IR out as a ``.onnx`` file (requires the ``onnx`` extra)."""
    onnx = require_onnx()
    onnx.save(to_onnx(graph), path)
