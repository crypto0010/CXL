"""ONNX graph ingestion with real shape resolution.

History: the original parser hardcoded ``output_tensor_bytes=0``.  That value
flowed into every manifest, and from there into the runtime, which passed
zero-byte operands to every executor.  The submitted paper's performance
numbers were therefore measurements of empty protocol round-trips.

The rule now is: a tensor whose size cannot be resolved is an *error*, never
a zero.  Callers that genuinely want to proceed with holes must opt in with
``strict=False`` and inspect ``LayerInfo.shape_resolved``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import shape_inference


class ShapeResolutionError(RuntimeError):
    """Raised when one or more tensors have no statically resolvable size."""


@dataclass
class LayerInfo:
    name: str
    op_type: str
    input_names: list[str]
    output_names: list[str]
    weight_bytes: int
    output_tensor_bytes: int
    input_activation_bytes: int = 0
    shape_resolved: bool = True
    # Shapes retained for the cost model (FLOP estimation needs M, K, N).
    weight_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shape: tuple[int, ...] = ()


def _itemsize(elem_type: int) -> int:
    try:
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).itemsize
    except Exception:  # pragma: no cover - fallback for exotic dtypes
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type].itemsize


def _concretize_dims(model: onnx.ModelProto, batch: int) -> onnx.ModelProto:
    """Replace symbolic / zero leading dims on graph inputs with ``batch``."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for vi in m.graph.input:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        for d in tt.shape.dim:
            if d.dim_param or d.dim_value == 0:
                d.ClearField("dim_param")
                d.dim_value = batch
    return m


def _value_info_bytes(vi: onnx.ValueInfoProto) -> tuple[int | None, tuple[int, ...]]:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None, ()
    dims: list[int] = []
    for d in tt.shape.dim:
        if d.dim_value <= 0 and not d.dim_param:
            # Genuinely unknown dimension.
            return None, ()
        if d.dim_param:
            return None, ()
        dims.append(d.dim_value)
    n = int(np.prod(dims)) if dims else 1
    return n * _itemsize(tt.elem_type), tuple(dims)


def parse_onnx_graph(model: onnx.ModelProto, batch: int = 1,
                     strict: bool = True) -> list[LayerInfo]:
    """Parse an ONNX graph into per-node LayerInfo with resolved tensor sizes.

    ``batch`` substitutes for any symbolic leading dimension on graph inputs.
    Weight sizes are computed from initializer dims alone, so this works on
    models loaded with ``load_external_data=False`` (multi-GB models).
    """
    concrete = _concretize_dims(model, batch)
    try:
        inferred = shape_inference.infer_shapes(concrete, strict_mode=False, data_prop=True)
    except Exception:
        # Shape inference can reject models with unknown ops outright; fall
        # back to whatever value_info the model already carries.
        inferred = concrete
    graph = inferred.graph

    init_bytes: dict[str, int] = {}
    init_shape: dict[str, tuple[int, ...]] = {}
    for init in graph.initializer:
        n = int(np.prod(init.dims)) if init.dims else 1
        init_bytes[init.name] = n * _itemsize(init.data_type)
        init_shape[init.name] = tuple(init.dims)

    tensor_bytes: dict[str, int | None] = {}
    tensor_shape: dict[str, tuple[int, ...]] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        b, s = _value_info_bytes(vi)
        tensor_bytes[vi.name] = b
        tensor_shape[vi.name] = s

    layers: list[LayerInfo] = []
    unresolved: list[str] = []
    for node in graph.node:
        weight = 0
        wshapes: list[tuple[int, ...]] = []
        act_in = 0
        resolved = True
        for inp in node.input:
            if not inp:
                continue
            if inp in init_bytes:
                weight += init_bytes[inp]
                wshapes.append(init_shape[inp])
            else:
                b = tensor_bytes.get(inp)
                if b is None:
                    resolved = False
                else:
                    act_in += b
        out = 0
        oshape: tuple[int, ...] = ()
        for o in node.output:
            if not o:
                continue
            b = tensor_bytes.get(o)
            if b is None:
                resolved = False
            else:
                out += b
                if not oshape:
                    oshape = tensor_shape.get(o, ())
        if not resolved:
            unresolved.append(node.name or node.op_type)
        layers.append(LayerInfo(
            name=node.name, op_type=node.op_type,
            input_names=list(node.input), output_names=list(node.output),
            weight_bytes=weight, output_tensor_bytes=out,
            input_activation_bytes=act_in, shape_resolved=resolved,
            weight_shapes=wshapes, output_shape=oshape,
        ))

    if unresolved and strict:
        raise ShapeResolutionError(
            f"{len(unresolved)} node(s) have unresolvable tensor sizes: "
            f"{unresolved[:8]}{' ...' if len(unresolved) > 8 else ''}. "
            "Refusing to default to 0 bytes (that defect invalidated the v1 "
            "evaluation).  Pass strict=False to proceed and inspect "
            "LayerInfo.shape_resolved.")
    return layers
