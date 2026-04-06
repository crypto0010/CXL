from dataclasses import dataclass
import onnx
import numpy as np

@dataclass
class LayerInfo:
    name: str
    op_type: str
    input_names: list[str]
    output_names: list[str]
    weight_bytes: int
    output_tensor_bytes: int

def parse_onnx_graph(model: onnx.ModelProto) -> list[LayerInfo]:
    graph = model.graph
    init_sizes: dict[str, int] = {}
    for init in graph.initializer:
        dtype = onnx._mapping.TENSOR_TYPE_MAP[init.data_type].np_dtype
        num_elements = int(np.prod(init.dims)) if init.dims else 0
        init_sizes[init.name] = num_elements * dtype.itemsize

    layers = []
    for node in graph.node:
        weight_bytes = sum(init_sizes.get(inp, 0) for inp in node.input)
        layers.append(LayerInfo(
            name=node.name, op_type=node.op_type,
            input_names=list(node.input), output_names=list(node.output),
            weight_bytes=weight_bytes, output_tensor_bytes=0,
        ))
    return layers
