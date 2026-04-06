import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto
from partitioner.graph import parse_onnx_graph, LayerInfo

def _make_simple_model():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 784])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [784, 128],
                            np.zeros([784, 128], dtype=np.float32).flatten().tolist())
    B1 = helper.make_tensor("B1", TensorProto.FLOAT, [128],
                            np.zeros([128], dtype=np.float32).tolist())
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, [128, 10],
                            np.zeros([128, 10], dtype=np.float32).flatten().tolist())
    B2 = helper.make_tensor("B2", TensorProto.FLOAT, [10],
                            np.zeros([10], dtype=np.float32).tolist())
    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["mm1"], name="fc1_matmul"),
        helper.make_node("Add", ["mm1", "B1"], ["fc1_out"], name="fc1_add"),
        helper.make_node("Relu", ["fc1_out"], ["relu_out"], name="relu1"),
        helper.make_node("MatMul", ["relu_out", "W2"], ["mm2"], name="fc2_matmul"),
        helper.make_node("Add", ["mm2", "B2"], ["Y"], name="fc2_add"),
    ]
    graph = helper.make_graph(nodes, "simple_fc", [X], [Y], initializer=[W1, B1, W2, B2])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

def test_parse_returns_layers():
    layers = parse_onnx_graph(_make_simple_model())
    assert len(layers) == 5

def test_layer_info_fields():
    layers = parse_onnx_graph(_make_simple_model())
    fc1 = layers[0]
    assert isinstance(fc1, LayerInfo)
    assert fc1.name == "fc1_matmul"
    assert fc1.op_type == "MatMul"
    assert fc1.weight_bytes == 784 * 128 * 4

def test_topological_order():
    layers = parse_onnx_graph(_make_simple_model())
    names = [l.name for l in layers]
    assert names.index("fc1_matmul") < names.index("relu1") < names.index("fc2_matmul")

def test_relu_has_zero_weights():
    layers = parse_onnx_graph(_make_simple_model())
    relu = [l for l in layers if l.name == "relu1"][0]
    assert relu.weight_bytes == 0
