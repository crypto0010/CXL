"""Regression tests for D1: output_tensor_bytes was hardcoded to 0.

The zero-byte defect invalidated every result in the submitted paper: it
propagated into every manifest, then into the runtime, which passed 0-byte
operands to every executor.  These tests make it impossible to reintroduce
silently.
"""
import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

from partitioner.graph import parse_onnx_graph, ShapeResolutionError


def _fc_model(batch="N"):
    """X[batch,784] -> MatMul W1[784,128] -> Add B1 -> Relu -> MatMul W2[128,10]."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch, 784])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, 10])
    inits = [
        helper.make_tensor("W1", TensorProto.FLOAT, [784, 128],
                           np.zeros(784 * 128, dtype=np.float32).tolist()),
        helper.make_tensor("B1", TensorProto.FLOAT, [128],
                           np.zeros(128, dtype=np.float32).tolist()),
        helper.make_tensor("W2", TensorProto.FLOAT, [128, 10],
                           np.zeros(128 * 10, dtype=np.float32).tolist()),
    ]
    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["mm1"], name="fc1"),
        helper.make_node("Add", ["mm1", "B1"], ["a1"], name="bias1"),
        helper.make_node("Relu", ["a1"], ["r1"], name="relu1"),
        helper.make_node("MatMul", ["r1", "W2"], ["Y"], name="fc2"),
    ]
    g = helper.make_graph(nodes, "fc", [X], [Y], initializer=inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def test_output_bytes_are_never_zero_for_real_activations():
    layers = parse_onnx_graph(_fc_model(), batch=1)
    for l in layers:
        assert l.output_tensor_bytes > 0, f"{l.name} reported a zero-byte output"


def test_output_bytes_match_hand_computation():
    layers = {l.name: l for l in parse_onnx_graph(_fc_model(), batch=1)}
    assert layers["fc1"].output_tensor_bytes == 1 * 128 * 4
    assert layers["bias1"].output_tensor_bytes == 1 * 128 * 4
    assert layers["relu1"].output_tensor_bytes == 1 * 128 * 4
    assert layers["fc2"].output_tensor_bytes == 1 * 10 * 4


def test_batch_scales_activations_but_not_weights():
    b1 = {l.name: l for l in parse_onnx_graph(_fc_model(), batch=1)}
    b8 = {l.name: l for l in parse_onnx_graph(_fc_model(), batch=8)}
    assert b8["fc1"].output_tensor_bytes == 8 * b1["fc1"].output_tensor_bytes
    assert b8["fc1"].weight_bytes == b1["fc1"].weight_bytes


def test_input_activation_bytes_exclude_initializers():
    layers = {l.name: l for l in parse_onnx_graph(_fc_model(), batch=1)}
    # fc1 consumes X (activation, 784 floats) and W1 (initializer).
    assert layers["fc1"].input_activation_bytes == 1 * 784 * 4
    assert layers["fc1"].weight_bytes == 784 * 128 * 4


def test_weights_still_reported():
    layers = {l.name: l for l in parse_onnx_graph(_fc_model(), batch=1)}
    assert layers["fc1"].weight_bytes == 784 * 128 * 4
    assert layers["relu1"].weight_bytes == 0


def test_unresolvable_shape_raises_rather_than_defaulting_to_zero():
    """The original bug was a silent default.  Loud failure is the fix."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    # A node ONNX shape inference cannot resolve (unknown custom domain op).
    node = helper.make_node("MysteryOp", ["X"], ["Y"], name="mystery", domain="custom.test")
    g = helper.make_graph([node], "mystery", [X], [Y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17),
                                            helper.make_opsetid("custom.test", 1)])
    # Y is a declared graph output, so it resolves; make the output internal instead.
    node2 = helper.make_node("MysteryOp", ["X"], ["hidden"], name="mystery2", domain="custom.test")
    g2 = helper.make_graph([node2], "mystery2", [X], [], value_info=[])
    m2 = helper.make_model(g2, opset_imports=[helper.make_opsetid("", 17),
                                              helper.make_opsetid("custom.test", 1)])
    with pytest.raises(ShapeResolutionError):
        parse_onnx_graph(m2, batch=1)


def test_permissive_mode_reports_unresolved_instead_of_raising():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("MysteryOp", ["X"], ["hidden"], name="m", domain="custom.test")
    g = helper.make_graph([node], "m", [X], [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17),
                                            helper.make_opsetid("custom.test", 1)])
    layers = parse_onnx_graph(m, batch=1, strict=False)
    assert layers[0].shape_resolved is False
