import json
import pytest
from partitioner.cost_model import CostModel, HardwareParams
from partitioner.graph import LayerInfo
from partitioner.manifest import generate_manifest
from partitioner.solver import partition_model


def _layers():
    return [
        LayerInfo("fc1", "MatMul", ["x", "W1"], ["y1"], 1024 * 1024, 4096,
                  input_activation_bytes=1024, weight_shapes=[(256, 1024)], output_shape=(1, 1024)),
        LayerInfo("relu1", "Relu", ["y1"], ["r1"], 0, 4096, input_activation_bytes=4096,
                  output_shape=(1, 1024)),
        LayerInfo("fc2", "MatMul", ["r1", "W2"], ["y2"], 32768, 32,
                  input_activation_bytes=4096, weight_shapes=[(1024, 8)], output_shape=(1, 8)),
    ]


def _doc(**hwkw):
    hw = HardwareParams.measured_uart_substrate(**hwkw)
    layers = _layers()
    r = partition_model(layers, CostModel(hw))
    return json.loads(generate_manifest(layers, r, hw=hw, batch=1, model_path="m.onnx"))


def test_manifest_has_real_byte_counts():
    d = _doc()
    for l in d["layers"]:
        assert l["output_tensor_bytes"] > 0
    assert d["summary"]["total_weight_bytes"] == 1024 * 1024 + 32768


def test_manifest_refuses_zero_byte_outputs():
    layers = _layers()
    layers[1].output_tensor_bytes = 0
    r = partition_model(layers, CostModel(HardwareParams.measured_uart_substrate()))
    with pytest.raises(ValueError):
        generate_manifest(layers, r)


def test_manifest_carries_placement_and_device():
    d = _doc()
    for l in d["layers"]:
        assert l["placement"] in ("gpu", "pool", "fpga")
        assert l["device"] in ("gpu", "fpga")


def test_manifest_carries_provenance():
    d = _doc()
    assert d["version"] == "2.0"
    assert d["hardware_params"]["link_bw_bytes_per_s"] == pytest.approx(11_520.0)
    assert "breakdown_ms" in d["summary"]
    assert len(d["decisions"]) == 3
    assert "candidates_ms" in d["decisions"][0]


def test_transfers_have_nonzero_bytes_when_present():
    d = _doc(host_weight_budget_bytes=0)   # forces off-host placements
    for t in d["transfers"]:
        assert t["tensor_bytes"] > 0
