import json, pytest
from partitioner.graph import LayerInfo
from partitioner.solver import PartitionResult
from partitioner.manifest import generate_manifest

def _data():
    layers = [
        LayerInfo("fc1","MatMul",["x","W1"],["y1"],50_000_000,512),
        LayerInfo("relu1","Relu",["y1"],["r1"],0,512),
        LayerInfo("fc2","MatMul",["r1","W2"],["y2"],1_000_000,256),
    ]
    result = PartitionResult(["fpga","fpga","gpu"], 5.0, 1_000_000, 50_000_000, 1)
    return layers, result

def test_valid_json():
    data = json.loads(generate_manifest(*_data()))
    assert "layers" in data and "summary" in data

def test_layer_count():
    assert len(json.loads(generate_manifest(*_data()))["layers"]) == 3

def test_layer_fields():
    l = json.loads(generate_manifest(*_data()))["layers"][0]
    assert l["name"] == "fc1" and l["device"] == "fpga"

def test_transfers():
    t = json.loads(generate_manifest(*_data()))["transfers"]
    assert len(t) == 1 and t[0]["from_device"] == "fpga"
