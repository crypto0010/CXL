import pytest
from partitioner.graph import LayerInfo
from partitioner.cost_model import CostModel, HardwareParams
from partitioner.solver import partition_model, PartitionResult

def _hw():
    return HardwareParams(gpu_gflops=100.0, fpga_int8_gops=6.4, fpga_ddr2_bw_gbps=1.3,
                          usb_bw_mbps=40.0, usb_latency_ms=1.0, fpga_ddr2_capacity_mb=128.0)

def _layers():
    return [
        LayerInfo("fc1","MatMul",["x","W1"],["y1"],50*1024*1024,512),
        LayerInfo("relu1","Relu",["y1"],["r1"],0,512),
        LayerInfo("fc2","MatMul",["r1","W2"],["y2"],1*1024*1024,256),
        LayerInfo("relu2","Relu",["y2"],["r2"],0,256),
        LayerInfo("fc3","MatMul",["r2","W3"],["y3"],512*1024,40),
    ]

def test_partition_returns_result():
    result = partition_model(_layers(), CostModel(_hw()))
    assert isinstance(result, PartitionResult)
    assert len(result.assignments) == 5

def test_all_assignments_valid():
    for a in partition_model(_layers(), CostModel(_hw())).assignments:
        assert a in ("gpu", "fpga")

def test_large_weight_prefers_fpga():
    assert partition_model(_layers(), CostModel(_hw())).assignments[0] == "fpga"

def test_capacity_respected():
    layers = _layers()
    layers[0] = LayerInfo("fc1","MatMul",["x","W1"],["y1"],100*1024*1024,512)
    layers[2] = LayerInfo("fc2","MatMul",["r1","W2"],["y2"],50*1024*1024,256)
    result = partition_model(layers, CostModel(_hw()))
    fpga_bytes = sum(layers[i].weight_bytes for i, a in enumerate(result.assignments) if a == "fpga")
    assert fpga_bytes <= 128 * 1024 * 1024

def test_total_latency():
    assert partition_model(_layers(), CostModel(_hw())).total_latency_ms > 0
