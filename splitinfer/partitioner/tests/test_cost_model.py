import pytest
from partitioner.graph import LayerInfo
from partitioner.cost_model import CostModel, HardwareParams

def _hw():
    return HardwareParams(gpu_gflops=100.0, fpga_int8_gops=6.4, fpga_ddr2_bw_gbps=1.3,
                          usb_bw_mbps=40.0, usb_latency_ms=1.0, fpga_ddr2_capacity_mb=128.0)

def test_gpu_cost_matmul():
    assert CostModel(_hw()).gpu_time_ms(
        LayerInfo("fc1","MatMul",["x","W"],["y"],784*128*4,128*4)) > 0

def test_fpga_cost_embedding():
    assert CostModel(_hw()).fpga_time_ms(
        LayerInfo("emb","Gather",["t","i"],["o"],10000*64,128*64)) > 0

def test_transfer_cost():
    assert abs(CostModel(_hw()).transfer_time_ms(2*1024*1024) - 51.0) < 1.0

def test_fpga_over_capacity():
    assert CostModel(_hw()).fpga_feasible(
        LayerInfo("big","MatMul",["x","W"],["y"],200*1024*1024,1024)) is False

def test_fpga_within_capacity():
    assert CostModel(_hw()).fpga_feasible(
        LayerInfo("sm","Gather",["t","i"],["o"],50*1024*1024,1024)) is True
