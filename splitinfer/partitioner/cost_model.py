from dataclasses import dataclass
from partitioner.graph import LayerInfo

@dataclass
class HardwareParams:
    gpu_gflops: float
    fpga_int8_gops: float
    fpga_ddr2_bw_gbps: float
    usb_bw_mbps: float
    usb_latency_ms: float
    fpga_ddr2_capacity_mb: float

FPGA_MEMORY_BOUND_OPS = {"Gather"}

class CostModel:
    def __init__(self, hw: HardwareParams):
        self.hw = hw

    def gpu_time_ms(self, layer: LayerInfo) -> float:
        flops = self._estimate_flops(layer)
        if flops == 0: return 0.01
        return (flops / (self.hw.gpu_gflops * 1e9)) * 1000.0

    def fpga_time_ms(self, layer: LayerInfo) -> float:
        if layer.op_type in FPGA_MEMORY_BOUND_OPS:
            read_bytes = layer.weight_bytes + layer.output_tensor_bytes
            return (read_bytes / (self.hw.fpga_ddr2_bw_gbps * 1e9)) * 1000.0
        ops = self._estimate_int8_ops(layer)
        if ops == 0: return 0.01
        return (ops / (self.hw.fpga_int8_gops * 1e9)) * 1000.0

    def transfer_time_ms(self, tensor_bytes: int) -> float:
        return (tensor_bytes / (self.hw.usb_bw_mbps * 1024 * 1024)) * 1000.0 + self.hw.usb_latency_ms

    def fpga_feasible(self, layer: LayerInfo) -> bool:
        return layer.weight_bytes <= self.hw.fpga_ddr2_capacity_mb * 1024 * 1024

    def _estimate_flops(self, layer: LayerInfo) -> float:
        if layer.op_type in ("MatMul", "Gemm"): return 2 * layer.weight_bytes / 4
        if layer.op_type in ("Relu", "Add"): return layer.output_tensor_bytes / 4
        if layer.op_type == "Gather": return 0
        return layer.weight_bytes / 4

    def _estimate_int8_ops(self, layer: LayerInfo) -> float:
        if layer.op_type in ("MatMul", "Gemm"): return 2 * layer.weight_bytes / 4
        if layer.op_type in ("Relu", "Add"): return layer.output_tensor_bytes
        return 0
