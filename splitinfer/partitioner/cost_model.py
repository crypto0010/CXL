"""Per-layer cost model for three placements over one substrate.

The v1 model had two defects that the reviewers found by inspection:
its default link bandwidth was 3,478x the real one, and it never priced the
weight traffic that the streaming capacity model implies.  This model prices
everything the runtime actually does, and it prices three placements — not
two — because the paper's new thesis is the comparison between them:

  ``gpu``   weights resident in host memory, compute on the GPU.
            Costs host memory.  The conventional deployment.
  ``pool``  weights resident in FPGA DDR2, *faulted into the host* through
            the cxlwin load/store window, compute on the GPU.
            Memory moves to compute.  Costs link bandwidth and one round
            trip per fault batch; costs no persistent host memory.
  ``fpga``  weights resident in (or streamed to) FPGA DDR2, compute on the
            near-memory engines.  Compute moves to memory.  Costs FPGA
            compute and one dispatch round trip; only results cross the link.

Every parameter is expected to be *calibrated from measurement*.  The
defaults below are the measured UART substrate, not aspirational values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from partitioner.graph import LayerInfo

PLACEMENTS = ("gpu", "pool", "fpga")
COMPUTE_LOCATION = {"gpu": "gpu", "pool": "gpu", "fpga": "fpga"}

# Ops whose FPGA cost is DDR2-traffic-bound rather than MAC-bound.
FPGA_MEMORY_BOUND_OPS = {"Gather"}
# Ops the NMC engines implement.  Anything else is infeasible on the FPGA.
FPGA_SUPPORTED_OPS = {"Gather", "MatMul", "Gemm", "Relu", "Add", "Mul"}


@dataclass
class HardwareParams:
    gpu_gflops: float                 # effective GPU rate at the chosen precision
    fpga_int8_gops: float             # effective NMC MAC-engine rate
    fpga_ddr2_bw_gbps: float          # DDR2 read bandwidth seen by the engines
    link_bw_bytes_per_s: float        # measured sustained link bandwidth
    link_rtt_ms: float                # measured per-message round trip
    fpga_ddr2_capacity_bytes: int
    host_weight_budget_bytes: int     # host memory available for resident weights
    page_bytes: int = 4096            # cxlwin fault granularity
    link_chunk_bytes: int = 4096      # DATA_WRITE / DATA_READ chunk size
    pool_prefetch_pages: int = 1      # pages fetched per fault (1 = demand paging)
    fpga_weights_resident: bool = False  # True: preloaded once, reused across inferences
    dispatch_msg_bytes: int = 33      # serialised NMC_EXEC

    @classmethod
    def measured_uart_substrate(cls, **overrides) -> "HardwareParams":
        """The Nexys 4 DDR over FT2232HQ UART at 115,200 baud, as measured.

        Link figures are MEASURED on the board (calibrate_link.py, 2026-09-07,
        evaluation/calibration/uart_measured_v1bit_lt1.json): 11,635 B/s
        sustained DATA_WRITE (the 8N1 ceiling is 11,520; the fit's slope is
        within noise of it) and an 8.0 ms SYNC_BARRIER round trip with the
        FTDI latency timer at 1 ms (16.0 ms at the 16 ms default).
        fpga_int8_gops is derived from simulation of the v2 MAC data path
        (0.684 MAC/cycle at 81.25 MHz = 0.111 GOPS; sim/v2/tb_mac_golden).
        gpu_gflops is a placeholder until the TensorRT calibration runs.
        """
        base = dict(
            gpu_gflops=2000.0,
            fpga_int8_gops=0.111,
            fpga_ddr2_bw_gbps=1.3,
            link_bw_bytes_per_s=11_520.0,
            link_rtt_ms=8.0,
            fpga_ddr2_capacity_bytes=128 * 1024 * 1024,
            host_weight_budget_bytes=6 * 1024 * 1024 * 1024,
        )
        base.update(overrides)
        return cls(**base)


class CostModel:
    def __init__(self, hw: HardwareParams):
        self.hw = hw

    # ── Primitive quantities ────────────────────────────────────────────────

    def flops(self, layer: LayerInfo) -> float:
        """Multiply-accumulate count x2, from real shapes when available."""
        if layer.op_type in ("MatMul", "Gemm"):
            if layer.weight_shapes and len(layer.weight_shapes[0]) == 2 and layer.output_shape:
                k, n = layer.weight_shapes[0]
                rows = math.prod(layer.output_shape[:-1]) if len(layer.output_shape) > 1 else 1
                return 2.0 * rows * k * n
            # Shape-less fallback: assume batch 1, FP32 weights.
            return 2.0 * layer.weight_bytes / 4
        if layer.op_type in ("Relu", "Add", "Mul"):
            return layer.output_tensor_bytes / 4
        return 0.0

    def arithmetic_intensity(self, layer: LayerInfo) -> float:
        """FLOPs per weight byte — the quantity that decides pool vs fpga."""
        if layer.weight_bytes == 0:
            return math.inf
        return self.flops(layer) / layer.weight_bytes

    def link_ms(self, nbytes: int, messages: int) -> float:
        return (nbytes / self.hw.link_bw_bytes_per_s) * 1000.0 + messages * self.hw.link_rtt_ms

    # ── Compute costs by location ───────────────────────────────────────────

    def gpu_compute_ms(self, layer: LayerInfo) -> float:
        f = self.flops(layer)
        if f == 0:
            return 0.005  # kernel-launch floor
        return (f / (self.hw.gpu_gflops * 1e9)) * 1000.0

    def fpga_compute_ms(self, layer: LayerInfo) -> float:
        ddr_bw = self.hw.fpga_ddr2_bw_gbps * 1e9
        if layer.op_type in FPGA_MEMORY_BOUND_OPS:
            traffic = layer.input_activation_bytes + layer.output_tensor_bytes * 2
            return (traffic / ddr_bw) * 1000.0
        f = self.flops(layer)
        if f == 0:
            return 0.001
        mac_ms = (f / (self.hw.fpga_int8_gops * 1e9)) * 1000.0
        # The MAC controller re-reads activations for every weight row, so the
        # engine is frequently DDR2-bound.  Take the larger of the two.
        traffic = layer.weight_bytes / 4 + layer.input_activation_bytes  # INT8 weights
        ddr_ms = (traffic / ddr_bw) * 1000.0
        return max(mac_ms, ddr_ms)

    # ── Data-movement costs ─────────────────────────────────────────────────

    def pool_weight_fault_ms(self, layer: LayerInfo) -> float:
        """Faulting this layer's weights into the host through cxlwin."""
        if layer.weight_bytes == 0:
            return 0.0
        pages = math.ceil(layer.weight_bytes / self.hw.page_bytes)
        messages = math.ceil(pages / max(1, self.hw.pool_prefetch_pages))
        return self.link_ms(layer.weight_bytes, messages)

    def nmc_weight_stream_ms(self, layer: LayerInfo) -> float:
        """Downloading this layer's weights to DDR2 for one inference."""
        if self.hw.fpga_weights_resident or layer.weight_bytes == 0:
            return 0.0
        chunks = math.ceil(layer.weight_bytes / self.hw.link_chunk_bytes)
        return self.link_ms(layer.weight_bytes, chunks)

    def nmc_dispatch_ms(self) -> float:
        return self.link_ms(self.hw.dispatch_msg_bytes, 1)

    def activation_transfer_ms(self, nbytes: int) -> float:
        if nbytes == 0:
            return 0.0
        chunks = math.ceil(nbytes / self.hw.link_chunk_bytes)
        return self.link_ms(nbytes, chunks)

    # ── Placement costs ─────────────────────────────────────────────────────

    def placement_cost_ms(self, layer: LayerInfo, placement: str) -> float:
        if placement == "gpu":
            return self.gpu_compute_ms(layer)
        if placement == "pool":
            return self.pool_weight_fault_ms(layer) + self.gpu_compute_ms(layer)
        if placement == "fpga":
            return (self.nmc_weight_stream_ms(layer) + self.nmc_dispatch_ms()
                    + self.fpga_compute_ms(layer))
        raise ValueError(placement)

    def boundary_cost_ms(self, prev: LayerInfo, prev_placement: str, placement: str) -> float:
        """Activation crossing when compute location changes."""
        if COMPUTE_LOCATION[prev_placement] == COMPUTE_LOCATION[placement]:
            return 0.0
        return self.activation_transfer_ms(prev.output_tensor_bytes)

    # ── Feasibility ─────────────────────────────────────────────────────────

    def fpga_feasible(self, layer: LayerInfo) -> bool:
        return (layer.op_type in FPGA_SUPPORTED_OPS
                and layer.weight_bytes <= self.hw.fpga_ddr2_capacity_bytes)

    def pool_feasible(self, layer: LayerInfo) -> bool:
        # Pool mode holds weights in DDR2 while faulting; a single layer must fit.
        return layer.weight_bytes <= self.hw.fpga_ddr2_capacity_bytes

    # ── Analysis ────────────────────────────────────────────────────────────

    def link_crossover_bytes_per_s(self, layer: LayerInfo) -> float | None:
        """Link bandwidth above which ``pool`` beats ``fpga`` for this layer.

        Solves  W/B + msgs*rtt + T_gpu  =  S(B) + T_disp + T_fpga  for B.
        With resident FPGA weights S = 0 and the crossover is finite whenever
        fpga's fixed cost exceeds pool's.  With streamed weights both sides pay
        W/B, so the comparison is bandwidth-independent.
        Returns None when there is no positive crossover (pool always wins),
        ``math.inf`` when fpga always wins, or a bandwidth in bytes/s.
        """
        if layer.weight_bytes == 0:
            return None
        w = layer.weight_bytes
        t_gpu = self.gpu_compute_ms(layer) / 1000.0
        t_fpga = (self.fpga_compute_ms(layer) + self.nmc_dispatch_ms()) / 1000.0
        pages = math.ceil(w / self.hw.page_bytes)
        msgs = math.ceil(pages / max(1, self.hw.pool_prefetch_pages))
        rtt = self.hw.link_rtt_ms / 1000.0
        if self.hw.fpga_weights_resident:
            denom = t_fpga - t_gpu - msgs * rtt
            if denom <= 0:
                return math.inf     # fpga's fixed cost is below pool's: fpga always wins
            return w / denom
        chunks = math.ceil(w / self.hw.link_chunk_bytes)
        return math.inf if (chunks * rtt + t_fpga) < (msgs * rtt + t_gpu) else None
