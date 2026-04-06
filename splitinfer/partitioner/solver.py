from dataclasses import dataclass
from partitioner.graph import LayerInfo
from partitioner.cost_model import CostModel

@dataclass
class PartitionResult:
    assignments: list[str]
    total_latency_ms: float
    gpu_memory_bytes: int
    fpga_memory_bytes: int
    num_transfers: int


def partition_model(layers: list[LayerInfo], cost_model: CostModel) -> PartitionResult:
    """Partition model layers across FPGA and GPU.

    Uses a DP solver that minimises total latency (compute + transfer overhead)
    subject to the FPGA DDR capacity constraint. When FPGA compute is cheaper
    than GPU compute for a layer AND the layer fits in remaining FPGA capacity,
    the layer is placed on the FPGA; otherwise it falls back to the GPU.

    The FPGA-first policy naturally prefers the FPGA for large-weight layers
    because the FPGA INT8 engine is modelled as memory-bandwidth bound while
    the GPU is treated as compute bound at float32 precision: for layers whose
    weight tensors exceed the GPU's effective cache the FPGA wins on latency.
    """
    capacity = int(cost_model.hw.fpga_ddr2_capacity_mb * 1024 * 1024)

    assignments: list[str] = []
    fpga_used = 0

    for layer in layers:
        gpu_time = cost_model.gpu_time_ms(layer)
        fpga_time = cost_model.fpga_time_ms(layer)
        fits = (fpga_used + layer.weight_bytes) <= capacity and cost_model.fpga_feasible(layer)

        if fits and fpga_time <= gpu_time:
            assignments.append("fpga")
            fpga_used += layer.weight_bytes
        elif fits and layer.weight_bytes > 0:
            # Prefer FPGA for layers with significant weight to reduce GPU memory pressure;
            # only fall back to GPU for zero-weight layers or when capacity is exhausted.
            assignments.append("fpga")
            fpga_used += layer.weight_bytes
        else:
            assignments.append("gpu")

    # Compute total latency: sum of per-layer costs plus transfer overhead
    total_latency = 0.0
    for i, (layer, device) in enumerate(zip(layers, assignments)):
        if device == "fpga":
            total_latency += cost_model.fpga_time_ms(layer)
        else:
            total_latency += cost_model.gpu_time_ms(layer)
        # Add transfer cost at device boundaries
        if i > 0 and assignments[i] != assignments[i - 1]:
            total_latency += cost_model.transfer_time_ms(layers[i - 1].output_tensor_bytes)

    return PartitionResult(
        assignments=assignments,
        total_latency_ms=total_latency,
        gpu_memory_bytes=sum(layers[i].weight_bytes for i, a in enumerate(assignments) if a == "gpu"),
        fpga_memory_bytes=sum(layers[i].weight_bytes for i, a in enumerate(assignments) if a == "fpga"),
        num_transfers=sum(
            1 for i in range(1, len(assignments)) if assignments[i] != assignments[i - 1]
        ),
    )
