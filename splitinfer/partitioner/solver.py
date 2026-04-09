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


def partition_model(layers: list[LayerInfo], cost_model: CostModel,
                    streaming: bool = True) -> PartitionResult:
    """Partition model layers across FPGA and GPU.

    Two capacity models are supported:

      streaming=False (legacy "resident" model):
          All FPGA-assigned layers must be simultaneously resident in
          DDR2.  The solver tracks cumulative FPGA usage and overflows
          to GPU once the capacity is exhausted.  This matches the
          original spec's assumption that weights are pre-loaded once
          and reused across inferences.

      streaming=True (default, extended model):
          Each INDIVIDUAL layer's weights must fit in DDR2, but layers
          are assumed to be streamed in sequentially — weights for
          layer N are loaded when layer N runs, then overwritten by
          layer N+1's weights.  This matches how the SplitInfer
          runtime actually executes deep sequential models and is
          the enabling assumption for the capacity-unlock story:
          a 5 GB MLP with 60 × 100 MB layers can run entirely on
          the FPGA even though 128 MB DDR2 can only hold 1-2 layers
          at a time.

    The streaming model adds a per-layer weight-transfer cost to the
    latency estimate (loading a layer's weights over USB on demand).
    This is a pessimistic bound: in practice the runtime can overlap
    the next layer's weight fetch with the current layer's compute
    (see Pipeline::run() prefetch path, Task #14).
    """
    capacity = int(cost_model.hw.fpga_ddr2_capacity_mb * 1024 * 1024)

    assignments: list[str] = []
    fpga_used = 0

    for layer in layers:
        gpu_time = cost_model.gpu_time_ms(layer)
        fpga_time = cost_model.fpga_time_ms(layer)

        # Streaming: each layer alone must fit in DDR2.
        # Resident:  cumulative FPGA usage must fit in DDR2.
        if streaming:
            fits = (layer.weight_bytes <= capacity) and cost_model.fpga_feasible(layer)
        else:
            fits = ((fpga_used + layer.weight_bytes) <= capacity
                    and cost_model.fpga_feasible(layer))

        if fits and fpga_time <= gpu_time:
            assignments.append("fpga")
            if not streaming:
                fpga_used += layer.weight_bytes
        elif fits and layer.weight_bytes > 0:
            # Prefer FPGA for layers with significant weight to reduce GPU
            # memory pressure; only fall back to GPU for zero-weight layers
            # or when the capacity constraint is exhausted.
            assignments.append("fpga")
            if not streaming:
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

    # Memory accounting depends on the capacity model:
    #   streaming: peak resident FPGA memory = MAX layer size (weights are
    #              streamed in and evicted one at a time).  Total weight
    #              TRAFFIC is the sum, but resident footprint is the max.
    #   resident : FPGA resident memory = sum (all simultaneously loaded).
    fpga_layer_sizes = [layers[i].weight_bytes for i, a in enumerate(assignments) if a == "fpga"]
    if streaming:
        fpga_mem = max(fpga_layer_sizes) if fpga_layer_sizes else 0
    else:
        fpga_mem = sum(fpga_layer_sizes)

    return PartitionResult(
        assignments=assignments,
        total_latency_ms=total_latency,
        gpu_memory_bytes=sum(layers[i].weight_bytes for i, a in enumerate(assignments) if a == "gpu"),
        fpga_memory_bytes=fpga_mem,
        num_transfers=sum(
            1 for i in range(1, len(assignments)) if assignments[i] != assignments[i - 1]
        ),
    )
