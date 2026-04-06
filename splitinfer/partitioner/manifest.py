import json
from partitioner.graph import LayerInfo
from partitioner.solver import PartitionResult

def generate_manifest(layers: list[LayerInfo], result: PartitionResult) -> str:
    layer_entries = [{"name": l.name, "op_type": l.op_type, "device": d,
                      "weight_bytes": l.weight_bytes, "output_tensor_bytes": l.output_tensor_bytes,
                      "inputs": l.input_names, "outputs": l.output_names}
                     for l, d in zip(layers, result.assignments)]

    transfers = [{"after_layer": layers[i-1].name, "before_layer": layers[i].name,
                  "from_device": result.assignments[i-1], "to_device": result.assignments[i],
                  "tensor_names": layers[i-1].output_names,
                  "tensor_bytes": layers[i-1].output_tensor_bytes}
                 for i in range(1, len(result.assignments))
                 if result.assignments[i] != result.assignments[i-1]]

    return json.dumps({"version": "1.0", "layers": layer_entries, "transfers": transfers,
        "summary": {"total_layers": len(layers),
                     "gpu_layers": sum(1 for a in result.assignments if a == "gpu"),
                     "fpga_layers": sum(1 for a in result.assignments if a == "fpga"),
                     "estimated_latency_ms": result.total_latency_ms,
                     "gpu_memory_bytes": result.gpu_memory_bytes,
                     "fpga_memory_bytes": result.fpga_memory_bytes,
                     "num_transfers": result.num_transfers}}, indent=2)
