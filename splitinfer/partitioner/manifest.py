"""Manifest generation.

The manifest is the contract between the offline partitioner and the C++
runtime.  v2 adds three things the v1 manifest lacked and the reviewers
noticed: real tensor byte counts, the placement (not just the device), and
provenance — the hardware parameters and cost breakdown the decision was
made under, so a reader can reproduce or dispute it.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from partitioner.cost_model import COMPUTE_LOCATION, HardwareParams
from partitioner.graph import LayerInfo
from partitioner.solver import PartitionResult

MANIFEST_VERSION = "2.0"


def _finite(x):
    if x is None:
        return None
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return "inf" if x > 0 else None
    return x


def generate_manifest(layers: list[LayerInfo], result: PartitionResult,
                      hw: HardwareParams | None = None, batch: int = 1,
                      model_path: str | None = None) -> str:
    if len(layers) != len(result.assignments):
        raise ValueError("layers / assignments length mismatch")
    zero = [l.name for l in layers if l.output_tensor_bytes == 0 and l.op_type != "Constant"]
    if zero:
        raise ValueError(f"refusing to emit a manifest with zero-byte outputs: {zero[:5]}")

    layer_entries = []
    for l, p in zip(layers, result.assignments):
        layer_entries.append({
            "name": l.name, "op_type": l.op_type,
            "placement": p, "device": COMPUTE_LOCATION[p],
            "weight_bytes": l.weight_bytes,
            "input_activation_bytes": l.input_activation_bytes,
            "output_tensor_bytes": l.output_tensor_bytes,
            "weight_shapes": [list(s) for s in l.weight_shapes],
            "output_shape": list(l.output_shape),
            "inputs": l.input_names, "outputs": l.output_names,
        })

    transfers = []
    for i in range(1, len(layers)):
        a, b = result.assignments[i - 1], result.assignments[i]
        if COMPUTE_LOCATION[a] != COMPUTE_LOCATION[b]:
            transfers.append({
                "after_layer": layers[i - 1].name, "before_layer": layers[i].name,
                "from_device": COMPUTE_LOCATION[a], "to_device": COMPUTE_LOCATION[b],
                "tensor_names": layers[i - 1].output_names,
                "tensor_bytes": layers[i - 1].output_tensor_bytes,
            })

    decisions = [{
        "name": d.name, "placement": d.placement, "cost_ms": d.cost_ms,
        "boundary_ms": d.boundary_ms, "candidates_ms": d.candidates_ms,
        "arithmetic_intensity": _finite(d.arithmetic_intensity),
        "link_crossover_bytes_per_s": _finite(d.link_crossover_bytes_per_s),
    } for d in result.decisions]

    doc = {
        "version": MANIFEST_VERSION,
        "model": model_path, "batch": batch,
        "layers": layer_entries, "transfers": transfers,
        "summary": {
            "total_layers": len(layers),
            "placements": result.placement_counts,
            "gpu_layers": sum(1 for a in result.assignments if COMPUTE_LOCATION[a] == "gpu"),
            "fpga_layers": sum(1 for a in result.assignments if COMPUTE_LOCATION[a] == "fpga"),
            "estimated_latency_ms": result.total_latency_ms,
            "breakdown_ms": result.breakdown_ms,
            "host_weight_bytes": result.gpu_memory_bytes,
            "gpu_memory_bytes": result.gpu_memory_bytes,       # legacy key
            "fpga_peak_resident_bytes": result.fpga_memory_bytes,
            "fpga_memory_bytes": result.fpga_memory_bytes,     # legacy key
            "fpga_weight_traffic_bytes": result.fpga_weight_traffic_bytes,
            "pool_fault_bytes": result.pool_fault_bytes,
            "activation_transfer_bytes": result.activation_transfer_bytes,
            "num_transfers": result.num_transfers,
            "total_weight_bytes": sum(l.weight_bytes for l in layers),
        },
        "decisions": decisions,
        "hardware_params": asdict(hw) if hw else None,
    }
    return json.dumps(doc, indent=2)
