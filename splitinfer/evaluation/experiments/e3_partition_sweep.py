#!/usr/bin/env python3
"""Experiment E3: Partition Sweep over FPGA DDR2 Capacity.

Sweeps FPGA DDR2 capacity from 0 to 128 MB in configurable steps.
At each capacity point, runs the partitioner and records:
  - cut_point (index of first GPU layer after FPGA layers)
  - gpu_layers count
  - fpga_layers count
  - estimated total latency (ms)
  - FPGA memory used (MB)

Outputs CSV to stdout and optionally to a file.

Usage:
    python e3_partition_sweep.py <model.onnx> [--step-mb N] [--output results.csv]
"""

import argparse
import csv
import os
import sys
import time

# Allow running from repo root without installing the package
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "partitioner"))

try:
    import onnx
    from partitioner.graph import parse_onnx_graph
    from partitioner.cost_model import CostModel, HardwareParams
    from partitioner.solver import partition_model
except ImportError as exc:
    print(f"ERROR: Cannot import partitioner: {exc}", file=sys.stderr)
    print("Make sure onnx is installed and the partitioner package is on the path.", file=sys.stderr)
    sys.exit(1)


# Jetson Orin Nano + Nexys 4 DDR baseline hardware params
BASE_HW = HardwareParams(
    gpu_gflops=1000.0,          # Jetson Orin Nano GPU
    fpga_int8_gops=6.4,         # Artix-7 INT8 compute estimate
    fpga_ddr2_bw_gbps=1.3,      # DDR2 128MB theoretical peak
    usb_bw_mbps=40.0,           # USB 2.0 HS sustained
    usb_latency_ms=1.0,
    fpga_ddr2_capacity_mb=0.0,  # swept parameter
)


def sweep(model_path: str, step_mb: float, output_path: str | None) -> None:
    print(f"Loading model: {model_path}", file=sys.stderr)
    model  = onnx.load(model_path)
    layers = parse_onnx_graph(model)
    print(f"  {len(layers)} layers parsed.", file=sys.stderr)

    capacities_mb = []
    cap = 0.0
    while cap <= 128.0:
        capacities_mb.append(round(cap, 2))
        cap += step_mb
    if capacities_mb[-1] < 128.0:
        capacities_mb.append(128.0)

    fieldnames = [
        "capacity_mb", "gpu_layers", "fpga_layers",
        "num_transfers", "estimated_latency_ms", "fpga_memory_mb",
        "solve_time_ms",
    ]

    rows = []
    print(f"Sweeping {len(capacities_mb)} capacity points...", file=sys.stderr)

    for cap_mb in capacities_mb:
        hw  = HardwareParams(
            gpu_gflops=BASE_HW.gpu_gflops,
            fpga_int8_gops=BASE_HW.fpga_int8_gops,
            fpga_ddr2_bw_gbps=BASE_HW.fpga_ddr2_bw_gbps,
            usb_bw_mbps=BASE_HW.usb_bw_mbps,
            usb_latency_ms=BASE_HW.usb_latency_ms,
            fpga_ddr2_capacity_mb=cap_mb,
        )
        cm  = CostModel(hw)

        t0  = time.perf_counter()
        res = partition_model(layers, cm)
        t1  = time.perf_counter()

        gpu_count  = sum(1 for a in res.assignments if a == "gpu")
        fpga_count = sum(1 for a in res.assignments if a == "fpga")

        # cut_point: index of the first GPU layer (first device switch boundary)
        cut_point = next(
            (i for i, a in enumerate(res.assignments) if a == "gpu"),
            len(res.assignments),
        )

        rows.append({
            "capacity_mb":           cap_mb,
            "gpu_layers":            gpu_count,
            "fpga_layers":           fpga_count,
            "num_transfers":         res.num_transfers,
            "estimated_latency_ms":  round(res.total_latency_ms, 4),
            "fpga_memory_mb":        round(res.fpga_memory_bytes / (1024 * 1024), 4),
            "solve_time_ms":         round((t1 - t0) * 1000.0, 3),
        })

    # Output CSV
    writer_target = open(output_path, "w", newline="") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(writer_target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output_path:
            writer_target.close()

    if output_path:
        print(f"Results written to: {output_path}", file=sys.stderr)

    # Print summary stats
    print("", file=sys.stderr)
    print("Summary:", file=sys.stderr)
    print(f"  Capacity 0 MB  -> FPGA layers: {rows[0]['fpga_layers']:3d} | latency: {rows[0]['estimated_latency_ms']:.2f} ms", file=sys.stderr)
    print(f"  Capacity 128 MB -> FPGA layers: {rows[-1]['fpga_layers']:3d} | latency: {rows[-1]['estimated_latency_ms']:.2f} ms", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: Partition sweep over FPGA DDR2 capacity")
    parser.add_argument("model",      help="Path to ONNX model file")
    parser.add_argument("--step-mb",  type=float, default=8.0,
                        help="Capacity sweep step size in MB (default: 8)")
    parser.add_argument("--output",   default=None,
                        help="Output CSV file path (default: stdout)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    sweep(args.model, args.step_mb, args.output)


if __name__ == "__main__":
    main()
