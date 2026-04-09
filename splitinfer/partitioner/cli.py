#!/usr/bin/env python3
"""SplitInfer partitioner CLI.

Usage:
    python3 -m partitioner.cli <model.onnx> --output <manifest.json> [options]
"""

import argparse
import json
import os
import sys

import onnx

from partitioner.graph import parse_onnx_graph
from partitioner.cost_model import CostModel, HardwareParams
from partitioner.solver import partition_model
from partitioner.manifest import generate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="SplitInfer model partitioner")
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("--output", "-o", required=True, help="Output manifest JSON path")
    parser.add_argument("--gpu-gflops", type=float, default=40.0,
                        help="GPU compute capability in GFLOPS (default: 40)")
    parser.add_argument("--fpga-int8-gops", type=float, default=0.5,
                        help="FPGA INT8 throughput in GOPS (default: 0.5)")
    parser.add_argument("--fpga-ddr2-bw-gbps", type=float, default=1.3,
                        help="FPGA DDR2 bandwidth in GB/s (default: 1.3)")
    parser.add_argument("--usb-bw-mbps", type=float, default=40.0,
                        help="USB 2.0 usable bandwidth in MB/s (default: 40)")
    parser.add_argument("--usb-latency-ms", type=float, default=0.5,
                        help="USB transfer latency overhead in ms (default: 0.5)")
    parser.add_argument("--fpga-ddr2-mb", type=float, default=128.0,
                        help="FPGA DDR2 capacity in MB (default: 128)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ONNX model: {args.model}")
    # IMPORTANT: load_external_data=False is required for multi-GB models.
    # The default load_external_data=True materializes every weight tensor
    # into RAM, which OOM-kills the partitioner on models larger than
    # available memory.  We only need initializer *shapes* (init.dims +
    # data_type) to compute weight_bytes per layer, never the actual data.
    # See parse_onnx_graph() — it uses init.dims × dtype.itemsize, which
    # works identically whether data_location is DEFAULT or EXTERNAL.
    model = onnx.load(args.model, load_external_data=False)
    layers = parse_onnx_graph(model)
    print(f"Parsed {len(layers)} layers")

    hw = HardwareParams(
        gpu_gflops=args.gpu_gflops,
        fpga_int8_gops=args.fpga_int8_gops,
        fpga_ddr2_bw_gbps=args.fpga_ddr2_bw_gbps,
        usb_bw_mbps=args.usb_bw_mbps,
        usb_latency_ms=args.usb_latency_ms,
        fpga_ddr2_capacity_mb=args.fpga_ddr2_mb,
    )
    cost = CostModel(hw)

    result = partition_model(layers, cost)
    print(f"Partition: {sum(1 for a in result.assignments if a == 'gpu')} GPU, "
          f"{sum(1 for a in result.assignments if a == 'fpga')} FPGA, "
          f"{result.num_transfers} transfers")
    print(f"Estimated latency: {result.total_latency_ms:.3f} ms")

    manifest_json = generate_manifest(layers, result)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(manifest_json)
    print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()
