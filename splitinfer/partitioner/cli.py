#!/usr/bin/env python3
"""SplitInfer partitioner CLI.

    python3 -m partitioner.cli <model.onnx> -o <manifest.json> [options]

Hardware parameters default to the *measured* UART substrate.  Supply
``--calibration <json>`` (the output of evaluation/calibrate_link.py and the
TensorRT/NMC microbenchmarks) to override them with measured values.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import onnx

from partitioner.cost_model import PLACEMENTS, CostModel, HardwareParams
from partitioner.graph import ShapeResolutionError, parse_onnx_graph
from partitioner.manifest import generate_manifest
from partitioner.solver import PartitionResult, partition_model


def _fmt_bw(b) -> str:
    if b is None:
        return "pool-always"
    if b == math.inf or b == "inf":
        return "fpga-always"
    for unit, div in (("GB/s", 1e9), ("MB/s", 1e6), ("KB/s", 1e3)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B/s"


def print_report(result: PartitionResult, hw: HardwareParams) -> None:
    print(f"\n{'layer':<16}{'op':<9}{'place':<6}{'cost ms':>10}{'gpu':>10}{'pool':>10}"
          f"{'fpga':>10}{'AI':>8}  crossover")
    for d in result.decisions:
        c = d.candidates_ms
        ai = "inf" if d.arithmetic_intensity == math.inf else f"{d.arithmetic_intensity:.2f}"
        print(f"{d.name:<16}{d.op_type:<9}{d.placement:<6}{d.cost_ms:>10.2f}"
              f"{c.get('gpu', math.nan):>10.2f}{c.get('pool', math.nan):>10.2f}"
              f"{c.get('fpga', math.nan):>10.2f}{ai:>8}  {_fmt_bw(d.link_crossover_bytes_per_s)}")
    b = result.breakdown_ms
    print(f"\nplacements: {result.placement_counts}  transfers: {result.num_transfers}")
    print(f"estimated latency: {result.total_latency_ms:.2f} ms  = "
          + " + ".join(f"{k} {v:.1f}" for k, v in b.items() if v > 0))
    print(f"host-resident weights: {result.gpu_memory_bytes / 2**20:.1f} MiB "
          f"(budget {hw.host_weight_budget_bytes / 2**20:.0f} MiB)")
    print(f"FPGA DDR2 peak resident: {result.fpga_memory_bytes / 2**20:.1f} MiB "
          f"({'resident' if hw.fpga_weights_resident else 'streamed'}); "
          f"per-inference weight traffic {result.fpga_weight_traffic_bytes / 2**20:.1f} MiB streamed, "
          f"{result.pool_fault_bytes / 2**20:.1f} MiB faulted")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SplitInfer model partitioner (v2, exact DP)")
    ap.add_argument("model")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--calibration", help="JSON of measured HardwareParams overrides")
    ap.add_argument("--allow", default="gpu,pool,fpga",
                    help="comma list of placements the solver may use")
    ap.add_argument("--resident", action="store_true",
                    help="FPGA weights preloaded once (no per-inference streaming)")
    ap.add_argument("--host-budget-mb", type=float)
    ap.add_argument("--link-bw", type=float, help="bytes/s")
    ap.add_argument("--link-rtt-ms", type=float)
    ap.add_argument("--gpu-gflops", type=float)
    ap.add_argument("--fpga-int8-gops", type=float)
    ap.add_argument("--prefetch-pages", type=int)
    ap.add_argument("--bucket-mb", type=float, default=None, help="fixed bucket (default: adaptive)")
    ap.add_argument("--report", action="store_true", help="print per-layer decision table")
    ap.add_argument("--permissive", action="store_true",
                    help="continue past unresolved shapes (NOT for evaluation runs)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: model not found: {args.model}")

    overrides = {}
    if args.calibration:
        with open(args.calibration) as f:
            overrides.update(json.load(f))
    for key, val in (("host_weight_budget_bytes", args.host_budget_mb and int(args.host_budget_mb * 2**20)),
                     ("link_bw_bytes_per_s", args.link_bw), ("link_rtt_ms", args.link_rtt_ms),
                     ("gpu_gflops", args.gpu_gflops), ("fpga_int8_gops", args.fpga_int8_gops),
                     ("pool_prefetch_pages", args.prefetch_pages)):
        if val is not None:
            overrides[key] = val
    if args.resident:
        overrides["fpga_weights_resident"] = True
    hw = HardwareParams.measured_uart_substrate(**overrides)

    print(f"Loading ONNX model: {args.model}")
    # load_external_data=False: multi-GB models — we only need initializer dims.
    model = onnx.load(args.model, load_external_data=False)
    try:
        layers = parse_onnx_graph(model, batch=args.batch, strict=not args.permissive)
    except ShapeResolutionError as e:
        sys.exit(f"ERROR: {e}")
    print(f"Parsed {len(layers)} layers, batch={args.batch}, "
          f"weights {sum(l.weight_bytes for l in layers) / 2**20:.1f} MiB, "
          f"activations {sum(l.output_tensor_bytes for l in layers) / 2**10:.1f} KiB/inference")

    allowed = tuple(p.strip() for p in args.allow.split(",") if p.strip() in PLACEMENTS)
    result = partition_model(layers, CostModel(hw), allowed=allowed,
                             bucket_bytes=(int(args.bucket_mb * 2**20) if args.bucket_mb else None))
    if args.report:
        print_report(result, hw)
    else:
        print(f"Partition: {result.placement_counts}, {result.num_transfers} transfers, "
              f"estimated {result.total_latency_ms:.2f} ms")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(generate_manifest(layers, result, hw=hw, batch=args.batch, model_path=args.model))
    print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()
