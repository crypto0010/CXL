#!/usr/bin/env python3
"""Experiment E5: Scalability Projection across Interconnect Bandwidths.

Takes a measured results JSON (from B1/B2 or splitinfer_run telemetry) and
projects total inference latency at multiple interconnect bandwidths:

  - USB 2.0 HS  :   40 MB/s
  - USB 3.0     :  400 MB/s
  - PCIe Gen3 x1: 1000 MB/s
  - PCIe Gen4 x1: 2000 MB/s
  - CXL-lite    : 4000 MB/s   (CXL 1.1 equivalent over thin link)
  - CXL 2.0     : 8000 MB/s   (CXL 2.0 x4 estimate)

The projection model decomposes total latency into:
  latency_total = latency_compute + latency_transfer(bw)

where latency_transfer = transfer_bytes / bandwidth + latency_overhead_ms

Usage:
    python e5_scalability.py <results.json> [--output results_e5.json]

Input JSON format (produced by splitinfer_run or estimated):
{
  "compute_latency_ms": <float>,   # Pure compute time (GPU + FPGA), independent of BW
  "transfer_bytes":     <int>,     # Total bytes transferred over interconnect
  "usb_latency_ms":     <float>,   # Per-transfer protocol overhead (default: 1.0 ms)
  "num_transfers":      <int>      # Number of host<->FPGA transfers per inference
}
"""

import argparse
import json
import os
import sys


# Interconnect configurations: (label, bandwidth_mb_per_s, latency_overhead_ms_per_transfer)
INTERCONNECTS = [
    ("USB 2.0 HS",  40.0,   1.0),
    ("USB 3.0",     400.0,  0.5),
    ("PCIe Gen3 x1", 1000.0, 0.1),
    ("PCIe Gen4 x1", 2000.0, 0.05),
    ("CXL-lite",    4000.0, 0.02),
    ("CXL 2.0",     8000.0, 0.01),
]


def project(results: dict) -> list[dict]:
    compute_ms    = float(results.get("compute_latency_ms", 0.0))
    transfer_bytes = int(results.get("transfer_bytes", 0))
    num_transfers  = int(results.get("num_transfers", 1))

    projections = []
    for label, bw_mb_s, overhead_ms in INTERCONNECTS:
        # Transfer time = bytes / bandwidth
        xfer_time_ms = (transfer_bytes / (bw_mb_s * 1_000_000)) * 1000.0
        # Protocol overhead per transfer
        overhead_total_ms = overhead_ms * num_transfers
        total_ms = compute_ms + xfer_time_ms + overhead_total_ms

        speedup_vs_usb2 = None  # filled in after first entry
        projections.append({
            "interconnect":        label,
            "bandwidth_mb_s":      bw_mb_s,
            "compute_latency_ms":  round(compute_ms, 4),
            "transfer_latency_ms": round(xfer_time_ms, 4),
            "overhead_ms":         round(overhead_total_ms, 4),
            "total_latency_ms":    round(total_ms, 4),
            "throughput_inf_s":    round(1000.0 / total_ms, 4) if total_ms > 0 else None,
        })

    # Compute speedup relative to USB 2.0 baseline
    usb2_latency = projections[0]["total_latency_ms"]
    for p in projections:
        p["speedup_vs_usb2"] = round(usb2_latency / p["total_latency_ms"], 3) if p["total_latency_ms"] > 0 else None

    return projections


def print_table(projections: list[dict]) -> None:
    header = f"{'Interconnect':<20} {'BW (MB/s)':>12} {'Compute (ms)':>14} {'Xfer (ms)':>12} {'Total (ms)':>12} {'Throughput':>12} {'Speedup':>9}"
    print(header)
    print("-" * len(header))
    for p in projections:
        print(f"{p['interconnect']:<20} {p['bandwidth_mb_s']:>12.0f} "
              f"{p['compute_latency_ms']:>14.3f} {p['transfer_latency_ms']:>12.3f} "
              f"{p['total_latency_ms']:>12.3f} {p['throughput_inf_s']:>11.2f}/s "
              f"{p['speedup_vs_usb2']:>8.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: Scalability projection across interconnect bandwidths")
    parser.add_argument("results", help="Path to measured results JSON file")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: stdout)")
    args = parser.parse_args()

    if not os.path.isfile(args.results):
        print(f"ERROR: results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    with open(args.results) as f:
        results = json.load(f)

    print("=== E5: Scalability Projection ===", file=sys.stderr)
    print(f"Input: {args.results}", file=sys.stderr)
    print(f"  compute_latency_ms : {results.get('compute_latency_ms', 0.0):.3f} ms", file=sys.stderr)
    print(f"  transfer_bytes     : {results.get('transfer_bytes', 0):,} bytes", file=sys.stderr)
    print(f"  num_transfers      : {results.get('num_transfers', 1)}", file=sys.stderr)
    print("", file=sys.stderr)

    projections = project(results)

    print_table(projections)
    print()

    output = {
        "experiment": "E5_scalability",
        "input": results,
        "projections": projections,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to: {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
