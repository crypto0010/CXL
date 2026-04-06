#!/usr/bin/env python3
"""Baseline B1: Full-precision (FP32) inference on Jetson using onnxruntime.

Measures latency over multiple runs with warmup, reports mean/median/p99/throughput.

Usage:
    python jetson_only_fp.py <model.onnx> [--warmup N] [--runs N]
"""

import argparse
import time
import sys
import os

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)


def build_dummy_inputs(session: ort.InferenceSession) -> dict:
    """Build random dummy inputs matching the model's input shapes."""
    feeds = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if inp.type == "tensor(float)":
            feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
        elif inp.type == "tensor(int64)":
            feeds[inp.name] = np.zeros(shape, dtype=np.int64)
        elif inp.type == "tensor(int32)":
            feeds[inp.name] = np.zeros(shape, dtype=np.int32)
        else:
            feeds[inp.name] = np.zeros(shape, dtype=np.float32)
    return feeds


def benchmark(model_path: str, warmup: int, runs: int) -> None:
    print(f"=== Baseline B1: Jetson-only FP32 ===")
    print(f"Model:   {model_path}")
    print(f"Warmup:  {warmup} runs")
    print(f"Measure: {runs} runs")
    print()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4

    # CPU provider for Jetson baseline (CUDA provider can be added via --cuda flag)
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

    feeds = build_dummy_inputs(session)
    output_names = [o.name for o in session.get_outputs()]

    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        session.run(output_names, feeds)

    # Benchmark
    print(f"Running {runs} measurement iterations...")
    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(output_names, feeds)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    mean_ms   = np.mean(latencies_ms)
    median_ms = np.median(latencies_ms)
    p99_ms    = np.percentile(latencies_ms, 99)
    total_s   = np.sum(latencies_ms) / 1000.0
    throughput = runs / total_s

    print()
    print("=== Results ===")
    print(f"Mean latency   : {mean_ms:.3f} ms")
    print(f"Median latency : {median_ms:.3f} ms")
    print(f"P99 latency    : {p99_ms:.3f} ms")
    print(f"Throughput     : {throughput:.2f} inf/s")
    print()

    # JSON-compatible summary for downstream scripts
    import json
    result = {
        "baseline": "B1_jetson_fp32",
        "model": os.path.basename(model_path),
        "warmup_runs": warmup,
        "measure_runs": runs,
        "mean_ms": round(float(mean_ms), 4),
        "median_ms": round(float(median_ms), 4),
        "p99_ms": round(float(p99_ms), 4),
        "throughput_inf_per_s": round(float(throughput), 4),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="B1: Jetson-only FP32 baseline")
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("--warmup", type=int, default=50,  help="Warmup iterations (default: 50)")
    parser.add_argument("--runs",   type=int, default=1000, help="Measurement iterations (default: 1000)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    benchmark(args.model, args.warmup, args.runs)


if __name__ == "__main__":
    main()
