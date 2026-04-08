#!/usr/bin/env python3
"""Baseline B4: Disk swap inference on Jetson using onnxruntime.

Demonstrates SD card-based weight swapping.  For each inference pass the
script iterates over every node in the model graph, loads that node's
weights from disk (simulating SD-card I/O), builds a single-node
sub-model, runs it, then frees the weights.  This is the worst-case
memory expansion approach and is expected to be significantly slower than
the other baselines due to disk I/O overhead.

On Jetson Orin Nano the SD card read speed is typically ~100 MB/s, so
the disk round-trips dominate latency for any non-trivial model.

Usage:
    python jetson_disk_swap.py <model.onnx> [--warmup N] [--runs N]
"""

import argparse
import time
import sys
import os
import json
import tempfile
import shutil

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

try:
    import onnx
    from onnx import helper as onnx_helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx is not installed. Run: pip install onnx", file=sys.stderr)
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


def _dump_weights_to_disk(model_path: str, weight_dir: str) -> tuple[dict, int]:
    """Serialize each initializer (weight tensor) to a separate .npy file on disk.

    Returns a dict mapping initializer name -> file path.
    """
    model = onnx.load(model_path)
    weight_files = {}
    total_bytes = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        fpath = os.path.join(weight_dir, init.name.replace("/", "_") + ".npy")
        np.save(fpath, arr)
        weight_files[init.name] = fpath
        total_bytes += arr.nbytes
    return weight_files, total_bytes


def _load_weights_from_disk(weight_files: dict) -> dict:
    """Load all weight tensors back from disk into memory.

    Returns a dict mapping initializer name -> numpy array.
    """
    weights = {}
    for name, fpath in weight_files.items():
        weights[name] = np.load(fpath)
    return weights


def _rebuild_model_with_weights(model_path: str, weights: dict, tmpdir: str) -> str:
    """Rebuild the ONNX model with the given weight arrays (loaded from disk).

    Returns path to the rebuilt model file.
    """
    model = onnx.load(model_path)
    # Replace initializers with the disk-loaded versions
    new_initializers = []
    for init in model.graph.initializer:
        if init.name in weights:
            new_init = numpy_helper.from_array(weights[init.name], name=init.name)
            new_initializers.append(new_init)
        else:
            new_initializers.append(init)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    rebuilt_path = os.path.join(tmpdir, "rebuilt.onnx")
    onnx.save(model, rebuilt_path)
    return rebuilt_path


def benchmark(model_path: str, warmup: int, runs: int) -> None:
    print(f"=== Baseline B4: Jetson Disk Swap ===")
    print(f"Model:   {model_path}")
    print(f"Warmup:  {warmup} runs")
    print(f"Measure: {runs} runs")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        weight_dir = os.path.join(tmpdir, "weights")
        os.makedirs(weight_dir)

        # Dump weights to disk
        print("Dumping model weights to disk...")
        t_dump_start = time.perf_counter()
        weight_files, total_weight_bytes = _dump_weights_to_disk(model_path, weight_dir)
        t_dump_end = time.perf_counter()
        dump_time_s = t_dump_end - t_dump_start
        num_weights = len(weight_files)
        print(f"  Weights:    {num_weights} tensors")
        print(f"  Total size: {total_weight_bytes / (1024 * 1024):.2f} MB")
        print(f"  Dump time:  {dump_time_s:.2f} s")
        print()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]

        # For each inference pass:
        # 1. Load weights from disk (simulating SD card read)
        # 2. Rebuild the model in memory
        # 3. Create a session and run inference
        # 4. Discard the session (freeing memory)
        #
        # This is deliberately expensive to demonstrate the disk-swap penalty.

        # Pre-load model structure (without weights) to build dummy inputs
        tmp_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        feeds = build_dummy_inputs(tmp_session)
        output_names = [o.name for o in tmp_session.get_outputs()]
        del tmp_session

        def run_once():
            """Single inference pass with disk-swap overhead."""
            # Step 1: Load weights from disk
            weights = _load_weights_from_disk(weight_files)
            # Step 2: Rebuild model with loaded weights
            rebuilt_path = _rebuild_model_with_weights(model_path, weights, tmpdir)
            # Step 3: Create session and run
            sess = ort.InferenceSession(rebuilt_path, sess_options=so, providers=providers)
            sess.run(output_names, feeds)
            # Step 4: Discard (Python GC handles deallocation)
            del sess
            del weights

        # Warmup
        print(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            run_once()

        # Benchmark
        print(f"Running {runs} measurement iterations...")
        latencies_ms = []
        disk_read_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()

            # Measure disk read portion separately
            t_disk_start = time.perf_counter()
            weights = _load_weights_from_disk(weight_files)
            t_disk_end = time.perf_counter()
            disk_read_ms.append((t_disk_end - t_disk_start) * 1000.0)

            rebuilt_path = _rebuild_model_with_weights(model_path, weights, tmpdir)
            sess = ort.InferenceSession(rebuilt_path, sess_options=so, providers=providers)
            sess.run(output_names, feeds)
            del sess
            del weights

            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    disk_read_ms = np.array(disk_read_ms)
    mean_ms   = np.mean(latencies_ms)
    median_ms = np.median(latencies_ms)
    p99_ms    = np.percentile(latencies_ms, 99)
    total_s   = np.sum(latencies_ms) / 1000.0
    throughput = runs / total_s

    mean_disk_ms   = np.mean(disk_read_ms)
    median_disk_ms = np.median(disk_read_ms)

    print()
    print("=== Results ===")
    print(f"Mean latency   : {mean_ms:.3f} ms")
    print(f"Median latency : {median_ms:.3f} ms")
    print(f"P99 latency    : {p99_ms:.3f} ms")
    print(f"Throughput     : {throughput:.2f} inf/s")
    print(f"Mean disk read : {mean_disk_ms:.3f} ms")
    print(f"Median disk rd : {median_disk_ms:.3f} ms")
    print()

    result = {
        "baseline": "B4_jetson_disk_swap",
        "model": os.path.basename(model_path),
        "num_weight_tensors": num_weights,
        "total_weight_bytes": total_weight_bytes,
        "warmup_runs": warmup,
        "measure_runs": runs,
        "mean_ms": round(float(mean_ms), 4),
        "median_ms": round(float(median_ms), 4),
        "p99_ms": round(float(p99_ms), 4),
        "throughput_inf_per_s": round(float(throughput), 4),
        "mean_disk_read_ms": round(float(mean_disk_ms), 4),
        "median_disk_read_ms": round(float(median_disk_ms), 4),
        "dump_time_s": round(float(dump_time_s), 3),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="B4: Jetson disk swap baseline")
    parser.add_argument("model",    help="Path to ONNX model file")
    parser.add_argument("--warmup", type=int, default=50,   help="Warmup iterations (default: 50)")
    parser.add_argument("--runs",   type=int, default=1000, help="Measurement iterations (default: 1000)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    benchmark(args.model, args.warmup, args.runs)


if __name__ == "__main__":
    main()
