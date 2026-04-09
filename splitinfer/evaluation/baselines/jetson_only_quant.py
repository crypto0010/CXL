#!/usr/bin/env python3
"""Baseline B2: INT8 quantized inference on Jetson using onnxruntime.

Quantizes the model to INT8 using onnxruntime.quantization (static or dynamic),
then benchmarks the same as B1.

Usage:
    python jetson_only_quant.py <model.onnx> [--warmup N] [--runs N] [--dynamic]
"""

import argparse
import time
import sys
import os
import json
import tempfile

import numpy as np

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    from onnxruntime.quantization import CalibrationDataReader
except ImportError:
    print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

try:
    import onnx
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


class DummyCalibrationReader(CalibrationDataReader):
    """Provides a small set of random calibration samples for static quantization."""

    def __init__(self, session_inputs: list, num_samples: int = 100):
        self._inputs   = session_inputs
        self._samples  = num_samples
        self._iter     = iter(range(num_samples))

    def get_next(self):
        try:
            next(self._iter)
        except StopIteration:
            return None
        feeds = {}
        for inp in self._inputs:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if inp.type == "tensor(float)":
                feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
            elif inp.type == "tensor(int64)":
                feeds[inp.name] = np.zeros(shape, dtype=np.int64)
            else:
                feeds[inp.name] = np.zeros(shape, dtype=np.float32)
        return feeds


def quantize_model(model_path: str, quant_path: str, dynamic: bool) -> None:
    """Quantize the model and write to quant_path."""
    if dynamic:
        print("  Quantization mode: dynamic (INT8 weights, FP32 activations)")
        quantize_dynamic(
            model_input=model_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
        )
    else:
        print("  Quantization mode: static (INT8 weights + activations, dummy calibration)")
        # Load model to get input metadata for calibration reader
        tmp_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        calib_reader = DummyCalibrationReader(tmp_session.get_inputs(), num_samples=100)
        quantize_static(
            model_input=model_path,
            model_output=quant_path,
            calibration_data_reader=calib_reader,
            weight_type=QuantType.QInt8,
        )


def benchmark(model_path: str, warmup: int, runs: int, dynamic: bool) -> None:
    mode_label = "dynamic INT8" if dynamic else "static INT8"
    print(f"=== Baseline B2: Jetson-only {mode_label} ===")
    print(f"Model:   {model_path}")
    print(f"Warmup:  {warmup} runs")
    print(f"Measure: {runs} runs")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        quant_path = os.path.join(tmpdir, "model_quant.onnx")

        print("Quantizing model...")
        t_quant_start = time.perf_counter()
        try:
            quantize_model(model_path, quant_path, dynamic)
        except Exception as exc:
            print(f"  WARNING: quantization failed ({exc}), falling back to FP32 model.")
            quant_path = model_path
        t_quant_end = time.perf_counter()
        quant_time_s = t_quant_end - t_quant_start
        print(f"  Quantization time: {quant_time_s:.2f} s")

        orig_size = os.path.getsize(model_path)
        quant_size = os.path.getsize(quant_path)
        print(f"  Original size : {orig_size / (1024*1024):.2f} MB")
        print(f"  Quantized size: {quant_size / (1024*1024):.2f} MB")
        print(f"  Compression   : {orig_size / quant_size:.2f}x")
        print()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4

        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(quant_path, sess_options=so, providers=providers)
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
    mean_ms   = float(np.mean(latencies_ms))
    std_ms    = float(np.std(latencies_ms, ddof=1)) if runs > 1 else 0.0
    min_ms    = float(np.min(latencies_ms))
    max_ms    = float(np.max(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p5_ms     = float(np.percentile(latencies_ms, 5))
    p95_ms    = float(np.percentile(latencies_ms, 95))
    p99_ms    = float(np.percentile(latencies_ms, 99))
    total_s   = float(np.sum(latencies_ms) / 1000.0)
    throughput = runs / total_s

    print()
    print("=== Results ===")
    print(f"Mean latency   : {mean_ms:.3f} ms  (std {std_ms:.3f})")
    print(f"Median latency : {median_ms:.3f} ms")
    print(f"p5  / p95      : {p5_ms:.3f} / {p95_ms:.3f} ms")
    print(f"P99 latency    : {p99_ms:.3f} ms")
    print(f"Throughput     : {throughput:.2f} inf/s")
    print()

    result = {
        "baseline":     "B2_jetson_int8",
        "model":        os.path.basename(model_path),
        "quant_mode":   "dynamic" if dynamic else "static",
        "warmup_runs":  warmup,
        "measure_runs": runs,
        "mean_ms":      round(mean_ms, 4),
        "std_ms":       round(std_ms,  4),
        "min_ms":       round(min_ms,  4),
        "p5_ms":        round(p5_ms,   4),
        "median_ms":    round(median_ms, 4),
        "p95_ms":       round(p95_ms,  4),
        "p99_ms":       round(p99_ms,  4),
        "max_ms":       round(max_ms,  4),
        "throughput_inf_per_s": round(float(throughput), 4),
        "quant_time_s": round(float(quant_time_s), 3),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="B2: Jetson-only INT8 quantized baseline")
    parser.add_argument("model",     help="Path to ONNX model file")
    parser.add_argument("--warmup",  type=int, default=50,   help="Warmup iterations (default: 50)")
    parser.add_argument("--runs",    type=int, default=1000, help="Measurement iterations (default: 1000)")
    parser.add_argument("--dynamic", action="store_true",    help="Use dynamic quantization (default: static)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    benchmark(args.model, args.warmup, args.runs, args.dynamic)


if __name__ == "__main__":
    main()
