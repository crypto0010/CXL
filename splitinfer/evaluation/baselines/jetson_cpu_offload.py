#!/usr/bin/env python3
"""Baseline B3: CPU offload inference on Jetson using onnxruntime.

Simulates naive heterogeneous compute by splitting execution: early layers
run on GPU (CUDAExecutionProvider), trailing layers fall back to CPU
(CPUExecutionProvider).  This represents the simplest "offload" approach
without an FPGA accelerator.

Approach: the model graph is partitioned into two sub-models via ONNX
helper utilities.  The first sub-model (GPU portion) is executed with
CUDAExecutionProvider; the second (CPU portion) is executed with
CPUExecutionProvider.  Total latency includes the intermediate tensor
transfer between the two sessions.

Usage:
    python jetson_cpu_offload.py <model.onnx> [--cpu-layers N] [--warmup N] [--runs N]
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


def _get_node_list(model: onnx.ModelProto) -> list:
    """Return the list of nodes in topological order."""
    return list(model.graph.node)


def _split_model(model_path: str, cpu_layers: int, tmpdir: str):
    """Split the ONNX model into a GPU part and a CPU part.

    Returns (gpu_model_path, cpu_model_path, split_index).
    If the model has fewer nodes than cpu_layers, the entire model runs on CPU.
    """
    model = onnx.load(model_path)
    nodes = _get_node_list(model)
    total = len(nodes)

    if cpu_layers <= 0 or cpu_layers >= total:
        # Everything on CPU — degenerate case
        cpu_path = os.path.join(tmpdir, "full_cpu.onnx")
        onnx.save(model, cpu_path)
        return None, cpu_path, 0

    split_idx = total - cpu_layers  # first split_idx nodes on GPU, rest on CPU

    # Collect all value_info / initializer names for easy lookup
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}

    # Collect outputs produced by GPU nodes — these become the intermediate tensors
    gpu_output_names = set()
    for node in nodes[:split_idx]:
        for out in node.output:
            if out:
                gpu_output_names.add(out)

    # Determine which GPU outputs are consumed by CPU nodes (the "cut" tensors)
    cpu_input_names = set()
    for node in nodes[split_idx:]:
        for inp in node.input:
            if inp and inp in gpu_output_names:
                cpu_input_names.add(inp)

    # Also include original graph inputs consumed by CPU nodes
    graph_input_names = {inp.name for inp in graph.input}
    for node in nodes[split_idx:]:
        for inp in node.input:
            if inp and inp in graph_input_names and inp not in initializer_names:
                cpu_input_names.add(inp)

    # Build value_info map for intermediate tensors
    vi_map = {vi.name: vi for vi in graph.value_info}
    for inp in graph.input:
        vi_map[inp.name] = inp
    for out in graph.output:
        vi_map[out.name] = out

    # --- GPU sub-model ---
    gpu_outputs = []
    for name in cpu_input_names:
        if name in vi_map:
            gpu_outputs.append(vi_map[name])
        else:
            # Create a placeholder value_info with unknown type
            gpu_outputs.append(onnx_helper.make_tensor_value_info(name, TensorProto.FLOAT, None))

    # Filter initializers to only those consumed by GPU nodes
    gpu_init_names = set()
    for node in nodes[:split_idx]:
        for inp in node.input:
            if inp and inp in initializer_names:
                gpu_init_names.add(inp)

    gpu_graph = onnx_helper.make_graph(
        nodes=list(nodes[:split_idx]),
        name="gpu_part",
        inputs=list(graph.input),
        outputs=gpu_outputs,
        initializer=[init for init in graph.initializer if init.name in gpu_init_names],
    )
    gpu_model = onnx_helper.make_model(gpu_graph)
    gpu_model.ir_version = model.ir_version
    gpu_model.opset_import.MergeFrom(model.opset_import)
    gpu_path = os.path.join(tmpdir, "gpu_part.onnx")
    onnx.save(gpu_model, gpu_path)

    # --- CPU sub-model ---
    # Inputs: cut tensors + original graph inputs needed + initializers consumed
    cpu_graph_inputs = []
    seen = set()
    for name in cpu_input_names:
        if name not in seen:
            if name in vi_map:
                cpu_graph_inputs.append(vi_map[name])
            else:
                cpu_graph_inputs.append(onnx_helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
            seen.add(name)

    # Initializers needed by CPU nodes
    cpu_init_names = set()
    for node in nodes[split_idx:]:
        for inp in node.input:
            if inp and inp in initializer_names:
                cpu_init_names.add(inp)

    cpu_graph = onnx_helper.make_graph(
        nodes=list(nodes[split_idx:]),
        name="cpu_part",
        inputs=cpu_graph_inputs,
        outputs=list(graph.output),
        initializer=[init for init in graph.initializer if init.name in cpu_init_names],
    )
    cpu_model = onnx_helper.make_model(cpu_graph)
    cpu_model.ir_version = model.ir_version
    cpu_model.opset_import.MergeFrom(model.opset_import)
    cpu_path = os.path.join(tmpdir, "cpu_part.onnx")
    onnx.save(cpu_model, cpu_path)

    return gpu_path, cpu_path, split_idx


def _auto_cpu_layers(model_path: str) -> int:
    """Heuristic: offload roughly the last 25 % of nodes to CPU."""
    model = onnx.load(model_path)
    total = len(model.graph.node)
    n = max(1, total // 4)
    return n


def benchmark(model_path: str, warmup: int, runs: int, cpu_layers: int) -> None:
    print(f"=== Baseline B3: Jetson CPU Offload ===")
    print(f"Model:      {model_path}")
    print(f"CPU layers: {cpu_layers}")
    print(f"Warmup:     {warmup} runs")
    print(f"Measure:    {runs} runs")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Splitting model graph...")
        t_split_start = time.perf_counter()
        gpu_path, cpu_path, split_idx = _split_model(model_path, cpu_layers, tmpdir)
        t_split_end = time.perf_counter()
        split_time_s = t_split_end - t_split_start
        print(f"  Split time: {split_time_s:.2f} s")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4

        # Try CUDA for GPU part; fall back to CPU if unavailable
        gpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        cpu_providers = ["CPUExecutionProvider"]

        if gpu_path is not None:
            print(f"  GPU sub-model: {split_idx} nodes")
            total_nodes = split_idx + cpu_layers
            print(f"  CPU sub-model: {total_nodes - split_idx} nodes")
            gpu_session = ort.InferenceSession(gpu_path, sess_options=so, providers=gpu_providers)
            cpu_session = ort.InferenceSession(cpu_path, sess_options=so, providers=cpu_providers)

            actual_gpu_provider = gpu_session.get_providers()[0]
            print(f"  GPU session provider: {actual_gpu_provider}")

            gpu_feeds = build_dummy_inputs(gpu_session)
            gpu_output_names = [o.name for o in gpu_session.get_outputs()]
            cpu_output_names = [o.name for o in cpu_session.get_outputs()]

            def run_once():
                gpu_results = gpu_session.run(gpu_output_names, gpu_feeds)
                # Build CPU feeds from GPU outputs
                cpu_feeds = {}
                for name, arr in zip(gpu_output_names, gpu_results):
                    cpu_feeds[name] = arr
                cpu_session.run(cpu_output_names, cpu_feeds)
        else:
            print("  All nodes assigned to CPU (no GPU sub-model)")
            cpu_session = ort.InferenceSession(cpu_path, sess_options=so, providers=cpu_providers)
            cpu_feeds = build_dummy_inputs(cpu_session)
            cpu_output_names = [o.name for o in cpu_session.get_outputs()]

            def run_once():
                cpu_session.run(cpu_output_names, cpu_feeds)

        print()

        # Warmup
        print(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            run_once()

        # Benchmark
        print(f"Running {runs} measurement iterations...")
        latencies_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            run_once()
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

    result = {
        "baseline": "B3_jetson_cpu_offload",
        "model": os.path.basename(model_path),
        "cpu_layers": cpu_layers,
        "split_index": split_idx,
        "warmup_runs": warmup,
        "measure_runs": runs,
        "mean_ms": round(float(mean_ms), 4),
        "median_ms": round(float(median_ms), 4),
        "p99_ms": round(float(p99_ms), 4),
        "throughput_inf_per_s": round(float(throughput), 4),
        "split_time_s": round(float(split_time_s), 3),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="B3: Jetson CPU offload baseline")
    parser.add_argument("model",        help="Path to ONNX model file")
    parser.add_argument("--cpu-layers", type=int, default=0,
                        help="Number of trailing layers to run on CPU (default: auto ~25%%)")
    parser.add_argument("--warmup",     type=int, default=50,   help="Warmup iterations (default: 50)")
    parser.add_argument("--runs",       type=int, default=1000, help="Measurement iterations (default: 1000)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    cpu_layers = args.cpu_layers if args.cpu_layers > 0 else _auto_cpu_layers(args.model)
    benchmark(args.model, args.warmup, args.runs, cpu_layers)


if __name__ == "__main__":
    main()
