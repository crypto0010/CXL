#!/usr/bin/env python3
"""B3 / B4: Jetson GPU baseline via TensorRT (FP16 / INT8).

The submitted paper's baselines used onnxruntime's CPUExecutionProvider —
because the installed onnxruntime has no CUDA provider — and reviewers
rightly rejected them as not the best Jetson path.  This runs the same ONNX
model through TensorRT on the GPU.

Latency is per inference INCLUDING host<->device copies (the CPU baseline
also includes feeding inputs), measured with a monotonic clock around
execute + synchronize.  Power is sampled with tegrastats; the measurement
window is stretched to at least --min-window-s so the power figure rests
on dozens of samples rather than two (v1 defect D11).

INT8 here uses TensorRT's implicit quantisation with trtexec's default
dynamic ranges: a fair TIMING baseline, not an accuracy claim.
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tegra_power import TegraStatsCollector

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def build_engine(onnx_path: str, precision: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    plan = os.path.join(cache_dir, os.path.basename(onnx_path) + f".{precision}.plan")
    if os.path.exists(plan) and os.path.getmtime(plan) > os.path.getmtime(onnx_path):
        return plan
    cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={plan}", "--iterations=1", "--warmUp=0"]
    if precision == "fp16": cmd.append("--fp16")
    elif precision == "int8": cmd += ["--int8", "--fp16"]
    print("building:", " ".join(cmd[:3]), "...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(plan):
        sys.exit(f"trtexec failed:\n{r.stdout[-1500:]}\n{r.stderr[-500:]}")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--runs", type=int, default=1000); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min-window-s", type=float, default=3.0)
    ap.add_argument("--power", action="store_true"); ap.add_argument("--cache", default="/tmp/trt_cache")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json-out")
    a = ap.parse_args()

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    plan = build_engine(a.model, a.precision, a.cache)
    logger = trt.Logger(trt.Logger.ERROR)
    with open(plan, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    stream = cuda.Stream()
    rng = np.random.default_rng(a.seed)

    # Buffers
    import onnx
    m = onnx.load(a.model, load_external_data=False)
    table_rows = {n.input[1]: next(i.dims[0] for i in m.graph.initializer if i.name == n.input[0])
                  for n in m.graph.node if n.op_type == "Gather"}
    host, dev = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(max(d, 1) for d in engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if np.issubdtype(dtype, np.integer):
                arr = rng.integers(0, table_rows.get(name, 1), size=shape).astype(dtype)
            else:
                arr = rng.standard_normal(shape).astype(dtype)
            ctx.set_input_shape(name, shape)
        else:
            arr = np.zeros(shape, dtype=dtype)
        host[name] = cuda.pagelocked_empty(arr.shape, dtype); host[name][...] = arr
        dev[name] = cuda.mem_alloc(host[name].nbytes)
        ctx.set_tensor_address(name, int(dev[name]))
    inputs = [n for n in host if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in host if n not in inputs]

    def infer():
        for n in inputs: cuda.memcpy_htod_async(dev[n], host[n], stream)
        ctx.execute_async_v3(stream.handle)
        for n in outputs: cuda.memcpy_dtoh_async(host[n], dev[n], stream)
        stream.synchronize()

    for _ in range(a.warmup): infer()
    # size the window
    t0 = time.perf_counter(); 
    for _ in range(20): infer()
    est_ms = (time.perf_counter() - t0) / 20 * 1000
    runs = max(a.runs, int(math.ceil(a.min_window_s * 1000 / max(est_ms, 1e-3))))

    lat = []
    collector = TegraStatsCollector(interval_ms=100, sudo=os.geteuid() != 0) if a.power else None
    if collector: collector.start()
    for _ in range(runs):
        t = time.perf_counter(); infer(); lat.append((time.perf_counter() - t) * 1000.0)
    power = None
    if collector:
        collector.stop(); power = collector.summary()
    lat = np.array(lat)
    res = {
        "baseline": f"B_tensorrt_{a.precision}", "model": os.path.basename(a.model), "precision": a.precision,
        "engine": plan, "warmup_runs": a.warmup, "measure_runs": int(runs),
        "stats": {"count": int(runs), "mean_ms": float(lat.mean()), "std_ms": float(lat.std(ddof=1)),
                  "min_ms": float(lat.min()), "p5_ms": float(np.percentile(lat, 5)), "median_ms": float(np.median(lat)),
                  "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)), "max_ms": float(lat.max()),
                  "cv_pct": float(100 * lat.std(ddof=1) / lat.mean())},
        "throughput_inf_per_s": float(runs / (lat.sum() / 1000)),
        "power": power,
    }
    if power and "VDD_IN" in power["rails"]:
        res["energy_per_inf_J"] = power["rails"]["VDD_IN"]["mean_mw"] / 1000 * float(lat.mean()) / 1000
        res["power_samples"] = power["rails"]["VDD_IN"]["samples"]
        if res["power_samples"] < 20:
            res["power_warning"] = f"only {res['power_samples']} tegrastats samples"
    s = res["stats"]
    print(f"{res['baseline']}: median {s['median_ms']:.3f} ms  p95 {s['p95_ms']:.3f}  p99 {s['p99_ms']:.3f}  cv {s['cv_pct']:.1f}%  runs {runs}"
          + (f"  power {power['rails']['VDD_IN']['mean_mw']:.0f} mW ({res['power_samples']} samples)  {res['energy_per_inf_J']*1000:.3f} mJ/inf" if power and 'VDD_IN' in power['rails'] else ""))
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w") as f: json.dump(res, f, indent=1)
    print(json.dumps(res))   # single-line JSON for orchestrators


if __name__ == "__main__":
    main()
