#!/usr/bin/env python3
"""E2 (v2): one workload, every execution path, honest reporting.

  B1  onnxruntime FP32, CPU            (v1 baseline, kept for continuity)
  B2  onnxruntime INT8 dynamic, CPU    (v1 baseline)
  B3  TensorRT FP16, GPU
  B4  TensorRT INT8, GPU
  SI-host  lowered INT8 program, host integer kernels (weights host-resident)
  SI-pool  weights in FPGA DDR2, host compute through cxlwin (memory -> compute)
  SI-nmc   weights in FPGA DDR2, NMC engines (compute -> memory)

Every cell is wrapped in a tegrastats window of at least --min-window-s.
A cell that fails or times out is recorded with that status and NEVER as
"ok" (v1 recorded 180 s timeouts as ok).  --transport emul labels SI cells
"emulated": true; --transport usb labels them "measured": true.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evaluation" / "baselines"
BIN = ROOT / "build" / "runtime" / "v2" / "splitinfer_v2"
sys.path.insert(0, str(BASE))
from tegra_power import TegraStatsCollector  # noqa: E402


def wrapped(cmd, min_window_s, timeout, parse_json_last_line=True, env=None):
    """Run cmd under a power window; if it finishes early, loop it until the window is long enough."""
    out = {"cmd": " ".join(map(str, cmd))}
    with TegraStatsCollector(interval_ms=100, sudo=os.geteuid() != 0) as col:
        t0 = time.monotonic(); runs = 0; last = None
        while True:
            try:
                p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, timeout=timeout, env=env)
            except subprocess.TimeoutExpired:
                out.update(status="timeout", duration_s=time.monotonic() - t0); return out
            runs += 1
            if p.returncode not in (0, 3):
                out.update(status="failed", exit_code=p.returncode, stderr=p.stderr[-600:]); return out
            last = p
            if time.monotonic() - t0 >= min_window_s: break
    out["duration_s"] = time.monotonic() - t0; out["process_runs"] = runs
    out["power"] = col.summary()
    try:
        lines = last.stdout.strip().splitlines()
        start = max(i for i, l in enumerate(lines) if l.startswith("{"))
        out["result"] = json.loads("\n".join(lines[start:]))
    except Exception:
        out.update(status="parse_failed", stdout=last.stdout[-800:]); return out
    st = out["result"].get("stats") or out["result"]
    out["status"] = "ok"
    if "VDD_IN" in out["power"]["rails"] and "mean_ms" in st:
        out["energy_per_inf_J"] = out["power"]["rails"]["VDD_IN"]["mean_mw"] / 1000 * st["mean_ms"] / 1000
        out["power_samples"] = out["power"]["rails"]["VDD_IN"]["samples"]
    corr = out["result"].get("correctness")
    if corr and corr.get("bit_exact") != corr.get("total"):
        out["status"] = "output_mismatch"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("program_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=30); ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--si-runs", type=int, help="iterations for SI_* cells (default: --runs)")
    ap.add_argument("--min-window-s", type=float, default=3.0)
    ap.add_argument("--transport", choices=["emul", "usb"], default="emul")
    ap.add_argument("--emul-link-bw", default="11520"); ap.add_argument("--emul-rtt-ms", default="5")
    ap.add_argument("--prefetch-pages", default="64")
    ap.add_argument("--skip", default="", help="comma list of cells to skip")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    skip = set(a.skip.split(",")) if a.skip else set()
    si_runs = a.si_runs or a.runs
    res = {"experiment": "E2_v2", "model": a.model, "program": a.program_dir, "runs": a.runs, "warmup": a.warmup,
           "transport": a.transport, "cells": {}}
    emul = ["--emul-link-bw", a.emul_link_bw, "--emul-rtt-ms", a.emul_rtt_ms, "--emul-clock-hz", "81.25e6"] if a.transport == "emul" else []
    cells = {
        "B1_ort_fp32_cpu": [sys.executable, BASE / "jetson_only_fp.py", a.model, "--warmup", a.warmup, "--runs", a.runs],
        "B2_ort_int8_cpu": [sys.executable, BASE / "jetson_only_quant.py", a.model, "--warmup", a.warmup, "--runs", a.runs],
        "B3_trt_fp16_gpu": [sys.executable, BASE / "jetson_tensorrt.py", a.model, "--precision", "fp16", "--warmup", a.warmup, "--runs", a.runs, "--min-window-s", "0", "--json-out", "/tmp/trt_b3.json"],
        "B4_trt_int8_gpu": [sys.executable, BASE / "jetson_tensorrt.py", a.model, "--precision", "int8", "--warmup", a.warmup, "--runs", a.runs, "--min-window-s", "0", "--json-out", "/tmp/trt_b4.json"],
        "SI_host":  [BIN, a.program_dir, "--mode", "host", "--iterations", si_runs, "--warmup", min(a.warmup, 2), "--json"],
        "SI_nmc":   [BIN, a.program_dir, "--mode", "nmc", "--transport", a.transport, "--iterations", si_runs, "--warmup", min(a.warmup, 2), "--json"] + emul,
        "SI_pool":  [BIN, a.program_dir, "--mode", "pool", "--transport", a.transport, "--iterations", si_runs, "--warmup", min(a.warmup, 2), "--json", "--prefetch-pages", a.prefetch_pages] + emul,
        "SI_pool_warm": [BIN, a.program_dir, "--mode", "pool", "--transport", a.transport, "--iterations", si_runs, "--warmup", min(a.warmup, 2), "--json", "--prefetch-pages", a.prefetch_pages, "--pool-warm"] + emul,
    }
    for name, cmd in cells.items():
        if name in skip: continue
        print(f"[{name}] ...", flush=True)
        r = wrapped(cmd, a.min_window_s, a.timeout)
        if name.startswith("SI_") and name != "SI_host":
            r["emulated"] = a.transport == "emul"; r["measured"] = a.transport == "usb"
        res["cells"][name] = r
        st = (r.get("result") or {}).get("stats") or (r.get("result") or {})
        print(f"[{name}] {r['status']}" + (f"  median {st.get('median_ms', float('nan')):.3f} ms  p99 {st.get('p99_ms', float('nan')):.3f}  "
              f"power {r['power']['rails'].get('VDD_IN', {}).get('mean_mw', 0):.0f} mW ({r.get('power_samples', 0)} samples)  "
              f"{r.get('energy_per_inf_J', 0)*1000:.3f} mJ/inf" if r["status"] == "ok" else ""), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f: json.dump(res, f, indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
