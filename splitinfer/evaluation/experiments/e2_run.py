#!/usr/bin/env python3
"""e2_run.py — multi-model, multi-batch performance & power sweep.

This is the Task #27 implementation of E2: a richer, more honest version
of the bash-based e2_performance.sh that produces a clean JSON results
table across model x batch x backend cells.

Configurations swept (default):
  Models   : dlrm (14 MB toy), yolov8n (13 MB) — both support arbitrary batch
             mobilebert (94 MB) — fixed seq_len=128, batch=1 only
  Batches  : 1, 64, 256 (where the model supports it)
  Backends : B1 FP32 (CPU EP), B2 INT8 (dynamic quant CPU EP),
             SplitInfer-real (host runtime + real FPGA via --real)

For each (model, batch, backend) cell:
  - 5 warmup runs
  - 30 measured runs wrapped in a TegraStatsCollector
  - Reports median / p95 / std latency, throughput (inf/s),
    mean power (mW), energy per inference (mJ)

Output:
  evaluation/experiments/results/e2/e2_summary.json
  Per-cell logs in evaluation/experiments/results/e2/<model>_b<N>_<backend>.log

Why not use the old e2_performance.sh:
  Bash data structures are awful for the cross-product needed here.
  Python gives us native dicts, JSON, numpy stats with much less code.
  The old shell script remains in place as a simpler reference.

Why we skip B3/B4 by default:
  B3 (CPU offload) and B4 (disk swap) are documented baselines that
  we have scripts for, but they don't change the SplitInfer comparison
  story (both are known to be strictly slower than B1), and running
  all five backends × all sweep cells × power-sampled would take hours.
  B1 + B2 + SplitInfer is the core comparison.  Enable B3/B4 with
  --include-cpu-offload and --include-disk-swap.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT     = Path(__file__).resolve().parents[2]
MODELS_DIR    = REPO_ROOT / "evaluation" / "models" / "generated"
BASELINES_DIR = REPO_ROOT / "evaluation" / "baselines"
RESULTS_DIR   = REPO_ROOT / "evaluation" / "experiments" / "results" / "e2"

# Make tegra_power importable
sys.path.insert(0, str(BASELINES_DIR))
from tegra_power import TegraStatsCollector  # noqa: E402


# ─── Cell definitions ──────────────────────────────────────────────────────────


def cells_for_model(model_name: str, batches: List[int]) -> List[Dict]:
    """Return the list of (model_path, batch) cells for a given model.

    Some models only support batch=1 because they were exported with
    fixed input shapes (e.g. mobilebert seq_len=128).  This function
    encodes that constraint and emits "skipped" entries for batch>1.
    """
    if model_name == "dlrm":
        return [{"model_name": "dlrm",
                 "batch": b,
                 "model_path": str(MODELS_DIR / "dlrm.onnx"),
                 "supports_batch": True}
                for b in batches]
    if model_name == "yolov8n":
        # YOLOv8n was exported with fixed [1, 3, 640, 640] input by ultralytics.
        # We can still report batch=1 for it; larger batches would need
        # re-export with --batch.  For now we report batch=1 only and skip
        # batch>1 cells.
        return [{"model_name": "yolov8n",
                 "batch": b,
                 "model_path": str(MODELS_DIR / "yolov8n.onnx"),
                 "supports_batch": (b == 1)}
                for b in batches]
    if model_name == "mobilebert":
        return [{"model_name": "mobilebert",
                 "batch": b,
                 "model_path": str(MODELS_DIR / "mobilebert.onnx"),
                 "supports_batch": (b == 1)}
                for b in batches]
    raise ValueError(f"unknown model: {model_name}")


# ─── Backend runners ───────────────────────────────────────────────────────────


def run_baseline_b1(cell: Dict, warmup: int, runs: int, log_path: Path) -> Dict:
    """Run jetson_only_fp.py and parse its JSON output."""
    cmd = [
        "python3", str(BASELINES_DIR / "jetson_only_fp.py"),
        cell["model_path"],
        "--warmup", str(warmup),
        "--runs", str(runs),
    ]
    return _run_baseline_subprocess(cmd, log_path)


def run_baseline_b2(cell: Dict, warmup: int, runs: int, log_path: Path) -> Dict:
    """Run jetson_only_quant.py (B2 INT8 dynamic) and parse output."""
    cmd = [
        "python3", str(BASELINES_DIR / "jetson_only_quant.py"),
        cell["model_path"],
        "--warmup", str(warmup),
        "--runs", str(runs),
        "--dynamic",
    ]
    return _run_baseline_subprocess(cmd, log_path)


def _run_baseline_subprocess(cmd: List[str], log_path: Path) -> Dict:
    """Run a baseline subprocess wrapped in tegra power sampling."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    with TegraStatsCollector(interval_ms=100) as col:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min hard cap per cell
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "duration_s": time.monotonic() - t0}
    wall_s = time.monotonic() - t0

    log_path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)

    if result.returncode != 0:
        return {
            "status": "failed",
            "exit_code": result.returncode,
            "duration_s": wall_s,
            "power": col.summary(),
        }

    # Parse the trailing JSON block from the baseline's stdout.
    parsed = _parse_trailing_json(result.stdout)
    if parsed is None:
        return {
            "status": "no_json",
            "duration_s": wall_s,
            "power": col.summary(),
        }

    # The upgraded baselines now emit count/std/min/p5/p95/max alongside
    # the legacy mean/median/p99.  Pass through whatever is present so
    # the summary JSON has a consistent schema regardless of which
    # baseline produced it.
    stats = {
        "count":     parsed.get("measure_runs"),
        "mean_ms":   parsed.get("mean_ms"),
        "std_ms":    parsed.get("std_ms"),
        "min_ms":    parsed.get("min_ms"),
        "p5_ms":     parsed.get("p5_ms"),
        "median_ms": parsed.get("median_ms"),
        "p95_ms":    parsed.get("p95_ms"),
        "p99_ms":    parsed.get("p99_ms"),
        "max_ms":    parsed.get("max_ms"),
    }
    if stats["mean_ms"] is not None and stats["std_ms"] is not None and stats["mean_ms"] > 0:
        stats["cv_pct"] = round(100.0 * stats["std_ms"] / stats["mean_ms"], 2)

    return {
        "status":     "ok",
        "stats":      stats,
        # Legacy top-level fields (still emitted for back-compat with the
        # _print_result helper and any downstream scripts).
        "mean_ms":    parsed.get("mean_ms"),
        "median_ms":  parsed.get("median_ms"),
        "p99_ms":     parsed.get("p99_ms"),
        "throughput": parsed.get("throughput_inf_per_s"),
        "duration_s": wall_s,
        "power":      col.summary(),
    }


def _parse_trailing_json(stdout: str) -> Optional[Dict]:
    """Find the last JSON object in stdout (the baselines print one at the end)."""
    # Look for the last '{' that starts a parseable JSON object.
    for i in range(len(stdout)):
        candidate = stdout[i:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
        except Exception:
            return None
    # Fallback: scan line by line for a {...} block
    lines = stdout.strip().split("\n")
    for start in range(len(lines)):
        for end in range(len(lines), start, -1):
            try:
                return json.loads("\n".join(lines[start:end]))
            except json.JSONDecodeError:
                pass
    return None


def run_splitinfer(cell: Dict, warmup: int, runs: int, log_path: Path,
                   prefetch: bool = True) -> Dict:
    """Partition + run splitinfer_run --real, repeated `runs` times.

    The current splitinfer_run executes ONE inference per invocation, so
    we loop in Python and aggregate.  This adds the per-process startup
    cost (USB transport open + close) to each iteration which is a
    realistic worst-case for the overhead-amortization story.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / f"{cell['model_name']}_b{cell['batch']}_manifest.json"

    # Step 1: partition
    partition_cmd = [
        "python3", "-m", "partitioner.cli",
        cell["model_path"],
        "--output", str(manifest_path),
        "--fpga-ddr2-mb", "128",
        "--gpu-gflops",   "1000",
        "--fpga-int8-gops", "6.4",
        "--usb-bw-mbps",   "40",
    ]
    p = subprocess.run(partition_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if p.returncode != 0:
        log_path.write_text("PARTITION FAILED\n" + p.stdout + p.stderr)
        return {"status": "partition_failed", "exit_code": p.returncode}

    # Step 2: run N inferences via splitinfer_run --real
    splitinfer_bin = REPO_ROOT / "build" / "runtime" / "splitinfer_run"
    if not splitinfer_bin.exists() or not os.access(splitinfer_bin, os.X_OK):
        return {"status": "binary_missing", "path": str(splitinfer_bin)}

    env = os.environ.copy()
    if not prefetch:
        env["SPLITINFER_NO_PREFETCH"] = "1"

    latencies = []
    log_lines = []

    # Generous per-call timeout — large batches on the MAC controller can
    # genuinely take several seconds per inference, and large M values
    # multiply the work.  600s per call is the absolute ceiling.
    PER_CALL_TIMEOUT = 600
    timeouts_in_warmup = 0
    flash_script = str(REPO_ROOT / "fpga" / "scripts" / "flash_fpga.sh")

    # Warmup (not measured) — separate runs to settle FPGA state.
    # If even one warmup call times out, we abort this cell rather than
    # waste 30 measurement iterations on a known-broken configuration.
    for _ in range(warmup):
        try:
            subprocess.run([flash_script], capture_output=True, text=True, timeout=60)
            time.sleep(1.0)
            subprocess.run(
                ["sudo", str(splitinfer_bin), str(manifest_path), "--real"],
                capture_output=True, text=True, env=env, timeout=PER_CALL_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            timeouts_in_warmup += 1
            break  # bail out of warmup early

    if timeouts_in_warmup > 0:
        log_path.write_text(
            f"WARMUP TIMED OUT after {PER_CALL_TIMEOUT}s — aborting cell.\n"
            f"This usually means the FPGA cannot handle this batch size in\n"
            f"reasonable time (MAC controller M loop is too long).\n")
        return {
            "status":     "warmup_timeout",
            "duration_s": 0.0,
            "iterations": 0,
            "warmup_timeouts": timeouts_in_warmup,
        }

    # FPGA state-reset strategy: re-flash the bitstream before every
    # measurement iteration.  This is the only reliable way to guarantee
    # the edgecoh_controller FSM starts from a clean S_IDLE state with
    # all CDC handshakes fully quiesced.  The cost is ~6 seconds per
    # iteration (bitstream load over JTAG).  For 30 iterations that's
    # ~3 minutes of overhead — acceptable for paper-quality measurement.
    #
    # Why simpler approaches failed:
    #   - Host-side sleep (0.5-2.0 s): insufficient for 36-layer models
    #     where 36 nmc_done CDC handshakes must fully drain
    #   - RTL watchdog (100 ms timeout): correctly resets the FSM but
    #     can't fix the nmc_dispatch busy-flag race across process exits
    #   - Both combined: still produces ~50% failure rate on iter 3+
    #
    # A full re-flash resets ALL FFs (including nmc_dispatch.busy,
    # the CDC toggle registers, and MIG calibration state) to their
    # initial values.  The 6-second cost comes from the JTAG bitstream
    # load (~3.8 MB at 6 MHz) plus MIG DDR2 calibration (~2 s).
    flash_script = str(REPO_ROOT / "fpga" / "scripts" / "flash_fpga.sh")

    t0 = time.monotonic()
    with TegraStatsCollector(interval_ms=100) as col:
        for i in range(runs):
            # Re-flash FPGA to guarantee clean state for this iteration.
            try:
                subprocess.run(
                    [flash_script],
                    capture_output=True, text=True, timeout=60)
                time.sleep(1.0)  # let MIG calibrate
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log_lines.append(f"iter {i}: REFLASH FAILED: {e}")
                continue

            try:
                r = subprocess.run(
                    ["sudo", str(splitinfer_bin), str(manifest_path), "--real"],
                    capture_output=True, text=True, env=env, timeout=PER_CALL_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                log_lines.append(f"iter {i}: TIMEOUT (>{PER_CALL_TIMEOUT}s)")
                continue
            log_lines.append(r.stdout[-200:])  # tail of each iter
            for line in r.stdout.splitlines():
                if "Mean latency" in line or "Total latency" in line:
                    try:
                        ms = float(line.split(":")[1].strip().split()[0])
                        latencies.append(ms)
                    except (ValueError, IndexError):
                        pass
                    break
    wall_s = time.monotonic() - t0
    log_path.write_text("\n--- per-iter tails ---\n".join(log_lines))

    if not latencies:
        return {
            "status":     "no_latencies",
            "duration_s": wall_s,
            "power":      col.summary(),
        }

    stats = compute_stats(latencies)
    return {
        "status":     "ok",
        "stats":      stats,
        # Legacy top-level fields so older consumers keep working.
        "mean_ms":    stats.get("mean_ms"),
        "median_ms":  stats.get("median_ms"),
        "p99_ms":     stats.get("p99_ms"),
        "throughput": runs / max(wall_s, 1e-6),
        "duration_s": wall_s,
        "power":      col.summary(),
        "iterations": len(latencies),
    }


def _percentile(values, p):
    """Linear-interpolation percentile (matches numpy / C++ splitinfer_run)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def compute_stats(latencies_ms: List[float]) -> Dict:
    """Uniform statistics dict for any list of per-iteration latencies.

    Returns the same schema the C++ splitinfer_run STATS_JSON emits AND
    the same fields the upgraded B1/B2 baselines now print in their JSON
    summary, so downstream consumers see a consistent shape regardless
    of where the distribution came from.
    """
    if not latencies_ms:
        return {"count": 0}
    n = len(latencies_ms)
    mean_v = statistics.mean(latencies_ms)
    std_v  = statistics.stdev(latencies_ms) if n > 1 else 0.0
    return {
        "count":     n,
        "mean_ms":   round(mean_v, 4),
        "std_ms":    round(std_v,  4),
        "min_ms":    round(min(latencies_ms), 4),
        "p5_ms":     round(_percentile(latencies_ms,  5), 4),
        "median_ms": round(_percentile(latencies_ms, 50), 4),
        "p95_ms":    round(_percentile(latencies_ms, 95), 4),
        "p99_ms":    round(_percentile(latencies_ms, 99), 4),
        "max_ms":    round(max(latencies_ms), 4),
        "cv_pct":    round(100.0 * std_v / mean_v, 2) if mean_v > 0 else 0.0,
    }


# ─── Sweep driver ──────────────────────────────────────────────────────────────


def regenerate_dlrm_for_batch(batch: int) -> Path:
    """Generate a fresh dlrm.onnx with the requested batch size."""
    out_path = MODELS_DIR / f"dlrm_b{batch}.onnx"
    if out_path.exists():
        return out_path
    cmd = [
        "python3", str(REPO_ROOT / "evaluation" / "models" / "gen_dlrm.py"),
        str(out_path),
        "--batch", str(batch),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="E2: SplitInfer perf+power sweep")
    parser.add_argument("--models", nargs="+",
                        default=["dlrm", "yolov8n", "mobilebert"],
                        help="Models to include")
    parser.add_argument("--batches", type=int, nargs="+",
                        default=[1, 64, 256],
                        help="Batch sizes to sweep")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs",   type=int, default=30)
    parser.add_argument("--include-splitinfer", action="store_true", default=True)
    parser.add_argument("--no-splitinfer", dest="include_splitinfer",
                        action="store_false")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing summary if present so --append semantics work by default.
    # This prevents a targeted re-run (e.g. for one model) from clobbering
    # data from a previous sweep.  To force a fresh start, delete the
    # e2_summary.json file manually before running.
    summary_path = RESULTS_DIR / "e2_summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            prior_results = existing.get("results", [])
            # Drop any prior cells that overlap with the (model, batch) pairs
            # we are about to run — they'll be re-measured.
            sweep_pairs = {(m, b) for m in args.models for b in args.batches}
            kept = [r for r in prior_results
                    if (r.get("model"), r.get("batch")) not in sweep_pairs]
            print(f"  [merge] kept {len(kept)} prior cells, "
                  f"overwriting {len(prior_results) - len(kept)} overlapping ones")
        except Exception as e:
            print(f"  [merge] could not parse existing summary ({e}); starting fresh")
            kept = []
    else:
        kept = []

    summary = {
        "experiment": "E2_performance_comparison",
        "models":     args.models,
        "batches":    args.batches,
        "warmup":     args.warmup,
        "runs":       args.runs,
        "results":    list(kept),
    }

    print("=" * 64)
    print(f"E2 sweep: {len(args.models)} models × {len(args.batches)} batches")
    print(f"Output:    {RESULTS_DIR}")
    print("=" * 64)

    for model_name in args.models:
        for batch in args.batches:
            # For DLRM we regenerate the model with the requested batch size.
            if model_name == "dlrm" and batch != 1:
                model_path = regenerate_dlrm_for_batch(batch)
            else:
                cells = cells_for_model(model_name, [batch])
                cell = cells[0]
                if not cell["supports_batch"]:
                    print(f"\n[skip] {model_name} batch={batch} (model has fixed batch=1)")
                    summary["results"].append({
                        "model": model_name, "batch": batch,
                        "status": "skipped (fixed batch=1)",
                    })
                    continue
                model_path = Path(cell["model_path"])

            print(f"\n[{model_name} batch={batch}] model={model_path.name}")

            cell_for_run = {"model_name": model_name, "batch": batch,
                            "model_path": str(model_path)}
            cell_results = {"model": model_name, "batch": batch,
                             "model_size_mb": model_path.stat().st_size // (1024 * 1024)}

            # B1
            print(f"  → B1 FP32 ...", end=" ", flush=True)
            log = RESULTS_DIR / f"{model_name}_b{batch}_b1.log"
            r1 = run_baseline_b1(cell_for_run, args.warmup, args.runs, log)
            cell_results["b1_fp32"] = r1
            _print_result(r1)

            # B2
            print(f"  → B2 INT8 ...", end=" ", flush=True)
            log = RESULTS_DIR / f"{model_name}_b{batch}_b2.log"
            r2 = run_baseline_b2(cell_for_run, args.warmup, args.runs, log)
            cell_results["b2_int8"] = r2
            _print_result(r2)

            # SplitInfer
            if args.include_splitinfer:
                print(f"  → SplitInfer (--real, prefetch=ON) ...", end=" ", flush=True)
                log = RESULTS_DIR / f"{model_name}_b{batch}_splitinfer.log"
                rs = run_splitinfer(cell_for_run, args.warmup, args.runs, log,
                                    prefetch=True)
                cell_results["splitinfer"] = rs
                _print_result(rs)

            summary["results"].append(cell_results)

            # Save partial summary after every cell so we don't lose work
            (RESULTS_DIR / "e2_summary.json").write_text(
                json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 64)
    print(f"E2 complete.  Summary written to {RESULTS_DIR / 'e2_summary.json'}")
    print("=" * 64)


def _print_result(r: Dict) -> None:
    if r["status"] == "ok":
        stats  = r.get("stats") or {}
        median = stats.get("median_ms") or r.get("median_ms") or 0.0
        std    = stats.get("std_ms")    or 0.0
        p95    = stats.get("p95_ms")    or 0.0
        cv     = stats.get("cv_pct")    or 0.0
        thr    = r.get("throughput") or 0.0
        pwr    = r.get("power", {}).get("rails", {}).get("VDD_IN", {}).get("mean_mw", 0)
        nrg    = r.get("power", {}).get("energy_J", 0)
        print(f"median={median:.2f}±{std:.2f}ms p95={p95:.2f}ms "
              f"cv={cv:.1f}%  thr={thr:.1f}/s  pwr={pwr:.0f}mW  nrg={nrg:.1f}J")
    else:
        print(f"[{r['status']}]")


if __name__ == "__main__":
    main()
