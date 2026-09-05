#!/usr/bin/env python3
"""E6 (EMULATED): pool-vs-NMC latency across link bandwidth / RTT.

Runs splitinfer_v2 on the virtual FPGA with an emulated link at each grid
point and records full distributions.  This is the crossover figure the
cost model predicts; the board run at the UART point is the MEASURED
anchor.  Results are labelled "emulated": true.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "build", "runtime", "v2", "splitinfer_v2")

GRID = [  # (name, bytes/s, rtt_ms)
    ("uart-115200",   11_520.0,       5.0),
    ("uart-921600",   92_160.0,       1.0),
    ("usb2-bulk",     30e6,           0.5),
    ("usb3",          300e6,          0.1),
    ("pcie3x1",       800e6,          0.01),
    ("pcie4x4",       6e9,            0.005),
]

def run(prog, mode, bw, rtt, iters, prefetch=1, warm=False):
    cmd = [BIN, prog, "--mode", mode, "--transport", "emul", "--iterations", str(iters),
           "--warmup", "1", "--json", "--emul-link-bw", str(bw), "--emul-rtt-ms", str(rtt),
           "--emul-clock-hz", "81.25e6", "--prefetch-pages", str(prefetch)]
    if warm: cmd.append("--pool-warm")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 3):
        return {"status": "failed", "rc": r.returncode, "stderr": r.stderr[-400:]}
    d = json.loads(r.stdout.strip().splitlines()[-1]); d["status"] = "ok"; d["wall_s"] = time.time() - t0
    return d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("program_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    grid = GRID[2:] if a.quick else GRID
    res = {"experiment": "E6_crossover", "emulated": True, "program": a.program_dir, "points": []}
    for name, bw, rtt in grid:
        # slow links: fewer iterations so the sweep finishes
        iters = a.iters if bw >= 1e6 else 1
        pt = {"link": name, "bytes_per_s": bw, "rtt_ms": rtt}
        pt["nmc"] = run(a.program_dir, "nmc", bw, rtt, iters)
        for pf in (1, 64, 512):
            pt[f"pool_pf{pf}"] = run(a.program_dir, "pool", bw, rtt, iters, prefetch=pf)
        pt["pool_warm"] = run(a.program_dir, "pool", bw, rtt, iters, prefetch=64, warm=True)
        res["points"].append(pt)
        n = pt["nmc"]; p1 = pt["pool_pf1"]; p64 = pt["pool_pf64"]
        print(f"{name:14s} nmc {n.get('stats',{}).get('median_ms',float('nan')):10.1f} ms | pool pf1 {p1.get('stats',{}).get('median_ms',float('nan')):10.1f} | pf64 {p64.get('stats',{}).get('median_ms',float('nan')):10.1f} | warm {pt['pool_warm'].get('stats',{}).get('median_ms',float('nan')):8.2f}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f: json.dump(res, f, indent=1)
    print("written", a.out)

if __name__ == "__main__":
    main()
