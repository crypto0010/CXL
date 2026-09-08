#!/usr/bin/env python3
"""Regenerate every data figure in the paper from the JSON artifacts."""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "evaluation", "experiments", "results")
P = os.path.join(ROOT, "paper")
os.makedirs(P, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"nmc": "#1f77b4", "pool1": "#d62728", "pool64": "#ff7f0e", "pool512": "#9467bd", "warm": "#2ca02c", "host": "#7f7f7f"}


def load(p):
    try: return json.load(open(p))
    except Exception: return None


def fig_crossover():
    d = load(os.path.join(R, "e6", "dlrm_crossover_emul.json"))
    if not d: return
    pts = d["points"]; bw = [p["bytes_per_s"] for p in pts]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    def series(k, label, color, marker):
        y = [p[k]["stats"]["median_ms"] if p.get(k, {}).get("status") == "ok" else math.nan for p in pts]
        ax.plot(bw, y, marker=marker, color=color, label=label, lw=1.2, ms=4)
    series("nmc", "NMC (compute $\\rightarrow$ memory)", C["nmc"], "o")
    series("pool_pf1", "Pool, demand paging", C["pool1"], "s")
    series("pool_pf64", "Pool, 64-page prefetch", C["pool64"], "^")
    series("pool_warm", "Pool, warm (host-cached)", C["warm"], "v")
    host = load(os.path.join(R, "e2", "e2_v2_dlrm_emul_uart.json"))
    if host and host["cells"].get("SI_host", {}).get("status") == "ok":
        h = host["cells"]["SI_host"]["result"]["stats"]["median_ms"]
        ax.axhline(h, color=C["host"], ls="--", lw=1, label=f"Host-resident ({h:.1f} ms)")
    # measured points from the board at the calibrated UART bandwidth
    cal = load(os.path.join(ROOT, "evaluation", "calibration", "uart_measured.json")) or {}
    bw_m = (cal.get("fit") or {}).get("link_bw_bytes_per_s")
    usb = load(os.path.join(R, "e2", "e2_v2_dlrm_usb.json")) or {}
    nmc_m = ((usb.get("cells", {}).get("SI_nmc") or {}).get("result") or {}).get("stats", {}).get("median_ms")
    import glob, re
    pool_m = None
    logs = sorted(glob.glob(os.path.join(R, "board_session_*.log")))
    if logs:
        txt = open(logs[-1], errors="replace").read(); a = txt.find("===== pool_exact ====="); b = txt.find("===== calibrate =====")
        mm = re.search(r"median ([\d.]+)", txt[a:b]) if a >= 0 and b > a else None
        if mm and "exit 0" in txt[a:b]: pool_m = float(mm.group(1))
    if bw_m and nmc_m: ax.plot([bw_m], [nmc_m], marker="*", ms=11, color=C["nmc"], mec="black", ls="none", label="NMC, measured on board")
    if bw_m and pool_m: ax.plot([bw_m], [pool_m], marker="*", ms=11, color=C["pool64"], mec="black", ls="none", label="Pool (64-pg), measured on board")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Link bandwidth (bytes/s)"); ax.set_ylabel("Latency per inference (ms)")
    ax.set_xticks(bw); ax.set_xticklabels([p["link"].replace("-", "\n") for p in pts], fontsize=5.5)
    ax.minorticks_off()
    ax.legend(fontsize=5.5, loc="upper right")
    ax.set_title("DLRM: emulated-link sweep (E6) with board measurements", fontsize=8)
    fig.savefig(os.path.join(P, "fig_crossover.pdf")); plt.close(fig)


def fig_e2():
    d = load(os.path.join(R, "e2", "e2_v2_dlrm_emul_uart.json"))
    if not d: return
    cells = d["cells"]
    order = [("B1_ort_fp32_cpu", "ORT FP32\nCPU"), ("B2_ort_int8_cpu", "ORT INT8\nCPU"), ("B3_trt_fp16_gpu", "TRT FP16\nGPU"),
             ("B4_trt_int8_gpu", "TRT INT8\nGPU"), ("SI_host", "SI host\nINT8"), ("SI_nmc", "SI NMC\n(UART)"), ("SI_pool", "SI pool\n(UART)")]
    names, med, p99, col = [], [], [], []
    for k, lab in order:
        c = cells.get(k) or {}
        if c.get("status") != "ok": continue
        st = c["result"].get("stats") or c["result"]
        names.append(lab); med.append(st["median_ms"]); p99.append(st["p99_ms"])
        col.append(C["nmc"] if k == "SI_nmc" else C["pool1"] if k == "SI_pool" else C["host"] if k == "SI_host" else "#bbbbbb")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = range(len(names))
    ax.bar(x, med, color=col, edgecolor="black", lw=0.5)
    ax.errorbar(x, med, yerr=[[0] * len(med), [b - a for a, b in zip(med, p99)]], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=5.5, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms), median + p99")
    for i, v in enumerate(med): ax.text(i, v * 1.15, f"{v:.3g}", ha="center", fontsize=6)
    ax.set_title("DLRM batch 1: every execution path (E2)", fontsize=8)
    fig.savefig(os.path.join(P, "fig_e2.pdf")); plt.close(fig)


def fig_layer_crossover():
    d = load(os.path.join(ROOT, "evaluation", "lowered", "dlrm_small_manifest.json"))
    if not d: return
    xs, ys, labels = [], [], []
    for dec in d["decisions"]:
        ai = dec["arithmetic_intensity"]; b = dec["link_crossover_bytes_per_s"]
        if ai in (None, "inf") or b in (None, "inf") or not isinstance(b, (int, float)): continue
        xs.append(dec["cost_ms"] and next(l["weight_bytes"] for l in d["layers"] if l["name"] == dec["name"])); ys.append(b); labels.append(dec["name"])
    if not xs: return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.scatter(xs, ys, s=14, color=C["nmc"])
    for x, y, l in zip(xs, ys, labels):
        if l.startswith("mlp"): ax.annotate(l, (x, y), fontsize=5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Layer weight bytes"); ax.set_ylabel("Crossover link bandwidth (B/s)")
    ax.set_title("Per-layer pool/NMC crossover (cost model)", fontsize=8)
    fig.savefig(os.path.join(P, "fig_layer_crossover.pdf")); plt.close(fig)


def fig_capacity():
    cap = load(os.path.join(R, "e1", "capacity_sweep", "capacity_unlock_summary.json"))
    if not cap: return
    ms = cap["baseline_fp32_measurements"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for m in ms:
        if m["b1_status"] == "oom_killed":
            ax.axvline(m["model_size_gb"], color="#d62728", ls=":", lw=1); ax.text(m["model_size_gb"], 500, "OOM-killed", rotation=90, fontsize=6, color="#d62728", va="bottom", ha="right")
        else:
            ax.plot(m["model_size_gb"], m["b1_inference_ms"], "o", color=C["host"])
            ax.annotate(f'{m["b1_inference_ms"]:.0f} ms\n{m["b1_peak_rss_mb"]} MB RSS', (m["model_size_gb"], m["b1_inference_ms"]), fontsize=5, xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel("Dense MLP size (GB, FP32)"); ax.set_ylabel("Jetson-only latency (ms)"); ax.set_yscale("log")
    ax.set_title("Host memory wall (measured, v1 E1)", fontsize=8)
    fig.savefig(os.path.join(P, "fig_capacity.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_crossover(); fig_e2(); fig_layer_crossover(); fig_capacity()
    print("figures:", [f for f in os.listdir(P) if f.endswith(".pdf")])
