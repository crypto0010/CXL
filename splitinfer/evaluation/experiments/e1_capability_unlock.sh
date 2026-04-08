#!/usr/bin/env bash
# evaluation/experiments/e1_capability_unlock.sh
#
# Experiment E1: Capability Unlock
#
# Demonstrates that a SCALED DLRM model (~1.3 GB of embedding tables)
# cannot run efficiently on Jetson alone (high latency or OOM at larger
# scales) but can be split across Jetson GPU + FPGA using SplitInfer.
#
# Model: 26 embedding tables x 200,000 rows x 64-dim FP32 = ~1.3 GB.
# This is large enough that loading the full model into onnxruntime
# strains the Jetson Orin Nano's 8 GB shared LPDDR5, while each
# individual table (~51 MB) still fits comfortably in the Nexys 4
# DDR's 128 MB DDR2 bank — exactly the partitioning sweet spot.
#
# Steps:
#   1. Run B1 (FP32) baseline   — measures Jetson-only latency / OOM
#   2. Run B2 (INT8) baseline   — measures quantized Jetson-only
#   3. Partition with SplitInfer partitioner
#   4. Run split inference via splitinfer_run --real (real FPGA)
#
# Hardware required: Jetson Orin Nano + Nexys 4 DDR (Artix-7 FPGA)
#                     connected via USB, programmed with SplitInfer bitstream.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS_DIR="${REPO_ROOT}/evaluation/models"
BASELINES_DIR="${REPO_ROOT}/evaluation/baselines"
RESULTS_DIR="${SCRIPT_DIR}/results/e1"
mkdir -p "${RESULTS_DIR}"

# E1 uses the SCALED DLRM (~1.3 GB).  Generate via:
#   python3 evaluation/models/gen_dlrm.py <out> --rows 200000
DLRM_MODEL="${MODELS_DIR}/generated/dlrm_scaled.onnx"
MANIFEST_OUT="${RESULTS_DIR}/dlrm_manifest.json"

# Reduced run counts for the scaled model to keep wall time reasonable.
# At ~600 MB GPU footprint per ORT run, even 50 iterations is plenty
# to characterize median latency.
WARMUP=5
RUNS=50

echo "============================================================"
echo "E1: Capability Unlock — DLRM on Jetson vs. SplitInfer"
echo "============================================================"
echo "Results dir: ${RESULTS_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Ensure scaled DLRM model exists
# ---------------------------------------------------------------------------
if [ ! -f "${DLRM_MODEL}" ]; then
    echo "[0/4] Generating SCALED DLRM model (26 x 200000 x 64 ≈ 1.3 GB)..."
    mkdir -p "$(dirname "${DLRM_MODEL}")"
    python3 "${MODELS_DIR}/gen_dlrm.py" "${DLRM_MODEL}" --rows 200000
else
    echo "[0/4] DLRM model already exists: ${DLRM_MODEL}"
fi
DLRM_SIZE_MB=$(du -m "${DLRM_MODEL}" | cut -f1)
echo "      Model size: ${DLRM_SIZE_MB} MB"

# ---------------------------------------------------------------------------
# Step 1: B1 baseline (FP32)
# Bounded to 5 minutes — if onnxruntime is still trying to load/run after
# that, we declare it as "did not complete in reasonable time" which counts
# as a capability failure for E1's purposes.
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Running B1 (FP32) baseline on DLRM (timeout: 5 min)..."
echo "      NOTE: ${DLRM_SIZE_MB} MB model may OOM or thrash on the 8 GB Jetson."
B1_LOG="${RESULTS_DIR}/b1_fp32.log"

if timeout 300 python3 "${BASELINES_DIR}/jetson_only_fp.py" \
        "${DLRM_MODEL}" \
        --warmup "${WARMUP}" \
        --runs "${RUNS}" \
        2>&1 | tee "${B1_LOG}"; then
    echo "[1/4] B1 completed (model fits — check reported latency in log)."
else
    rc=$?
    if [ $rc -eq 124 ]; then
        echo "[1/4] B1 TIMED OUT after 5 min (capability failure as expected)."
    else
        echo "[1/4] B1 FAILED with exit $rc (likely OOM — expected for large DLRM)."
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: B2 baseline (INT8)
# Note: dynamic quantization needs to load the entire FP32 model into memory
# during the quantize_dynamic() call, which itself can OOM before any
# inference happens.  Same 5-min ceiling.
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Running B2 (INT8) baseline on DLRM (timeout: 5 min)..."
B2_LOG="${RESULTS_DIR}/b2_int8.log"

if timeout 300 python3 "${BASELINES_DIR}/jetson_only_quant.py" \
        "${DLRM_MODEL}" \
        --warmup "${WARMUP}" \
        --runs "${RUNS}" \
        --dynamic \
        2>&1 | tee "${B2_LOG}"; then
    echo "[2/4] B2 completed (model fits — check reported latency in log)."
else
    rc=$?
    if [ $rc -eq 124 ]; then
        echo "[2/4] B2 TIMED OUT after 5 min (capability failure as expected)."
    else
        echo "[2/4] B2 FAILED with exit $rc (likely OOM during quantization)."
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Partition with SplitInfer
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Partitioning DLRM with SplitInfer..."
PARTITION_LOG="${RESULTS_DIR}/partition.log"

if python3 -m partitioner.cli \
        "${DLRM_MODEL}" \
        --output "${MANIFEST_OUT}" \
        --fpga-ddr2-mb 128 \
        --gpu-gflops 1000 \
        --fpga-int8-gops 6.4 \
        --usb-bw-mbps 40 \
        2>&1 | tee "${PARTITION_LOG}"; then
    echo "[3/4] Partition manifest written: ${MANIFEST_OUT}"
else
    echo "[3/4] Partition failed — check that splitinfer Python package is installed."
    echo "      Run: cd ${REPO_ROOT} && pip install -e ."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 4: Run SplitInfer runtime with manifest
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Running split inference via splitinfer_run..."
RUNTIME_LOG="${RESULTS_DIR}/runtime.log"

SPLITINFER_BIN="${REPO_ROOT}/build/runtime/splitinfer_run"
if [ ! -x "${SPLITINFER_BIN}" ]; then
    echo "  WARNING: splitinfer_run not found at ${SPLITINFER_BIN}"
    echo "  Build first: cd ${REPO_ROOT}/build && cmake .. -DBUILD_TESTS=ON && make -j4"
    echo "  Skipping runtime step."
else
    # --real engages the FpgaExecutor (requires FPGA connected via USB).
    # Without --real, splitinfer_run uses stub executors and the result
    # would not be a real comparison.
    if sudo "${SPLITINFER_BIN}" "${MANIFEST_OUT}" --real 2>&1 | tee "${RUNTIME_LOG}"; then
        echo "[4/4] SplitInfer runtime completed successfully."
        echo "      RESULT: DLRM inference succeeded with GPU+FPGA split!"
    else
        echo "[4/4] Runtime failed — ensure FPGA is connected and programmed."
    fi
fi

echo ""
echo "============================================================"
echo "E1 Complete. Results saved in: ${RESULTS_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 5: Aggregate results into a single JSON file for downstream analysis
# ---------------------------------------------------------------------------
SUMMARY_JSON="${RESULTS_DIR}/e1_summary.json"

python3 - "${RESULTS_DIR}" "${DLRM_SIZE_MB}" "${SUMMARY_JSON}" << 'PYEOF'
import json, os, re, sys

results_dir, dlrm_size_mb, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def parse_baseline_log(path, baseline_name):
    """Extract latency / status from a B1 / B2 log file."""
    if not os.path.exists(path):
        return {"status": "missing", "log": path}
    txt = open(path).read()
    if "Mean latency" in txt:
        m = re.search(r"Mean latency\s*[:=]\s*([0-9.]+)\s*ms", txt)
        return {
            "status": "completed",
            "mean_latency_ms": float(m.group(1)) if m else None,
        }
    if "timed out" in txt.lower() or "out of memory" in txt.lower():
        return {"status": "oom_or_timeout"}
    return {"status": "unknown_failure"}

def parse_runtime_log(path):
    if not os.path.exists(path):
        return {"status": "missing"}
    txt = open(path).read()
    if "Status: OK" in txt:
        m = re.search(r"Total latency\s*:\s*([0-9.]+)\s*ms", txt)
        return {
            "status": "completed",
            "total_latency_ms": float(m.group(1)) if m else None,
        }
    return {"status": "failed"}

summary = {
    "experiment": "E1_capability_unlock",
    "model": "scaled DLRM",
    "model_size_mb": dlrm_size_mb,
    "results": {
        "B1_fp32_jetson": parse_baseline_log(
            os.path.join(results_dir, "b1_fp32.log"), "B1"),
        "B2_int8_jetson": parse_baseline_log(
            os.path.join(results_dir, "b2_int8.log"), "B2"),
        "splitinfer_real": parse_runtime_log(
            os.path.join(results_dir, "runtime.log")),
    },
}

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nE1 summary written to {out_path}")
print(json.dumps(summary, indent=2))
PYEOF
