#!/usr/bin/env bash
# evaluation/experiments/e1_capability_unlock.sh
#
# Experiment E1: Capability Unlock
#
# Demonstrates that the DLRM model cannot run on Jetson alone (OOM) but can be
# split across Jetson GPU + FPGA using SplitInfer.
#
# Steps:
#   1. Run B1 (FP32) baseline on DLRM — expected: OOM or very high latency
#   2. Run B2 (INT8) baseline on DLRM — expected: OOM or very high latency
#   3. Partition DLRM with SplitInfer partitioner
#   4. Run split inference via splitinfer_run — expected: success
#
# Hardware required: Jetson Orin Nano + Nexys 4 DDR (Artix-7 FPGA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS_DIR="${REPO_ROOT}/evaluation/models"
BASELINES_DIR="${REPO_ROOT}/evaluation/baselines"
RESULTS_DIR="${SCRIPT_DIR}/results/e1"
mkdir -p "${RESULTS_DIR}"

DLRM_MODEL="${MODELS_DIR}/dlrm_synthetic.onnx"
MANIFEST_OUT="${RESULTS_DIR}/dlrm_manifest.json"

WARMUP=10
RUNS=100

echo "============================================================"
echo "E1: Capability Unlock — DLRM on Jetson vs. SplitInfer"
echo "============================================================"
echo "Results dir: ${RESULTS_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Ensure DLRM model exists
# ---------------------------------------------------------------------------
if [ ! -f "${DLRM_MODEL}" ]; then
    echo "[0/4] Generating DLRM model..."
    python3 "${MODELS_DIR}/gen_dlrm.py" "${DLRM_MODEL}"
else
    echo "[0/4] DLRM model already exists: ${DLRM_MODEL}"
fi

# ---------------------------------------------------------------------------
# Step 1: B1 baseline (FP32)
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Running B1 (FP32) baseline on DLRM..."
echo "      NOTE: This may OOM on Jetson with limited GPU memory."
B1_LOG="${RESULTS_DIR}/b1_fp32.log"

if python3 "${BASELINES_DIR}/jetson_only_fp.py" \
        "${DLRM_MODEL}" \
        --warmup "${WARMUP}" \
        --runs "${RUNS}" \
        2>&1 | tee "${B1_LOG}"; then
    echo "[1/4] B1 completed (model fits in memory — check latency)."
else
    echo "[1/4] B1 FAILED (likely OOM — this is expected for large DLRM)."
    echo "      EXPECTED: OOM or inference failure without FPGA offloading."
fi

# ---------------------------------------------------------------------------
# Step 2: B2 baseline (INT8)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Running B2 (INT8) baseline on DLRM..."
B2_LOG="${RESULTS_DIR}/b2_int8.log"

if python3 "${BASELINES_DIR}/jetson_only_quant.py" \
        "${DLRM_MODEL}" \
        --warmup "${WARMUP}" \
        --runs "${RUNS}" \
        --dynamic \
        2>&1 | tee "${B2_LOG}"; then
    echo "[2/4] B2 completed (model fits in memory — check latency)."
else
    echo "[2/4] B2 FAILED (likely OOM — this is expected for large DLRM)."
    echo "      EXPECTED: OOM or inference failure without FPGA offloading."
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
    if "${SPLITINFER_BIN}" "${MANIFEST_OUT}" 2>&1 | tee "${RUNTIME_LOG}"; then
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
