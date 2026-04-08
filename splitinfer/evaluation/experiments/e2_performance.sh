#!/usr/bin/env bash
# evaluation/experiments/e2_performance.sh
#
# Experiment E2: Performance Comparison
#
# Measures latency, throughput, and peak memory for each model across all
# baselines (B1-B4) and SplitInfer.
#
# Models: DLRM, MobileBERT, YOLOv8-nano
# Baselines: B1 (FP32), B2 (INT8), B3 (CPU offload), B4 (disk swap)
#
# Hardware required: Jetson Orin Nano + Nexys 4 DDR (Artix-7 FPGA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS_DIR="${REPO_ROOT}/evaluation/models"
BASELINES_DIR="${REPO_ROOT}/evaluation/baselines"
RESULTS_DIR="${SCRIPT_DIR}/results/e2"
mkdir -p "${RESULTS_DIR}"

SPLITINFER_BIN="${REPO_ROOT}/build/runtime/splitinfer_run"

WARMUP=50
RUNS=1000

# Models to evaluate
declare -A MODEL_PATHS=(
    ["dlrm"]="${MODELS_DIR}/dlrm_synthetic.onnx"
    ["mobilebert"]="${MODELS_DIR}/mobilebert.onnx"
    ["yolov8n"]="${MODELS_DIR}/yolov8n.onnx"
)

declare -A MODEL_GENERATORS=(
    ["dlrm"]="${MODELS_DIR}/gen_dlrm.py"
    ["mobilebert"]="${MODELS_DIR}/gen_mobilebert.py"
    ["yolov8n"]="${MODELS_DIR}/gen_yolov8.py"
)

echo "============================================================"
echo "E2: Performance Comparison — All Models x All Baselines"
echo "============================================================"
echo "Results dir : ${RESULTS_DIR}"
echo "Warmup      : ${WARMUP}"
echo "Runs        : ${RUNS}"
echo ""

# ---------------------------------------------------------------------------
# Helper: capture peak GPU memory from tegra_stats or nvidia-smi
# ---------------------------------------------------------------------------
capture_peak_memory() {
    local log_file="$1"
    # Try nvidia-smi first, then tegra sysfs
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
            >> "${log_file}" || true
    elif [ -f /sys/devices/platform/host1x/tegradc.0/smartdimmer/sw_settings ]; then
        cat /sys/kernel/debug/nvmap/iovmm/allocations 2>/dev/null \
            >> "${log_file}" || true
    fi
}

# ---------------------------------------------------------------------------
# Helper: run a single baseline and capture results
# ---------------------------------------------------------------------------
run_baseline() {
    local model_name="$1"
    local baseline_tag="$2"
    local baseline_script="$3"
    local model_path="$4"
    shift 4
    local extra_args=("$@")

    local log_file="${RESULTS_DIR}/${model_name}_${baseline_tag}.log"
    local json_file="${RESULTS_DIR}/${model_name}_${baseline_tag}.json"

    echo "  Running ${baseline_tag} on ${model_name}..."

    if python3 "${baseline_script}" \
            "${model_path}" \
            --warmup "${WARMUP}" \
            --runs "${RUNS}" \
            "${extra_args[@]}" \
            2>&1 | tee "${log_file}"; then
        echo "  ${baseline_tag} completed."
    else
        echo "  ${baseline_tag} FAILED (OOM or unsupported — this may be expected)."
    fi

    capture_peak_memory "${log_file}"

    # Write a minimal JSON result (scripts are expected to emit timing to stdout)
    python3 -c "
import json, re, sys

log = open('${log_file}').read()
result = {
    'model': '${model_name}',
    'baseline': '${baseline_tag}',
    'status': 'ok' if 'latency' in log.lower() or 'throughput' in log.lower() else 'failed',
}
# Try to extract mean latency (ms) from log
m = re.search(r'mean[_ ]latency[:\s]*([\d.]+)\s*ms', log, re.IGNORECASE)
if m:
    result['mean_latency_ms'] = float(m.group(1))
m = re.search(r'throughput[:\s]*([\d.]+)', log, re.IGNORECASE)
if m:
    result['throughput_ips'] = float(m.group(1))
m = re.search(r'peak[_ ]memory[:\s]*([\d.]+)\s*MB', log, re.IGNORECASE)
if m:
    result['peak_memory_mb'] = float(m.group(1))
json.dump(result, open('${json_file}', 'w'), indent=2)
" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Step 0: Ensure all models exist
# ---------------------------------------------------------------------------
echo "[0] Ensuring models exist..."
for model_name in dlrm mobilebert yolov8n; do
    model_path="${MODEL_PATHS[${model_name}]}"
    gen_script="${MODEL_GENERATORS[${model_name}]}"
    if [ ! -f "${model_path}" ]; then
        echo "  Generating ${model_name}..."
        python3 "${gen_script}" "${model_path}" || true
    else
        echo "  ${model_name} already exists: ${model_path}"
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Check splitinfer_run binary
# ---------------------------------------------------------------------------
echo ""
echo "[1] Checking splitinfer_run binary..."
if [ ! -x "${SPLITINFER_BIN}" ]; then
    echo "  WARNING: splitinfer_run not found at ${SPLITINFER_BIN}"
    echo "  Build first: cd ${REPO_ROOT}/build && cmake .. -DBUILD_TESTS=ON && make -j4"
    echo "  SplitInfer runs will be skipped."
    SPLITINFER_BIN=""
fi

# ---------------------------------------------------------------------------
# Step 2: Run all baselines and SplitInfer for each model
# ---------------------------------------------------------------------------
for model_name in dlrm mobilebert yolov8n; do
    model_path="${MODEL_PATHS[${model_name}]}"

    echo ""
    echo "============================================================"
    echo "Model: ${model_name}"
    echo "============================================================"

    if [ ! -f "${model_path}" ]; then
        echo "  SKIP: model file not found: ${model_path}"
        continue
    fi

    # --- B1: FP32 baseline ---
    run_baseline "${model_name}" "b1_fp32" \
        "${BASELINES_DIR}/jetson_only_fp.py" "${model_path}" || true

    # --- B2: INT8 baseline ---
    run_baseline "${model_name}" "b2_int8" \
        "${BASELINES_DIR}/jetson_only_quant.py" "${model_path}" --dynamic || true

    # --- B3: CPU offload baseline ---
    run_baseline "${model_name}" "b3_cpu_offload" \
        "${BASELINES_DIR}/jetson_cpu_offload.py" "${model_path}" || true

    # --- B4: Disk swap baseline ---
    run_baseline "${model_name}" "b4_disk_swap" \
        "${BASELINES_DIR}/jetson_disk_swap.py" "${model_path}" || true

    # --- SplitInfer ---
    echo "  Partitioning ${model_name} with SplitInfer..."
    MANIFEST="${RESULTS_DIR}/${model_name}_manifest.json"
    PARTITION_LOG="${RESULTS_DIR}/${model_name}_partition.log"
    RUNTIME_LOG="${RESULTS_DIR}/${model_name}_splitinfer.log"
    RUNTIME_JSON="${RESULTS_DIR}/${model_name}_splitinfer.json"

    if python3 -m partitioner.cli \
            "${model_path}" \
            --output "${MANIFEST}" \
            --fpga-ddr2-mb 128 \
            --gpu-gflops 1000 \
            --fpga-int8-gops 6.4 \
            --usb-bw-mbps 40 \
            2>&1 | tee "${PARTITION_LOG}"; then
        echo "  Partition manifest: ${MANIFEST}"
    else
        echo "  Partitioning FAILED for ${model_name}."
        continue
    fi

    if [ -n "${SPLITINFER_BIN}" ] && [ -x "${SPLITINFER_BIN}" ] && [ -f "${MANIFEST}" ]; then
        echo "  Running SplitInfer runtime on ${model_name}..."
        if "${SPLITINFER_BIN}" "${MANIFEST}" 2>&1 | tee "${RUNTIME_LOG}"; then
            echo "  SplitInfer completed for ${model_name}."
        else
            echo "  SplitInfer runtime FAILED for ${model_name}."
        fi
        capture_peak_memory "${RUNTIME_LOG}"

        # Extract SplitInfer results to JSON
        python3 -c "
import json, re
log = open('${RUNTIME_LOG}').read()
result = {
    'model': '${model_name}',
    'baseline': 'splitinfer',
    'status': 'ok' if 'latency' in log.lower() or 'complete' in log.lower() else 'failed',
}
m = re.search(r'mean[_ ]latency[:\s]*([\d.]+)\s*ms', log, re.IGNORECASE)
if m:
    result['mean_latency_ms'] = float(m.group(1))
m = re.search(r'throughput[:\s]*([\d.]+)', log, re.IGNORECASE)
if m:
    result['throughput_ips'] = float(m.group(1))
m = re.search(r'peak[_ ]memory[:\s]*([\d.]+)\s*MB', log, re.IGNORECASE)
if m:
    result['peak_memory_mb'] = float(m.group(1))
json.dump(result, open('${RUNTIME_JSON}', 'w'), indent=2)
" 2>/dev/null || true
    else
        echo "  [SKIP] splitinfer_run not available — skipping runtime for ${model_name}."
    fi
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "E2: Summary Table"
echo "============================================================"
echo ""
printf "%-12s %-15s %-10s %-15s %-15s %-12s\n" \
    "Model" "Config" "Status" "Latency(ms)" "Throughput" "Memory(MB)"
printf "%-12s %-15s %-10s %-15s %-15s %-12s\n" \
    "--------" "-----------" "------" "-----------" "-----------" "----------"

for json_file in "${RESULTS_DIR}"/*.json; do
    [ -f "${json_file}" ] || continue
    python3 -c "
import json, sys
d = json.load(open('${json_file}'))
lat = str(d.get('mean_latency_ms', '-'))
thr = str(d.get('throughput_ips', '-'))
mem = str(d.get('peak_memory_mb', '-'))
print(f\"{d['model']:<12s} {d['baseline']:<15s} {d['status']:<10s} {lat:<15s} {thr:<15s} {mem:<12s}\")
" 2>/dev/null || true
done

echo ""
echo "============================================================"
echo "E2 Complete. Per-model JSON results in: ${RESULTS_DIR}"
echo "============================================================"
