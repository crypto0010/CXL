#!/usr/bin/env bash
# evaluation/experiments/e4_ablation.sh
#
# Experiment E4: Ablation Study
#
# Measures the contribution of each SplitInfer component by selectively
# disabling features and comparing latency/throughput.
#
# Configurations tested (on DLRM model):
#   1. Full SplitInfer       — all features enabled
#   2. No prefetch           — SPLITINFER_NO_PREFETCH=1
#   3. No pipelining         — SPLITINFER_NO_PIPELINE=1
#   4. No NMC                — SPLITINFER_NO_NMC=1
#   5. GPU-only quantized    — B2 (INT8) baseline for reference
#
# Hardware required: Jetson Orin Nano + Nexys 4 DDR (Artix-7 FPGA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS_DIR="${REPO_ROOT}/evaluation/models"
BASELINES_DIR="${REPO_ROOT}/evaluation/baselines"
RESULTS_DIR="${SCRIPT_DIR}/results/e4"
mkdir -p "${RESULTS_DIR}"

SPLITINFER_BIN="${REPO_ROOT}/build/runtime/splitinfer_run"
DLRM_MODEL="${MODELS_DIR}/dlrm_synthetic.onnx"
MANIFEST="${RESULTS_DIR}/dlrm_manifest.json"

WARMUP=50
RUNS=1000

echo "============================================================"
echo "E4: Ablation Study — Component Contributions"
echo "============================================================"
echo "Results dir : ${RESULTS_DIR}"
echo "Warmup      : ${WARMUP}"
echo "Runs        : ${RUNS}"
echo ""

# ---------------------------------------------------------------------------
# Helper: run splitinfer_run with given env vars, capture results
# ---------------------------------------------------------------------------
run_splitinfer_config() {
    local config_name="$1"
    shift
    local env_vars=("$@")

    local log_file="${RESULTS_DIR}/${config_name}.log"
    local json_file="${RESULTS_DIR}/${config_name}.json"

    echo "  Running config: ${config_name}..."

    if env "${env_vars[@]}" "${SPLITINFER_BIN}" "${MANIFEST}" \
            2>&1 | tee "${log_file}"; then
        echo "  ${config_name} completed."
    else
        echo "  ${config_name} FAILED."
    fi

    # Extract results to JSON
    python3 -c "
import json, re
log = open('${log_file}').read()
result = {
    'config': '${config_name}',
    'status': 'ok' if 'latency' in log.lower() or 'complete' in log.lower() else 'failed',
}
m = re.search(r'mean[_ ]latency[:\s]*([\d.]+)\s*ms', log, re.IGNORECASE)
if m:
    result['mean_latency_ms'] = float(m.group(1))
m = re.search(r'p50[_ ]latency[:\s]*([\d.]+)\s*ms', log, re.IGNORECASE)
if m:
    result['p50_latency_ms'] = float(m.group(1))
m = re.search(r'p99[_ ]latency[:\s]*([\d.]+)\s*ms', log, re.IGNORECASE)
if m:
    result['p99_latency_ms'] = float(m.group(1))
m = re.search(r'throughput[:\s]*([\d.]+)', log, re.IGNORECASE)
if m:
    result['throughput_ips'] = float(m.group(1))
json.dump(result, open('${json_file}', 'w'), indent=2)
" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Step 0: Ensure DLRM model exists
# ---------------------------------------------------------------------------
echo "[0/6] Ensuring DLRM model exists..."
if [ ! -f "${DLRM_MODEL}" ]; then
    echo "  Generating DLRM model..."
    python3 "${MODELS_DIR}/gen_dlrm.py" "${DLRM_MODEL}"
else
    echo "  DLRM model already exists: ${DLRM_MODEL}"
fi

# ---------------------------------------------------------------------------
# Step 1: Check splitinfer_run binary
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Checking splitinfer_run binary..."
if [ ! -x "${SPLITINFER_BIN}" ]; then
    echo "  ERROR: splitinfer_run not found at ${SPLITINFER_BIN}"
    echo "  Build first: cd ${REPO_ROOT}/build && cmake .. -DBUILD_TESTS=ON && make -j4"
    echo "  Cannot run ablation without the runtime binary."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Partition DLRM
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Partitioning DLRM with SplitInfer..."
PARTITION_LOG="${RESULTS_DIR}/partition.log"

if python3 -m partitioner.cli \
        "${DLRM_MODEL}" \
        --output "${MANIFEST}" \
        --fpga-ddr2-mb 128 \
        --gpu-gflops 1000 \
        --fpga-int8-gops 6.4 \
        --usb-bw-mbps 40 \
        2>&1 | tee "${PARTITION_LOG}"; then
    echo "  Partition manifest: ${MANIFEST}"
else
    echo "  Partitioning FAILED."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Full SplitInfer (all features enabled)
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Config 1: Full SplitInfer (all features enabled)..."
run_splitinfer_config "full_splitinfer" \
    "SPLITINFER_NO_PREFETCH=0" \
    "SPLITINFER_NO_PIPELINE=0" \
    "SPLITINFER_NO_NMC=0"

# ---------------------------------------------------------------------------
# Step 4: Ablation variants
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Config 2: No prefetch..."
run_splitinfer_config "no_prefetch" \
    "SPLITINFER_NO_PREFETCH=1" \
    "SPLITINFER_NO_PIPELINE=0" \
    "SPLITINFER_NO_NMC=0"

echo ""
echo "[4/6] Config 3: No pipelining..."
run_splitinfer_config "no_pipeline" \
    "SPLITINFER_NO_PREFETCH=0" \
    "SPLITINFER_NO_PIPELINE=1" \
    "SPLITINFER_NO_NMC=0"

echo ""
echo "[4/6] Config 4: No NMC..."
run_splitinfer_config "no_nmc" \
    "SPLITINFER_NO_PREFETCH=0" \
    "SPLITINFER_NO_PIPELINE=0" \
    "SPLITINFER_NO_NMC=1"

# ---------------------------------------------------------------------------
# Step 5: B2 baseline (GPU-only INT8) for reference
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Config 5: GPU-only quantized (B2 INT8 baseline)..."
B2_LOG="${RESULTS_DIR}/b2_int8.log"
B2_JSON="${RESULTS_DIR}/b2_int8.json"

if python3 "${BASELINES_DIR}/jetson_only_quant.py" \
        "${DLRM_MODEL}" \
        --warmup "${WARMUP}" \
        --runs "${RUNS}" \
        --dynamic \
        2>&1 | tee "${B2_LOG}"; then
    echo "  B2 INT8 completed."
else
    echo "  B2 INT8 FAILED (OOM expected for DLRM)."
fi

python3 -c "
import json, re
log = open('${B2_LOG}').read()
result = {
    'config': 'b2_int8_reference',
    'status': 'ok' if 'latency' in log.lower() or 'throughput' in log.lower() else 'failed',
}
m = re.search(r'mean[_ ]latency[:\s]*([\d.]+)\s*ms', log, re.IGNORECASE)
if m:
    result['mean_latency_ms'] = float(m.group(1))
m = re.search(r'throughput[:\s]*([\d.]+)', log, re.IGNORECASE)
if m:
    result['throughput_ips'] = float(m.group(1))
json.dump(result, open('${B2_JSON}', 'w'), indent=2)
" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 6: Summary table
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "E4: Ablation Summary"
echo "============================================================"
echo ""
printf "%-22s %-10s %-15s %-15s %-15s\n" \
    "Configuration" "Status" "Latency(ms)" "P99(ms)" "Throughput"
printf "%-22s %-10s %-15s %-15s %-15s\n" \
    "-------------------" "------" "-----------" "-----------" "-----------"

for json_file in "${RESULTS_DIR}"/*.json; do
    [ -f "${json_file}" ] || continue
    python3 -c "
import json
d = json.load(open('${json_file}'))
name = d.get('config', '?')
status = d.get('status', '?')
lat = str(d.get('mean_latency_ms', '-'))
p99 = str(d.get('p99_latency_ms', '-'))
thr = str(d.get('throughput_ips', '-'))
print(f'{name:<22s} {status:<10s} {lat:<15s} {p99:<15s} {thr:<15s}')
" 2>/dev/null || true
done

# Compute speedup relative to slowest ablation (if data available)
echo ""
echo "Speedup analysis (relative to full SplitInfer):"
python3 -c "
import json, glob, os

results_dir = '${RESULTS_DIR}'
configs = {}
for f in glob.glob(os.path.join(results_dir, '*.json')):
    d = json.load(open(f))
    if 'mean_latency_ms' in d:
        configs[d.get('config', os.path.basename(f))] = d['mean_latency_ms']

if 'full_splitinfer' in configs:
    base = configs['full_splitinfer']
    for name, lat in sorted(configs.items()):
        if base > 0:
            ratio = lat / base
            delta = ((lat - base) / base) * 100
            print(f'  {name:<22s}  {lat:8.2f} ms  ({delta:+.1f}%)')
else:
    print('  (Full SplitInfer latency not available — cannot compute speedups)')
" 2>/dev/null || true

echo ""
echo "============================================================"
echo "E4 Complete. Results saved in: ${RESULTS_DIR}"
echo "============================================================"
