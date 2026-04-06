#!/usr/bin/env bash
# evaluation/experiments/smoke_test.sh
#
# End-to-End Smoke Test for SplitInfer Pipeline
#
# Steps:
#   1. Generate a tiny 3-layer ONNX model (MatMul -> Relu -> MatMul)
#   2. Run the SplitInfer partitioner to produce a manifest
#   3. Run splitinfer_run CLI with the manifest (stub executors)
#   4. Run all C unit tests via ctest
#
# Exits 0 if all steps pass, non-zero on first failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="${SCRIPT_DIR}/smoke_test_tmp"
mkdir -p "${WORK_DIR}"

TINY_MODEL="${WORK_DIR}/tiny_model.onnx"
MANIFEST="${WORK_DIR}/tiny_manifest.json"
BUILD_DIR="${REPO_ROOT}/build"

PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "============================================================"
echo "SplitInfer End-to-End Smoke Test"
echo "============================================================"
echo "Repo root : ${REPO_ROOT}"
echo "Work dir  : ${WORK_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Generate tiny 3-layer ONNX model
# ---------------------------------------------------------------------------
echo "[1/4] Generating tiny 3-layer ONNX model..."

python3 - "${TINY_MODEL}" << 'PYEOF'
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

out_path = sys.argv[1]

# fc1: (1,128) x (128,64) -> (1,64)
W1 = np.zeros((128, 64), dtype=np.float32)
# fc2: (1,64) x (64,10) -> (1,10)
W2 = np.zeros((64, 10), dtype=np.float32)

inits = [
    numpy_helper.from_array(W1, name="W1"),
    numpy_helper.from_array(W2, name="W2"),
]
nodes = [
    helper.make_node("MatMul", ["input", "W1"], ["mm1"],  name="fc1"),
    helper.make_node("Relu",   ["mm1"],          ["relu"], name="relu1"),
    helper.make_node("MatMul", ["relu",  "W2"], ["output"], name="fc2"),
]
g = helper.make_graph(
    nodes, "tiny_model",
    [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 128])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
    initializer=inits,
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, out_path)
print(f"  Written: {out_path}")
PYEOF

if [ -f "${TINY_MODEL}" ]; then
    pass "Tiny ONNX model generated"
else
    fail "Tiny ONNX model NOT generated"
    echo "Abort: cannot continue without model."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Run partitioner
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Running partitioner..."

cd "${REPO_ROOT}"
if python3 -m partitioner.cli \
        "${TINY_MODEL}" \
        --output "${MANIFEST}" \
        --fpga-ddr2-mb 128 \
        --gpu-gflops 1000 \
        --fpga-int8-gops 6.4 \
        --usb-bw-mbps 40 \
        2>&1; then
    pass "Partitioner produced manifest"
else
    fail "Partitioner failed"
    echo "  Ensure the partitioner package is installed: pip install -e ."
fi

if [ -f "${MANIFEST}" ]; then
    pass "Manifest file exists: ${MANIFEST}"
else
    fail "Manifest file missing"
fi

# ---------------------------------------------------------------------------
# Step 3: Run splitinfer_run CLI with manifest
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Running splitinfer_run CLI..."

SPLITINFER_BIN="${BUILD_DIR}/runtime/splitinfer_run"

if [ ! -x "${SPLITINFER_BIN}" ]; then
    echo "  WARNING: splitinfer_run not found at ${SPLITINFER_BIN}"
    echo "  Building now..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    if cmake "${REPO_ROOT}" -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 \
       && make -j"$(nproc)" splitinfer_run > /dev/null 2>&1; then
        pass "splitinfer_run built successfully"
    else
        fail "splitinfer_run build failed"
        echo "  Skipping runtime step."
        SPLITINFER_BIN=""
    fi
    cd "${REPO_ROOT}"
fi

if [ -n "${SPLITINFER_BIN}" ] && [ -x "${SPLITINFER_BIN}" ] && [ -f "${MANIFEST}" ]; then
    if "${SPLITINFER_BIN}" "${MANIFEST}" 2>&1; then
        pass "splitinfer_run completed successfully"
    else
        fail "splitinfer_run returned non-zero exit"
    fi
else
    echo "  [SKIP] splitinfer_run not available"
fi

# ---------------------------------------------------------------------------
# Step 4: Run all C unit tests via ctest
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Running C unit tests via ctest..."

if [ ! -d "${BUILD_DIR}" ]; then
    echo "  Build directory not found: ${BUILD_DIR}"
    echo "  Run: mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR} && cmake .. -DBUILD_TESTS=ON && make -j4"
    fail "Build directory missing"
else
    cd "${BUILD_DIR}"
    # Ensure tests are built
    if make -j"$(nproc)" 2>&1 | tail -5; then
        if ctest --output-on-failure 2>&1; then
            pass "All ctest unit tests passed"
        else
            fail "One or more ctest tests failed"
        fi
    else
        fail "Build failed before ctest could run"
    fi
    cd "${REPO_ROOT}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Smoke Test Summary"
echo "============================================================"
echo "  PASSED: ${PASS}"
echo "  FAILED: ${FAIL}"
echo ""

# Cleanup
rm -rf "${WORK_DIR}"

if [ "${FAIL}" -gt 0 ]; then
    echo "RESULT: SMOKE TEST FAILED"
    exit 1
else
    echo "RESULT: SMOKE TEST PASSED"
    exit 0
fi
