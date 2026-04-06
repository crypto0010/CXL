#!/usr/bin/env bash
# evaluation/models/download_models.sh
# Generates/downloads ML models for SplitInfer evaluation.
# Requires: python3, onnx, numpy.  MobileBERT and YOLOv8 require optional deps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}"

echo "=== SplitInfer Model Download / Generation ==="
echo "Output directory: ${MODELS_DIR}"

# ---------------------------------------------------------------------------
# 1. Synthetic DLRM-style model
#    26 embedding tables x 1000 rows x 64-dim  +  top MLP (4096->1024->256->1)
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Generating synthetic DLRM model..."

python3 "${SCRIPT_DIR}/gen_dlrm.py" "${MODELS_DIR}/dlrm_synthetic.onnx"

echo "[1/3] DLRM model generated."

# ---------------------------------------------------------------------------
# 2. MobileBERT (requires transformers + torch)
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Attempting MobileBERT export..."

python3 "${SCRIPT_DIR}/gen_mobilebert.py" "${MODELS_DIR}/mobilebert.onnx" \
    || echo "  [SKIP] MobileBERT export skipped (missing deps: transformers, torch)"

# ---------------------------------------------------------------------------
# 3. YOLOv8-nano (requires ultralytics)
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Attempting YOLOv8-nano export..."

python3 "${SCRIPT_DIR}/gen_yolov8.py" "${MODELS_DIR}/yolov8n.onnx" \
    || echo "  [SKIP] YOLOv8-nano export skipped (missing deps: ultralytics)"

echo ""
echo "=== Done.  Available models in ${MODELS_DIR}: ==="
ls -lh "${MODELS_DIR}"/*.onnx 2>/dev/null || echo "  (none)"
