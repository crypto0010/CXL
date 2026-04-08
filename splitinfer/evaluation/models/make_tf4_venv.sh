#!/usr/bin/env bash
# make_tf4_venv.sh — set up a transformers<5 venv for MobileBERT export.
#
# Why this exists:
#   transformers >= 5.5 has an internal bug that crashes during ONNX trace
#   for MobileBERT (see gen_mobilebert.py docstring for details).  The fix
#   is to use transformers 4.x in an isolated venv, without disturbing the
#   system-wide install.
#
# Usage:
#   ./make_tf4_venv.sh                  # creates /tmp/tf4_venv
#   ./make_tf4_venv.sh /path/to/venv    # creates at custom location
#
# After setup, generate MobileBERT with:
#   /tmp/tf4_venv/bin/python evaluation/models/gen_mobilebert.py \
#       evaluation/models/generated/mobilebert.onnx

set -euo pipefail

VENV="${1:-/tmp/tf4_venv}"

if [[ -d "${VENV}" ]]; then
    echo "venv already exists at ${VENV}"
    "${VENV}/bin/python" -c "import transformers; print('transformers:', transformers.__version__)"
    exit 0
fi

echo "[1/3] Creating venv at ${VENV} (with --system-site-packages so torch is inherited)..."
python3 -m venv --system-site-packages "${VENV}"

echo "[2/3] Installing transformers<5..."
"${VENV}/bin/pip" install --quiet 'transformers<5'

echo "[3/3] Verifying..."
"${VENV}/bin/python" -c "
import transformers, torch, onnx
print(f'  transformers: {transformers.__version__}')
print(f'  torch:        {torch.__version__}')
print(f'  onnx:         {onnx.__version__}')
"
echo ""
echo "Done.  Generate MobileBERT with:"
echo "  ${VENV}/bin/python $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gen_mobilebert.py \\"
echo "      $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/generated/mobilebert.onnx"
