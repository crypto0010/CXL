#!/usr/bin/env bash
# flash_fpga.sh — convert top.bit to .bin and flash to Nexys 4 DDR via JTAG.
#
# Usage:
#   ./flash_fpga.sh                       # uses ../output/top.bit by default
#   ./flash_fpga.sh path/to/some.bit      # explicit bit file
#
# Requires:
#   - python3 (for bit2bin.py)
#   - openFPGALoader on PATH (sudo apt install openfpgaloader, or build from source)
#   - Nexys 4 DDR connected via USB
#
# Safe to re-run; flashes to volatile SRAM (not flash) so power-cycle reverts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BIT="${SCRIPT_DIR}/../output/top.bit"
BIT_FILE="${1:-${DEFAULT_BIT}}"

if [[ ! -f "${BIT_FILE}" ]]; then
    echo "ERROR: bitstream not found: ${BIT_FILE}" >&2
    echo "Usage: $0 [path/to/file.bit]" >&2
    exit 1
fi

BIN_FILE="${BIT_FILE%.bit}.bin"

echo "[1/2] Converting .bit -> .bin (strip Vivado 2025.2 metadata)..."
python3 "${SCRIPT_DIR}/bit2bin.py" "${BIT_FILE}" "${BIN_FILE}"

echo ""
echo "[2/2] Flashing to Nexys 4 DDR via JTAG..."
sudo openFPGALoader -b nexys_a7_100 --file-type bin "${BIN_FILE}"

echo ""
echo "Done. The FPGA is now running the new bitstream (volatile SRAM)."
echo "To make it persistent across power cycles, re-run with:"
echo "  sudo openFPGALoader -b nexys_a7_100 --write-flash --file-type bin ${BIN_FILE}"
