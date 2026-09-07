#!/usr/bin/env bash
# board_session.sh — the v2 on-board session, end to end, from the Jetson.
#
#   fpga/scripts/board_session.sh [path/to/top.bit]
#
# 1. flash the bitstream (volatile SRAM) over JTAG with openFPGALoader
# 2. protocol loopback (framing/ACK)
# 3. bit-exact check of the lowered DLRM program in NMC and pool modes
# 4. measured link calibration -> evaluation/calibration/uart_measured.json
# 5. E2 v2 with --transport usb (MEASURED SplitInfer cells)
# 6. regenerate paper macros and figures
# Every step's exit code is recorded; the script continues past a failing
# measurement step so partial results are kept, and prints a summary.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SI="$(cd "$HERE/../.." && pwd)"
BIT="${1:-$SI/fpga/output/top.bit}"
LOG="$SI/evaluation/experiments/results/board_session_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
declare -A RC
step() { local name="$1"; shift; echo; echo "===== $name =====" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; RC[$name]=${PIPESTATUS[0]}; echo "[$name] exit ${RC[$name]}" | tee -a "$LOG"; }

cd "$SI"
[[ -f "$BIT" ]] || { echo "no bitstream at $BIT"; exit 1; }
echo "bitstream: $BIT ($(stat -c '%y' "$BIT"))" | tee -a "$LOG"

step build      cmake --build build -j4
step bit2bin    python3 fpga/scripts/bit2bin.py "$BIT" /tmp/top_v2.bin
step flash      openFPGALoader -b nexys_a7_100 --file-type bin /tmp/top_v2.bin
sleep 2
step loopback   timeout 60 ./build/protocol/test_loopback
step nmc_exact  timeout 900 ./build/runtime/v2/splitinfer_v2 evaluation/lowered/dlrm_small --mode nmc  --transport usb --iterations 3 --warmup 1
step pool_exact timeout 1800 ./build/runtime/v2/splitinfer_v2 evaluation/lowered/dlrm_small --mode pool --transport usb --iterations 1 --warmup 0 --prefetch-pages 64
step calibrate  timeout 600 python3 evaluation/experiments/calibrate_link.py --transport usb --out evaluation/calibration/uart_measured.json --reps 30
if [[ ${RC[nmc_exact]} -eq 0 ]]; then
  step e2_usb   python3 evaluation/experiments/e2_v2.py evaluation/models/out/dlrm_small.onnx evaluation/lowered/dlrm_small \
                  --out evaluation/experiments/results/e2/e2_v2_dlrm_usb.json --runs 30 --warmup 5 --si-runs 30 --transport usb \
                  --skip B1_ort_fp32_cpu,B2_ort_int8_cpu,B3_trt_fp16_gpu,B4_trt_int8_gpu,SI_host,SI_pool
else
  echo "skipping E2 usb: NMC bit-exact check failed" | tee -a "$LOG"
fi
step results    python3 evaluation/make_results_tex.py
step figures    python3 evaluation/make_figures.py

echo; echo "===== SUMMARY =====" | tee -a "$LOG"
for k in build bit2bin flash loopback nmc_exact pool_exact calibrate e2_usb results figures; do [[ -v RC[$k] ]] && printf "  %-11s %s\n" "$k" "$([[ ${RC[$k]} -eq 0 ]] && echo OK || echo "FAIL (${RC[$k]})")" | tee -a "$LOG"; done
echo "log: $LOG"
