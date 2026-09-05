# Vivado Session — SplitInfer v2 data path

**Prepared:** 2026-09-05 · **Target:** HEAD of `main` (v2 RTL, commit `fa17bd3` or later)
**Board:** Digilent Nexys 4 DDR (XC7A100T-1CSG324C) · **Tool:** Vivado 2025.x on the Windows workstation

## Why a re-synthesis is required

Golden-vector simulation (`fpga/sim/v2/`) showed the v1 bitstream could not
preserve data through DDR2 in either direction, and that the MAC engine was
not an 8×8 array.  Every module below changed; `top.v` port wiring changed.
The v1 bitstream must not be used for any v2 measurement.

| File | Change |
|---|---|
| `src/ddr2_arbiter.v` | Rewritten: byte→MIG address conversion, DMA byte bridge with `app_wdf_mask`, 2-deep write queues, hold-until-accepted MIG handshake, owner-tagged single outstanding read |
| `src/mac_array_8x8.v` | Rewritten: 8 independent lanes (`a_rows[511:0]`, `b[63:0]`) |
| `src/mac_controller.v` | Rewritten: 8 rows/pass, 16-byte reads, correct output stride |
| `src/elementwise.v`, `src/eltwise_controller.v` | 3-bit op, second operand stream, fused EPILOGUE |
| `src/embedding_lookup.v` | Index lane select |
| `src/nmc_dispatch.v` | Routes `table_cols`→`addr2`, `input_len[15:0]`→`mult`, 3-bit op |
| `src/edgecoh_controller.v` | DMA read byte latch (`dma_rd_have`) |
| `src/top.v` | `app_wdf_mask` from arbiter; DMA FIFO pops gated by arbiter ready; MAC/elt port wiring |

`usb_interface.v`, `cdc_*.v`, constraints and the MIG IP are unchanged.

## Steps

1. `git pull origin main` on the workstation; confirm `git log -1` shows the
   v2 RTL commit and `git status` is clean.
2. Open `splitinfer/fpga/splitinfer.xpr`.  **Refresh sources** — `top.v`'s
   port list changed; if Vivado reports stale hierarchy, remove and re-add
   `src/*.v` (do not add anything from `sim/`).
3. Run synthesis.  Expected: no new critical warnings.  The DSP count should
   rise (64 multipliers in the MAC array vs 8 in v1) — verify `DSP48E1`
   usage is ~64–80 of 240.  BRAM unchanged.
4. Run implementation.  Timing: the MAC array's stage-1 adds two products
   per lane and stage-2 adds four partial sums — both at 81.25 MHz
   (`ui_clk`); should close with margin.  If `mac_array_8x8` fails timing,
   register the four stage-1 partial sums individually (already the
   structure) and report the slack figure back.
5. Generate bitstream; `program.tcl` as before.
6. Confirm LED[0] (calibration) and LED[3] (heartbeat) as in v1.

## On-board validation (Jetson side, after flashing)

All of these send the **same bytes** the emulator tests send.

```bash
cd ~/cxl/splitinfer && cmake --build build -j4
# 1. protocol + data path (was never data-checked in v1)
./build/protocol/test_loopback                      # framing / ACK
./build/runtime/v2/splitinfer_v2 evaluation/lowered/dlrm_small --mode nmc  --transport usb --iterations 3 --warmup 1
./build/runtime/v2/splitinfer_v2 evaluation/lowered/dlrm_small --mode pool --transport usb --iterations 1 --prefetch-pages 64
```
Both must report `N/N bit-exact vs expected`.  A mismatch localises as:
- `nmc` wrong, `pool` right → an engine (compare per-layer with `--mode host`).
- both wrong → DMA byte path (`DATA_WRITE`/`DATA_READ`) — run
  `test_loopback` with a data pattern.

## Measurements to take (the MEASURED anchor for the paper)

```bash
python3 evaluation/experiments/e2_v2.py evaluation/models/out/dlrm_small.onnx evaluation/lowered/dlrm_small \
    --out evaluation/experiments/results/e2/e2_v2_dlrm_usb.json --runs 30 --warmup 5 --si-runs 30 --transport usb
python3 evaluation/experiments/calibrate_link.py --out evaluation/calibration/uart_measured.json
```
Then re-partition with `--calibration evaluation/calibration/uart_measured.json`
and re-lower; the cost-model figures in the paper must come from that file.

Do **not** re-flash between iterations.  If the multi-inference run stalls,
that is a finding to report, not a thing to work around.
