# Re-synthesis Trip Instructions — Multi-Inference FSM Fix

**Date prepared:** 2026-04-09
**Target commit:** `aec4772` (`fpga: fix multi-inference state leak + add regression testbench`)
**Reason:** Restore correct behavior across back-to-back `splitinfer_run --real`
invocations without needing a bitstream re-flash between iterations. This
unblocks Task #32 (re-run DLRM batch sweep with statistical rigor) and
enables realistic E4 ablation runs where prefetch-on vs prefetch-off
comparison must execute hundreds of inferences without manual intervention.

---

## What changed

Exactly **one source file** needs to be re-synthesized:

- `splitinfer/fpga/src/edgecoh_controller.v`
  `+122 insertions, -8 deletions`

The changes are:

1. Added `stall_cnt[23:0]` register and `WATCHDOG_LIMIT = 10_000_000`
   localparam near the top of the module.
2. Added two wire declarations (`progress`, `in_wait_state`) just before
   the main `always` block.
3. Extended the reset block to also clear `payload_len`, `dma_wr_base`,
   `dma_rd_base`, and `stall_cnt`.
4. Added a watchdog counter-maintenance block and an early-exit watchdog
   trigger that goes before the `case (state)` statement. The trigger
   uses an `else case` pattern — the `else` is load-bearing, don't drop it.
5. Rewrote the three exit paths to S_IDLE (in `S_ACK_WAIT`, `S_DMA_READ_REQ`,
   `S_DMA_READ_RESP`) to explicitly clear all per-message transient state
   before the state update.

No other files need to change. Neither `top.v`, `usb_interface.v`, nor any
constraints file were touched.

---

## Verification on the Windows box (before flashing)

1. **Pull the latest main branch** on the Windows workstation:
   ```
   git pull origin main
   ```
   You should end up with `aec4772` or later as HEAD.

2. **Sanity-check the diff** actually arrived:
   ```
   git show --stat aec4772
   ```
   Expected output:
   ```
    splitinfer/fpga/src/edgecoh_controller.v          | 130 ++++++++++++++++---
    splitinfer/fpga/sim/tb_edgecoh_multi_inference.v  | 321 +++++++++++++++++++
    2 files changed, 443 insertions(+), 8 deletions(-)
   ```

3. **Open the Vivado project** and run synthesis + implementation:
   - Open `splitinfer/fpga/splitinfer.xpr`
   - Click **Run Synthesis** (or **Generate Bitstream** to do the whole flow)
   - Expected wall time: ~10-15 minutes (similar to previous builds)

4. **Review the timing report** after implementation:
   - Open *Reports → Timing → Report Timing Summary*
   - Look at **Worst Negative Slack (WNS)** — must be ≥ 0 ns
   - Look at **Worst Hold Slack (WHS)** — must be ≥ 0 ns
   - Expected: roughly the same slack as before (our changes add a
     single 24-bit counter + a few comparators, well under 100 LUTs
     of additional logic — should not touch any existing critical paths)

5. **Review the utilization report** to confirm the delta is small:
   - Expected increase: **+20 to +60 LUTs, +24 to +30 FFs** (just
     the stall_cnt register and its increment logic)
   - If you see a much larger delta (e.g., +500 LUTs), something
     interpreted the watchdog logic badly — stop and investigate
     before flashing.

6. **Generate the bitstream** if timing is clean.

---

## Flashing and verifying on the Jetson

1. **Carry back** `splitinfer/fpga/output/top.bit` (or push via git — but
   keep in mind `.bit` is gitignored by default, so you'd need to
   temporarily bypass that).

2. **On the Jetson**, convert `.bit` → `.bin` and flash:
   ```
   cd /home/csdf/cxl/splitinfer
   ./fpga/scripts/flash_fpga.sh
   ```

3. **Run the existing T19 loopback test** to confirm baseline sanity:
   ```
   cd build
   sudo ./protocol/test_loopback
   ```
   Expected: passes as before (this verifies the fix didn't break
   single-inference behavior).

4. **Run the critical multi-iteration test** — this is the whole reason
   for the trip:
   ```
   sudo ./runtime/splitinfer_run \
       ../evaluation/experiments/results/e2/dlrm_b1_manifest.json \
       --real --warmup 3 --iterations 10 --json
   ```

   **Expected result (post-fix):**
   - All 10 iterations complete successfully
   - Per-iteration median latency ≈ 595 ms (consistent with E2 cell)
   - Final STATS_JSON line emitted to stdout

   **Pre-fix behavior (before this trip):**
   - Iter 1 would succeed
   - Iter 2-3 would partially succeed with wrong ACK timing
   - Iter 4+ would all time out on "gather_0 ACK timeout"

   If you see the pre-fix behavior after flashing, STOP — it means
   the bitstream didn't pick up the RTL change, probably because
   Vivado cached a stale compile. Force a clean synthesis with
   `launch_runs synth_1 -force_up_to_date` in the Vivado Tcl console.

5. **Run the full DLRM batch sweep with the new methodology:**
   ```
   cd /home/csdf/cxl/splitinfer
   python3 evaluation/experiments/e2_run.py \
       --models dlrm --batches 1 64 256 \
       --warmup 3 --runs 30
   ```

   Expected wall time: ~30-60 minutes depending on how b64/b256 fare
   (the MAC controller slowness on those cells is unrelated to this fix).

---

## Rollback plan if something breaks

If the new bitstream misbehaves in any unexpected way:

1. **Re-flash the OLD bitstream** (whichever top.bit was in
   `splitinfer/fpga/output/` before this trip) via
   `./fpga/scripts/flash_fpga.sh path/to/old/top.bit`.

2. **Git-revert the RTL change** locally:
   ```
   git revert aec4772
   ```

   This creates a new commit that undoes the fix. The testbench stays
   (harmless — no DUT references in the wild), only the controller
   reverts.

3. **Fall back to the re-flash-per-iteration workaround** that e2_run.py
   was about to adopt before we decided to fix at the RTL level. It's
   slower (~6s per iteration) but proven reliable.

---

## Confidence level

- **iverilog simulation** (on the Jetson, `11.0`): the new
  `tb_edgecoh_multi_inference.v` testbench runs two back-to-back
  `SYNC_BARRIER + NMC_EXEC` sequences with NO reset between them
  and both iterations produce the correct ACK markers with
  correct tensor_ids. **PASS** (committed output in `aec4772`).

- **Real hardware**: not yet tested. ~85% confidence the fix
  transfers cleanly to silicon; the remaining 15% is Vivado-specific
  synthesis subtleties that simulation can't catch.

- **Fallback readiness**: e2_run.py still has the
  one-process-per-iteration pattern from the original E2 sweep as
  a safety net. If the fix misbehaves, we can fall back to the
  pre-fix measurement approach at the cost of 3-4x longer sweep
  wall time.
