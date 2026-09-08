#!/bin/bash
# Run all v2 golden testbenches under Icarus Verilog.  Exit non-zero on any failure.
cd "$(dirname "$0")"
SRC=../../src
fail=0
for tb in tb_dma_roundtrip tb_mac_golden tb_embedding_golden tb_epilogue_golden; do
  iverilog -g2012 -o /tmp/$tb.vvp ddr2_ui_model.v $tb.v $SRC/ddr2_arbiter.v $SRC/mac_array_8x8.v $SRC/mac_controller.v $SRC/embedding_lookup.v $SRC/elementwise.v $SRC/eltwise_controller.v 2>&1 | grep -v "warning: Port" || true
  out=$(timeout 300 vvp -n /tmp/$tb.vvp 2>&1)
  echo "$out" | grep -E "===|MISMATCH|clobbered|row |idx |MODEL ERROR|cycles" | head -20
  echo "$out" | grep -q "TEST PASSED" || fail=1
done
# epilogue with ReLU off and a different shift: exposes sign errors the ReLU case masks
iverilog -g2012 -DEPI_MULT="16'd23170" -DEPI_SHIFT="5'd9" -DEPI_RELU="1'b0" -o /tmp/tb_epi2.vvp ddr2_ui_model.v tb_epilogue_golden.v $SRC/ddr2_arbiter.v $SRC/elementwise.v $SRC/eltwise_controller.v 2>&1 | grep -v "warning: Port" || true
out=$(timeout 300 vvp -n /tmp/tb_epi2.vvp 2>&1); echo "$out" | grep -E "===|m=" | head -8; echo "$out" | grep -q "TEST PASSED" || fail=1
# full-chip: UART -> controller -> CDC FIFOs -> arbiter -> behavioural MIG -> UART
iverilog -g2012 -o /tmp/tb_top_dma.vvp -I $SRC tb_top_dma.v mig_7series_0_beh.v ddr2_ui_model.v xilinx_prims_stub.v $SRC/*.v 2>&1 | grep -v "warning: Port" || true
out=$(timeout 600 vvp -n /tmp/tb_top_dma.vvp 2>&1); echo "$out" | grep -E "===|\[" | head -10; echo "$out" | grep -q "TEST PASSED" || fail=1
exit $fail
