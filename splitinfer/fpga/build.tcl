# ==============================================================================
# SplitInfer FPGA Build Script — Vivado Non-Project Mode
# Target: Nexys 4 DDR (XC7A100T-1CSG324C)
# Usage:  vivado -mode batch -source build.tcl
# ==============================================================================

# --- Configuration -----------------------------------------------------------
set PART        xc7a100tcsg324-1
set TOP         top
set BUILD_DIR   [file dirname [file normalize [info script]]]
set SRC_DIR     ${BUILD_DIR}/src
set XDC_DIR     ${BUILD_DIR}/constraints
set IP_DIR      ${BUILD_DIR}/ip
set OUT_DIR     ${BUILD_DIR}/output
set RPT_DIR     ${OUT_DIR}/reports
set MIG_RTL     ${IP_DIR}/mig_7series_0/mig_7series_0/user_design/rtl
set MIG_XDC     ${IP_DIR}/mig_7series_0/mig_7series_0/user_design/constraints

file mkdir ${OUT_DIR}
file mkdir ${RPT_DIR}

# --- Verify MIG IP exists ----------------------------------------------------
if {![file exists ${MIG_RTL}/mig_7series_0.v]} {
    puts "ERROR: MIG RTL not found. Run generate_mig.tcl first."
    exit 1
}

# --- Read design sources -----------------------------------------------------
read_verilog [glob ${SRC_DIR}/*.v]

# --- Read MIG RTL sources (all subdirectories) --------------------------------
puts "========== LOADING MIG RTL =========="
foreach vfile [glob -nocomplain ${MIG_RTL}/*.v ${MIG_RTL}/*/*.v] {
    # Skip simulation-only files
    if {[string match "*_sim.v" $vfile]} { continue }
    read_verilog $vfile
}
puts "  MIG RTL files loaded"

# --- Read MIG constraints (pin locations, timing) ----------------------------
foreach xdc [glob -nocomplain ${MIG_XDC}/*.xdc] {
    if {[string match "*_ooc.xdc" $xdc]} { continue }
    puts "  Reading MIG XDC: [file tail $xdc]"
    read_xdc $xdc
}

# --- Read user constraints ----------------------------------------------------
read_xdc ${XDC_DIR}/nexys4ddr.xdc

# --- Synthesis ----------------------------------------------------------------
puts "========== SYNTHESIS =========="
synth_design -top ${TOP} -part ${PART} \
    -flatten_hierarchy rebuilt

# Post-synthesis reports
report_utilization -file ${RPT_DIR}/post_synth_utilization.rpt
report_timing_summary -file ${RPT_DIR}/post_synth_timing.rpt
write_checkpoint -force ${OUT_DIR}/post_synth.dcp

# --- Optimization -------------------------------------------------------------
puts "========== OPTIMIZATION =========="
opt_design
place_design
report_clock_utilization -file ${RPT_DIR}/post_place_clock_util.rpt

# Post-place timing
phys_opt_design
route_design

# --- Post-route reports -------------------------------------------------------
puts "========== REPORTS =========="
report_utilization -file ${RPT_DIR}/post_route_utilization.rpt
report_utilization -hierarchical -file ${RPT_DIR}/post_route_utilization_hier.rpt
report_timing_summary -file ${RPT_DIR}/post_route_timing.rpt
report_timing -sort_by group -max_paths 10 -path_type summary \
    -file ${RPT_DIR}/post_route_timing_detail.rpt
report_io -file ${RPT_DIR}/post_route_io.rpt
report_power -file ${RPT_DIR}/post_route_power.rpt
report_drc -file ${RPT_DIR}/post_route_drc.rpt

write_checkpoint -force ${OUT_DIR}/post_route.dcp

# --- Bitstream generation -----------------------------------------------------
puts "========== BITSTREAM =========="
write_bitstream -force ${OUT_DIR}/${TOP}.bit

puts "========== BUILD COMPLETE =========="
puts "Bitstream: ${OUT_DIR}/${TOP}.bit"
puts "Reports:   ${RPT_DIR}/"
