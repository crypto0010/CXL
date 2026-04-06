# ==============================================================================
# MIG DDR2 IP Generation Script for Nexys 4 DDR
# Uses new flow (default) with Digilent-reference PRJ format
# Usage:  vivado -mode batch -source generate_mig.tcl
# ==============================================================================

set BUILD_DIR [file dirname [file normalize [info script]]]
set IP_DIR    ${BUILD_DIR}/ip
set PRJ_FILE  [file normalize ${IP_DIR}/mig_nexys4ddr.prj]
set PART      xc7a100tcsg324-1

# Clean previous
set proj_dir ${IP_DIR}/mig_project
if {[file exists ${proj_dir}]} { file delete -force ${proj_dir} }
if {[file exists ${IP_DIR}/mig_7series_0]} { file delete -force ${IP_DIR}/mig_7series_0 }

# Use default (new) flow
create_project mig_project ${proj_dir} -part ${PART} -force
set_property target_language Verilog [current_project]

create_ip -name mig_7series -vendor xilinx.com -library ip -version 4.2 \
    -module_name mig_7series_0

puts "========== CONFIGURING MIG FROM PRJ =========="
set_property CONFIG.XML_INPUT_FILE ${PRJ_FILE} [get_ips mig_7series_0]

puts "========== GENERATING OUTPUT PRODUCTS =========="
generate_target all [get_ips mig_7series_0]

puts "========== OOC SYNTHESIS =========="
synth_ip [get_ips mig_7series_0]

# Copy outputs
set src_dir [get_property IP_DIR [get_ips mig_7series_0]]
puts "MIG IP directory: ${src_dir}"
file copy -force ${src_dir} ${IP_DIR}/mig_7series_0

puts "========== MIG IP GENERATION COMPLETE =========="
close_project
