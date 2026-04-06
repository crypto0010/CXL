# ==============================================================================
# SplitInfer FPGA Programming Script
# Target: Nexys 4 DDR (XC7A100T-1CSG324C)
# Usage:  vivado -mode batch -source program.tcl
# ==============================================================================

set BUILD_DIR [file dirname [file normalize [info script]]]
set BIT_FILE  ${BUILD_DIR}/output/top.bit

if {![file exists ${BIT_FILE}]} {
    puts "ERROR: Bitstream not found at ${BIT_FILE}"
    puts "       Run build.tcl first."
    exit 1
}

open_hw_manager
connect_hw_server -allow_non_jtag

# Auto-detect the target
open_hw_target

set device [lindex [get_hw_devices xc7a*] 0]
current_hw_device $device
set_property PROGRAM.FILE ${BIT_FILE} $device

puts "Programming ${device} with ${BIT_FILE} ..."
program_hw_devices $device

puts "Programming complete."
close_hw_target
disconnect_hw_server
close_hw_manager
