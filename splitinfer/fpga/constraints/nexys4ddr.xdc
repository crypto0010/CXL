## splitinfer/fpga/constraints/nexys4ddr.xdc
## Digilent Nexys 4 DDR (XC7A100T-1CSG324C)
## Pin assignments verified against official Nexys-4-DDR-Master.xdc

## System clock (100 MHz crystal oscillator)
## We use explicit IBUF + BUFG in RTL; pin location and clock still needed.
set_property -dict {PACKAGE_PIN E3 IOSTANDARD LVCMOS33} [get_ports sys_clk]
create_clock -period 10.000 -name sys_clk [get_ports sys_clk]

## CDC constraints: declare sys_clk and MIG-derived clocks as asynchronous.
## All crossings use handshake synchronizers or async FIFOs in RTL.
set_clock_groups -asynchronous -group [get_clocks sys_clk] \
    -group [get_clocks -quiet -include_generated_clocks -of_objects [get_pins -quiet -hierarchical -filter {NAME =~ *u_mig*plle2_i/CLKFBOUT}]]

## False paths for CDC synchronizer inputs (2FF chains)
set_false_path -to [get_pins -quiet -hierarchical -filter {NAME =~ *_sync_reg[0]/D}]
set_false_path -to [get_pins -quiet -hierarchical -filter {NAME =~ *calib_sync_reg[0]/D}]

## False path for handshake CDC data hold (stable during handshake)
set_false_path -from [get_pins -quiet -hierarchical -filter {NAME =~ *cdc_*/src_data_hold_reg*/C}] \
               -to   [get_pins -quiet -hierarchical -filter {NAME =~ *cdc_*/dst_data_reg*/D}]

## False paths for async FIFO gray-code pointer synchronizers
set_false_path -from [get_pins -quiet -hierarchical -filter {NAME =~ *_async_fifo*/wr_ptr_gray_reg*/C}] \
               -to   [get_pins -quiet -hierarchical -filter {NAME =~ *_async_fifo*/wr_ptr_gray_sync_reg*/D}]
set_false_path -from [get_pins -quiet -hierarchical -filter {NAME =~ *_async_fifo*/rd_ptr_gray_reg*/C}] \
               -to   [get_pins -quiet -hierarchical -filter {NAME =~ *_async_fifo*/rd_ptr_gray_sync_reg*/D}]

## Reset button (active-low)
set_property -dict {PACKAGE_PIN C12 IOSTANDARD LVCMOS33} [get_ports sys_rst_n]

## USB-UART (FTDI FT2232HQ Channel B)
## uart_rx = FPGA receives data FROM FTDI (signal name: uart_txd_in in Digilent XDC)
## uart_tx = FPGA transmits data TO FTDI (signal name: uart_rxd_out in Digilent XDC)
set_property -dict {PACKAGE_PIN C4 IOSTANDARD LVCMOS33} [get_ports uart_rx]
set_property -dict {PACKAGE_PIN D4 IOSTANDARD LVCMOS33} [get_ports uart_tx]

## LEDs (using first 4 of 16 standard LEDs)
set_property -dict {PACKAGE_PIN H17 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN K15 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN J13 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN N14 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

## Configuration bank voltage (Nexys 4 DDR uses 3.3V for config bank 0)
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## NOTE: DDR2 pin constraints are generated automatically by the MIG IP core.
## Do NOT manually constrain DDR2 pins — MIG handles this via its own UCF/XDC.
